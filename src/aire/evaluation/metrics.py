"""Built-in evaluation metrics.

A metric is any async callable ``(case, output, ctx) -> MetricResult`` where
``ctx`` carries latency, usage, context and (for judges) a model. Register
custom metrics with ``register_metric(name, fn)``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from aire.core.errors import NotFoundError
from aire.evaluation.types import EvalCase, MetricResult
from aire.rag.store import tokenize

if TYPE_CHECKING:
    from aire.models.base import Model


class MetricContext(Protocol):
    """Execution context handed to metrics."""

    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    context: str | None
    judge: Model | None
    embedder: Any | None


MetricFn = Any  # async (EvalCase, str, MetricContext) -> MetricResult


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn, *, replace: bool = False) -> None:
    if name in _REGISTRY and not replace:
        from aire.core.errors import PluginError

        raise PluginError(f"metric {name!r} already registered", code="registry.duplicate")
    _REGISTRY[name] = fn


def get_metric(name: str) -> MetricFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise NotFoundError("metric", name, context={"available": sorted(_REGISTRY)}) from None


def metric_names() -> list[str]:
    return sorted(_REGISTRY)


# -- exact / lexical ---------------------------------------------------------------


async def exact_match(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    expected = (case.expected or "").strip()
    score = 1.0 if output.strip() == expected else 0.0
    return MetricResult(name="exact_match", score=score)


async def contains(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    expected = (case.expected or "").strip().lower()
    score = 1.0 if expected and expected in output.lower() else 0.0
    return MetricResult(name="contains", score=score)


async def semantic_overlap(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """F1 over content tokens between output and expected (embedding-free)."""
    expected_terms = set(tokenize(case.expected or ""))
    output_terms = set(tokenize(output))
    if not expected_terms:
        return MetricResult(name="semantic_overlap", score=0.0)
    overlap = len(expected_terms & output_terms)
    precision = overlap / len(output_terms) if output_terms else 0.0
    recall = overlap / len(expected_terms)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricResult(
        name="semantic_overlap", score=f1, detail={"precision": precision, "recall": recall}
    )


async def json_valid(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    try:
        json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return MetricResult(name="json_valid", score=0.0)
    return MetricResult(name="json_valid", score=1.0)


async def regex_match(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    pattern = case.metadata.get("pattern") or case.expected or ""
    try:
        found = re.search(pattern, output, re.DOTALL) is not None
    except re.error:
        return MetricResult(name="regex_match", score=0.0, detail={"error": "invalid pattern"})
    return MetricResult(name="regex_match", score=1.0 if found else 0.0)


# -- groundedness / retrieval -------------------------------------------------------


async def groundedness(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Fraction of output content tokens supported by the provided context."""
    context = ctx.context or case.context or ""
    context_terms = set(tokenize(context))
    output_terms = [t for t in tokenize(output) if len(t) > 3]
    if not output_terms:
        return MetricResult(name="groundedness", score=1.0, detail={"reason": "empty output"})
    supported = sum(1 for t in output_terms if t in context_terms)
    return MetricResult(
        name="groundedness",
        score=supported / len(output_terms),
        detail={"supported": supported, "total": len(output_terms)},
    )


async def faithfulness(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Sentence-level groundedness: fraction of output sentences supported by context."""
    context = ctx.context or case.context or ""
    context_terms = set(tokenize(context))
    sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
    if not sentences:
        return MetricResult(name="faithfulness", score=1.0, detail={"reason": "empty output"})
    supported = 0
    for sentence in sentences:
        terms = [t for t in tokenize(sentence) if len(t) > 3]
        if not terms:
            supported += 1
            continue
        hit = sum(1 for t in terms if t in context_terms)
        if hit / len(terms) >= 0.4:
            supported += 1
    score = supported / len(sentences)
    return MetricResult(
        name="faithfulness",
        score=score,
        detail={"supported_sentences": supported, "total_sentences": len(sentences)},
    )


async def embedding_similarity(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Cosine similarity between output and expected via ctx.embedder (falls back to lexical F1)."""
    expected = (case.expected or "").strip()
    if not expected:
        return MetricResult(name="embedding_similarity", score=0.0, detail={"error": "no expected"})
    embedder = getattr(ctx, "embedder", None)
    if embedder is None:
        lexical = await semantic_overlap(case, output, ctx)
        return MetricResult(
            name="embedding_similarity",
            score=lexical.score,
            detail={"fallback": "semantic_overlap", **(lexical.detail or {})},
        )
    from aire.rag.store import cosine_similarity

    vectors = await embedder.embed_texts([expected, output])
    score = cosine_similarity(vectors[0], vectors[1])
    # map [-1, 1] → [0, 1]
    normalized = max(0.0, min(1.0, (score + 1.0) / 2.0))
    return MetricResult(
        name="embedding_similarity",
        score=normalized,
        detail={"cosine": score},
    )


async def bleu(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Offline unigram+bigram BLEU-ish F1 against expected (no external deps)."""
    ref = tokenize(case.expected or "")
    hyp = tokenize(output)
    if not ref:
        return MetricResult(name="bleu", score=0.0)
    scores: list[float] = []
    for n in (1, 2):
        ref_ng = _ngrams(ref, n)
        hyp_ng = _ngrams(hyp, n)
        if not ref_ng:
            continue
        ref_counts: dict[tuple[str, ...], int] = {}
        for g in ref_ng:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        hyp_counts: dict[tuple[str, ...], int] = {}
        for g in hyp_ng:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1
        overlap = sum(min(hyp_counts.get(g, 0), c) for g, c in ref_counts.items())
        precision = overlap / len(hyp_ng) if hyp_ng else 0.0
        recall = overlap / len(ref_ng)
        scores.append(_f1(precision, recall))
    score = sum(scores) / len(scores) if scores else 0.0
    return MetricResult(name="bleu", score=score, detail={"orders": len(scores)})


async def rouge_l(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Longest-common-subsequence F1 (ROUGE-L style) against expected."""
    ref = tokenize(case.expected or "")
    hyp = tokenize(output)
    if not ref or not hyp:
        return MetricResult(name="rouge_l", score=0.0)
    # classic DP LCS length
    prev = [0] * (len(hyp) + 1)
    for _i, rt in enumerate(ref, start=1):
        curr = [0]
        for j, ht in enumerate(hyp, start=1):
            if rt == ht:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return MetricResult(
        name="rouge_l",
        score=_f1(precision, recall),
        detail={"lcs": lcs, "precision": precision, "recall": recall},
    )


# -- operational ---------------------------------------------------------------------


async def latency(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """1.0 under the threshold (default 2000ms), decaying linearly to 0 at 10x."""
    threshold = float(case.metadata.get("latency_threshold_ms", 2000.0))
    ratio = ctx.latency_ms / threshold if threshold else 1.0
    score = max(0.0, min(1.0, (10.0 - ratio) / 9.0))
    return MetricResult(name="latency", score=score, detail={"latency_ms": ctx.latency_ms})


async def cost(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """1.0 at zero cost, 0.0 at or beyond the budget (default $0.01/case)."""
    budget = float(case.metadata.get("cost_budget_usd", 0.01))
    score = max(0.0, 1.0 - ctx.cost_usd / budget) if budget else 0.0
    return MetricResult(name="cost", score=score, detail={"cost_usd": ctx.cost_usd})


async def accuracy(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Alias for exact_match when expected is short, else semantic_overlap."""
    if case.expected and len(case.expected) <= 64:
        result = await exact_match(case, output, ctx)
        return MetricResult(name="accuracy", score=result.score)
    result = await semantic_overlap(case, output, ctx)
    return MetricResult(name="accuracy", score=result.score, detail=result.detail)


# -- model-based judging ---------------------------------------------------------------


_JUDGE_PROMPT = (
    "You are a strict evaluator. Score the OUTPUT against the EXPECTED answer "
    "for the QUESTION on a 0-10 scale for {criterion}. "
    "Respond with only the integer score.\n\n"
    "QUESTION: {question}\nEXPECTED: {expected}\nOUTPUT: {output}\nSCORE:"
)


async def model_judge(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Score output with a judge model (requires ctx.judge)."""
    criterion = case.metadata.get("criterion", "correctness and completeness")
    if ctx.judge is None:
        return MetricResult(
            name="model_judge", score=0.0, detail={"error": "no judge model configured"}
        )
    prompt = _JUDGE_PROMPT.format(
        criterion=criterion,
        question=case.input,
        expected=case.expected or "(none provided)",
        output=output,
    )
    text = await ctx.judge.ask(prompt, max_tokens=8)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        raw = float(match.group())
        # accept 0-10 or already-normalized 0-1
        score = raw if raw <= 1.0 else min(10.0, max(0.0, raw)) / 10.0
    else:
        score = 0.0
    return MetricResult(
        name="model_judge", score=score, detail={"raw": text, "criterion": criterion}
    )


def _register_builtins() -> None:
    for name, fn in {
        "exact_match": exact_match,
        "accuracy": accuracy,
        "contains": contains,
        "semantic_overlap": semantic_overlap,
        "embedding_similarity": embedding_similarity,
        "bleu": bleu,
        "rouge_l": rouge_l,
        "json_valid": json_valid,
        "regex_match": regex_match,
        "groundedness": groundedness,
        "faithfulness": faithfulness,
        "latency": latency,
        "cost": cost,
        "model_judge": model_judge,
    }.items():
        register_metric(name, fn, replace=True)


_register_builtins()
