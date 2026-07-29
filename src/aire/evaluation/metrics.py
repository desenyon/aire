"""Built-in evaluation metrics.

A metric is any async callable ``(case, output, ctx) -> MetricResult`` where
``ctx`` carries latency, usage, context and (for judges) a model. Register
custom metrics with ``register_metric(name, fn)``.
"""

from __future__ import annotations

import json
import math
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
    """Lexical token-set F1 between output and expected (not embedding/semantic similarity).

    Public id kept as ``semantic_overlap`` for API stability; prefer ``token_overlap``.
    """
    expected_terms = set(tokenize(case.expected or ""))
    output_terms = set(tokenize(output))
    if not expected_terms:
        return MetricResult(name="semantic_overlap", score=0.0)
    overlap = len(expected_terms & output_terms)
    precision = overlap / len(output_terms) if output_terms else 0.0
    recall = overlap / len(expected_terms)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricResult(
        name="semantic_overlap",
        score=f1,
        detail={"precision": precision, "recall": recall, "kind": "lexical_token_f1"},
    )


async def token_overlap(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Honest alias for lexical token-set F1 (same as ``semantic_overlap``)."""
    result = await semantic_overlap(case, output, ctx)
    return MetricResult(name="token_overlap", score=result.score, detail=result.detail)


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
    """Lexical token overlap: fraction of output tokens (len>3) present in context.

    Not an NLI/entailment groundedness judge — offline lexical support only.
    """
    context = ctx.context or case.context or ""
    context_terms = set(tokenize(context))
    output_terms = [t for t in tokenize(output) if len(t) > 3]
    if not output_terms:
        return MetricResult(name="groundedness", score=1.0, detail={"reason": "empty output"})
    supported = sum(1 for t in output_terms if t in context_terms)
    return MetricResult(
        name="groundedness",
        score=supported / len(output_terms),
        detail={
            "supported": supported,
            "total": len(output_terms),
            "kind": "lexical_token_support",
        },
    )


async def faithfulness(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Sentence-level lexical support: fraction of sentences with ≥40% tokens in context.

    Not model-based faithfulness/NLI — offline lexical heuristic only.
    Prefer :func:`nli_faithfulness` when a judge model is available.
    """
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
        detail={
            "supported_sentences": supported,
            "total_sentences": len(sentences),
            "kind": "lexical_sentence_support",
        },
    )


_NLI_PROMPT = (
    "Does the CONTEXT entail the CLAIM? Answer with only YES or NO.\n\n"
    "CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nANSWER:"
)


async def nli_faithfulness(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Model-based faithfulness: fraction of output sentences entailed by context.

    Requires ``ctx.judge``. Does not silently fall back to lexical scoring.
    """
    from aire.core.errors import ConfigurationError

    if getattr(ctx, "judge", None) is None:
        raise ConfigurationError(
            "nli_faithfulness requires ctx.judge; pass judge= to EvalRunner",
            code="eval.judge_required",
        )
    context = (ctx.context or case.context or "").strip()
    if not context:
        return MetricResult(
            name="nli_faithfulness",
            score=0.0,
            detail={"error": "no context", "kind": "nli_entailment"},
        )
    sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
    if not sentences:
        return MetricResult(
            name="nli_faithfulness",
            score=1.0,
            detail={"reason": "empty output", "kind": "nli_entailment"},
        )
    entailed = 0
    for claim in sentences:
        prompt = _NLI_PROMPT.format(context=context[:6000], claim=claim[:1000])
        judge = ctx.judge
        assert judge is not None
        text = (await judge.ask(prompt, max_tokens=4) or "").strip().lower()
        if text.startswith("y") or "yes" in text.split()[:2]:
            entailed += 1
    score = entailed / len(sentences)
    return MetricResult(
        name="nli_faithfulness",
        score=score,
        detail={
            "entailed_sentences": entailed,
            "total_sentences": len(sentences),
            "kind": "nli_entailment",
        },
    )


async def embedding_similarity(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Cosine similarity between output and expected via ``ctx.embedder``.

    Requires an embedder on the evaluation context — does not silently fall back
    to lexical F1 (that would inflate scores under a semantic label).
    """
    from aire.core.errors import ConfigurationError

    expected = (case.expected or "").strip()
    if not expected:
        return MetricResult(name="embedding_similarity", score=0.0, detail={"error": "no expected"})
    embedder = getattr(ctx, "embedder", None)
    if embedder is None:
        raise ConfigurationError(
            "embedding_similarity requires ctx.embedder; pass embedder= to EvalRunner",
            code="eval.embedder_required",
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


def _count_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for g in _ngrams(tokens, n):
        counts[g] = counts.get(g, 0) + 1
    return counts


def _modified_precision(hyp: list[str], ref: list[str], n: int) -> tuple[int, int]:
    """Return (clipped_overlap, hyp_ngram_count) for BLEU modified precision."""
    hyp_counts = _count_ngrams(hyp, n)
    ref_counts = _count_ngrams(ref, n)
    if not hyp_counts:
        return 0, 0
    clipped = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
    return clipped, sum(hyp_counts.values())


def _bleu4(hyp: list[str], ref: list[str]) -> tuple[float, dict[str, Any]]:
    """Corpus-style sentence BLEU-4 with brevity penalty (Papineni et al.)."""
    if not ref:
        return 0.0, {"error": "empty reference"}
    if not hyp:
        return 0.0, {"brevity_penalty": 0.0, "kind": "bleu4_with_bp"}
    precisions: list[float] = []
    for n in range(1, 5):
        clipped, total = _modified_precision(hyp, ref, n)
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)
    # geometric mean; if any order is zero, BLEU is zero
    if any(p <= 0.0 for p in precisions):
        geo = 0.0
    else:
        log_sum = sum(math.log(p) for p in precisions) / 4.0
        geo = math.exp(log_sum)
    ref_len = len(ref)
    hyp_len = len(hyp)
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len:
        bp = math.exp(1.0 - ref_len / hyp_len)
    else:
        bp = 0.0
    score = bp * geo
    return score, {
        "precisions": precisions,
        "brevity_penalty": bp,
        "kind": "bleu4_with_bp",
    }


async def bleu(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Sentence BLEU-4 with brevity penalty against ``expected``.

    Pure-Python Papineni BLEU (orders 1-4). Prefer ``sacrebleu`` metric when the
    optional ``aire[eval]`` extra is installed for sacreBLEU tokenization.
    """
    ref = tokenize(case.expected or "")
    hyp = tokenize(output)
    score, detail = _bleu4(hyp, ref)
    return MetricResult(name="bleu", score=score, detail=detail)


async def bleu_approx(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Legacy unigram/bigram F1 approximation (pre-0.3.5 ``bleu`` behaviour)."""
    ref = tokenize(case.expected or "")
    hyp = tokenize(output)
    if not ref:
        return MetricResult(name="bleu_approx", score=0.0)
    scores: list[float] = []
    for n in (1, 2):
        ref_ng = _ngrams(ref, n)
        hyp_ng = _ngrams(hyp, n)
        if not ref_ng:
            continue
        ref_counts = _count_ngrams(ref, n)
        hyp_counts = _count_ngrams(hyp, n)
        overlap = sum(min(hyp_counts.get(g, 0), c) for g, c in ref_counts.items())
        precision = overlap / len(hyp_ng) if hyp_ng else 0.0
        recall = overlap / len(ref_ng)
        scores.append(_f1(precision, recall))
    score = sum(scores) / len(scores) if scores else 0.0
    return MetricResult(
        name="bleu_approx",
        score=score,
        detail={"orders": len(scores), "kind": "approx_unigram_bigram_f1"},
    )


async def sacrebleu(case: EvalCase, output: str, ctx: MetricContext) -> MetricResult:
    """Official sacreBLEU score (requires ``pip install 'aire[eval]'``)."""
    from aire.core.errors import ConfigurationError

    try:
        import sacrebleu as sb
    except ImportError as exc:
        raise ConfigurationError(
            "sacrebleu metric requires the eval extra: pip install 'aire[eval]'",
            code="eval.sacrebleu_missing",
        ) from exc
    expected = case.expected or ""
    bleu_obj = sb.corpus_bleu([output], [[expected]])
    score = float(bleu_obj.score) / 100.0
    return MetricResult(
        name="sacrebleu",
        score=max(0.0, min(1.0, score)),
        detail={"sacrebleu": float(bleu_obj.score), "kind": "sacrebleu"},
    )


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
    """Score output with a judge model (requires ``ctx.judge``).

    Raises :class:`~aire.core.errors.ConfigurationError` when no judge is configured
    instead of returning a misleading 0.0 score.
    """
    from aire.core.errors import ConfigurationError

    criterion = case.metadata.get("criterion", "correctness and completeness")
    if ctx.judge is None:
        raise ConfigurationError(
            "model_judge requires ctx.judge; pass judge= to EvalRunner",
            code="eval.judge_required",
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
        "token_overlap": token_overlap,
        "embedding_similarity": embedding_similarity,
        "bleu": bleu,
        "bleu_approx": bleu_approx,
        "sacrebleu": sacrebleu,
        "rouge_l": rouge_l,
        "json_valid": json_valid,
        "regex_match": regex_match,
        "groundedness": groundedness,
        "faithfulness": faithfulness,
        "nli_faithfulness": nli_faithfulness,
        "latency": latency,
        "cost": cost,
        "model_judge": model_judge,
    }.items():
        register_metric(name, fn, replace=True)


_register_builtins()
