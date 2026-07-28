"""The Evaluator: run a target against a dataset of cases and score it.

Targets are anything that maps ``str -> Awaitable[str]`` — an agent, a
Knowledge pipeline, a model, or a plain function. Results preserve the full
execution evidence (input, output, expected, scores, usage, latency, errors).
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from aire.core.errors import DataError, ensure_aire_error
from aire.core.serialization import iter_jsonl, write_json_file
from aire.evaluation.metrics import get_metric
from aire.evaluation.types import CaseResult, EvalCase, EvalReport
from aire.models.base import Model

Target = Callable[[str], str | Awaitable[str]] | Any


class _Ctx:
    def __init__(
        self,
        *,
        latency_ms: float,
        usage: Any,
        context: str | None,
        judge: Model | None,
        embedder: Any | None = None,
    ) -> None:
        self.latency_ms = latency_ms
        self.input_tokens = getattr(usage, "input_tokens", 0)
        self.output_tokens = getattr(usage, "output_tokens", 0)
        self.cost_usd = getattr(usage, "cost_usd", 0.0)
        self.context = context
        self.judge = judge
        self.embedder = embedder


def load_cases(source: str | Path | list[dict[str, Any]] | list[EvalCase]) -> list[EvalCase]:
    """Load eval cases from a JSONL file or in-memory values."""
    if isinstance(source, list):
        if source and isinstance(source[0], EvalCase):
            return list(source)
        return [EvalCase.model_validate(row) for row in cast("list[dict[str, Any]]", source)]
    path = Path(source)
    if not path.is_file():
        raise DataError(f"evaluation file not found: {path}", context={"path": str(path)})
    return [EvalCase.model_validate(row) for row in iter_jsonl(path)]


class Evaluator:
    """Runs evaluation suites and produces reports."""

    def __init__(
        self,
        *,
        judge: Model | None = None,
        embedder: Any | None = None,
        name: str = "evaluation",
    ) -> None:
        self.judge = judge
        self.embedder = embedder
        self.name = name

    async def run(
        self,
        target: Target,
        dataset: str | Path | list[dict[str, Any]] | list[EvalCase],
        *,
        metrics: list[str] | None = None,
        concurrency: int = 4,
    ) -> EvalReport:
        """Evaluate ``target`` on every case with the requested metrics."""
        import asyncio

        cases = load_cases(dataset)
        metric_fns = [get_metric(m) for m in (metrics or ["accuracy"])]
        report = EvalReport(name=self.name, target=_target_name(target))
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run_case(case: EvalCase) -> CaseResult:
            async with semaphore:
                return await self._evaluate_case(target, case, metric_fns)

        results = await asyncio.gather(*(_run_case(c) for c in cases))
        report.results = list(results)
        report.finished = time.time()
        return report

    async def _evaluate_case(
        self, target: Target, case: EvalCase, metric_fns: list[Any]
    ) -> CaseResult:
        started = time.perf_counter()
        output = ""
        usage: Any = None
        context: str | None = case.context
        error: str | None = None
        error_category: str | None = None
        try:
            output, usage, context = await _invoke(target, case.input, context)
        except Exception as exc:
            err = ensure_aire_error(exc)
            error = err.message
            error_category = err.code
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = CaseResult(
            case=case,
            output=output,
            usage=usage if usage is not None else _zero_usage(),
            latency_ms=latency_ms,
            error=error,
            error_category=error_category,
        )
        if error is None:
            ctx = _Ctx(
                latency_ms=latency_ms,
                usage=result.usage,
                context=context,
                judge=self.judge,
                embedder=self.embedder,
            )
            for metric_fn in metric_fns:
                try:
                    result.metrics.append(await metric_fn(case, output, ctx))
                except Exception as exc:
                    from aire.evaluation.types import MetricResult

                    result.metrics.append(
                        MetricResult(
                            name=getattr(metric_fn, "__name__", "metric"),
                            score=0.0,
                            detail={"error": str(exc)},
                        )
                    )
        return result

    def run_sync(
        self,
        target: Target,
        dataset: str | Path | list[dict[str, Any]] | list[EvalCase],
        **kwargs: Any,
    ) -> EvalReport:
        from aire.models.base import run_sync

        return run_sync(self.run(target, dataset, **kwargs))


async def _invoke(target: Target, input: str, context: str | None) -> tuple[str, Any, str | None]:
    """Normalize different target kinds to (output, usage, context)."""
    # Model: generate directly
    if isinstance(target, Model):
        from aire.models.types import GenerationRequest

        gen = await target.generate(GenerationRequest.of(input))
        return gen.text, gen.usage, context
    # Agent-like / Workflow-like: .run(input) -> result with .output and .usage
    run = getattr(target, "run", None)
    if callable(run):
        result = await run(input)
        return str(getattr(result, "output", result)), getattr(result, "usage", None), context
    # Knowledge-like: .ask(question) -> Answer with citations
    ask = getattr(target, "ask", None)
    if callable(ask):
        answer = await ask(input)
        text = getattr(answer, "text", answer)
        usage = getattr(answer, "usage", None)
        citations = getattr(answer, "citations", None)
        ctx = "\n".join(c.excerpt for c in citations) if citations else context
        return str(text), usage, ctx
    # Plain callable
    if callable(target):
        outcome = target(input)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return str(outcome), None, context
    raise DataError(f"unsupported evaluation target type: {type(target).__name__}")


def _target_name(target: Target) -> str:
    if isinstance(target, Model):
        return target.info.ref
    return getattr(target, "name", type(target).__name__)


def _zero_usage() -> Any:
    from aire.core.types import Usage

    return Usage()


def save_report(report: EvalReport, path: str | Path) -> Path:
    """Persist a report as JSON."""
    return write_json_file(path, report)
