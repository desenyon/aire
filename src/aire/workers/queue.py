"""In-process, local file-queue, and Redis workflow workers.

``FileQueueWorker`` is intentionally **local-only**: claim uses an atomic
rename (or ``O_EXCL`` lock file) on a single filesystem. Do not use it as a
distributed queue across hosts or network mounts without a real broker.

``RedisQueueWorker`` uses a Redis list (``LPUSH`` / ``BRPOP``) and requires
``aire[redis]``. SQS is not bundled — see :class:`SQSQueueWorker`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.core.errors import ConfigurationError
from aire.core.serialization import read_json_file, write_json_file
from aire.workflows.graph import Workflow
from aire.workflows.types import WorkflowResult


class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow: str
    input: Any = None
    status: str = "queued"  # queued | running | completed | failed
    result: Any = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None


class WorkerResult(BaseModel):
    job: Job
    workflow_result: WorkflowResult | None = None


class InProcessWorker:
    """Run workflows immediately in the calling process."""

    def __init__(self, workflows: dict[str, Workflow] | None = None) -> None:
        self.workflows: dict[str, Workflow] = dict(workflows or {})
        self.history: list[Job] = []

    def register(self, name: str, workflow: Workflow) -> None:
        self.workflows[name] = workflow

    async def submit(self, workflow: str | Workflow, input: Any = None) -> WorkerResult:
        if isinstance(workflow, Workflow):
            name = workflow.name
            wf = workflow
        else:
            name = workflow
            if name not in self.workflows:
                raise ConfigurationError(
                    f"unknown workflow {name!r}",
                    code="workers.workflow_missing",
                    context={"available": sorted(self.workflows)},
                )
            wf = self.workflows[name]
        job = Job(workflow=name, input=input, status="running")
        try:
            result = await wf.run(input)
            job.status = "completed" if result.ok else "failed"
            job.result = result.output
            if not result.ok:
                job.error = result.error
            job.finished_at = time.time()
            self.history.append(job)
            return WorkerResult(job=job, workflow_result=result)
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()
            self.history.append(job)
            return WorkerResult(job=job)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "in_process_worker",
            "workflows": sorted(self.workflows),
            "jobs": len(self.history),
        }


def _claim_job_file(path: Path) -> Path | None:
    """Atomically claim a queued job for local-only workers.

    Prefer rename into a ``.claimed`` sibling; fall back to ``O_EXCL`` lock file
    then rename. Returns the claimed path, or ``None`` if another worker won.
    """
    claimed = path.with_suffix(path.suffix + ".claimed")
    try:
        os.rename(path, claimed)
        return claimed
    except FileNotFoundError:
        return None
    except OSError:
        # Fallback: exclusive lock file then rename
        lock = path.with_suffix(path.suffix + ".lock")
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return None
        try:
            os.rename(path, claimed)
            return claimed
        except FileNotFoundError:
            return None
        finally:
            lock.unlink(missing_ok=True)


class FileQueueWorker:
    """Simple durable local file queue: drop JSON jobs into a directory, drain them.

    Local-only — not safe across hosts or network filesystems without a broker.
    """

    def __init__(
        self,
        directory: str | Path,
        workflows: dict[str, Workflow] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "queued").mkdir(exist_ok=True)
        (self.directory / "done").mkdir(exist_ok=True)
        (self.directory / "failed").mkdir(exist_ok=True)
        self.inner = InProcessWorker(workflows)

    def register(self, name: str, workflow: Workflow) -> None:
        self.inner.register(name, workflow)

    def enqueue(self, workflow: str, input: Any = None) -> Job:
        job = Job(workflow=workflow, input=input)
        path = self.directory / "queued" / f"{job.id}.json"
        write_json_file(path, job)
        return job

    async def drain(self, *, max_jobs: int = 10) -> list[WorkerResult]:
        results: list[WorkerResult] = []
        queued = sorted((self.directory / "queued").glob("*.json"))[:max_jobs]
        for path in queued:
            claimed = _claim_job_file(path)
            if claimed is None:
                continue
            try:
                job = Job.model_validate(read_json_file(claimed))
                result = await self.inner.submit(job.workflow, job.input)
                result.job.id = job.id
                dest_dir = "done" if result.job.status == "completed" else "failed"
                write_json_file(self.directory / dest_dir / f"{job.id}.json", result.job)
                results.append(result)
            finally:
                claimed.unlink(missing_ok=True)
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local_file_queue_worker",
            "local_only": True,
            "directory": str(self.directory),
            "queued": len(list((self.directory / "queued").glob("*.json"))),
            "workflows": sorted(self.inner.workflows),
            "claim": "atomic rename / O_EXCL lock",
        }


def _require_redis() -> Any:
    if importlib.util.find_spec("redis") is None:
        raise ConfigurationError(
            "redis is required for RedisQueueWorker: pip install 'aire[redis]'",
            code="workers.redis_missing",
            context={"extra": "aire[redis]", "package": "redis"},
        )
    import redis  # type: ignore[import-not-found]

    return redis


class RedisQueueWorker:
    """Distributed job queue backed by a Redis list (``LPUSH`` / ``BRPOP``).

    Requires ``pip install 'aire[redis]'``. Job payloads are JSON-serialized
    :class:`Job` objects.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        key: str = "aire:jobs",
        workflows: dict[str, Workflow] | None = None,
        client: Any | None = None,
    ) -> None:
        redis_mod = _require_redis()
        self.url = url
        self.key = key
        self.inner = InProcessWorker(workflows)
        self._client = client or redis_mod.Redis.from_url(url, decode_responses=True)

    def register(self, name: str, workflow: Workflow) -> None:
        self.inner.register(name, workflow)

    def enqueue(self, workflow: str, input: Any = None) -> Job:
        job = Job(workflow=workflow, input=input)
        payload = job.model_dump_json()
        self._client.lpush(self.key, payload)
        return job

    async def process_one(self, *, block_seconds: float = 1.0) -> WorkerResult | None:
        """Block up to ``block_seconds`` for one job via ``BRPOP``."""
        block = max(1, int(block_seconds))
        popped = await asyncio.to_thread(self._client.brpop, self.key, block)
        if popped is None:
            return None
        _key, raw = popped
        job = Job.model_validate(json.loads(raw))
        result = await self.inner.submit(job.workflow, job.input)
        result.job.id = job.id
        result.job.created_at = job.created_at
        return result

    async def drain(
        self, *, max_jobs: int = 10, block_seconds: float = 1.0
    ) -> list[WorkerResult]:
        results: list[WorkerResult] = []
        for _ in range(max_jobs):
            one = await self.process_one(block_seconds=block_seconds)
            if one is None:
                break
            results.append(one)
        return results

    def describe(self) -> dict[str, Any]:
        try:
            depth = int(self._client.llen(self.key))
        except Exception:
            depth = -1
        return {
            "kind": "redis_queue_worker",
            "url": self.url,
            "key": self.key,
            "queued": depth,
            "workflows": sorted(self.inner.workflows),
            "transport": "redis list LPUSH/BRPOP",
        }


class SQSQueueWorker:
    """SQS backend is not bundled — raises with an install / alternative hint."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ConfigurationError(
            "SQS worker requires boto3 and is not bundled; use redis or file workers",
            code="workers.sqs_unavailable",
            context={"alternatives": ["redis", "file", "in_process"]},
        )


WorkerFactory = Callable[..., InProcessWorker | FileQueueWorker | RedisQueueWorker]


def create_worker(
    kind: str = "in_process", **options: Any
) -> InProcessWorker | FileQueueWorker | RedisQueueWorker:
    if kind in {"in_process", "memory", "local"}:
        return InProcessWorker(**options)
    if kind in {"file", "file_queue", "local_file"}:
        directory = options.pop("directory", None) or options.pop("path", None)
        if not directory:
            raise ConfigurationError(
                "file worker requires directory=",
                code="workers.directory_missing",
            )
        return FileQueueWorker(directory, **options)
    if kind in {"redis", "redis_queue"}:
        url = options.pop("url", None) or "redis://localhost:6379/0"
        return RedisQueueWorker(url, **options)
    if kind in {"sqs", "sqs_queue"}:
        raise ConfigurationError(
            "SQS worker requires boto3 and is not bundled; use redis or file workers",
            code="workers.sqs_unavailable",
            context={"alternatives": ["redis", "file", "in_process"]},
        )
    raise ConfigurationError(
        f"unknown worker kind {kind!r}",
        code="workers.kind_unknown",
        context={"available": ["in_process", "file", "redis", "sqs"]},
    )
