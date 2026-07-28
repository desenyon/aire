"""In-process and file-queue workflow workers."""

from __future__ import annotations

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


class FileQueueWorker:
    """Simple durable queue: drop JSON jobs into a directory, drain them."""

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
            job = Job.model_validate(read_json_file(path))
            path.unlink(missing_ok=True)
            result = await self.inner.submit(job.workflow, job.input)
            # Preserve original job id
            result.job.id = job.id
            dest_dir = "done" if result.job.status == "completed" else "failed"
            write_json_file(self.directory / dest_dir / f"{job.id}.json", result.job)
            results.append(result)
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "file_queue_worker",
            "directory": str(self.directory),
            "queued": len(list((self.directory / "queued").glob("*.json"))),
            "workflows": sorted(self.inner.workflows),
        }


WorkerFactory = Callable[..., InProcessWorker | FileQueueWorker]


def create_worker(kind: str = "in_process", **options: Any) -> InProcessWorker | FileQueueWorker:
    if kind in {"in_process", "memory", "local"}:
        return InProcessWorker(**options)
    if kind in {"file", "file_queue"}:
        directory = options.pop("directory", None) or options.pop("path", None)
        if not directory:
            raise ConfigurationError(
                "file worker requires directory=",
                code="workers.directory_missing",
            )
        return FileQueueWorker(directory, **options)
    raise ConfigurationError(
        f"unknown worker kind {kind!r}",
        code="workers.kind_unknown",
        context={"available": ["in_process", "file"]},
    )
