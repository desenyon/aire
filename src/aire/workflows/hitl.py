"""Workflow human-in-the-loop helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from aire.workflows.graph import Workflow
from aire.workflows.types import NodeFn

ApproverFn = Callable[[str], bool | Awaitable[bool]]


class NodeInteractiveApprover:
    """Prompt on stdin for workflow node approval (y/n/always/never)."""

    def __init__(self, *, auto: dict[str, bool] | None = None) -> None:
        self._remembered: dict[str, bool] = dict(auto or {})

    async def __call__(self, node_name: str) -> bool:
        if node_name in self._remembered:
            return self._remembered[node_name]
        return await asyncio.to_thread(self._prompt, node_name)

    def _prompt(self, node_name: str) -> bool:
        print(f"\n[aire] workflow approval requested for node {node_name!r}")
        while True:
            answer = input("  allow? [y]es / [n]o / [a]lways / n[e]ver: ").strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            if answer in {"a", "always"}:
                self._remembered[node_name] = True
                return True
            if answer in {"e", "never"}:
                self._remembered[node_name] = False
                return False


def always_approve(node_name: str) -> bool:
    return True


def always_deny(node_name: str) -> bool:
    return False


def hitl_node(
    workflow: Workflow,
    name: str,
    fn: NodeFn,
    *,
    retries: int = 0,
    timeout_seconds: float | None = None,
) -> Workflow:
    """Add a node that requires human approval before execution."""
    return workflow.add(
        name,
        fn,
        retries=retries,
        timeout_seconds=timeout_seconds,
        requires_approval=True,
    )


def describe() -> dict[str, Any]:
    return {
        "kind": "workflow_hitl",
        "helpers": ["hitl_node", "NodeInteractiveApprover", "always_approve"],
    }
