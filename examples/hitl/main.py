"""HITL approvals: RuleApprover decides tool calls by side-effect severity.

Offline — no stdin prompts. For interactive HITL use
``AI.agents.approver("interactive")``.
"""

from __future__ import annotations

from aire import AI
from aire.models.types import ToolCall
from aire.tools.types import SideEffect, ToolSpec


def main() -> None:
    approver = AI.agents.approver(
        "rule",
        auto_approve_below=SideEffect.REVERSIBLE_WRITE,
        deny={"send_email"},
    )
    specs = [
        ToolSpec(name="calculator", side_effect=SideEffect.READ_ONLY),
        ToolSpec(name="write_file", side_effect=SideEffect.REVERSIBLE_WRITE),
        ToolSpec(name="send_email", side_effect=SideEffect.EXTERNAL_SIDE_EFFECT),
        ToolSpec(name="drop_db", side_effect=SideEffect.HIGH_IMPACT),
    ]
    for spec in specs:
        call = ToolCall(id="c1", name=spec.name, arguments={})
        ok = approver(call, spec)
        print(f"{spec.name:12} {str(spec.side_effect):22} approved={ok}")
    print("audit:", approver.decisions)
    print("for stdin HITL:", 'AI.agents.approver("interactive")')


if __name__ == "__main__":
    main()
