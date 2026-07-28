"""Execution context, budgets, and multimodal content types."""

from __future__ import annotations

import pytest

from aire.core.content import ImageContent, Message, TextContent, coerce_messages
from aire.core.context import Budget, ExecutionContext
from aire.core.errors import BudgetExceededError, DataError


def test_budget_token_limit() -> None:
    from aire.core.types import Usage

    budget = Budget(max_tokens=100)
    budget.check(Usage(input_tokens=60, output_tokens=30), steps=0, started=0)
    with pytest.raises(BudgetExceededError):
        budget.check(Usage(input_tokens=60, output_tokens=50), steps=0, started=0)


def test_budget_step_limit() -> None:
    budget = Budget(max_steps=2)
    budget.check(Usage_zero(), steps=2, started=0)
    with pytest.raises(BudgetExceededError):
        budget.check(Usage_zero(), steps=3, started=0)


def Usage_zero():
    from aire.core.types import Usage

    return Usage()


def test_context_records_usage_and_ticks() -> None:
    from aire.core.types import Usage

    ctx = ExecutionContext(budget=Budget(max_tokens=10, max_steps=2))
    ctx.tick()
    ctx.record_usage(Usage(input_tokens=4, output_tokens=4))
    with pytest.raises(BudgetExceededError):
        ctx.tick()
        ctx.tick()


def test_context_child_inherits() -> None:
    parent = ExecutionContext(permissions={"a", "b"}, user_id="u1")
    child = parent.child(extra="yes")
    assert child.permissions == {"a", "b"}
    assert child.user_id == "u1"
    assert child.parent_id == parent.run_id
    assert child.metadata["extra"] == "yes"


def test_ambient_context() -> None:
    with ExecutionContext() as ctx:
        assert ExecutionContext.current() is ctx
    fresh = ExecutionContext.current()
    assert fresh is not ctx


def test_coerce_messages() -> None:
    assert coerce_messages("hi")[0].role == "user"
    msg = Message.text("system", "rule")
    assert coerce_messages(msg) == [msg]
    assert coerce_messages([msg])[0].text_content == "rule"


def test_message_text_content() -> None:
    msg = Message(role="user", content=[TextContent(text="a"), TextContent(text="b")])
    assert msg.text_content == "ab"
    assert msg.describe()["kinds"] == ["text", "text"]


def test_image_content_from_file(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "pixel.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    image = ImageContent.from_file(path)
    assert image.data is not None
    assert image.media_type == "image/png"
    assert image.as_base64()


def test_image_content_missing_file() -> None:
    with pytest.raises(DataError):
        ImageContent.from_file("/nonexistent/image.png")


def test_usage_addition() -> None:
    from aire.core.types import Usage

    total = Usage(input_tokens=1, output_tokens=2, cost_usd=0.1) + Usage(
        input_tokens=3, output_tokens=4, cost_usd=0.2
    )
    assert total.total_tokens == 10
    assert total.cost_usd == pytest.approx(0.3)
