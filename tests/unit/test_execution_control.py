import asyncio
import threading

import pytest

from aire.agents.runtime import AgentExecutor
from aire.agents.streaming import run_stream, stream_tokens
from aire.agents.types import AgentConfig
from aire.core.context import ExecutionContext
from aire.core.errors import BudgetExceededError
from aire.models.base import run_sync
from aire.models.builtin import EchoModel
from aire.tools.registry import ToolRegistry
from aire.tools.tool import Tool
from aire.tools.types import RetryPolicy


class CountModel(EchoModel):
    calls = 0

    async def generate(self, request):
        self.calls += 1
        return await super().generate(request)

    async def stream(self, request):
        self.calls += 1
        async for chunk in super().stream(request):
            yield chunk


@pytest.mark.parametrize("mode", ["run", "events", "tokens"])
def test_zero_budget_never_invokes_provider(mode):
    async def check():
        model = CountModel()
        executor = AgentExecutor(model, ToolRegistry(), config=AgentConfig(token_budget=0))
        if mode == "run":
            assert (await executor.run("hello")).status == "budget_exceeded"
        elif mode == "events":
            events = [event async for event in run_stream(executor, "hello")]
            assert events[-1].status == "budget_exceeded"
        else:
            with pytest.raises(BudgetExceededError):
                _ = [part async for part in stream_tokens(executor, "hello")]
        assert model.calls == 0

    run_sync(check())


def test_pre_cancelled_agent_makes_no_provider_call():
    async def check():
        model = CountModel()
        ctx = ExecutionContext()
        ctx.cancel()
        result = await AgentExecutor(model, ToolRegistry()).run("hello", context=ctx)
        assert result.status == "cancelled"
        assert model.calls == 0

    run_sync(check())


def test_sync_tool_timeout_keeps_loop_responsive_and_never_retries():
    async def check():
        release = threading.Event()
        started = threading.Event()
        calls = []

        def blocking_write():
            calls.append(1)
            started.set()
            release.wait(timeout=1)
            return "done"

        task = asyncio.create_task(
            Tool(blocking_write, timeout_seconds=0.02, retry=RetryPolicy(attempts=3)).execute()
        )
        try:
            await asyncio.sleep(0.005)
            assert started.is_set()
            assert not task.done()
            result = await task
            assert not result.ok
            assert "timeout" in result.error_code
            assert calls == [1]
        finally:
            release.set()

    run_sync(check())


def test_event_stream_and_run_return_provider_failure():
    class Broken(EchoModel):
        async def generate(self, request):
            raise RuntimeError("provider stopped")

    async def check():
        executor = AgentExecutor(Broken(), ToolRegistry())
        result = await executor.run("hello")
        assert result.status == "failed"
        assert "provider stopped" in result.error
        events = [event async for event in run_stream(executor, "hello")]
        assert events[-1].status == "failed"
        assert "provider stopped" in events[-1].error

    run_sync(check())


def test_cancellation_between_tools_prevents_second_effect():
    from aire.models.types import ToolCall

    async def check():
        context = ExecutionContext()
        calls = []

        async def first():
            context.cancel()
            calls.append("first")

        async def second():
            calls.append("second")

        registry = ToolRegistry()
        registry.register(Tool(first))
        registry.register(Tool(second))
        model = EchoModel()
        model.scripted_tool_calls = [
            ToolCall(id="a", name="first"),
            ToolCall(id="b", name="second"),
        ]
        result = await AgentExecutor(model, registry).run("go", context=context)
        assert result.status == "cancelled"
        assert calls == ["first"]

    run_sync(check())


def test_closing_token_stream_closes_provider():
    class Model(EchoModel):
        closed = False

        async def stream(self, request):
            from aire.models.types import GenerationChunk

            try:
                yield GenerationChunk(text="one")
                yield GenerationChunk(text="two")
            finally:
                self.closed = True

    async def check():
        model = Model()
        stream = stream_tokens(AgentExecutor(model, ToolRegistry()), "go")
        assert await anext(stream) == "one"
        await stream.aclose()
        assert model.closed

    run_sync(check())
