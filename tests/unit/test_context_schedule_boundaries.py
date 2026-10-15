import asyncio
import math

import pytest

from aire.core.context import ExecutionContext
from aire.core.errors import ConfigurationError
from aire.models.base import run_sync
from aire.schedule.scheduler import Scheduler
from aire.workflows.graph import Workflow


def test_reentrant_context_restores_parent_after_exception():
    parent, context = ExecutionContext(), ExecutionContext()
    with parent:
        with context:
            with pytest.raises(ValueError), context:
                assert ExecutionContext.current() is context
                raise ValueError("expected")
            assert ExecutionContext.current() is context
        assert ExecutionContext.current() is parent


def test_shared_context_restores_separate_overlapping_tasks():
    async def check():
        shared = ExecutionContext()
        first_entered, second_entered, first_exited = (asyncio.Event() for _ in range(3))

        async def first():
            parent = ExecutionContext()
            with parent:
                with shared:
                    first_entered.set()
                    await second_entered.wait()
                assert ExecutionContext.current() is parent
                first_exited.set()

        async def second():
            await first_entered.wait()
            parent = ExecutionContext()
            with parent:
                with shared:
                    second_entered.set()
                    await first_exited.wait()
                    assert ExecutionContext.current() is shared
                assert ExecutionContext.current() is parent

        await asyncio.wait_for(asyncio.gather(first(), second()), timeout=1)

    run_sync(check())


def test_schedule_replacement_does_not_call_old_handler():
    async def check():
        calls = []

        async def old():
            calls.append("old")

        scheduler = Scheduler()
        scheduler.every(1, old, name="job")
        scheduler.register_workflow("new", Workflow().add("new", lambda value, ctx: "new result"))
        scheduler.every(1, "new", name="job")
        records = await scheduler.tick(now=1)
        assert calls == []
        assert records[0]["ok"] and records[0]["output"] == "new result"

    run_sync(check())


def test_failed_registration_keeps_existing_handler():
    async def check():
        calls = []

        async def good():
            calls.append("good")

        async def bad():
            calls.append("bad")

        scheduler = Scheduler()
        scheduler.every(1, good, name="job")
        with pytest.raises(ValueError):
            scheduler.every(-1, bad, name="job")
        assert (await scheduler.tick(now=1))[0]["ok"]
        assert calls == ["good"]

    run_sync(check())


@pytest.mark.parametrize("interval", [math.inf, math.nan, 0, -1])
def test_schedule_rejects_nonfinite_or_nonpositive_interval(interval):
    with pytest.raises(ValueError):
        Scheduler().every(interval, "workflow")


@pytest.mark.parametrize("period", [math.inf, math.nan, 0, -1])
def test_polling_rejects_nonfinite_or_nonpositive_period(period):
    with pytest.raises(ConfigurationError):
        run_sync(Scheduler().run_forever(poll_seconds=period, max_ticks=0))


def test_completed_contexts_do_not_accumulate_ambient_bindings():
    import contextvars
    import gc
    import weakref

    # Establish the shared context variables before measuring growth.
    with ExecutionContext():
        pass
    before = len(contextvars.copy_context())
    references = []
    for _ in range(1000):
        context = ExecutionContext()
        references.append(weakref.ref(context))
        with context:
            pass
    del context
    gc.collect()
    assert len(contextvars.copy_context()) == before
    assert all(reference() is None for reference in references)
