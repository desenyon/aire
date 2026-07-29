# Agents

Agents combine a **model**, **tools**, optional **memory/session**, **approver**, and **policy**. Prefer `AI.agents.create_sync(...)` or the fluent builder.

## Create

```python
from aire import AI, AgentConfig

agent = AI.agents.create_sync(
    "mock:echo",
    tools=[],           # Tool objects or registered names
    builtins=False,     # include calculator, http_*, web_search, …
    config=AgentConfig(max_steps=8, planning=False),
    name="demo",
)
result = agent.run_sync("Say hello")
assert result.ok
print(result.output, result.status, len(result.steps))
```

### `AgentConfig` defaults

| Field | Default |
|-------|---------|
| `max_steps` | `12` |
| `planning` | `False` |
| `approval_levels` | `["external_side_effect", "high_impact", "prohibited"]` |
| `parallel_tools` | `False` |

When `planning=True` (or `run(..., use_planning=True)`), the agent delegates to `PlanActVerify` (plan → act → verify). Nested runs pass `use_planning=False` to avoid recursion.

## Builder & patterns

```python
agent = (
    AI.agents.builder("research")
    .model("mock:echo")
    .system("Be careful.")
    .tools(["calculator"])
    .planning(False)
    .build_sync()
)

# Named patterns: research | coder | critic | planner | rag
agent = AI.agents.pattern("coder").model("mock:echo").build_sync()
```

Toolkits: `AI.agents.toolkit("web"|"code"|"data"|"filesystem")`.

Approvers: `AI.agents.approver("rule"|"interactive"|"workflow")`.

## Sessions

`DurableSession` persists goal, messages, steps, and result to JSON:

```python
from aire.agents.session import DurableSession
from aire.agents.types import AgentStep, StepKind

session = DurableSession("/tmp/aire-sess.json")
session.state.goal = "demo"
session.append_step(AgentStep(index=0, kind=StepKind.OBSERVATION, detail={"note": "hi"}))
session.save()
restored = DurableSession("/tmp/aire-sess.json")
state = restored.hydrate_agent_state()
```

`Agent(..., session=path)` can resume when status is `paused`/`running` with prior messages. Mid-run resume coverage is still evolving — check `describe()` / GAPS.

## Policy

```python
from aire import AI
from aire.tools.types import SideEffect

engine = AI.safety.policy()  # default_engine: deny prohibited, approve external+
assert engine.decide(side_effect=SideEffect.PROHIBITED) == "deny"
assert engine.decide(side_effect=SideEffect.EXTERNAL_SIDE_EFFECT) == "require_approval"
```

Pass `policy=` into `Agent` / builder. Default approval levels on the agent also gate tool side-effects via the approver path.

## Teams & topologies

- `AI.agents.team(members, supervisor=...)` — supervisor-routed team
- `AI.agents.swarm` / `debate` and `AI.topologies()` — auction, blackboard, etc.

## Streaming

`agent.run_stream(input)` / `AI.agents.run_stream(agent, input)` yield `AgentEvent`s. With tools present, some stream helpers may collapse to a one-shot run — see honesty notes.
