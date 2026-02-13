# Multi‑Agent Orchestration

The SDK exposes a clean multi‑agent API that composes multiple `Agent` instances
and routes work across them. This hides LangGraph details and lets you think in
terms of agents and handoffs.

## When to Use

- You need specialized agents (planner, coder, renderer, safety)
- You want explicit handoffs and shared state
- You want a simple supervisor that coordinates multiple agents

## Core Types

```python
from isa_agent_sdk import Agent, MultiAgentOrchestrator, ISAAgentOptions
```

## Minimal Example

```python
from isa_agent_sdk import Agent, MultiAgentOrchestrator, ISAAgentOptions

planner = Agent("planner", ISAAgentOptions(skills=["documentation"]))
renderer = Agent("renderer", ISAAgentOptions(skills=["documentation"]))

# Simple handoff: planner -> renderer -> end

def router(state, last_result):
    outputs = state.get("outputs", {})
    if "planner" not in outputs:
        return "planner"
    if "renderer" not in outputs:
        return "renderer"
    return None

orchestrator = MultiAgentOrchestrator(
    agents={"planner": planner, "renderer": renderer},
    entry_agent="planner",
    router=router,
)

result = await orchestrator.run("Create a 30s storyboard")
print(result.outputs["planner"].text)
print(result.outputs["renderer"].text)
```

## Router Contract

The router receives:
- `state["shared_state"]` — mutable shared data
- `state["outputs"]` — per‑agent results
- `state["last_agent"]` — last agent that ran

It should return the next agent name or `None` to stop.

## Streaming Across Agents

```python
async for msg in orchestrator.stream("Design a creative workflow"):
    agent = msg.metadata.get("agent")
    if msg.is_text:
        print(f"[{agent}] {msg.content}")
```

## When to Use

`MultiAgentOrchestrator` is best when you have a **fixed routing pattern** — you know
at design time which agent runs after which. The router function gives you full control.

For **dynamic handoff** where agents decide at runtime when to hand off, or for
**DAG-based task pipelines** across agents, use [`SwarmOrchestrator`](./swarm.md).

## Notes

- Each agent uses the standard `query()` API internally
- Skills, tools, and execution_mode are configured per agent
- Use routing + shared state to enforce handoffs and guardrails
- For dynamic handoff and DAG execution, see [Swarm Orchestration](./swarm.md)
