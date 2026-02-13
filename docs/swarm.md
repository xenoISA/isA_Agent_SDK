# Swarm Multi-Agent Orchestration

Dynamic agent handoff orchestration where agents decide when to hand off control based on their specialization. Includes DAG-aware task execution for dependency-ordered workflows across agents.

## What Was Verified

Verified end-to-end against running isA services (MCP 8081, Model 8082):

- **Swarm handoff**: researcher agent ran with handoff prompt injected, orchestrator parsed `[HANDOFF:]` and `[COMPLETE]` directives correctly
- **Streaming**: `swarm_agent_start` and `swarm_handoff` lifecycle events emitted, all messages annotated with source agent
- **DAG execution**: 2-wavefront pipeline (research -> write) — researcher produced 435 chars of facts, writer produced 342-char summary from dependency results
- **DAG parallel wavefront**: 3 tasks (Python fact + Rust fact -> combine) — two research tasks ran in wavefront 0, writer combined results in wavefront 1. Final output: *"Python emphasizes readability by using indentation... Rust guarantees memory safety without a garbage collector..."*
- **Options immutability**: original agent options unchanged after handoff prompt injection
- **33 unit tests** + **4 live integration tests** passing

## When to Use

| Use Case | Use This |
|----------|----------|
| Fixed routing (planner -> renderer -> end) | `MultiAgentOrchestrator` |
| Agents dynamically decide when to hand off | `SwarmOrchestrator` |
| Tasks with dependencies across agents | `SwarmOrchestrator.run_dag()` |

## Core Types

```python
from isa_agent_sdk import Agent, ISAAgentOptions
from isa_agent_sdk.agents import SwarmOrchestrator, SwarmAgent, SwarmRunResult, SwarmState
```

## Basic Swarm

Agents hand off control dynamically. The orchestrator injects peer information into each agent's system prompt and parses responses for `[HANDOFF: agent_name]` or `[COMPLETE]` directives.

```python
from isa_agent_sdk import Agent, ISAAgentOptions
from isa_agent_sdk.agents import SwarmOrchestrator, SwarmAgent

researcher = SwarmAgent(
    agent=Agent("researcher", ISAAgentOptions(
        system_prompt="You are a research agent. Gather facts only.",
    )),
    description="Research and information gathering",
)

writer = SwarmAgent(
    agent=Agent("writer", ISAAgentOptions(
        system_prompt="You are a technical writer. Write polished summaries.",
    )),
    description="Writing and documentation",
)

swarm = SwarmOrchestrator(
    agents=[researcher, writer],
    entry_agent="researcher",
    max_handoffs=10,
)

result = await swarm.run("Research quantum computing and write a summary")
print(result.text)           # Clean text (directives stripped)
print(result.final_agent)    # "writer"
print(result.handoff_trace)  # [{"from": "researcher", "to": "writer", "reason": "..."}]
```

### How Handoff Works

1. The orchestrator appends an **Agent Collaboration** section to each agent's system prompt listing peer agents and the handoff syntax
2. Each agent runs normally via `Agent.run()`
3. The orchestrator parses the last 500 chars of the response for directives:
   - `[HANDOFF: agent_name] reason` — switch to the named agent
   - `[COMPLETE]` — finish the swarm
   - No directive — treated as complete (safe default)
4. On handoff, the next agent receives the original prompt plus context from the previous agent

The injected prompt section looks like:

```
## Agent Collaboration

You are part of a multi-agent team. The following peer agents are available:
- **writer**: Writing and documentation

### Handoff Protocol
When you determine that another agent is better suited to continue,
end your response with:
[HANDOFF: agent_name] reason for handoff

When your part of the task is fully complete, end your response with:
[COMPLETE]
```

## Streaming

Stream messages with agent annotations and lifecycle events:

```python
async for msg in swarm.stream("Research and write about AI"):
    if msg.metadata.get("event") == "swarm_agent_start":
        print(f"\n--- {msg.metadata['agent']} started ---")
    elif msg.metadata.get("event") == "swarm_handoff":
        print(f"\n--- handoff: {msg.metadata['from_agent']} -> {msg.metadata['to_agent']} ---")
    elif msg.is_text:
        agent = msg.metadata.get("agent", "?")
        print(f"[{agent}] {msg.content}", end="")
```

Lifecycle events emitted:

| Event | Metadata | When |
|-------|----------|------|
| `swarm_agent_start` | `agent` | Before each agent runs |
| `swarm_handoff` | `from_agent`, `to_agent`, `reason` | When handoff is detected |

## DAG-Aware Execution

For workflows with task dependencies, use `run_dag()`. Each task specifies an `agent` field. Tasks in the same wavefront on different agents run concurrently.

```python
result = await swarm.run_dag([
    {
        "id": "research",
        "title": "Research Python",
        "description": "Gather key facts about Python.",
        "agent": "researcher",
    },
    {
        "id": "write_summary",
        "title": "Write Summary",
        "description": "Write a summary based on the research.",
        "agent": "writer",
        "depends_on": ["research"],
    },
])

# Both agents ran in dependency order
print(result.agent_outputs["researcher:research"].text)
print(result.agent_outputs["writer:write_summary"].text)
```

### Parallel Wavefronts

Independent tasks on different agents execute concurrently:

```python
result = await swarm.run_dag([
    # Wavefront 0: these two run concurrently (different agents)
    {"id": "fact_python", "title": "Python", "description": "One fact about Python.", "agent": "researcher"},
    {"id": "fact_rust", "title": "Rust", "description": "One fact about Rust.", "agent": "analyst"},

    # Wavefront 1: runs after both above complete
    {"id": "combine", "title": "Combine", "description": "Combine into comparison.",
     "agent": "writer", "depends_on": ["fact_python", "fact_rust"]},
])
```

Execution order:
```
Wavefront 0: [fact_python (researcher), fact_rust (analyst)]  <- concurrent
Wavefront 1: [combine (writer)]                               <- sequential, gets both results
```

Each downstream task receives truncated results from its dependencies in the prompt.

## Constructor Options

```python
SwarmOrchestrator(
    agents,           # List[SwarmAgent] or Dict[str, Agent]
    entry_agent,      # str — first agent to run (default: first in list)
    max_handoffs,     # int — safety cap (default: 10)
    shared_state,     # Dict — initial shared state passed across agents
)
```

### Dict Shorthand

For quick setup without descriptions:

```python
swarm = SwarmOrchestrator(
    agents={
        "researcher": Agent("researcher", ISAAgentOptions(...)),
        "writer": Agent("writer", ISAAgentOptions(...)),
    }
)
```

Agent descriptions default to the agent name. For better handoff behavior, use `SwarmAgent` with explicit descriptions.

## SwarmRunResult

```python
@dataclass
class SwarmRunResult:
    text: str                              # Final clean text (directives stripped)
    messages: List[AgentMessage]           # All messages from all agents
    final_agent: str                       # Agent that produced the final response
    handoff_trace: List[Dict[str, str]]    # [{"from", "to", "reason", "step"}]
    shared_state: Dict[str, Any]           # Accumulated shared state
    agent_outputs: Dict[str, AgentRunResult]  # Per-agent results
```

For DAG execution, `agent_outputs` keys are `"agent_name:task_id"`.

## SystemPromptConfig Support

Handoff injection works with all system prompt configurations:

```python
# String prompt — handoff section appended
Agent("a", ISAAgentOptions(system_prompt="You are a researcher."))

# SystemPromptConfig with preset — handoff added to append
Agent("a", ISAAgentOptions(
    system_prompt=SystemPromptConfig(preset="reason", append="Custom instructions.")
))

# SystemPromptConfig with replace — handoff added to replace text
Agent("a", ISAAgentOptions(
    system_prompt=SystemPromptConfig(replace="Full custom prompt.")
))
```

Original options are never mutated — the orchestrator clones options before injection.

## Shared State

State accumulates across handoffs and is available in the result:

```python
swarm = SwarmOrchestrator(
    agents=[...],
    shared_state={"project": "my-project"},
)

result = await swarm.run("do something")
print(result.shared_state)  # {"project": "my-project", ...merged from agents...}
```

Each agent's `AgentRunResult.shared_state` is merged into the swarm state after it runs.

## Configuration Tips

- **`max_iterations`**: Set to 15+ per agent. The SmartAgentGraph needs multiple LangGraph steps per turn (Sense -> Reason -> Response). With `max_iterations=3` the graph hits its recursion limit before producing output.
- **`max_handoffs`**: Safety cap on total handoffs. Set to 3-5 for production, 10 for development.
- **Agent descriptions**: Specific descriptions produce better handoff decisions. "Gathers facts only — never writes final content" is better than "Research agent".
- **System prompts**: Tell agents explicitly when to hand off. "After gathering facts, hand off to the writer agent" improves handoff reliability.

## Comparison with MultiAgentOrchestrator

| Feature | `MultiAgentOrchestrator` | `SwarmOrchestrator` |
|---------|--------------------------|---------------------|
| Routing | Fixed `router` function | Dynamic via LLM response |
| Handoff | Explicit in router code | Agent decides via `[HANDOFF:]` |
| DAG support | No | Yes (`run_dag()`) |
| Streaming events | Agent annotation only | Agent annotation + lifecycle events |
| Agent wrapper | `Dict[str, Agent]` | `List[SwarmAgent]` or `Dict[str, Agent]` |
| Shared state | Yes | Yes |
| Max steps | `max_steps` (default 6) | `max_handoffs` (default 10) |

## Testing

Run the unit tests:

```bash
python -m pytest tests/test_swarm.py -v  # 33 unit tests
```

Run the end-to-end smoke tests (requires running MCP + Model services):

```bash
# From isA_Agent directory with dev.env loaded
source deployment/environments/dev.env
python tests/test_swarm_e2e.py --live
```
