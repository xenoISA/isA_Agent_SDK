# Examples

Practical examples for common use cases.

## Basic Agent Query

Simple agent interaction:

```python
from isa_agent_sdk import query

async for msg in query("What files are in the current directory?"):
    if msg.is_text:
        print(msg.content, end="")
```

## File Operations Agent

Agent that reads and edits files:

```python
from isa_agent_sdk import query, ISAAgentOptions

async for msg in query(
    "Read config.json and update the version to 2.0.0",
    options=ISAAgentOptions(
        allowed_tools=["Read", "Edit"],
        execution_mode="collaborative",
    )
):
    if msg.is_text:
        print(msg.content)
    elif msg.is_tool_use:
        print(f"Using tool: {msg.tool_name}")
```

## Code Analysis Agent

Analyze code and provide suggestions:

```python
from isa_agent_sdk import query, ISAAgentOptions

async for msg in query(
    "Analyze the code in src/main.py and suggest improvements",
    options=ISAAgentOptions(
        allowed_tools=["Read", "Glob", "Grep"],
        system_prompt="You are a code review expert. Focus on best practices.",
    )
):
    if msg.is_text:
        print(msg.content)
```

## Web Research Agent

Search and summarize web content:

```python
from isa_agent_sdk import query, ISAAgentOptions

async for msg in query(
    "Research the latest Python 3.13 features",
    options=ISAAgentOptions(
        allowed_tools=["WebSearch", "WebFetch"],
        max_iterations=15,
    )
):
    if msg.is_text:
        print(msg.content, end="")
```

## Structured Output

Get structured data from the agent:

```python
from isa_agent_sdk import query, ISAAgentOptions, OutputFormat
from pydantic import BaseModel

class CodeAnalysis(BaseModel):
    language: str
    lines_of_code: int
    issues_found: int
    suggestions: list[str]

async for msg in query(
    "Analyze the code quality in src/",
    options=ISAAgentOptions(
        output_format=OutputFormat.from_pydantic(CodeAnalysis),
    )
):
    if msg.is_result:
        analysis = CodeAnalysis.model_validate_json(msg.content)
        print(f"Language: {analysis.language}")
        print(f"LOC: {analysis.lines_of_code}")
        print(f"Issues: {analysis.issues_found}")
        for suggestion in analysis.suggestions:
            print(f"  - {suggestion}")
```

## Multi-Agent Specialist Team

Coordinate specialized agents:

```python
from isa_agent_sdk import Agent, ISAAgentOptions
from isa_agent_sdk.agents import SwarmOrchestrator, SwarmAgent

# Define specialist agents
researcher = SwarmAgent(
    agent=Agent("researcher", ISAAgentOptions(
        allowed_tools=["WebSearch", "WebFetch", "Read"],
        system_prompt="""You are a research specialist. Your job is to:
        1. Gather accurate facts from reliable sources
        2. Organize information clearly
        3. Hand off to the writer when you have sufficient information
        Always end with [HANDOFF: writer] when done researching.""",
    )),
    description="Gathers facts from web and files",
)

writer = SwarmAgent(
    agent=Agent("writer", ISAAgentOptions(
        allowed_tools=[],
        system_prompt="""You are a technical writer. Your job is to:
        1. Take research findings and write clear, concise summaries
        2. Use proper markdown formatting
        3. End with [COMPLETE] when the summary is done""",
    )),
    description="Writes polished technical documentation",
)

# Create swarm
swarm = SwarmOrchestrator(
    agents=[researcher, writer],
    entry_agent="researcher",
    max_handoffs=5,
)

# Run with streaming
async for msg in swarm.stream("Research Rust async runtime and write a summary"):
    if msg.metadata.get("event") == "swarm_agent_start":
        print(f"\n=== {msg.metadata['agent'].upper()} STARTED ===")
    elif msg.metadata.get("event") == "swarm_handoff":
        print(f"\n>>> Handoff: {msg.metadata['from_agent']} → {msg.metadata['to_agent']}")
    elif msg.is_text:
        agent = msg.metadata.get("agent", "?")
        print(f"[{agent}] {msg.content}", end="")

# Or non-streaming
result = await swarm.run("Research quantum computing and write a summary")
print(result.text)
print(f"\nFinal agent: {result.final_agent}")
print(f"Handoffs: {len(result.handoff_trace)}")
```

## DAG Task Pipeline

Execute tasks with dependencies:

```python
from isa_agent_sdk import Agent, ISAAgentOptions
from isa_agent_sdk.agents import SwarmOrchestrator, SwarmAgent

# Setup agents
researcher = SwarmAgent(
    agent=Agent("researcher", ISAAgentOptions(
        allowed_tools=["WebSearch", "Read"],
    )),
    description="Research specialist",
)

analyst = SwarmAgent(
    agent=Agent("analyst", ISAAgentOptions(
        allowed_tools=["Read"],
    )),
    description="Data analyst",
)

writer = SwarmAgent(
    agent=Agent("writer", ISAAgentOptions()),
    description="Technical writer",
)

swarm = SwarmOrchestrator(
    agents=[researcher, analyst, writer],
)

# Define DAG
result = await swarm.run_dag([
    # Wavefront 0: Research tasks (run in parallel)
    {
        "id": "research_python",
        "title": "Research Python",
        "description": "Find key facts about Python language.",
        "agent": "researcher",
    },
    {
        "id": "research_rust",
        "title": "Research Rust",
        "description": "Find key facts about Rust language.",
        "agent": "researcher",
    },

    # Wavefront 1: Analysis (runs after research completes)
    {
        "id": "compare",
        "title": "Compare Languages",
        "description": "Compare Python and Rust based on research.",
        "agent": "analyst",
        "depends_on": ["research_python", "research_rust"],
    },

    # Wavefront 2: Writing (runs after analysis)
    {
        "id": "write_report",
        "title": "Write Report",
        "description": "Write a comprehensive comparison report.",
        "agent": "writer",
        "depends_on": ["compare"],
    },
])

# Access results by task
print("=== Research Python ===")
print(result.agent_outputs["researcher:research_python"].text)

print("\n=== Research Rust ===")
print(result.agent_outputs["researcher:research_rust"].text)

print("\n=== Comparison ===")
print(result.agent_outputs["analyst:compare"].text)

print("\n=== Final Report ===")
print(result.agent_outputs["writer:write_report"].text)
```

## Human-in-the-Loop Workflow

Request approval for sensitive operations:

```python
from isa_agent_sdk import query, ISAAgentOptions, request_tool_permission, checkpoint

async def deploy_workflow():
    # Agent analyzes deployment
    async for msg in query(
        "Review the deployment config and prepare for deployment",
        options=ISAAgentOptions(
            allowed_tools=["Read", "Bash"],
            execution_mode="collaborative",
        )
    ):
        if msg.is_text:
            print(msg.content)

    # Create checkpoint before deployment
    await checkpoint("pre_deployment", {
        "version": "2.0.0",
        "environment": "production",
        "timestamp": "2026-02-14T10:30:00Z",
    })

    # Request permission for deployment
    authorized = await request_tool_permission(
        "bash",
        {"command": "kubectl apply -f deployment.yaml"}
    )

    if authorized:
        # Proceed with deployment
        async for msg in query(
            "Deploy to production",
            options=ISAAgentOptions(
                allowed_tools=["Bash"],
            )
        ):
            if msg.is_text:
                print(msg.content)

        # Create post-deployment checkpoint
        await checkpoint("post_deployment", {
            "status": "success",
            "version": "2.0.0",
        })
    else:
        print("Deployment cancelled by user")

await deploy_workflow()
```

## Custom Skills

Create and use custom skills:

```python
# .isa/skills/my-skill/SKILL.md
"""
---
name: my-custom-skill
description: Custom skill for specific task
---

# My Custom Skill

This skill helps with [specific purpose].

## Usage
[Instructions for when and how to use this skill]

## Parameters
- param1: Description
- param2: Description
"""

# Use the skill
from isa_agent_sdk import query, ISAAgentOptions

async for msg in query(
    "Use my-custom-skill to process the data",
    options=ISAAgentOptions(
        skills=["my-custom-skill"],
    )
):
    if msg.is_text:
        print(msg.content)
```

## FastAPI Integration

Build an agent service:

```python
from fastapi import FastAPI
from isa_agent_sdk import query, ISAAgentOptions

app = FastAPI()

@app.post("/query")
async def agent_query(prompt: str):
    responses = []
    async for msg in query(
        prompt,
        options=ISAAgentOptions(
            max_iterations=15,
        )
    ):
        if msg.is_text:
            responses.append(msg.content)

    return {"response": "".join(responses)}

@app.post("/analyze")
async def analyze_code(file_path: str):
    from pydantic import BaseModel

    class Analysis(BaseModel):
        language: str
        issues: list[str]
        score: int

    async for msg in query(
        f"Analyze the code quality in {file_path}",
        options=ISAAgentOptions(
            allowed_tools=["Read"],
            output_format=OutputFormat.from_pydantic(Analysis),
        )
    ):
        if msg.is_result:
            return Analysis.model_validate_json(msg.content)
```

## A2A Agent Integration

Connect to other agents:

```python
from isa_agent_sdk import A2AClient, A2AAgentCard

# Connect to remote agent
client = A2AClient("https://remote-agent.example.com")

response = await client.send_message(
    "https://remote-agent.example.com/a2a",
    "Analyze the repository at https://github.com/user/repo"
)

print(response)
```

## Error Handling

Handle errors gracefully:

```python
from isa_agent_sdk import query, ISAAgentOptions

try:
    async for msg in query(
        "Complex task that might fail",
        options=ISAAgentOptions(
            max_iterations=20,
            execution_mode="collaborative",
        )
    ):
        if msg.is_error:
            print(f"Error: {msg.content}")
        elif msg.is_text:
            print(msg.content)
except Exception as e:
    print(f"Fatal error: {e}")
```

## Next Steps

- [API Reference](./api-reference.md) - Complete API documentation
- [Configuration](./configuration.md) - Detailed configuration options
- [Multi-Agent](./multi-agent.md) - Multi-agent patterns
- [Swarm](./swarm.md) - Swarm orchestration guide
- [Testing](./testing.md) - Testing your agent code
