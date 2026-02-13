# Skills System

Skills are specialized prompt injections that give the agent expert knowledge and behavior patterns for specific tasks.

## Overview

Skills provide:
- **Expert personas** - Code reviewer, debugger, technical writer
- **Domain knowledge** - Best practices, patterns, methodologies
- **Consistent behavior** - Reliable, repeatable approaches

## Project Setup

### The `.isa` Directory

The isA Agent SDK uses a `.isa` directory for agent-specific configuration and skills, similar to how Claude Code uses `.claude`.

```
my-project/
├── .isa/
│   ├── config.json          # Project configuration
│   ├── settings.local.json  # MCP servers, permissions
│   └── skills/              # Agent-specific skills
│       ├── my-skill/
│       │   └── SKILL.md
│       └── another-skill/
│           └── SKILL.md
├── src/
└── ...
```

### Configuration Files

**`.isa/config.json`** - Project configuration:
```json
{
  "project": {
    "name": "my-project",
    "type": "agent",
    "framework": "isA-Agent-SDK",
    "version": "0.1.0"
  },
  "skills": {
    "directory": "skills",
    "enabled": true
  },
  "agent": {
    "name": "My Agent",
    "model": "claude-opus-4-5-20251101"
  }
}
```

**`.isa/settings.local.json`** - MCP servers and permissions:
```json
{
  "permissions": {
    "allow": [],
    "deny": [],
    "ask": []
  },
  "mcpServers": {
    "isa_mcp": {
      "type": "http",
      "url": "http://localhost:8081/mcp/"
    }
  }
}
```

### Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ISA_SKILLS_DIR` | `.isa` | Directory name for skills (e.g., `.isa`, `.claude`) |

Example: Use `.claude` directory instead of `.isa`:
```bash
export ISA_SKILLS_DIR=.claude
```

## Skill Loading

### Local-First Strategy

Skills are loaded with a **local-first** strategy:

1. **Local skills** - Check `.isa/skills/{name}/SKILL.md`
2. **MCP fallback** - If not found locally, fetch from `vibe://skill/{name}`

This allows:
- **Agent-specific skills** - Customize skills per project
- **Shared skills** - Fall back to centralized MCP skills
- **Overrides** - Local skills take precedence over MCP

### Skill File Structure

Each skill is a directory with a `SKILL.md` file:

```
.isa/skills/my-skill/
├── SKILL.md           # Main skill content (required)
├── guides/            # Optional detailed guides
│   └── usage.md
└── templates/         # Optional templates
    └── example.py
```

**SKILL.md format:**
```markdown
---
name: my-skill
description: Brief description of what this skill does
---

# My Skill

## Purpose
Describe the skill's purpose and when to use it.

## Workflow
Step-by-step instructions for the agent.

## Examples
Show example usage and expected outputs.
```

## Using Skills

### Activate Skills via Options

```python
from isa_agent_sdk import query, ISAAgentOptions

options = ISAAgentOptions(
    skills=["code-review", "debug"]
)

async for msg in query("Review this function for issues", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Programmatic Skill Loading

```python
from isa_agent_sdk import load_skill, activate_skills, list_builtin_skills

# List available skills
skills = list_builtin_skills()
print(skills)
# Output: ['code-review', 'debug', 'refactor', 'test-writer', 'documentation', 'security-audit']

# Load a single skill (async)
skill = await load_skill("code-review")
print(f"Skill: {skill.name}")
print(f"Description: {skill.description}")
print(f"Triggers: {skill.triggers}")
# Output:
# Skill: code-review
# Description: Expert code reviewer focusing on quality, security, and best practices
# Triggers: ['review', 'code review', 'check this code', 'review my code']

# Activate multiple skills (note: uses *args, not list)
injection = activate_skills("code-review", "debug", "refactor")
print(f"Injection length: {len(injection)} chars")
```

## Built-in Skills

### code-review

Expert code reviewer focusing on:
- Code quality and readability
- Best practices adherence
- Security vulnerabilities
- Performance issues

```python
options = ISAAgentOptions(skills=["code-review"])

async for msg in query("""
Review this code:

def process(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result
""", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### debug

Systematic debugger that:
- Analyzes error messages
- Traces execution flow
- Identifies root causes
- Suggests fixes

```python
options = ISAAgentOptions(skills=["debug"])

async for msg in query("""
Debug this error:

TypeError: cannot unpack non-iterable NoneType object
  File "app.py", line 42, in process_user
    name, email = get_user_info(user_id)
""", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### refactor

Refactoring specialist that:
- Improves code structure
- Reduces complexity
- Applies design patterns
- Maintains functionality

```python
options = ISAAgentOptions(skills=["refactor"])

async for msg in query("Refactor this 200-line function into smaller pieces", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### test-writer

Test coverage specialist that:
- Writes comprehensive tests
- Covers edge cases
- Uses appropriate frameworks
- Follows testing best practices

```python
options = ISAAgentOptions(skills=["test-writer"])

async for msg in query("Write tests for the UserService class", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### documentation

Technical documentation writer that:
- Creates clear API docs
- Writes helpful README files
- Documents complex systems
- Maintains consistency

```python
options = ISAAgentOptions(skills=["documentation"])

async for msg in query("Document the authentication module", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

## Combining Skills

Skills can be combined for complex tasks:

```python
# Code review + security focus
options = ISAAgentOptions(skills=["code-review", "debug"])

# Full development workflow
options = ISAAgentOptions(skills=["code-review", "refactor", "test-writer"])

# Documentation + code quality
options = ISAAgentOptions(skills=["documentation", "code-review"])
```

## Skill Manager

### Advanced Skill Management

```python
from isa_agent_sdk import get_skill_manager

manager = get_skill_manager()

# Load skills
await manager.load("code-review")
await manager.load("debug")

# Get current injection
injection = manager.get_injection()

# List loaded skills
loaded = manager.list_loaded()
print(f"Active skills: {loaded}")
```

### List Available Skills

```python
from isa_agent_sdk import list_builtin_skills

skills = list_builtin_skills()
for skill_name in skills:
    print(f"- {skill_name}")
```

## Skill Structure

### Skill Properties

```python
from isa_agent_sdk import Skill

# Skill data structure
skill = Skill(
    name="code-review",           # Unique identifier
    description="Expert code reviewer",  # What it does
    prompt="You are an expert...",       # Prompt injection
    triggers=["review", "check"],        # Keywords that trigger
    category="coding",                   # Category
    source="builtin",                    # Where loaded from
    metadata={"version": "1.0"}          # Additional data
)
```

### Access Skill Details

```python
from isa_agent_sdk import load_skill

skill = load_skill("debug")

print(f"Name: {skill.name}")
print(f"Description: {skill.description}")
print(f"Category: {skill.category}")
print(f"Triggers: {skill.triggers}")
print(f"Prompt preview: {skill.prompt[:200]}...")
```

## Custom Skills

### Define Custom Skills

```python
from isa_agent_sdk import Skill, get_skill_manager

# Create custom skill
security_skill = Skill(
    name="security-audit",
    description="Security vulnerability scanner",
    prompt="""You are a security expert specializing in:
- OWASP Top 10 vulnerabilities
- SQL injection detection
- XSS prevention
- Authentication flaws
- Authorization issues

When reviewing code:
1. Check all user inputs
2. Verify authentication flows
3. Review authorization logic
4. Look for injection points
5. Check for sensitive data exposure
""",
    triggers=["security", "vulnerability", "audit"],
    category="security",
    source="custom",
    metadata={"author": "security-team"}
)

# Register with manager
manager = get_skill_manager()
manager.register_skill(security_skill)

# Use in query
options = ISAAgentOptions(skills=["security-audit"])
```

### Load Skills from Local `.isa` Directory

Skills placed in `.isa/skills/` are automatically discovered:

```
.isa/skills/
├── api-design/
│   └── SKILL.md
└── security-audit/
    └── SKILL.md
```

**`.isa/skills/api-design/SKILL.md`:**
```markdown
---
name: api-design
description: REST API design expert
---

# API Design Expert

You are an API design expert following:
- RESTful principles
- OpenAPI specification
- Versioning best practices
- Error handling standards
- Rate limiting patterns

## Workflow
1. Analyze the API requirements
2. Design resource endpoints
3. Define request/response schemas
4. Document with OpenAPI spec
```

Use the skill:
```python
options = ISAAgentOptions(skills=["api-design"])

async for msg in query("Design an API for user management", options=options):
    print(msg.content, end="" if msg.is_text else "\n")
```

### Load Skills from YAML Files

```yaml
# skills/api-design.yaml
name: api-design
description: REST API design expert
category: architecture
triggers:
  - api
  - rest
  - endpoint
prompt: |
  You are an API design expert following:
  - RESTful principles
  - OpenAPI specification
  - Versioning best practices
  - Error handling standards
  - Rate limiting patterns
```

```python
import yaml
from isa_agent_sdk import Skill, get_skill_manager

with open("skills/api-design.yaml") as f:
    data = yaml.safe_load(f)

skill = Skill(**data, source="file")
manager = get_skill_manager()
manager.register_skill(skill)
```

## Skill Triggers

Skills can be auto-activated based on query content:

```python
from isa_agent_sdk import get_skill_manager

manager = get_skill_manager()

# Check if query triggers any skills
query = "Can you review this code for bugs?"
triggered = manager.get_triggered_skills(query)
print(f"Auto-triggered: {triggered}")  # ['code-review', 'debug']
```

## Best Practices

1. **Combine complementary skills** - code-review + debug for thorough analysis
2. **Use specific skills** - Don't load all skills, pick relevant ones
3. **Create domain skills** - Build skills for your specific needs
4. **Keep prompts focused** - Each skill should have a clear purpose
5. **Test skill combinations** - Ensure skills work well together

## Skill Categories

| Category | Skills | Use Case |
|----------|--------|----------|
| `coding` | code-review, debug, refactor | Code quality |
| `testing` | test-writer | Test coverage |
| `documentation` | documentation | Technical writing |
| `security` | (custom) | Security audits |
| `architecture` | (custom) | System design |

## Skill Loading Priority

When a skill is requested, the SDK loads it using this priority:

```
1. Local: .isa/skills/{name}/SKILL.md
   ↓ (if not found)
2. MCP: vibe://skill/{name}
   ↓ (if not found)
3. Built-in: BUILTIN_SKILLS dict
```

### Logging

Skill loading is logged for debugging:

```
[PHASE:CONTEXT] skill_loaded_local | skill=cdd | path=/project/.isa/skills/cdd/SKILL.md | length=3996
[PHASE:CONTEXT] skill_loaded_mcp | skill=external-skill | length=2500
[PHASE:CONTEXT] skills_loaded | requested=2 | loaded=2 | local=1 | mcp=1 | duration_ms=45
```

### Override MCP Skills Locally

To customize a shared MCP skill for your project:

1. Create the skill directory:
   ```bash
   mkdir -p .isa/skills/cdd
   ```

2. Copy or create your custom `SKILL.md`:
   ```bash
   # Your custom version takes precedence
   echo "# Custom CDD Skill" > .isa/skills/cdd/SKILL.md
   ```

3. The local version will be used instead of MCP.

## Next Steps

- [Options](./options.md) - Configure skill activation
- [Tools](./tools.md) - Combine skills with tools
- [Human-in-the-Loop](./human-in-the-loop.md) - Review skill outputs
