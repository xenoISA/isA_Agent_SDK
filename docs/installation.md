# Installation

## Installation

```bash
pip install isa-agent-sdk

## Project Setup

Initialize a `.isa` directory for project-specific configuration:

```bash
mkdir -p .isa/skills
```

```
my-project/
├── .isa/
│   ├── config.json          # Project configuration
│   ├── settings.local.json  # MCP servers, permissions
│   └── skills/              # Local skills (loaded first)
│       └── my-skill/
│           └── SKILL.md
└── ...
```

Skills in `.isa/skills/` are loaded before MCP, allowing project-specific overrides.

# 1. Install systemd services
sudo ./deployment/staging/scripts/manage_workers.sh install

## References

- [README.md](./README.md)
- [deployment-guide.md](./deployment-guide.md)
- [desktop-execution.md](./desktop-execution.md)
- [isa_agent_sdk/services/background_jobs/docs/DOCKER_TEST_RESULTS.md](./isa_agent_sdk/services/background_jobs/docs/DOCKER_TEST_RESULTS.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
- [long-running-tasks.md](./long-running-tasks.md)
- [skills.md](./skills.md)
- [triggers.md](./triggers.md)
