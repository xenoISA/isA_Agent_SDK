# Testing

## Automated Test Run (2026-02-09 16:53:59)

- Command: `python -m pytest`
- Status: `timeout`
- Exit: `None`
- Duration: `120.11s`
- Stdout (tail):
```
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/xenodennis/Documents/Fun/isA/isA_Agent_SDK
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.12.1, asyncio-1.2.0, langsmith-0.6.4, cov-7.0.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 132 items

tests/test_client.py ..........                                          [  7%]
tests/test_dag.py ...................................................... [ 48%]
...                                                                      [ 50%]
tests/test_e2e_execution_modes.py ...
```

## References

- [isa_agent_sdk/nodes/docs/entry_node.md](./isa_agent_sdk/nodes/docs/entry_node.md)
- [isa_agent_sdk/nodes/docs/failsafe_node.md](./isa_agent_sdk/nodes/docs/failsafe_node.md)
- [isa_agent_sdk/nodes/docs/guardrail_node.md](./isa_agent_sdk/nodes/docs/guardrail_node.md)
- [isa_agent_sdk/nodes/docs/model_node.md](./isa_agent_sdk/nodes/docs/model_node.md)
- [isa_agent_sdk/nodes/docs/revise_node.md](./isa_agent_sdk/nodes/docs/revise_node.md)
- [isa_agent_sdk/nodes/docs/router_node.md](./isa_agent_sdk/nodes/docs/router_node.md)
- [isa_agent_sdk/nodes/docs/tool_node.md](./isa_agent_sdk/nodes/docs/tool_node.md)
- [isa_agent_sdk/services/auto_detection/DESIGN.md](./isa_agent_sdk/services/auto_detection/DESIGN.md)
- [isa_agent_sdk/services/auto_detection/MCP_PROGRESS_INTEGRATION.md](./isa_agent_sdk/services/auto_detection/MCP_PROGRESS_INTEGRATION.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
- [isa_agent_sdk/services/feedback/README.md](./isa_agent_sdk/services/feedback/README.md)
- [isa_agent_sdk/services/human_in_the_loop/README.md](./isa_agent_sdk/services/human_in_the_loop/README.md)
- [isa_agent_sdk/services/human_in_the_loop/old/FEATURE_COMPARISON.md](./isa_agent_sdk/services/human_in_the_loop/old/FEATURE_COMPARISON.md)
- [isa_agent_sdk/services/trace/migrations/README.md](./isa_agent_sdk/services/trace/migrations/README.md)
- [memory.md](./memory.md)
- [swarm.md](./swarm.md)
