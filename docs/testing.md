# Testing

## Test Coverage

The SDK has comprehensive test coverage across all major subsystems:

- **Total Tests**: 234 passing tests
- **Test Framework**: pytest with asyncio support
- **Coverage Areas**:
  - DAG system (37 tests in `test_dag.py`)
  - Swarm orchestration (33 tests in `test_swarm.py`)
  - Audit coverage (44 tests in `test_audit_coverage.py`)
  - Client operations (10 tests in `test_client.py`)
  - End-to-end execution modes
  - Graph building and routing
  - Options validation
  - Resume and state management
  - Concurrent SQLite operations

## Running Tests

Run all tests:

```bash
python -m pytest
```

Run specific test file:

```bash
python -m pytest tests/test_dag.py -v
```

Run tests matching a pattern:

```bash
python -m pytest tests/ -k "swarm" -v
```

Run with coverage report:

```bash
python -m pytest --cov=isa_agent_sdk --cov-report=html
```

## Test Configuration

The project uses `pyproject.toml` for pytest configuration:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

## Key Test Suites

### DAG System Tests (`test_dag.py`)

Tests for dependency-ordered task execution:

- DAG building from task lists
- Cycle detection with Kahn's algorithm
- Wavefront computation for parallel execution
- Failure cascade propagation
- Task status transitions
- Self-dependency detection
- Multi-agent DAG execution

### Swarm Orchestration Tests (`test_swarm.py`)

Tests for dynamic multi-agent handoffs:

- Handoff directive parsing (`[HANDOFF:]`, `[COMPLETE]`)
- Max handoffs termination
- Handoff trace recording
- Circular handoff detection
- System prompt injection
- DAG-aware swarm execution
- Streaming lifecycle events

### Audit Coverage Tests (`test_audit_coverage.py`)

Tests added for security audit findings (Issue #24):

- Concurrent SQLite write serialization
- ISAAgentOptions validation boundaries
- Swarm circular handoff handling
- Sync-from-async context detection
- DAG deadlock and cycle handling
- OutputFormat.from_pydantic() validation
- Resume error handling with state corruption

### Client Tests (`test_client.py`)

Tests for HTTP client operations:

- Session creation and management
- Message streaming
- Checkpoint operations
- Error handling

## Test Utilities

Common test patterns:

### AsyncMock for Async Functions

```python
from unittest.mock import AsyncMock

mock_runner = AsyncMock()
mock_runner.return_value = expected_result
```

### Testing Concurrent Operations

```python
import asyncio

tasks = [async_function() for _ in range(10)]
results = await asyncio.gather(*tasks)
assert len(results) == 10
```

### Patching sys.modules

```python
import sys

original = sys.modules.get("module_name")
sys.modules["module_name"] = mock_module
try:
    # test code
finally:
    if original:
        sys.modules["module_name"] = original
    else:
        sys.modules.pop("module_name", None)
```

## Continuous Integration

All tests run on every push via GitHub Actions. The test suite must pass before PRs can be merged.

## Test Quality Standards

- **Isolation**: Each test is independent and can run in any order
- **Cleanup**: Tests clean up resources (temp files, mocks, patches)
- **Clarity**: Test names clearly describe what is being tested
- **Coverage**: Critical paths have multiple test cases covering edge cases
- **Performance**: Tests complete in under 2 minutes total

## Related Documentation

- [Swarm Testing](./swarm.md#testing) - Swarm-specific test patterns
- [DAG Scheduler](../isa_agent_sdk/dag/scheduler.py) - DAG implementation details
- [Examples](./examples.md) - Usage examples that can guide test writing

## Audit Completion

All security audit findings (Issues #13-#26) have been resolved with comprehensive test coverage:

- ✅ Issue #24: Added 44 tests for audit coverage gaps
- ✅ Issues #13-#23, #25-#26: Fixed with tests in earlier PRs
- ✅ 234 tests passing across entire test suite
