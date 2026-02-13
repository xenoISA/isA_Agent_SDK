#!/usr/bin/env python3
"""
Tests for ISAAgentClient - Bidirectional Client

Tests the new Claude SDK-compatible bidirectional client that supports:
1. Async context manager pattern
2. query() / receive() separation
3. ask() convenience method
4. Session management and forking
5. Local (LangGraph) and Remote (HTTP) modes
6. Sync wrapper

Run with:
    python tests/test_client.py

Or with pytest:
    pytest tests/test_client.py -v
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment
os.environ.setdefault("ISA_MODEL_URL", "http://localhost:8082")
os.environ.setdefault("ISA_MCP_URL", "http://localhost:8081")


class TestResult:
    """Test result holder"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.duration_ms = 0


# ============================================================================
# Unit Tests (No external dependencies)
# ============================================================================

async def test_client_import() -> TestResult:
    """
    Test 1: Import ISAAgentClient

    Verifies all client components can be imported.
    """
    result = TestResult("Import ISAAgentClient")
    start = time.time()

    try:
        from isa_agent_sdk import (
            ISAAgentClient,
            ISAAgentClientSync,
            ClientMode,
            ISAAgentOptions
        )

        result.details["ISAAgentClient"] = ISAAgentClient is not None
        result.details["ISAAgentClientSync"] = ISAAgentClientSync is not None
        result.details["ClientMode"] = ClientMode is not None
        result.details["modes"] = [m.value for m in ClientMode]

        result.passed = all([
            ISAAgentClient is not None,
            ISAAgentClientSync is not None,
            ClientMode is not None
        ])

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_client_initialization() -> TestResult:
    """
    Test 2: Client Initialization

    Verifies client can be created with various options.
    """
    result = TestResult("Client Initialization")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient, ISAAgentOptions, ClientMode

        # Default initialization (local mode)
        client1 = ISAAgentClient()
        result.details["default_mode"] = client1.mode.value
        result.details["default_session_id"] = client1.session_id[:20] + "..."
        result.details["has_session_id"] = client1.session_id is not None

        # With options
        options = ISAAgentOptions(
            model="gpt-4.1-nano",
            allowed_tools=["Read", "Write"]
        )
        client2 = ISAAgentClient(options=options)
        result.details["with_options_mode"] = client2.mode.value

        # Remote mode initialization
        client3 = ISAAgentClient(base_url="http://example.com")
        result.details["remote_mode"] = client3.mode.value
        result.details["remote_is_remote"] = client3.mode == ClientMode.REMOTE

        # With explicit session_id and user_id
        client4 = ISAAgentClient(
            session_id="custom_session_123",
            user_id="test_user"
        )
        result.details["custom_session_id"] = client4.session_id
        result.details["custom_user_id"] = client4.user_id

        result.passed = (
            client1.mode == ClientMode.LOCAL and
            client3.mode == ClientMode.REMOTE and
            client4.session_id == "custom_session_123" and
            client4.user_id == "test_user"
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_client_properties() -> TestResult:
    """
    Test 3: Client Properties

    Verifies client properties work correctly.
    """
    result = TestResult("Client Properties")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient, ClientMode

        client = ISAAgentClient(
            session_id="test_session",
            user_id="test_user"
        )

        result.details["session_id"] = client.session_id
        result.details["user_id"] = client.user_id
        result.details["message_count"] = client.message_count
        result.details["mode"] = client.mode.value
        result.details["is_connected"] = client.is_connected

        # get_session_info()
        info = client.get_session_info()
        result.details["session_info_keys"] = list(info.keys())
        result.details["session_info"] = info

        result.passed = (
            client.session_id == "test_session" and
            client.user_id == "test_user" and
            client.message_count == 0 and
            client.mode == ClientMode.LOCAL and
            client.is_connected == False and  # Not initialized yet
            "session_id" in info and
            "message_count" in info
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_client_fork() -> TestResult:
    """
    Test 4: Session Forking

    Verifies client can fork sessions.
    """
    result = TestResult("Session Forking")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient

        # Create original client
        original = ISAAgentClient(
            session_id="original_session",
            user_id="test_user"
        )

        # Fork with auto-generated ID
        forked1 = original.fork()
        result.details["forked1_session_id"] = forked1.session_id
        result.details["forked1_has_fork_suffix"] = "_fork_" in forked1.session_id
        result.details["forked1_user_id"] = forked1.user_id

        # Fork with custom ID
        forked2 = original.fork(new_session_id="custom_fork_123")
        result.details["forked2_session_id"] = forked2.session_id

        result.passed = (
            "_fork_" in forked1.session_id and
            forked1.user_id == original.user_id and
            forked2.session_id == "custom_fork_123"
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_error_classes() -> TestResult:
    """
    Test 5: Error Classes

    Verifies all error classes are importable and work correctly.
    """
    result = TestResult("Error Classes")
    start = time.time()

    try:
        from isa_agent_sdk import (
            ISASDKError,
            ConnectionError,
            TimeoutError,
            ExecutionError,
            ToolExecutionError,
            SessionError,
            SessionNotFoundError,
            ValidationError,
            SchemaError,
            PermissionError,
            MCPError
        )

        # Test base error
        base_err = ISASDKError("Test error", {"key": "value"})
        result.details["base_message"] = base_err.message
        result.details["base_details"] = base_err.details
        result.details["base_str"] = str(base_err)

        # Test tool execution error
        tool_err = ToolExecutionError(
            "Tool failed",
            tool_name="read_file",
            tool_args={"path": "/test.txt"},
            session_id="sess_123"
        )
        result.details["tool_err_tool_name"] = tool_err.tool_name
        result.details["tool_err_session_id"] = tool_err.session_id

        # Test session error
        sess_err = SessionNotFoundError(
            "Session not found",
            session_id="sess_456"
        )
        result.details["sess_err_session_id"] = sess_err.session_id

        # Test inheritance
        result.details["tool_err_is_execution"] = isinstance(tool_err, ExecutionError)
        result.details["tool_err_is_base"] = isinstance(tool_err, ISASDKError)
        result.details["sess_err_is_session"] = isinstance(sess_err, SessionError)

        result.passed = (
            base_err.message == "Test error" and
            tool_err.tool_name == "read_file" and
            sess_err.session_id == "sess_456" and
            isinstance(tool_err, ExecutionError) and
            isinstance(tool_err, ISASDKError)
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


# ============================================================================
# Integration Tests (Require running services)
# ============================================================================

async def test_client_context_manager() -> TestResult:
    """
    Test 6: Async Context Manager

    Verifies client works with async context manager.
    Requires: ISA_MODEL_URL, ISA_MCP_URL
    """
    result = TestResult("Async Context Manager")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient, ISAAgentOptions

        options = ISAAgentOptions(
            model="gpt-4.1-nano",
            max_iterations=5
        )

        async with ISAAgentClient(options=options) as client:
            result.details["is_connected_inside"] = client.is_connected
            result.details["session_id"] = client.session_id
            result.details["mode"] = client.mode.value

        # After exiting, client should be closed
        result.details["is_connected_after"] = client.is_connected

        result.passed = (
            result.details["is_connected_inside"] == True and
            result.details["is_connected_after"] == False
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_query_receive_pattern() -> TestResult:
    """
    Test 7: Query/Receive Pattern

    Verifies the bidirectional query() and receive() pattern.
    Requires: ISA_MODEL_URL, ISA_MCP_URL
    """
    result = TestResult("Query/Receive Pattern")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient, ISAAgentOptions

        options = ISAAgentOptions(
            model="gpt-4.1-nano",
            max_iterations=5
        )

        messages = []
        text_content = ""

        async with ISAAgentClient(options=options) as client:
            # Send query
            await client.query("What is 2 + 2? Answer with just the number.")
            result.details["message_count_after_query"] = client.message_count

            # Receive response
            async for msg in client.receive():
                messages.append({
                    "type": msg.type,
                    "has_content": msg.content is not None
                })
                if hasattr(msg, 'is_text') and msg.is_text and msg.content:
                    text_content += msg.content
                elif hasattr(msg, 'is_complete') and msg.is_complete and msg.content:
                    text_content += msg.content

            result.details["session_id"] = client.session_id

        result.details["total_messages"] = len(messages)
        result.details["message_types"] = list(set(m["type"] for m in messages))
        result.details["text_content_preview"] = text_content[:100] if text_content else None
        result.details["has_response"] = len(text_content) > 0

        result.passed = (
            len(messages) > 0 and
            len(text_content) > 0
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_ask_convenience() -> TestResult:
    """
    Test 8: ask() Convenience Method

    Verifies the ask() method returns complete response.
    Requires: ISA_MODEL_URL, ISA_MCP_URL
    """
    result = TestResult("ask() Convenience Method")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient, ISAAgentOptions

        options = ISAAgentOptions(
            model="gpt-4.1-nano",
            max_iterations=5
        )

        async with ISAAgentClient(options=options) as client:
            response = await client.ask("Say 'hello world' and nothing else.")

            result.details["response_type"] = type(response).__name__
            result.details["response_length"] = len(response)
            result.details["response_preview"] = response[:100] if response else None
            result.details["session_id"] = client.session_id
            result.details["message_count"] = client.message_count

        result.passed = (
            isinstance(response, str) and
            len(response) > 0
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_multi_turn_conversation() -> TestResult:
    """
    Test 9: Multi-turn Conversation

    Verifies multiple queries in same session.
    Requires: ISA_MODEL_URL, ISA_MCP_URL
    """
    result = TestResult("Multi-turn Conversation")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClient, ISAAgentOptions

        options = ISAAgentOptions(
            model="gpt-4.1-nano",
            max_iterations=5
        )

        responses = []

        async with ISAAgentClient(options=options) as client:
            # First turn
            await client.query("Remember this number: 42")
            first_response = ""
            async for msg in client.receive():
                if hasattr(msg, 'is_text') and msg.is_text and msg.content:
                    first_response += msg.content
            responses.append(first_response)
            result.details["turn1_message_count"] = client.message_count

            # Second turn
            await client.query("What number did I ask you to remember?")
            second_response = ""
            async for msg in client.receive():
                if hasattr(msg, 'is_text') and msg.is_text and msg.content:
                    second_response += msg.content
            responses.append(second_response)
            result.details["turn2_message_count"] = client.message_count

            result.details["session_id"] = client.session_id

        result.details["turn1_preview"] = responses[0][:100] if responses[0] else None
        result.details["turn2_preview"] = responses[1][:100] if responses[1] else None
        result.details["turn2_mentions_42"] = "42" in responses[1] if len(responses) > 1 else False

        result.passed = (
            len(responses) == 2 and
            len(responses[0]) > 0 and
            len(responses[1]) > 0
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


async def test_sync_wrapper() -> TestResult:
    """
    Test 10: Synchronous Wrapper (Unit Test Only)

    Verifies ISAAgentClientSync can be instantiated.
    Note: Full sync integration test should run from sync context (e.g., separate script).
    Running sync wrapper from async context causes event loop conflicts.
    """
    result = TestResult("Synchronous Wrapper (Unit)")
    start = time.time()

    try:
        from isa_agent_sdk import ISAAgentClientSync, ISAAgentOptions

        # Unit test: verify class is importable and instantiable
        options = ISAAgentOptions(
            model="gpt-4.1-nano",
            max_iterations=5
        )

        # Create instance (don't enter context - that requires event loop)
        sync_client = ISAAgentClientSync(options=options)

        result.details["class_instantiated"] = True
        result.details["has_async_client"] = sync_client._async_client is not None
        result.details["session_id"] = sync_client.session_id
        result.details["note"] = "Full sync test requires running from sync context (not async test runner)"

        result.passed = (
            sync_client is not None and
            sync_client._async_client is not None
        )

    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()

    result.duration_ms = int((time.time() - start) * 1000)
    return result


# ============================================================================
# Test Runner
# ============================================================================

async def run_all_tests(integration: bool = False):
    """Run all tests and print results"""
    print("=" * 70)
    print("isA Agent SDK - ISAAgentClient Tests")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Integration tests: {'enabled' if integration else 'disabled'}")
    if integration:
        print(f"ISA_MODEL_URL: {os.environ.get('ISA_MODEL_URL', 'not set')}")
        print(f"ISA_MCP_URL: {os.environ.get('ISA_MCP_URL', 'not set')}")
    print("=" * 70)

    # Unit tests (always run)
    unit_tests = [
        ("1. Import", test_client_import),
        ("2. Initialization", test_client_initialization),
        ("3. Properties", test_client_properties),
        ("4. Fork", test_client_fork),
        ("5. Error Classes", test_error_classes),
    ]

    # Integration tests (require services)
    integration_tests = [
        ("6. Context Manager", test_client_context_manager),
        ("7. Query/Receive", test_query_receive_pattern),
        ("8. ask() Method", test_ask_convenience),
        ("9. Multi-turn", test_multi_turn_conversation),
        ("10. Sync Wrapper", test_sync_wrapper),
    ]

    tests = unit_tests
    if integration:
        tests += integration_tests

    results = []

    for test_name, test_func in tests:
        print(f"\n{'─' * 60}")
        print(f"Running: {test_name}")
        print(f"{'─' * 60}")

        try:
            result = await test_func()
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"\n{status} | {result.name} | {result.duration_ms}ms")

            if result.error:
                print(f"   Error: {result.error}")

            for key, value in result.details.items():
                if key != "traceback":
                    print(f"   {key}: {value}")

            if not result.passed and "traceback" in result.details:
                print(f"\n   Traceback:\n{result.details['traceback']}")

        except Exception as e:
            print(f"TEST CRASHED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    print()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name} ({r.duration_ms}ms)")

    print("\n" + "=" * 70)

    return passed == len(results)


if __name__ == "__main__":
    # Check for --integration flag
    integration = "--integration" in sys.argv or "-i" in sys.argv

    success = asyncio.run(run_all_tests(integration=integration))
    sys.exit(0 if success else 1)
