"""
Tests for query_sync, ask_sync, and resume_sync async context detection.

These tests verify that the sync wrappers properly detect async contexts
and raise RuntimeError when called incorrectly.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from isa_agent_sdk._query import query_sync, ask_sync, resume_sync
from isa_agent_sdk._messages import AgentMessage


# ---------------------------------------------------------------------------
# Test: query_sync from async context raises RuntimeError
# ---------------------------------------------------------------------------

class TestQuerySyncFromAsyncContext:
    """Test that query_sync raises RuntimeError when called from async context."""

    @pytest.mark.asyncio
    async def test_query_sync_from_async_context(self):
        """query_sync() should raise RuntimeError when called from async context."""
        with pytest.raises(RuntimeError) as exc_info:
            # This should raise because we're in an async context
            list(query_sync("test prompt"))

        assert "cannot be called from an async context" in str(exc_info.value)
        assert "Use query() (async) instead" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ask_sync_from_async_context(self):
        """ask_sync() should raise RuntimeError when called from async context."""
        with pytest.raises(RuntimeError) as exc_info:
            ask_sync("test prompt")

        assert "cannot be called from an async context" in str(exc_info.value)
        assert "Use ask() (async) instead" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resume_sync_from_async_context(self):
        """resume_sync() should raise RuntimeError when called from async context."""
        with pytest.raises(RuntimeError) as exc_info:
            list(resume_sync("session_123"))

        assert "cannot be called from an async context" in str(exc_info.value)
        assert "Use resume() (async) instead" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: query_sync concurrent calls (no deadlock)
# ---------------------------------------------------------------------------

class TestQuerySyncConcurrentCalls:
    """Test that query_sync can be called multiple times without deadlock."""

    def test_query_sync_concurrent_calls(self):
        """Multiple calls to query_sync should not deadlock.

        This test verifies that:
        1. query_sync properly cleans up its event loop
        2. Subsequent calls work correctly
        3. No deadlocks occur from stale event loop state
        """
        call_count = 0

        async def mock_query_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            yield AgentMessage.text(f"Response {call_count}", session_id="test")
            yield AgentMessage.session_end("test")

        with patch('isa_agent_sdk._query.query', mock_query_impl):
            # First call
            messages1 = list(query_sync("prompt 1"))
            assert len(messages1) == 2
            assert "Response 1" in messages1[0].content

            # Second call (should not deadlock)
            messages2 = list(query_sync("prompt 2"))
            assert len(messages2) == 2
            assert "Response 2" in messages2[0].content

            # Third call (verifying cleanup is consistent)
            messages3 = list(query_sync("prompt 3"))
            assert len(messages3) == 2
            assert "Response 3" in messages3[0].content

    def test_ask_sync_concurrent_calls(self):
        """Multiple calls to ask_sync should not deadlock."""
        call_count = 0

        async def mock_ask_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"Answer {call_count}"

        with patch('isa_agent_sdk._query.ask', mock_ask_impl):
            result1 = ask_sync("question 1")
            assert result1 == "Answer 1"

            result2 = ask_sync("question 2")
            assert result2 == "Answer 2"

            result3 = ask_sync("question 3")
            assert result3 == "Answer 3"


# ---------------------------------------------------------------------------
# Test: Message collection pattern works correctly
# ---------------------------------------------------------------------------

class TestMessageCollectionPattern:
    """Test that the collect-then-yield pattern preserves messages."""

    def test_query_sync_preserves_all_messages(self):
        """query_sync should yield all messages collected from async generator."""
        expected_messages = [
            AgentMessage.session_start("test"),
            AgentMessage.text("Hello", session_id="test"),
            AgentMessage.text("World", session_id="test"),
            AgentMessage.tool_use("test_tool", {"arg": "value"}, session_id="test"),
            AgentMessage.tool_result("test_tool", {"result": "ok"}, session_id="test"),
            AgentMessage.result("Final answer", session_id="test"),
            AgentMessage.session_end("test"),
        ]

        async def mock_query_impl(*args, **kwargs):
            for msg in expected_messages:
                yield msg

        with patch('isa_agent_sdk._query.query', mock_query_impl):
            collected = list(query_sync("test"))

            assert len(collected) == len(expected_messages)
            for i, msg in enumerate(collected):
                assert msg.type == expected_messages[i].type
                assert msg.content == expected_messages[i].content

    def test_resume_sync_preserves_all_messages(self):
        """resume_sync should yield all messages collected from async generator."""
        expected_messages = [
            AgentMessage.session_start("session_123", resumed=True),
            AgentMessage.text("Resumed execution", session_id="session_123"),
            AgentMessage.session_end("session_123", resumed=True),
        ]

        async def mock_resume_impl(*args, **kwargs):
            for msg in expected_messages:
                yield msg

        with patch('isa_agent_sdk._query.resume', mock_resume_impl):
            collected = list(resume_sync("session_123", {"continue": True}))

            assert len(collected) == len(expected_messages)
            for i, msg in enumerate(collected):
                assert msg.type == expected_messages[i].type


# ---------------------------------------------------------------------------
# Test: Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    """Test that errors in async code propagate correctly."""

    def test_query_sync_propagates_errors(self):
        """Errors from query() should propagate through query_sync()."""
        async def mock_query_error(*args, **kwargs):
            yield AgentMessage.session_start("test")
            raise ValueError("Test error from async code")

        with patch('isa_agent_sdk._query.query', mock_query_error):
            with pytest.raises(ValueError) as exc_info:
                list(query_sync("test"))

            assert "Test error from async code" in str(exc_info.value)

    def test_ask_sync_propagates_errors(self):
        """Errors from ask() should propagate through ask_sync()."""
        async def mock_ask_error(*args, **kwargs):
            raise RuntimeError("Async ask failed")

        with patch('isa_agent_sdk._query.ask', mock_ask_error):
            with pytest.raises(RuntimeError) as exc_info:
                ask_sync("test")

            assert "Async ask failed" in str(exc_info.value)
