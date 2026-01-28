"""
Test script for HIL Timeout Handler
File: app/services/human_in_the_loop/tests/test_timeout_handler.py

Tests timeout functionality for Human-in-the-Loop scenarios.

Usage:
    python app/services/human_in_the_loop/tests/test_timeout_handler.py

    # Or with pytest
    python -m pytest app/services/human_in_the_loop/tests/test_timeout_handler.py -v
"""

import sys
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, '/Users/xenodennis/Documents/Fun/isA_Agent')

from isa_agent_sdk.services.human_in_the_loop.timeout_handler import (
    TimeoutHandler,
    get_timeout_handler,
    reset_timeout_handler,
    TIMEOUT_CONFIG
)


class TestTimeoutConfig:
    """Test timeout configuration"""

    def test_timeout_config_values(self):
        """Test that timeout config has expected values"""
        assert TIMEOUT_CONFIG["execution_choice"] == 30
        assert TIMEOUT_CONFIG["input"] == 60
        assert TIMEOUT_CONFIG["authorization"] == 120
        assert TIMEOUT_CONFIG["review"] == 120
        assert TIMEOUT_CONFIG["input_with_auth"] == 120
        print("✅ Timeout config values are correct")

    def test_timeout_config_reasonable(self):
        """Test that timeout values are reasonable (not too short/long)"""
        for scenario, timeout in TIMEOUT_CONFIG.items():
            assert 10 <= timeout <= 300, f"Timeout for {scenario} is unreasonable: {timeout}s"
        print("✅ All timeout values are reasonable (10-300s)")


class TestTimeoutHandler:
    """Test TimeoutHandler class"""

    def test_get_timeout_default(self):
        """Test getting default timeout"""
        handler = TimeoutHandler()

        timeout = handler.get_timeout("authorization")
        assert timeout == 120

        timeout = handler.get_timeout("input")
        assert timeout == 60

        timeout = handler.get_timeout("execution_choice")
        assert timeout == 30

        print("✅ Get default timeout works")

    def test_get_timeout_custom(self):
        """Test getting custom timeout"""
        handler = TimeoutHandler()

        # Custom timeout should override default
        timeout = handler.get_timeout("authorization", custom_timeout=180)
        assert timeout == 180

        timeout = handler.get_timeout("input", custom_timeout=90)
        assert timeout == 90

        print("✅ Custom timeout override works")

    def test_get_timeout_unknown_scenario(self):
        """Test getting timeout for unknown scenario (should return default 60s)"""
        handler = TimeoutHandler()

        timeout = handler.get_timeout("unknown_scenario")
        assert timeout == 60  # Default fallback

        print("✅ Unknown scenario returns default 60s")

    async def test_with_timeout_success(self):
        """Test successful execution within timeout"""
        handler = TimeoutHandler()

        async def quick_task():
            await asyncio.sleep(0.1)
            return "success"

        result = await handler.with_timeout(
            coro=quick_task(),
            timeout=1,
            scenario="test_scenario",
            default_value="default"
        )

        assert result == "success"
        print("✅ with_timeout success case works")

    async def test_with_timeout_timeout(self):
        """Test timeout case (slow task)"""
        handler = TimeoutHandler()

        async def slow_task():
            await asyncio.sleep(2)
            return "should_not_reach"

        result = await handler.with_timeout(
            coro=slow_task(),
            timeout=0.5,  # Short timeout
            scenario="test_scenario",
            default_value="default_value"
        )

        assert result == "default_value"
        assert len(handler.timeout_events) == 1
        assert handler.timeout_events[0]["scenario"] == "test_scenario"
        print("✅ with_timeout timeout case works")

    def test_get_default_behavior_execution_choice(self):
        """Test default behavior for execution_choice"""
        handler = TimeoutHandler()

        data = {"recommendation": "background"}
        result = handler.get_default_behavior("execution_choice", data)

        assert result == "background"
        print("✅ execution_choice default behavior works")

    def test_get_default_behavior_input(self):
        """Test default behavior for input"""
        handler = TimeoutHandler()

        # With default_value
        data = {"default_value": "test_default"}
        result = handler.get_default_behavior("input", data)
        assert result == "test_default"

        # Without default_value
        data = {}
        result = handler.get_default_behavior("input", data)
        assert result is None

        print("✅ input default behavior works")

    def test_get_default_behavior_authorization(self):
        """Test default behavior for authorization (should reject)"""
        handler = TimeoutHandler()

        data = {}
        result = handler.get_default_behavior("authorization", data)

        assert result is False  # Reject on timeout
        print("✅ authorization default behavior works (rejects)")

    def test_get_default_behavior_review(self):
        """Test default behavior for review"""
        handler = TimeoutHandler()

        data = {"content": "original_content"}
        result = handler.get_default_behavior("review", data)

        assert result["approved"] is True
        assert result["edited_content"] == "original_content"
        assert result["action"] == "approve_by_timeout"
        print("✅ review default behavior works")

    def test_timeout_stats(self):
        """Test timeout statistics"""
        handler = TimeoutHandler()

        # Manually add timeout events
        handler.timeout_events = [
            {"scenario": "authorization", "timeout": 120, "timestamp": "2025-01-01T10:00:00"},
            {"scenario": "authorization", "timeout": 120, "timestamp": "2025-01-01T10:01:00"},
            {"scenario": "input", "timeout": 60, "timestamp": "2025-01-01T10:02:00"},
        ]

        stats = handler.get_timeout_stats()

        assert stats["total"] == 3
        assert stats["by_scenario"]["authorization"] == 2
        assert stats["by_scenario"]["input"] == 1
        assert len(stats["recent_events"]) == 3

        print("✅ Timeout statistics work")

    def test_clear_timeout_history(self):
        """Test clearing timeout history"""
        handler = TimeoutHandler()

        handler.timeout_events = [{"test": "data"}]
        handler.clear_timeout_history()

        assert len(handler.timeout_events) == 0
        print("✅ Clear timeout history works")


class TestGlobalTimeoutHandler:
    """Test global timeout handler singleton"""

    def test_get_timeout_handler_singleton(self):
        """Test that get_timeout_handler returns singleton"""
        reset_timeout_handler()  # Reset first

        handler1 = get_timeout_handler()
        handler2 = get_timeout_handler()

        assert handler1 is handler2  # Same instance
        print("✅ Global timeout handler is singleton")

    def test_reset_timeout_handler(self):
        """Test resetting global timeout handler"""
        handler1 = get_timeout_handler()
        reset_timeout_handler()
        handler2 = get_timeout_handler()

        assert handler1 is not handler2  # Different instances
        print("✅ Reset timeout handler works")


class TestTimeoutIntegration:
    """Integration tests with realistic scenarios"""

    async def test_authorization_timeout_scenario(self):
        """Test authorization timeout (should reject)"""
        handler = TimeoutHandler()

        async def wait_for_human_authorization():
            # Simulate waiting for human (never responds)
            await asyncio.sleep(5)
            return True

        # Authorization with 0.5s timeout
        result = await handler.with_timeout(
            coro=wait_for_human_authorization(),
            timeout=0.5,
            scenario="authorization",
            default_value=False  # Reject on timeout
        )

        assert result is False  # Should be rejected
        print("✅ Authorization timeout rejects (safety first)")

    async def test_execution_choice_timeout_scenario(self):
        """Test execution choice timeout (should use recommendation)"""
        handler = TimeoutHandler()

        async def wait_for_execution_choice():
            # Simulate waiting for user choice
            await asyncio.sleep(5)
            return "comprehensive"

        # Get timeout for execution_choice
        timeout = handler.get_timeout("execution_choice")
        assert timeout == 30  # Should be 30s

        # Execution choice with 0.5s timeout
        result = await handler.with_timeout(
            coro=wait_for_execution_choice(),
            timeout=0.5,
            scenario="execution_choice",
            default_value="background"  # Default recommendation
        )

        assert result == "background"
        print("✅ Execution choice timeout uses recommendation")

    async def test_input_with_default_value(self):
        """Test input timeout with default value"""
        handler = TimeoutHandler()

        async def wait_for_input():
            await asyncio.sleep(5)
            return "user_input"

        # Input with default value
        default_value = "default_api_key"
        result = await handler.with_timeout(
            coro=wait_for_input(),
            timeout=0.5,
            scenario="input",
            default_value=default_value
        )

        assert result == "default_api_key"
        print("✅ Input timeout uses default_value")


def run_sync_tests():
    """Run synchronous tests"""
    print("\n" + "="*60)
    print("Running Timeout Handler Tests (Sync)")
    print("="*60 + "\n")

    # Config tests
    print("📋 Testing timeout configuration...")
    config_tests = TestTimeoutConfig()
    config_tests.test_timeout_config_values()
    config_tests.test_timeout_config_reasonable()

    # Handler tests
    print("\n⏱️ Testing TimeoutHandler...")
    handler_tests = TestTimeoutHandler()
    handler_tests.test_get_timeout_default()
    handler_tests.test_get_timeout_custom()
    handler_tests.test_get_timeout_unknown_scenario()
    handler_tests.test_get_default_behavior_execution_choice()
    handler_tests.test_get_default_behavior_input()
    handler_tests.test_get_default_behavior_authorization()
    handler_tests.test_get_default_behavior_review()
    handler_tests.test_timeout_stats()
    handler_tests.test_clear_timeout_history()

    # Global handler tests
    print("\n🌍 Testing global timeout handler...")
    global_tests = TestGlobalTimeoutHandler()
    global_tests.test_get_timeout_handler_singleton()
    global_tests.test_reset_timeout_handler()


async def run_async_tests():
    """Run asynchronous tests"""
    print("\n" + "="*60)
    print("Running Timeout Handler Tests (Async)")
    print("="*60 + "\n")

    print("⏱️ Testing async timeout functionality...")
    handler_tests = TestTimeoutHandler()
    await handler_tests.test_with_timeout_success()
    await handler_tests.test_with_timeout_timeout()

    print("\n🔗 Testing integration scenarios...")
    integration_tests = TestTimeoutIntegration()
    await integration_tests.test_authorization_timeout_scenario()
    await integration_tests.test_execution_choice_timeout_scenario()
    await integration_tests.test_input_with_default_value()


def main():
    """Run all tests"""
    print("\n🧪 Testing Timeout Handler for HIL Service")
    print("=" * 60)

    # Run sync tests
    run_sync_tests()

    # Run async tests
    asyncio.run(run_async_tests())

    print("\n" + "="*60)
    print("✅ All Timeout Handler Tests Passed!")
    print("="*60 + "\n")

    # Print summary
    print("📊 Test Summary:")
    print("   - Timeout configuration: ✅")
    print("   - TimeoutHandler methods: ✅")
    print("   - Global singleton: ✅")
    print("   - Async timeout handling: ✅")
    print("   - Integration scenarios: ✅")
    print("\n🎉 Timeout handler is working correctly!\n")


if __name__ == "__main__":
    main()
