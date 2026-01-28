"""
Comprehensive test script for HIL Service (Refactored)
File: app/services/human_in_the_loop/tests/test_hil_service.py

This script tests all functionality of the refactored HIL service.

Usage:
    python -m pytest app/services/human_in_the_loop/tests/test_hil_service.py -v
    
    # Or run directly with Python
    python app/services/human_in_the_loop/tests/test_hil_service.py
"""

import sys
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import pytest

# Add parent directory to path
sys.path.insert(0, '/Users/xenodennis/Documents/Fun/isA_Agent')

from isa_agent_sdk.services.human_in_the_loop import (
    HILService,
    get_hil_service,
    reset_hil_service,
    InterruptType,
    InterventionType,
    SecurityLevel,
    HILValidator,
    ValidationRulesBuilder
)


class TestDataModels:
    """Test data models"""
    
    def test_interrupt_type_enum(self):
        """Test InterruptType enum"""
        assert InterruptType.APPROVAL.value == "approval"
        assert InterruptType.ASK_HUMAN.value == "ask_human"
        assert InterruptType.TOOL_AUTHORIZATION.value == "tool_authorization"
        print("✓ InterruptType enum works correctly")
    
    def test_intervention_type_enum(self):
        """Test InterventionType enum"""
        assert InterventionType.CAPTCHA.value == "captcha"
        assert InterventionType.LOGIN.value == "login"
        assert InterventionType.PAYMENT.value == "payment"
        print("✓ InterventionType enum works correctly")
    
    def test_security_level_enum(self):
        """Test SecurityLevel enum"""
        assert SecurityLevel.LOW.value == "LOW"
        assert SecurityLevel.HIGH.value == "HIGH"
        assert SecurityLevel.CRITICAL.value == "CRITICAL"
        print("✓ SecurityLevel enum works correctly")


class TestValidators:
    """Test validation logic"""
    
    def test_is_approved_bool(self):
        """Test approval detection with boolean"""
        validator = HILValidator()
        assert validator.is_approved(True) == True
        assert validator.is_approved(False) == False
        print("✓ Boolean approval detection works")
    
    def test_is_approved_dict(self):
        """Test approval detection with dict"""
        validator = HILValidator()
        assert validator.is_approved({"action": "approve"}) == True
        assert validator.is_approved({"approved": True}) == True
        assert validator.is_approved({"action": "reject"}) == False
        print("✓ Dict approval detection works")
    
    def test_is_approved_string(self):
        """Test approval detection with string"""
        validator = HILValidator()
        assert validator.is_approved("yes") == True
        assert validator.is_approved("approve") == True
        assert validator.is_approved("no") == False
        assert validator.is_approved("reject") == False
        print("✓ String approval detection works")
    
    def test_validate_input_type_int(self):
        """Test input validation - integer type"""
        validator = HILValidator()
        result = validator.validate_input("25", {"type": "int"})
        assert result["valid"] == True
        assert result["value"] == 25
        print("✓ Integer type validation works")
    
    def test_validate_input_range(self):
        """Test input validation - range"""
        validator = HILValidator()
        
        # Valid range
        result = validator.validate_input(50, {"type": "int", "min": 0, "max": 100})
        assert result["valid"] == True
        
        # Out of range (too high)
        result = validator.validate_input(150, {"type": "int", "max": 100})
        assert result["valid"] == False
        assert "<=" in result["error"] or "100" in result["error"]
        
        # Out of range (too low)
        result = validator.validate_input(-5, {"type": "int", "min": 0})
        assert result["valid"] == False
        assert ">=" in result["error"] or "0" in result["error"]
        
        print("✓ Range validation works")
    
    def test_validate_input_pattern(self):
        """Test input validation - regex pattern"""
        validator = HILValidator()
        
        # Valid email pattern
        result = validator.validate_input(
            "test@example.com",
            {"type": "str", "pattern": r".*@.*\..*"}
        )
        assert result["valid"] == True
        
        # Invalid email
        result = validator.validate_input(
            "notanemail",
            {"type": "str", "pattern": r".*@.*\..*"}
        )
        assert result["valid"] == False
        
        print("✓ Pattern validation works")
    
    def test_validation_rules_builder(self):
        """Test ValidationRulesBuilder"""
        rules = (ValidationRulesBuilder()
            .type("int")
            .min(0)
            .max(120)
            .build())
        
        assert rules["type"] == "int"
        assert rules["min"] == 0
        assert rules["max"] == 120
        print("✓ ValidationRulesBuilder works")


class TestInterruptManager:
    """Test interrupt manager"""
    
    def test_simple_interrupt_logging(self):
        """Test that interrupts are logged"""
        from isa_agent_sdk.services.human_in_the_loop.interrupt_manager import InterruptManager
        
        manager = InterruptManager()
        
        # Mock the interrupt function
        with patch('app.services.human_in_the_loop.interrupt_manager.interrupt', return_value="test response"):
            response = manager.simple_interrupt("Test question?")
            assert response == "test response"
            assert len(manager.interrupt_history) == 1
            assert manager.interrupt_history[0]["question"] == "Test question?"
        
        print("✓ Simple interrupt logging works")
    
    def test_get_interrupt_stats(self):
        """Test interrupt statistics"""
        from isa_agent_sdk.services.human_in_the_loop.interrupt_manager import InterruptManager
        
        manager = InterruptManager()
        
        # Add some interrupts manually
        manager.interrupt_history = [
            {"type": "approval", "node_source": "tool_node", "timestamp": "2025-01-01"},
            {"type": "approval", "node_source": "tool_node", "timestamp": "2025-01-02"},
            {"type": "ask_human", "node_source": "input_node", "timestamp": "2025-01-03"}
        ]
        
        stats = manager.get_interrupt_stats()
        assert stats.total == 3
        assert stats.by_type["approval"] == 2
        assert stats.by_type["ask_human"] == 1
        assert stats.by_node["tool_node"] == 2
        assert stats.by_node["input_node"] == 1
        
        print("✓ Interrupt statistics work")
    
    def test_clear_history(self):
        """Test clearing interrupt history"""
        from isa_agent_sdk.services.human_in_the_loop.interrupt_manager import InterruptManager
        
        manager = InterruptManager()
        manager.interrupt_history = [{"test": "data"}]
        
        manager.clear_history()
        assert len(manager.interrupt_history) == 0
        
        print("✓ Clear history works")


class TestHILService:
    """Test main HIL service"""
    
    def test_service_initialization(self):
        """Test service initialization"""
        reset_hil_service()
        hil = HILService()
        
        assert hil.interrupt_manager is not None
        assert hil.scenario_handler is not None
        assert hil.validator is not None
        
        print("✓ Service initialization works")
    
    def test_get_hil_service_singleton(self):
        """Test singleton pattern"""
        reset_hil_service()
        
        hil1 = get_hil_service()
        hil2 = get_hil_service()
        
        assert hil1 is hil2
        print("✓ Singleton pattern works")
    
    @pytest.mark.asyncio
    async def test_collect_user_input_mock(self):
        """Test collect_user_input with mock"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt function
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="test@example.com"):
            result = await hil.collect_user_input("What is your email?")
            assert result == "test@example.com"
        
        print("✓ collect_user_input works")
    
    @pytest.mark.asyncio
    async def test_collect_user_input_with_validation(self):
        """Test collect_user_input with validation"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt to return a valid integer
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="25"):
            result = await hil.collect_user_input(
                "What is your age?",
                validation_rules={"type": "int", "min": 0, "max": 120}
            )
            assert result == 25
        
        print("✓ collect_user_input with validation works")
    
    @pytest.mark.asyncio
    async def test_request_tool_permission_approved(self):
        """Test request_tool_permission - approved"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt to return approval
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="approve"):
            authorized = await hil.request_tool_permission(
                tool_name="web_crawl",
                tool_args={"url": "https://example.com"},
                security_level="HIGH",
                reason="Need to scrape data"
            )
            assert authorized == True
        
        print("✓ request_tool_permission (approved) works")
    
    @pytest.mark.asyncio
    async def test_request_tool_permission_denied(self):
        """Test request_tool_permission - denied"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt to return denial
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="reject"):
            authorized = await hil.request_tool_permission(
                tool_name="web_crawl",
                tool_args={"url": "https://example.com"},
                security_level="HIGH",
                reason="Need to scrape data"
            )
            assert authorized == False
        
        print("✓ request_tool_permission (denied) works")
    
    @pytest.mark.asyncio
    async def test_request_oauth_authorization(self):
        """Test request_oauth_authorization"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt to return auth code
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="auth_code_123"):
            result = await hil.request_oauth_authorization(
                provider="gmail",
                oauth_url="https://accounts.google.com/o/oauth2/auth",
                scopes=["read", "send"]
            )
            assert result["provider"] == "gmail"
            assert result["authorized"] == True
            assert result["response"] == "auth_code_123"
        
        print("✓ request_oauth_authorization works")
    
    @pytest.mark.asyncio
    async def test_request_credential_usage(self):
        """Test request_credential_usage"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt to return approval
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="yes"):
            approved = await hil.request_credential_usage(
                provider="google",
                credential_preview={
                    "vault_id": "vault_123",
                    "stored_at": "2025-01-15"
                },
                auth_type="social"
            )
            assert approved == True
        
        print("✓ request_credential_usage works")
    
    @pytest.mark.asyncio
    async def test_request_manual_intervention(self):
        """Test request_manual_intervention"""
        reset_hil_service()
        hil = get_hil_service()
        
        # Mock the interrupt to return completion signal
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="completed"):
            result = await hil.request_manual_intervention(
                intervention_type="captcha",
                provider="google",
                instructions="Please solve the reCAPTCHA",
                screenshot_path="/tmp/captcha.png"
            )
            assert result == "completed"
        
        print("✓ request_manual_intervention works")
    
    def test_get_interrupt_stats(self):
        """Test get_interrupt_stats"""
        reset_hil_service()
        hil = get_hil_service()
        
        stats = hil.get_interrupt_stats()
        assert stats.total >= 0
        assert isinstance(stats.by_type, dict)
        assert isinstance(stats.by_node, dict)
        
        print("✓ get_interrupt_stats works")


class TestLegacyMethods:
    """Test legacy methods for backward compatibility"""
    
    @pytest.mark.asyncio
    async def test_ask_human_via_mcp_with_interrupt(self):
        """Test legacy ask_human_via_mcp_with_interrupt"""
        reset_hil_service()
        hil = get_hil_service()
        
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="legacy response"):
            result = await hil.ask_human_via_mcp_with_interrupt("Legacy question?")
            assert result == "legacy response"
        
        print("✓ Legacy ask_human_via_mcp_with_interrupt works")
    
    @pytest.mark.asyncio
    async def test_request_tool_authorization_legacy(self):
        """Test legacy request_tool_authorization"""
        reset_hil_service()
        hil = get_hil_service()
        
        with patch('app.services.human_in_the_loop.scenario_handlers.interrupt', return_value="approve"):
            result = await hil.request_tool_authorization(
                tool_name="test_tool",
                tool_args={},
                reason="test"
            )
            assert result["authorized"] == True
        
        print("✓ Legacy request_tool_authorization works")


def run_sync_tests():
    """Run synchronous tests"""
    print("\n" + "="*60)
    print("Running Synchronous Tests")
    print("="*60 + "\n")
    
    # Data models
    print("Testing Data Models...")
    test_models = TestDataModels()
    test_models.test_interrupt_type_enum()
    test_models.test_intervention_type_enum()
    test_models.test_security_level_enum()
    
    # Validators
    print("\nTesting Validators...")
    test_validators = TestValidators()
    test_validators.test_is_approved_bool()
    test_validators.test_is_approved_dict()
    test_validators.test_is_approved_string()
    test_validators.test_validate_input_type_int()
    test_validators.test_validate_input_range()
    test_validators.test_validate_input_pattern()
    test_validators.test_validation_rules_builder()
    
    # Interrupt Manager
    print("\nTesting Interrupt Manager...")
    test_manager = TestInterruptManager()
    test_manager.test_simple_interrupt_logging()
    test_manager.test_get_interrupt_stats()
    test_manager.test_clear_history()
    
    # HIL Service (sync methods)
    print("\nTesting HIL Service...")
    test_service = TestHILService()
    test_service.test_service_initialization()
    test_service.test_get_hil_service_singleton()
    test_service.test_get_interrupt_stats()


async def run_async_tests():
    """Run asynchronous tests"""
    print("\n" + "="*60)
    print("Running Asynchronous Tests")
    print("="*60 + "\n")
    
    test_service = TestHILService()
    
    print("Testing Scenario Methods...")
    await test_service.test_collect_user_input_mock()
    await test_service.test_collect_user_input_with_validation()
    await test_service.test_request_tool_permission_approved()
    await test_service.test_request_tool_permission_denied()
    await test_service.test_request_oauth_authorization()
    await test_service.test_request_credential_usage()
    await test_service.test_request_manual_intervention()
    
    print("\nTesting Legacy Methods...")
    test_legacy = TestLegacyMethods()
    await test_legacy.test_ask_human_via_mcp_with_interrupt()
    await test_legacy.test_request_tool_authorization_legacy()


def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("HIL Service Refactored - Comprehensive Test Suite")
    print("="*60)
    
    try:
        # Run synchronous tests
        run_sync_tests()
        
        # Run asynchronous tests
        asyncio.run(run_async_tests())
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")
        
        print("Summary:")
        print("  - Data Models: ✓")
        print("  - Validators: ✓")
        print("  - Interrupt Manager: ✓")
        print("  - HIL Service: ✓")
        print("  - Scenario Methods: ✓")
        print("  - Legacy Methods: ✓")
        print("\nThe refactored HIL service is working correctly!")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

