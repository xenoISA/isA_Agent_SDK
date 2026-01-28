"""
Validation logic for Human-in-the-Loop service
File: app/services/human_in_the_loop/validators.py

This module handles all input validation and response processing.
"""

from typing import Dict, Any, Optional, List
import re


class HILValidator:
    """
    Validator for HIL inputs and responses
    
    Example:
        validator = HILValidator()
        result = validator.validate_input("25", {"type": "int", "min": 0, "max": 120})
        if result["valid"]:
            age = result["value"]  # 25
    """
    
    @staticmethod
    def is_approved(response: Any) -> bool:
        """
        Check if response indicates approval
        
        Args:
            response: User response (bool, dict, or str)
            
        Returns:
            True if approved, False otherwise
            
        Example:
            is_approved(True)  # True
            is_approved({"action": "approve"})  # True
            is_approved("yes")  # True
            is_approved("no")  # False
        """
        if isinstance(response, bool):
            return response
        if isinstance(response, dict):
            return response.get("action") == "approve" or response.get("approved", False)
        if isinstance(response, str):
            return response.lower() in ["yes", "approve", "approved", "accept", "ok"]
        return False
    
    @staticmethod
    def validate_input(input_value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate human input against rules
        
        Args:
            input_value: Value to validate
            rules: Validation rules
                - type: Expected type (int, float, str)
                - min: Minimum value (for numbers)
                - max: Maximum value (for numbers)
                - pattern: Regex pattern (for strings)
                - validator: Custom validation function
                
        Returns:
            Dict with 'valid' (bool), 'value' (if valid), or 'error' (if invalid)
            
        Example:
            # Type validation
            validate_input("25", {"type": "int"})
            # {"valid": True, "value": 25}
            
            # Range validation
            validate_input(150, {"type": "int", "max": 120})
            # {"valid": False, "error": "Value must be <= 120"}
            
            # Pattern validation
            validate_input("test@email", {"type": "str", "pattern": r".*@.*\..*"})
            # {"valid": False, "error": "Value must match pattern: .*@.*\\..*"}
        """
        try:
            # Type validation
            if "type" in rules:
                expected_type = rules["type"]
                if expected_type == "int":
                    value = int(input_value)
                elif expected_type == "float":
                    value = float(input_value)
                elif expected_type == "str":
                    value = str(input_value)
                else:
                    value = input_value
            else:
                value = input_value
            
            # Range validation
            if "min" in rules and value < rules["min"]:
                return {"valid": False, "error": f"Value must be >= {rules['min']}"}
            if "max" in rules and value > rules["max"]:
                return {"valid": False, "error": f"Value must be <= {rules['max']}"}
            
            # Pattern validation
            if "pattern" in rules:
                if not re.match(rules["pattern"], str(value)):
                    return {"valid": False, "error": f"Value must match pattern: {rules['pattern']}"}
            
            # Custom validation
            if "validator" in rules:
                validator_result = rules["validator"](value)
                if not validator_result:
                    return {"valid": False, "error": "Custom validation failed"}
            
            return {"valid": True, "value": value}
            
        except (ValueError, TypeError) as e:
            return {"valid": False, "error": f"Invalid input: {e}"}
    
    @staticmethod
    def validate_edited_content(
        content: Any,
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate edited content structure
        
        Args:
            content: Content to validate
            required_fields: List of required fields (for dict content)
            
        Returns:
            Dict with validation result
            
        Example:
            validate_edited_content(
                {"name": "John", "age": 25},
                required_fields=["name", "age"]
            )
            # {"edited_content": {"name": "John", "age": 25}, "valid": True}
            
            validate_edited_content(
                {"name": "John"},
                required_fields=["name", "age"]
            )
            # {"error": "Missing fields: ['age']", "valid": False}
        """
        if not isinstance(content, dict):
            return {"edited_content": str(content), "valid": True}
        
        if required_fields:
            missing_fields = [field for field in required_fields if field not in content]
            if missing_fields:
                return {"error": f"Missing fields: {missing_fields}", "valid": False}
        
        return {"edited_content": content, "valid": True}
    
    @staticmethod
    def process_interrupt_response(
        interrupt_data: Dict[str, Any],
        response: Any
    ) -> Dict[str, Any]:
        """
        Process response for specific interrupt type
        
        Args:
            interrupt_data: Original interrupt data
            response: User response
            
        Returns:
            Processed response
            
        Example:
            process_interrupt_response(
                {"type": "approval", "question": "Approve?"},
                "yes"
            )
            # {"approved": True}
        """
        interrupt_type = interrupt_data.get("type")
        
        if interrupt_type == "approval":
            return {"approved": HILValidator.is_approved(response)}
        elif interrupt_type == "review_edit":
            return HILValidator.validate_edited_content(
                response,
                interrupt_data.get("required_fields")
            )
        elif interrupt_type == "input_validation":
            return HILValidator.validate_input(
                response,
                interrupt_data.get("validation_rules", {})
            )
        else:
            return {"response": response, "processed": True}


class ValidationRulesBuilder:
    """
    Builder for validation rules
    
    Example:
        rules = (ValidationRulesBuilder()
            .type("int")
            .min(0)
            .max(120)
            .build())
        # {"type": "int", "min": 0, "max": 120}
    """
    
    def __init__(self):
        self._rules: Dict[str, Any] = {}
    
    def type(self, type_name: str) -> "ValidationRulesBuilder":
        """Set expected type"""
        self._rules["type"] = type_name
        return self
    
    def min(self, min_value: Any) -> "ValidationRulesBuilder":
        """Set minimum value"""
        self._rules["min"] = min_value
        return self
    
    def max(self, max_value: Any) -> "ValidationRulesBuilder":
        """Set maximum value"""
        self._rules["max"] = max_value
        return self
    
    def pattern(self, regex_pattern: str) -> "ValidationRulesBuilder":
        """Set regex pattern"""
        self._rules["pattern"] = regex_pattern
        return self
    
    def validator(self, validator_func) -> "ValidationRulesBuilder":
        """Set custom validator function"""
        self._rules["validator"] = validator_func
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build validation rules"""
        return self._rules

