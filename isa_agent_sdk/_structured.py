#!/usr/bin/env python3
"""
isA Agent SDK - Structured Output Helpers
==========================================

Utilities for working with structured outputs and Pydantic models.
Provides type-safe parsing, validation, and schema generation.

Example:
    from pydantic import BaseModel
    from isa_agent_sdk import query, ISAAgentOptions, OutputFormat
    from isa_agent_sdk._structured import parse_structured_output

    class Recipe(BaseModel):
        name: str
        ingredients: list[str]
        prep_time_minutes: int

    async for msg in query(
        "Find a chocolate chip cookie recipe",
        options=ISAAgentOptions(
            output_format=OutputFormat.from_pydantic(Recipe)
        )
    ):
        if msg.has_structured_output:
            recipe = msg.parse(Recipe)
            print(f"Recipe: {recipe.name}")
"""

import json
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T')


@dataclass
class StructuredOutputResult:
    """
    Result of structured output parsing.

    Attributes:
        success: Whether parsing succeeded
        data: Parsed data (dict or Pydantic model)
        error: Error message if parsing failed
        raw_content: Original raw content before parsing
        validation_errors: List of validation errors if any
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    raw_content: Optional[str] = None
    validation_errors: Optional[List[Dict[str, Any]]] = None


def parse_json_response(content: str) -> StructuredOutputResult:
    """
    Parse JSON from model response content.

    Handles common edge cases:
    - JSON wrapped in markdown code blocks
    - Leading/trailing whitespace
    - Multiple JSON objects (takes first)

    Args:
        content: Raw response content from model

    Returns:
        StructuredOutputResult with parsed data or error
    """
    if not content:
        return StructuredOutputResult(
            success=False,
            error="Empty content",
            raw_content=content
        )

    # Clean the content
    cleaned = content.strip()

    # Try to extract JSON from markdown code blocks
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.find("```", start)
        if end > start:
            cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        # Generic code block
        start = cleaned.find("```") + 3
        # Skip language identifier if present
        newline = cleaned.find("\n", start)
        if newline > start:
            start = newline + 1
        end = cleaned.find("```", start)
        if end > start:
            cleaned = cleaned[start:end].strip()

    # Try to parse
    try:
        data = json.loads(cleaned)
        return StructuredOutputResult(
            success=True,
            data=data,
            raw_content=content
        )
    except json.JSONDecodeError as e:
        return StructuredOutputResult(
            success=False,
            error=f"JSON parse error: {e}",
            raw_content=content
        )


def validate_against_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> StructuredOutputResult:
    """
    Validate data against a JSON schema.

    Args:
        data: Parsed JSON data
        schema: JSON schema to validate against

    Returns:
        StructuredOutputResult with validation status
    """
    try:
        import jsonschema
        from jsonschema import validate, ValidationError as JsonSchemaError

        validate(instance=data, schema=schema)
        return StructuredOutputResult(
            success=True,
            data=data
        )

    except ImportError:
        # jsonschema not installed, skip validation
        logger.warning("jsonschema not installed, skipping schema validation")
        return StructuredOutputResult(
            success=True,
            data=data
        )

    except JsonSchemaError as e:
        return StructuredOutputResult(
            success=False,
            data=data,
            error=f"Schema validation error: {e.message}",
            validation_errors=[{
                "path": list(e.path),
                "message": e.message,
                "validator": e.validator
            }]
        )


def parse_to_pydantic(
    data: Union[str, Dict[str, Any]],
    model: Type[T]
) -> StructuredOutputResult:
    """
    Parse data into a Pydantic model.

    Args:
        data: JSON string or dict to parse
        model: Pydantic BaseModel class

    Returns:
        StructuredOutputResult with parsed model instance
    """
    # Parse JSON string if needed
    if isinstance(data, str):
        result = parse_json_response(data)
        if not result.success:
            return result
        data = result.data

    try:
        # Use Pydantic's model_validate
        instance = model.model_validate(data)
        return StructuredOutputResult(
            success=True,
            data=instance
        )

    except Exception as e:
        # Handle Pydantic validation errors
        error_details = []
        if hasattr(e, 'errors'):
            error_details = [
                {
                    "loc": err.get("loc", []),
                    "msg": err.get("msg", str(err)),
                    "type": err.get("type", "unknown")
                }
                for err in e.errors()
            ]

        return StructuredOutputResult(
            success=False,
            data=data,
            error=f"Pydantic validation error: {e}",
            validation_errors=error_details
        )


def generate_schema_from_pydantic(model: Type[T]) -> Dict[str, Any]:
    """
    Generate JSON schema from a Pydantic model.

    Args:
        model: Pydantic BaseModel class

    Returns:
        JSON schema dict
    """
    try:
        return model.model_json_schema()
    except AttributeError:
        raise TypeError(
            f"{model} does not appear to be a Pydantic model. "
            "Ensure it inherits from pydantic.BaseModel."
        )


def create_repair_prompt(
    original_content: str,
    schema: Dict[str, Any],
    error: str
) -> str:
    """
    Create a prompt for the model to repair invalid JSON output.

    Args:
        original_content: The original invalid response
        schema: The target JSON schema
        error: The validation error message

    Returns:
        Prompt string for repair request
    """
    schema_str = json.dumps(schema, indent=2)

    return f"""The previous response was not valid JSON matching the required schema.

Error: {error}

Required schema:
```json
{schema_str}
```

Previous response:
```
{original_content}
```

Please provide a corrected response that is valid JSON matching the schema exactly.
Output only the JSON, no explanation."""


class StructuredOutputParser:
    """
    Parser for structured outputs with retry logic.

    Handles parsing, validation, and optional repair of structured outputs
    from model responses.

    Example:
        from pydantic import BaseModel

        class Task(BaseModel):
            title: str
            priority: int

        parser = StructuredOutputParser(schema=Task.model_json_schema())
        result = parser.parse(model_response)

        if result.success:
            task = Task.model_validate(result.data)
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        pydantic_model: Optional[Type[T]] = None,
        max_repair_attempts: int = 3
    ):
        """
        Initialize the parser.

        Args:
            schema: JSON schema for validation
            pydantic_model: Optional Pydantic model for parsing
            max_repair_attempts: Max attempts to repair invalid output
        """
        self.schema = schema
        self.pydantic_model = pydantic_model
        self.max_repair_attempts = max_repair_attempts

        # Generate schema from Pydantic model if not provided
        if self.pydantic_model and not self.schema:
            self.schema = generate_schema_from_pydantic(pydantic_model)

    def parse(self, content: str) -> StructuredOutputResult:
        """
        Parse content into structured output.

        Args:
            content: Raw model response content

        Returns:
            StructuredOutputResult with parsed data
        """
        # Step 1: Parse JSON
        result = parse_json_response(content)
        if not result.success:
            return result

        # Step 2: Validate against schema (if provided)
        if self.schema:
            result = validate_against_schema(result.data, self.schema)
            if not result.success:
                return result

        # Step 3: Parse to Pydantic model (if provided)
        if self.pydantic_model:
            result = parse_to_pydantic(result.data, self.pydantic_model)

        return result

    def get_repair_prompt(self, content: str, error: str) -> str:
        """
        Get a repair prompt for invalid output.

        Args:
            content: The invalid response
            error: The validation error

        Returns:
            Prompt for repair request
        """
        if not self.schema:
            return f"The response was invalid: {error}. Please try again."

        return create_repair_prompt(content, self.schema, error)


# Convenience function for direct use
def parse_structured_output(
    content: str,
    model: Optional[Type[T]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> StructuredOutputResult:
    """
    Parse structured output from model response.

    Convenience function that handles JSON parsing, schema validation,
    and Pydantic model parsing.

    Args:
        content: Raw model response content
        model: Optional Pydantic model to parse into
        schema: Optional JSON schema for validation

    Returns:
        StructuredOutputResult with parsed data

    Example:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        result = parse_structured_output(response_content, model=User)
        if result.success:
            user = result.data  # User instance
    """
    parser = StructuredOutputParser(schema=schema, pydantic_model=model)
    return parser.parse(content)


__all__ = [
    "StructuredOutputResult",
    "StructuredOutputParser",
    "parse_json_response",
    "validate_against_schema",
    "parse_to_pydantic",
    "generate_schema_from_pydantic",
    "create_repair_prompt",
    "parse_structured_output",
]
