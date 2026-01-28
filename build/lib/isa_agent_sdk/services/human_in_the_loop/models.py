"""
Data models for Human-in-the-Loop service
File: app/services/human_in_the_loop/models.py

This module defines all data structures used by the HIL service.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class InterruptType(Enum):
    """Types of interrupts supported by HIL service"""
    APPROVAL = "approval"
    REVIEW_EDIT = "review_edit"
    INPUT_VALIDATION = "input_validation"
    AUTHORIZATION = "authorization"
    ASK_HUMAN = "ask_human"
    COLLECT_USER_INPUT = "collect_user_input"
    TOOL_AUTHORIZATION = "tool_authorization"
    OAUTH_AUTHORIZATION = "oauth_authorization"
    CREDENTIAL_AUTHORIZATION = "credential_authorization"
    MANUAL_INTERVENTION = "manual_intervention"


class InterventionType(Enum):
    """Types of manual interventions"""
    CAPTCHA = "captcha"
    LOGIN = "login"
    PAYMENT = "payment"
    WALLET = "wallet"
    VERIFICATION = "verification"


class SecurityLevel(Enum):
    """Security levels for tool authorization"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class InterruptData:
    """
    Base interrupt data structure
    
    Example:
        interrupt = InterruptData(
            id="123",
            type=InterruptType.ASK_HUMAN,
            question="What is your email?",
            node_source="tool_node"
        )
    """
    id: str
    type: InterruptType
    node_source: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    question: Optional[str] = None
    context: Optional[str] = None
    user_id: str = "default"
    instruction: Optional[str] = None
    mcp_request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "node_source": self.node_source,
            "timestamp": self.timestamp,
            "user_id": self.user_id
        }
        
        # Add optional fields if present
        if self.question:
            data["question"] = self.question
        if self.context:
            data["context"] = self.context
        if self.instruction:
            data["instruction"] = self.instruction
        if self.mcp_request_id:
            data["mcp_request_id"] = self.mcp_request_id
            
        return data


@dataclass
class ApprovalInterruptData(InterruptData):
    """
    Interrupt for approval/rejection workflows
    
    Example:
        interrupt = ApprovalInterruptData(
            id="123",
            type=InterruptType.APPROVAL,
            question="Approve this action?",
            node_source="tool_node",
            approval_options=["approve", "reject"],
            context={"action": "delete_file"}
        )
    """
    approval_options: List[str] = field(default_factory=lambda: ["approve", "reject"])
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["options"] = self.approval_options
        return data


@dataclass
class ReviewEditInterruptData(InterruptData):
    """
    Interrupt for review and edit workflows
    
    Example:
        interrupt = ReviewEditInterruptData(
            id="123",
            type=InterruptType.REVIEW_EDIT,
            task="Review API response",
            content="{'status': 'ok'}",
            node_source="response_node",
            required_fields=["status", "data"]
        )
    """
    task: str = ""
    content: str = ""
    required_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["task"] = self.task
        data["content"] = self.content
        data["required_fields"] = self.required_fields
        return data


@dataclass
class ValidationInterruptData(InterruptData):
    """
    Interrupt for input validation with retry
    
    Example:
        interrupt = ValidationInterruptData(
            id="123",
            type=InterruptType.INPUT_VALIDATION,
            question="Enter your age:",
            node_source="input_node",
            validation_rules={"type": "int", "min": 0, "max": 120},
            retry_count=0,
            max_retries=3
        )
    """
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["validation_rules"] = self.validation_rules
        data["retry_count"] = self.retry_count
        data["max_retries"] = self.max_retries
        return data


@dataclass
class ToolAuthorizationInterruptData(InterruptData):
    """
    Interrupt for tool authorization
    
    Example:
        interrupt = ToolAuthorizationInterruptData(
            id="123",
            type=InterruptType.TOOL_AUTHORIZATION,
            tool_name="web_crawl",
            tool_args={"url": "https://example.com"},
            security_level=SecurityLevel.HIGH,
            reason="Need to scrape data",
            node_source="tool_node"
        )
    """
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "MEDIUM"
    reason: str = ""
    scenario: str = "tool_permission"
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["tool_name"] = self.tool_name
        data["tool_args"] = self.tool_args
        data["security_level"] = self.security_level
        data["reason"] = self.reason
        data["scenario"] = self.scenario
        return data


@dataclass
class OAuthAuthorizationInterruptData(InterruptData):
    """
    Interrupt for OAuth authorization
    
    Example:
        interrupt = OAuthAuthorizationInterruptData(
            id="123",
            type=InterruptType.OAUTH_AUTHORIZATION,
            provider="gmail",
            oauth_url="https://accounts.google.com/o/oauth2/auth?...",
            scopes=["read", "send"],
            node_source="tool_node"
        )
    """
    provider: str = ""
    oauth_url: str = ""
    scopes: Optional[List[str]] = None
    scenario: str = "oauth_permission"
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["provider"] = self.provider
        data["oauth_url"] = self.oauth_url
        data["scopes"] = self.scopes
        data["scenario"] = self.scenario
        return data


@dataclass
class CredentialAuthorizationInterruptData(InterruptData):
    """
    Interrupt for stored credential usage authorization
    
    Example:
        interrupt = CredentialAuthorizationInterruptData(
            id="123",
            type=InterruptType.CREDENTIAL_AUTHORIZATION,
            provider="google",
            credential_preview={"vault_id": "vault_123", "stored_at": "2025-01-15"},
            auth_type="social",
            node_source="tool_node"
        )
    """
    provider: str = ""
    credential_preview: Dict[str, Any] = field(default_factory=dict)
    auth_type: str = ""
    scenario: str = "credential_usage"
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["provider"] = self.provider
        data["credential_preview"] = self.credential_preview
        data["auth_type"] = self.auth_type
        data["scenario"] = self.scenario
        return data


@dataclass
class ManualInterventionInterruptData(InterruptData):
    """
    Interrupt for manual user intervention
    
    Example:
        interrupt = ManualInterventionInterruptData(
            id="123",
            type=InterruptType.MANUAL_INTERVENTION,
            intervention_type="captcha",
            provider="google",
            instructions="Please solve the reCAPTCHA",
            screenshot_path="/tmp/captcha.png",
            node_source="tool_node"
        )
    """
    intervention_type: str = ""
    provider: str = ""
    instructions: str = ""
    screenshot_path: Optional[str] = None
    oauth_url: Optional[str] = None
    scenario: str = "manual_action"
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["intervention_type"] = self.intervention_type
        data["provider"] = self.provider
        data["instructions"] = self.instructions
        data["scenario"] = self.scenario
        if self.screenshot_path:
            data["screenshot_path"] = self.screenshot_path
        if self.oauth_url:
            data["oauth_url"] = self.oauth_url
        return data


@dataclass
class InterruptStats:
    """Statistics about interrupts"""
    total: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_node: Dict[str, int] = field(default_factory=dict)
    latest: Optional[str] = None


# =============================================================================
# 4 CORE HIL REQUEST MODELS (Aligned with MCP HIL Integration Guide)
# =============================================================================

@dataclass
class AuthorizationRequest:
    """
    Core HIL Type 1: Authorization Request

    MCP Response Format:
        status: "authorization_requested"
        hil_type: "authorization"

    Scenarios:
        - tool_authorization: High/Critical security tools
        - payment: Financial transactions
        - deletion: Destructive operations
        - deployment: Production deployments

    Example:
        req = AuthorizationRequest(
            action="Process $5000 payment",
            reason="Complete vendor payment",
            risk_level="high",
            scenario="payment",
            context={"vendor": "Acme", "amount": 5000}
        )
    """
    action: str
    reason: str
    risk_level: str = "high"  # low, medium, high, critical
    scenario: str = "generic"
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 120  # 2 minutes (authorization needs time to think)

    def to_interrupt_data(self) -> Dict[str, Any]:
        """Convert to interrupt data for LangGraph"""
        return {
            "type": "authorization",
            "scenario": self.scenario,
            "question": f"Authorize: {self.action}",
            "message": f"**Authorization Required ({self.risk_level.upper()} risk)**\n\n{self.reason}",
            "context": {**self.context, "risk_level": self.risk_level},
            "options": ["approve", "reject"],
            "timeout": self.timeout,
            "data": {
                "request_type": "authorization",
                "action": self.action,
                "reason": self.reason,
                "risk_level": self.risk_level,
                "scenario": self.scenario
            }
        }


@dataclass
class InputRequest:
    """
    Core HIL Type 2: Input Request

    MCP Response Format:
        status: "human_input_requested"
        hil_type: "input"

    Scenarios:
        - credentials: API keys, passwords
        - selection: Choose from options
        - augmentation: Add missing details

    Example:
        req = InputRequest(
            input_type="credentials",
            prompt="Enter OpenAI API Key",
            description="Provide your API key",
            scenario="credentials",
            schema={"type": "string", "pattern": "^sk-"}
        )
    """
    input_type: str  # text, credentials, file, selection, augmentation
    prompt: str
    description: str
    scenario: str = "generic"
    schema: Optional[Dict] = None
    current_data: Optional[Any] = None
    suggestions: Optional[List[str]] = None
    default_value: Optional[Any] = None
    timeout: int = 60  # 1 minute (input collection)

    def to_interrupt_data(self) -> Dict[str, Any]:
        """Convert to interrupt data for LangGraph"""
        return {
            "type": "input",
            "scenario": self.scenario,
            "question": self.prompt,
            "message": self.description,
            "context": {
                "has_suggestions": bool(self.suggestions),
                "has_current_data": bool(self.current_data),
                "input_type": self.input_type
            },
            "options": ["submit", "skip", "cancel"],
            "timeout": self.timeout,
            "data": {
                "request_type": "input",
                "input_type": self.input_type,
                "prompt": self.prompt,
                "schema": self.schema,
                "current_data": self.current_data,
                "suggestions": self.suggestions,
                "default_value": self.default_value,
                "scenario": self.scenario
            }
        }


@dataclass
class ReviewRequest:
    """
    Core HIL Type 3: Review Request

    MCP Response Format:
        status: "human_input_requested"
        hil_type: "review"

    Scenarios:
        - execution_plan: Review task execution plan
        - code: Review generated code
        - config: Review configuration

    Example:
        req = ReviewRequest(
            content=execution_plan,
            content_type="execution_plan",
            instructions="Review before execution",
            scenario="execution_plan",
            editable=True
        )
    """
    content: Any
    content_type: str  # plan, code, config, document
    instructions: str
    scenario: str = "generic"
    editable: bool = True
    timeout: int = 120  # 2 minutes (review content)

    def to_interrupt_data(self) -> Dict[str, Any]:
        """Convert to interrupt data for LangGraph"""
        content_str = str(self.content) if not isinstance(self.content, str) else self.content

        return {
            "type": "review",
            "scenario": self.scenario,
            "question": f"Review {self.content_type}",
            "message": self.instructions,
            "context": {
                "content_type": self.content_type,
                "editable": self.editable,
                "content_length": len(content_str)
            },
            "options": ["approve", "edit", "reject"] if self.editable else ["approve", "reject"],
            "timeout": self.timeout,
            "data": {
                "request_type": "review",
                "content_type": self.content_type,
                "content": self.content,
                "instructions": self.instructions,
                "editable": self.editable,
                "scenario": self.scenario
            }
        }


@dataclass
class InputWithAuthRequest:
    """
    Core HIL Type 4: Input + Authorization Request

    MCP Response Format:
        status: "authorization_requested"
        hil_type: "input_with_authorization"

    Scenarios:
        - payment_with_amount: Enter amount + authorize payment
        - deploy_with_config: Enter config + authorize deployment

    Example:
        req = InputWithAuthRequest(
            input_prompt="Enter payment amount",
            input_description="Specify amount to pay vendor",
            authorization_reason="Authorize payment after entering amount",
            input_type="number",
            risk_level="high",
            scenario="payment_with_amount",
            schema={"type": "number", "minimum": 0}
        )
    """
    input_prompt: str
    input_description: str
    authorization_reason: str
    input_type: str = "text"
    risk_level: str = "high"
    scenario: str = "generic"
    schema: Optional[Dict] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 120  # 2 minutes (input + authorization)

    def to_interrupt_data(self) -> Dict[str, Any]:
        """Convert to interrupt data for LangGraph"""
        return {
            "type": "input_with_authorization",
            "scenario": self.scenario,
            "question": f"{self.input_prompt} (requires authorization)",
            "message": f"**Input Required with Authorization ({self.risk_level.upper()} risk)**\n\n{self.input_description}\n\n{self.authorization_reason}",
            "context": {
                **self.context,
                "risk_level": self.risk_level,
                "requires_input": True
            },
            "options": ["approve_with_input", "cancel"],
            "timeout": self.timeout,
            "data": {
                "request_type": "input_with_authorization",
                "input_prompt": self.input_prompt,
                "input_type": self.input_type,
                "input_schema": self.schema,
                "authorization_reason": self.authorization_reason,
                "risk_level": self.risk_level,
                "scenario": self.scenario
            }
        }


@dataclass
class ExecutionChoiceRequest:
    """
    CORE HIL METHOD 5: Execution Choice - Choose execution mode for long-running tasks

    Triggered by: ToolNode (not MCP tools)

    Scenarios:
        - long_running_web_crawl: Multiple web pages to crawl
        - long_running_web_search: Multiple searches
        - long_running_mixed: Mixed operations

    Options:
        - quick: Limited resources, fast response (~30s)
        - comprehensive: Full execution, wait for all (~60s+)
        - background: Queue to background jobs, get job_id immediately

    Example:
        req = ExecutionChoiceRequest(
            prompt="Long-running task detected: crawling 5 web pages (~45s total)",
            estimated_time_seconds=45.6,
            tool_count=5,
            task_type="web_crawling",
            recommendation="background",
            options=[...],
            context={"tool_breakdown": {...}}
        )
    """
    prompt: str
    estimated_time_seconds: float
    tool_count: int
    task_type: str  # "web_crawling", "web_searching", "mixed"
    recommendation: str  # "quick" | "comprehensive" | "background"
    options: List[Dict[str, Any]]  # [{"value": "quick", "label": "Quick", "description": "..."}]
    scenario: str = "long_running_task"
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30  # 30 seconds (quick execution choice)

    def to_interrupt_data(self) -> Dict[str, Any]:
        """Convert to interrupt data for LangGraph"""
        return {
            "type": "execution_choice",
            "scenario": self.scenario,
            "question": "How do you want to execute this task?",
            "message": self.prompt,
            "context": {
                **self.context,
                "estimated_time_seconds": self.estimated_time_seconds,
                "tool_count": self.tool_count,
                "task_type": self.task_type
            },
            "options": [opt["value"] for opt in self.options],  # ["quick", "comprehensive", "background"]
            "recommendation": self.recommendation,
            "timeout": self.timeout,
            "data": {
                "request_type": "execution_choice",
                "estimated_time_seconds": self.estimated_time_seconds,
                "tool_count": self.tool_count,
                "task_type": self.task_type,
                "recommendation": self.recommendation,
                "options": self.options,  # Full option details with descriptions
                "scenario": self.scenario
            }
        }

