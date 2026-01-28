#!/usr/bin/env python3
"""
Guardrail Node - Clean compliance checking with default resources
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from .base_node import BaseNode
from isa_agent_sdk.services.human_in_the_loop import get_hil_service
from isa_agent_sdk.utils.logger import agent_logger

hil_service = get_hil_service()
logger = agent_logger  # Use centralized logger for Loki integration


class GuardrailNode(BaseNode):
    """Clean guardrail compliance checking node"""
    
    def __init__(self, guardrail_mode: str = "moderate"):
        super().__init__("GuardrailNode")
        self.guardrail_mode = guardrail_mode
    
    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Execute guardrail compliance check using default resources
        Enhanced for Co-comply: Critical task approval gates
        
        Args:
            state: Current agent state with final response or task information
            config: Runtime config with default_resources
            
        Returns:
            Updated state with guardrail/compliance results
        """
        # CO-COMPLY: Check if this is a critical task approval request
        if self._is_critical_task_check(state):
            return await self._handle_critical_task_approval(state, config)
        
        # Original guardrail logic for final response checking
        # Get final response content
        final_response = ""
        if state.get("messages"):
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                final_response = str(last_message.content)
        
        # Get guardrail rules from context
        context = self.get_runtime_context(config)
        default_resources = context.get('default_resources', [])
        
        # Apply guardrail checks
        check_result = self._apply_guardrail_checks(final_response, default_resources)
        
        action = check_result.get("action", "ALLOW")
        violations = check_result.get("violations", [])
        
        # Process guardrail decision
        if action == "BLOCK":
            compliance_message = AIMessage(
                content=check_result.get("message", "❌ Output blocked due to compliance violations.")
            )
            state["messages"][-1] = compliance_message
            
        elif action == "SANITIZE":
            sanitized_content = check_result.get("sanitized_text", final_response)
            sanitized_message = AIMessage(
                content=sanitized_content + "\n\n⚠️ Content was sanitized for compliance."
            )
            state["messages"][-1] = sanitized_message
        
        # Store guardrail result
        state["guardrail_result"] = {
            "action": action,
            "violations": violations,
            "risk_score": check_result.get("risk_score", 0)
        }
        
        # Stream guardrail completion
        self.stream_custom({
            "guardrail_check": {
                "status": "completed",
                "action": action,
                "violations_found": len(violations),
                "mode": self.guardrail_mode
            }
        })
        
        return state
    
    def _apply_guardrail_checks(self, text: str, default_resources: list) -> Dict[str, Any]:
        """
        Apply guardrail checks using default resources
        
        Args:
            text: Text content to check
            default_resources: Available guardrail resources
            
        Returns:
            Guardrail check results
        """
        violations = []
        sanitized_text = text
        
        # Extract guardrail rules from resources
        guardrail_rules = self._extract_guardrail_rules(default_resources)
        
        # Check PII patterns
        pii_patterns = guardrail_rules.get("pii_patterns", {})
        for pii_type, pattern in pii_patterns.items():
            if pattern in text.lower():
                violations.append({
                    "type": "PII_EXPOSURE",
                    "category": pii_type,
                    "severity": "HIGH"
                })
                sanitized_text = sanitized_text.replace(pattern, f"[REDACTED_{pii_type.upper()}]")
        
        # Check sensitive keywords
        sensitive_keywords = guardrail_rules.get("sensitive_keywords", [])
        text_lower = text.lower()
        detected_keywords = [kw for kw in sensitive_keywords if kw in text_lower]
        
        if detected_keywords:
            violations.append({
                "type": "SENSITIVE_CONTENT",
                "detected_terms": detected_keywords,
                "severity": "MEDIUM"
            })
        
        # Calculate risk score
        risk_score = len([v for v in violations if v["severity"] == "HIGH"]) * 3 + \
                    len([v for v in violations if v["severity"] == "MEDIUM"]) * 1
        
        # Determine action based on mode
        if self.guardrail_mode == "strict" and violations:
            action = "BLOCK"
            message = "Output blocked due to compliance violations"
        elif self.guardrail_mode == "moderate" and any(v["severity"] == "HIGH" for v in violations):
            action = "SANITIZE"
            message = "Output sanitized to remove high-risk information"
        else:
            action = "ALLOW"
            message = "Output approved"
        
        return {
            "action": action,
            "message": message,
            "sanitized_text": sanitized_text,
            "violations": violations,
            "risk_score": risk_score
        }
    
    def _extract_guardrail_rules(self, default_resources: list) -> Dict[str, Any]:
        """
        Extract guardrail rules from default resources
        
        Args:
            default_resources: List of available resources
            
        Returns:
            Dictionary of guardrail rules
        """
        rules = {
            "pii_patterns": {},
            "sensitive_keywords": []
        }
        
        # Look for guardrail-related resources
        for resource in default_resources:
            if isinstance(resource, dict):
                resource_type = resource.get("type", "")
                resource_name = resource.get("name", "")
                
                if "guardrail" in resource_type.lower() or "compliance" in resource_name.lower():
                    # Extract rules from resource
                    if "pii" in resource_name.lower():
                        rules["pii_patterns"] = resource.get("patterns", {})
                    elif "sensitive" in resource_name.lower():
                        rules["sensitive_keywords"] = resource.get("keywords", [])
        
        # Fallback to basic rules if no resources found
        if not rules["pii_patterns"] and not rules["sensitive_keywords"]:
            rules = {
                "pii_patterns": {
                    "email": "@",
                    "phone": "phone",
                    "ssn": "ssn"
                },
                "sensitive_keywords": ["password", "secret", "private", "confidential"]
            }
        
        return rules
    
    # ========== CO-COMPLY: Critical Task Approval Methods ==========
    
    def _is_critical_task_check(self, state: AgentState) -> bool:
        """
        Check if this is a critical task requiring approval
        
        Args:
            state: Current agent state
            
        Returns:
            True if this is a critical task check request
        """
        # Check if we have task data and need approval
        task_list = state.get("task_list", [])
        current_task_index = state.get("current_task_index", 0)
        
        if not task_list or current_task_index >= len(task_list):
            return False
            
        # Check if guardrail is being called for task approval (not output checking)
        return state.get("compliance_check_requested", False) or \
               state.get("next_action") == "guardrail" and "messages" not in state
    
    async def _handle_critical_task_approval(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Handle critical task approval using HIL service
        
        Args:
            state: Current agent state with task information
            config: Runtime config
            
        Returns:
            Updated state with compliance decision
        """
        task_list = state.get("task_list", [])
        current_task_index = state.get("current_task_index", 0)
        current_task = task_list[current_task_index]
        
        # Assess task criticality using existing guardrail patterns
        risk_assessment = self._assess_task_criticality(current_task, state)
        
        # Stream compliance check start
        self.stream_custom({
            "compliance_check": {
                "status": "analyzing",
                "task_title": current_task.get("title", "Unknown task"),
                "risk_level": risk_assessment["risk_level"]
            }
        })
        
        # Check if human approval is required
        if risk_assessment["requires_approval"]:
            # Use existing HIL service for approval
            approval_result = await self._request_task_approval(current_task, risk_assessment)
            
            if approval_result == "approved":
                state["compliance_approved"] = True
                state["next_action"] = "agent_executor"
                self.logger.info(f"Critical task approved: {current_task.get('title')}")
            else:
                state["compliance_approved"] = False
                state["next_action"] = "call_model"  # Return to reasoning
                self.logger.info(f"Critical task rejected: {current_task.get('title')}")
                
                # Remove rejected task
                state["task_list"] = task_list[:current_task_index] + task_list[current_task_index + 1:]
        else:
            # Auto-approve low-risk tasks
            state["compliance_approved"] = True
            return {"next_action": "agent_executor"}
    
    def _assess_task_criticality(self, task: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """
        Assess criticality of a task using guardrail patterns
        
        Args:
            task: Task to assess
            state: Current agent state
            
        Returns:
            Risk assessment dictionary
        """
        task_title = task.get("title", "")
        task_description = task.get("description", "")
        task_tools = task.get("tools", [])
        
        # Critical task patterns (similar to sensitive keywords)
        critical_patterns = {
            "data_deletion": ["delete", "remove", "drop", "truncate", "clear", "wipe"],
            "system_admin": ["sudo", "admin", "root", "system", "config", "install"],
            "external_comm": ["email", "send", "post", "publish", "share", "upload"],
            "financial": ["payment", "purchase", "buy", "order", "transaction", "billing"],
            "database": ["database", "sql", "query", "insert", "update", "modify"]
        }
        
        # Combine all task text
        task_text = f"{task_title} {task_description} {' '.join(task_tools)}".lower()
        
        # Find matches
        matched_patterns = {}
        total_matches = 0
        
        for category, patterns in critical_patterns.items():
            matches = [p for p in patterns if p in task_text]
            if matches:
                matched_patterns[category] = matches
                total_matches += len(matches)
        
        # Determine risk level
        risk_level = "low"
        if total_matches >= 3:
            risk_level = "critical"
        elif total_matches >= 2:
            risk_level = "high"  
        elif total_matches >= 1:
            risk_level = "medium"
        
        # High-risk tools automatically elevate risk
        high_risk_tools = ["system_admin", "database_operations", "external_communications"]
        if any(tool in high_risk_tools for tool in task_tools):
            risk_level = "high"
        
        # User-marked critical tasks
        if task.get("requires_approval", False) or task.get("critical", False):
            risk_level = "critical"
        
        return {
            "risk_level": risk_level,
            "total_matches": total_matches,
            "matched_patterns": matched_patterns,
            "requires_approval": risk_level in ["high", "critical"],
            "assessment_reason": f"Found {total_matches} critical patterns"
        }
    
    async def _request_task_approval(self, task: Dict[str, Any], risk_assessment: Dict[str, Any]) -> str:
        """
        Request human approval for critical task using HIL service
        
        Args:
            task: Task requiring approval
            risk_assessment: Risk assessment data
            
        Returns:
            "approved" or "rejected"
        """
        # Create approval summary
        approval_summary = f"""🚨 **Critical Task Approval Required**

**Task:** {task.get('title', 'Unknown task')}
**Description:** {task.get('description', 'No description')}
**Tools:** {', '.join(task.get('tools', []))}
**Risk Level:** {risk_assessment['risk_level'].upper()}

**Why flagged:** {risk_assessment['assessment_reason']}
**Critical patterns:** {', '.join(risk_assessment['matched_patterns'].keys())}

This task requires explicit approval due to {risk_assessment['risk_level']} risk level.

Type 'approve' to authorize or 'reject' to deny:"""
        
        try:
            human_response = hil_service.ask_human_with_interrupt(
                question=approval_summary,
                context=f"Critical task approval - Risk: {risk_assessment['risk_level']}",
                node_source="guardrail_node"
            )
            
            response_str = str(human_response).lower().strip() if human_response else "reject"
            return "approved" if response_str in ["approve", "approved", "yes"] else "rejected"
            
        except Exception as e:
            self.logger.error(f"Task approval failed: {e}")
            return "rejected"  # Default to reject for safety


# Factory function for compatibility
def create_guardrail_node(guardrail_mode: str = "moderate") -> GuardrailNode:
    """Create GuardrailNode instance"""
    return GuardrailNode(guardrail_mode)