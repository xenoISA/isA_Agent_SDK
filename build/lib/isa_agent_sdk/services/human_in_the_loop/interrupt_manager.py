"""
Core interrupt management for Human-in-the-Loop service
File: app/services/human_in_the_loop/interrupt_manager.py

This module handles core interrupt patterns and execution control.
"""

from typing import Dict, Any, Optional, List
from langgraph.types import interrupt, Command
import uuid
from datetime import datetime

from .models import (
    InterruptData,
    ApprovalInterruptData,
    ReviewEditInterruptData,
    ValidationInterruptData,
    InterruptType,
    InterruptStats
)
from .validators import HILValidator
from isa_agent_sdk.utils.logger import api_logger


class InterruptManager:
    """
    Manages LangGraph interrupts for HIL scenarios
    
    Example:
        manager = InterruptManager()
        command = manager.approve_or_reject(
            question="Delete this file?",
            context={"file": "data.txt"},
            node_source="file_node"
        )
    """
    
    def __init__(self):
        self.interrupt_history: List[Dict[str, Any]] = []
        self.validator = HILValidator()
    
    def approve_or_reject(
        self,
        question: str,
        context: Dict[str, Any],
        node_source: str = "unknown",
        approval_options: Optional[List[str]] = None
    ) -> Command:
        """
        Approve/reject pattern - returns Command for routing
        
        Args:
            question: Question to ask human
            context: Context data to display
            node_source: Source node requesting approval
            approval_options: Custom approval options
            
        Returns:
            Command with goto directive based on approval
            
        Example:
            command = approve_or_reject(
                question="Execute this tool?",
                context={"tool": "web_crawl", "url": "https://example.com"},
                node_source="tool_node"
            )
        """
        interrupt_id = str(uuid.uuid4())
        
        interrupt_data = ApprovalInterruptData(
            id=interrupt_id,
            type=InterruptType.APPROVAL,
            question=question,
            context=str(context),
            node_source=node_source,
            approval_options=approval_options or ["approve", "reject"]
        )
        
        # Log interrupt
        self._log_interrupt(interrupt_data.to_dict())
        
        # Trigger LangGraph interrupt
        response = interrupt(interrupt_data.to_dict())
        
        # Process approval response
        if self.validator.is_approved(response):
            api_logger.info(f"HIL: Approved by human - {question[:50]}...")
            return Command(goto="approved_path")
        else:
            api_logger.info(f"HIL: Rejected by human - {question[:50]}...")
            return Command(goto="rejected_path")
    
    def review_and_edit(
        self,
        content_to_review: str,
        task_description: str,
        node_source: str = "unknown",
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Review and edit pattern - returns edited content
        
        Args:
            content_to_review: Content for human to review
            task_description: Description of review task
            node_source: Source node requesting review
            required_fields: Required fields in response
            
        Returns:
            Edited content from human
            
        Example:
            edited = review_and_edit(
                content_to_review='{"status": "pending"}',
                task_description="Review API response",
                node_source="api_node",
                required_fields=["status", "message"]
            )
        """
        interrupt_id = str(uuid.uuid4())
        
        interrupt_data = ReviewEditInterruptData(
            id=interrupt_id,
            type=InterruptType.REVIEW_EDIT,
            task=task_description,
            content=content_to_review,
            node_source=node_source,
            required_fields=required_fields or []
        )
        
        self._log_interrupt(interrupt_data.to_dict())
        
        # Trigger interrupt for review
        edited_content = interrupt(interrupt_data.to_dict())
        
        api_logger.info(f"HIL: Content reviewed and edited - {task_description[:50]}...")
        return self.validator.validate_edited_content(edited_content, required_fields)
    
    def validate_input_with_retry(
        self,
        initial_question: str,
        validation_rules: Dict[str, Any],
        max_retries: int = 3,
        node_source: str = "unknown"
    ) -> Any:
        """
        Input validation with retry pattern
        
        Args:
            initial_question: Initial question to ask
            validation_rules: Validation rules to apply
            max_retries: Maximum retry attempts
            node_source: Source node requesting input
            
        Returns:
            Validated input from human
            
        Example:
            age = validate_input_with_retry(
                initial_question="Enter your age:",
                validation_rules={"type": "int", "min": 0, "max": 120},
                max_retries=3
            )
        """
        question = initial_question
        retry_count = 0
        
        while retry_count < max_retries:
            interrupt_id = str(uuid.uuid4())
            
            interrupt_data = ValidationInterruptData(
                id=interrupt_id,
                type=InterruptType.INPUT_VALIDATION,
                question=question,
                node_source=node_source,
                validation_rules=validation_rules,
                retry_count=retry_count,
                max_retries=max_retries
            )
            
            self._log_interrupt(interrupt_data.to_dict())
            
            # Get human input
            human_input = interrupt(interrupt_data.to_dict())
            
            # Validate input
            validation_result = self.validator.validate_input(human_input, validation_rules)
            
            if validation_result["valid"]:
                api_logger.info(f"HIL: Input validated successfully after {retry_count + 1} attempts")
                return validation_result["value"]
            
            # Prepare retry question
            retry_count += 1
            question = f"Invalid input: {validation_result['error']}. Please try again ({retry_count}/{max_retries}): {initial_question}"
        
        # Max retries reached
        api_logger.warning(f"HIL: Input validation failed after {max_retries} attempts")
        raise ValueError(f"Input validation failed after {max_retries} attempts")
    
    def simple_interrupt(
        self,
        question: str,
        context: str = "",
        user_id: str = "default",
        node_source: str = "unknown",
        interrupt_type: InterruptType = InterruptType.ASK_HUMAN
    ) -> Any:
        """
        Simple interrupt that pauses execution and waits for human response
        
        Args:
            question: Question to ask
            context: Additional context
            user_id: User identifier
            node_source: Source node
            interrupt_type: Type of interrupt
            
        Returns:
            Human response
            
        Example:
            response = simple_interrupt(
                question="What should we do next?",
                context="Process paused at step 3",
                node_source="decision_node"
            )
        """
        interrupt_id = str(uuid.uuid4())
        
        interrupt_data = InterruptData(
            id=interrupt_id,
            type=interrupt_type,
            question=question,
            context=context,
            user_id=user_id,
            node_source=node_source,
            instruction="Graph execution paused. Provide input via /api/chat/resume to continue."
        )
        
        self._log_interrupt(interrupt_data.to_dict())
        
        api_logger.info(f"HIL: Triggering interrupt for question: {question[:50]}...")
        api_logger.info(f"HIL: Graph will PAUSE until resume with interrupt_id: {interrupt_id}")
        
        # Trigger LangGraph interrupt
        human_response = interrupt(interrupt_data.to_dict())
        
        api_logger.info(f"HIL: Human response received: {str(human_response)[:100]}...")
        return human_response
    
    def resume_multiple_interrupts(
        self,
        interrupt_responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resume multiple interrupts with single invocation
        
        Args:
            interrupt_responses: Map of interrupt_id -> response
            
        Returns:
            Processing results
            
        Example:
            results = resume_multiple_interrupts({
                "interrupt_1": "yes",
                "interrupt_2": {"approved": True}
            })
        """
        results = {}
        
        for interrupt_id, response in interrupt_responses.items():
            try:
                # Find interrupt in history
                interrupt_data = self._find_interrupt_by_id(interrupt_id)
                if interrupt_data:
                    processed_response = self.validator.process_interrupt_response(
                        interrupt_data,
                        response
                    )
                    results[interrupt_id] = processed_response
                else:
                    results[interrupt_id] = {"error": "Interrupt not found"}
                    
            except Exception as e:
                results[interrupt_id] = {"error": str(e)}
        
        return results
    
    # ========== History and Logging ==========
    
    def _log_interrupt(self, interrupt_data: Dict[str, Any]):
        """Log interrupt for history tracking"""
        self.interrupt_history.append(interrupt_data)
        interrupt_type = interrupt_data.get("type")
        node_source = interrupt_data.get("node_source", "unknown")
        api_logger.info(f"HIL: Interrupt logged - {interrupt_type} from {node_source}")
    
    def _find_interrupt_by_id(self, interrupt_id: str) -> Optional[Dict[str, Any]]:
        """Find interrupt in history by ID"""
        for interrupt_data in self.interrupt_history:
            if interrupt_data.get("id") == interrupt_id:
                return interrupt_data
        return None
    
    def get_interrupt_stats(self) -> InterruptStats:
        """
        Get statistics about interrupts
        
        Returns:
            InterruptStats object
            
        Example:
            stats = get_interrupt_stats()
            print(f"Total interrupts: {stats.total}")
            print(f"By type: {stats.by_type}")
        """
        if not self.interrupt_history:
            return InterruptStats()

        by_type = {}
        by_node = {}

        for interrupt in self.interrupt_history:
            interrupt_type = interrupt.get("type", "unknown")
            node_source = interrupt.get("node_source", "unknown")

            by_type[interrupt_type] = by_type.get(interrupt_type, 0) + 1
            by_node[node_source] = by_node.get(node_source, 0) + 1

        return InterruptStats(
            total=len(self.interrupt_history),
            by_type=by_type,
            by_node=by_node,
            latest=self.interrupt_history[-1]["timestamp"] if self.interrupt_history else None
        )
    
    def clear_history(self):
        """Clear interrupt history"""
        self.interrupt_history.clear()
        api_logger.info("HIL: Interrupt history cleared")

