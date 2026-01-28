#!/usr/bin/env python3
"""
Failsafe Node - AI confidence assessment and graceful failure handling

This node evaluates AI responses for confidence and completeness,
implementing graceful degradation when the AI is uncertain about its answer.
"""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from .base_node import BaseNode
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger  # Use centralized logger for Loki integration


class FailsafeNode(BaseNode):
    """
    Failsafe node for AI confidence assessment and graceful failure handling
    
    Features:
    - Confidence assessment of AI responses
    - Error categorization and handling
    - Graceful degradation strategies
    - Safety mechanisms for uncertain responses
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize failsafe node
        
        Args:
            confidence_threshold: Minimum confidence threshold (0.0-1.0)
        """
        super().__init__("FailsafeNode")
        self.confidence_threshold = confidence_threshold
        
        # Error categories for classification
        self.error_categories = {
            "INSUFFICIENT_INFO": "Not enough information to provide a reliable answer",
            "AMBIGUOUS_QUERY": "Query is ambiguous or unclear", 
            "TECHNICAL_LIMITATION": "Technical limitation preventing accurate response",
            "UNCERTAINTY": "High uncertainty in response accuracy",
            "TOOL_FAILURE": "Tool execution failed",
            "TIMEOUT": "Request timeout occurred"
        }
        
        # Confidence indicators to look for in responses
        self.uncertainty_indicators = [
            "i'm not sure", "i don't know", "uncertain", "maybe", "possibly",
            "i think", "i believe", "might be", "could be", "not certain",
            "unclear", "ambiguous", "difficult to determine", "hard to say",
            "i cannot", "i can't", "unable to", "insufficient information"
        ]
        
        # Partial response indicators
        self.partial_indicators = [
            "partial", "incomplete", "some of", "part of", "limited",
            "only able to", "partially", "to some extent"
        ]

    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Execute failsafe assessment with unified mode support (Reactive/Collaborative/Proactive)
        
        Args:
            state: Current agent state with AI response
            config: Runtime config with context
            
        Returns:
            Updated state with failsafe assessment and potential modifications
        """
        # Get context for mode determination
        context = self.get_runtime_context(config)
        
        # Determine and apply failsafe mode enhancements
        failsafe_mode = self._determine_failsafe_mode(state, context)
        
        if failsafe_mode == "proactive":
            state = await self._apply_proactive_failsafe_enhancements(state, context)
            self.logger.info("FailsafeNode executing in PROACTIVE mode")
        elif failsafe_mode == "collaborative":
            state = await self._apply_collaborative_failsafe_enhancements(state, context)
            self.logger.info("FailsafeNode executing in COLLABORATIVE mode")
        else:
            self.logger.info("FailsafeNode executing in REACTIVE mode")
        
        messages = state.get("messages", [])
        
        if not messages:
            return self._handle_empty_response(state, config)
        
        # Check for unhandled tool calls to prevent message chain corruption
        if self._has_unhandled_tool_calls(messages):
            self.logger.warning("Failsafe: Skipping assessment due to unhandled tool calls in message chain")
            # Pass through with high confidence to avoid recursive model calls
            state["failsafe_metadata"] = {
                "confidence_score": 0.9,
                "assessment": "SKIPPED_TOOL_CALLS",
                "timestamp": self._get_timestamp(),
                "reason": "Unhandled tool calls detected"
            }
            return state
        
        # Get the last AI response
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage):
            # Not an AI response, pass through
            return state
        
        response_content = str(last_message.content) if last_message.content else ""
        
        # Perform confidence assessment
        confidence_score = await self._assess_confidence(response_content, config)
        
        # Categorize any potential issues
        error_category = self._categorize_response_issues(response_content)
        
        # Log assessment results
        self.logger.info(f"Confidence assessment: {confidence_score:.2f} (threshold: {self.confidence_threshold})")
        if error_category:
            self.logger.warning(f"Response issue detected: {error_category}")
        
        # Determine if failsafe action is needed
        if confidence_score < self.confidence_threshold or error_category:
            return await self._handle_low_confidence_response(
                state, config, confidence_score, error_category, response_content
            )
        
        # High confidence response, pass through with metadata
        state["failsafe_metadata"] = {
            "confidence_score": confidence_score,
            "assessment": "PASSED",
            "timestamp": self._get_timestamp()
        }
        
        return state

    async def _assess_confidence(self, response_content: str, config: RunnableConfig) -> float:
        """
        Assess confidence level of the AI response
        
        Args:
            response_content: The AI response content
            config: Runtime config
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Basic rule-based confidence assessment
        confidence_score = 1.0
        
        response_lower = response_content.lower()
        
        # Check for uncertainty indicators
        uncertainty_count = sum(1 for indicator in self.uncertainty_indicators 
                              if indicator in response_lower)
        
        # Reduce confidence based on uncertainty indicators
        if uncertainty_count > 0:
            confidence_score -= min(0.5, uncertainty_count * 0.1)
        
        # Check for partial response indicators
        partial_count = sum(1 for indicator in self.partial_indicators 
                           if indicator in response_lower)
        
        if partial_count > 0:
            confidence_score -= min(0.3, partial_count * 0.1)
        
        # Check response length (very short responses might indicate issues)
        if len(response_content.strip()) < 50:
            confidence_score -= 0.2
        
        # Check for error messages or failure indicators
        error_keywords = ["error", "failed", "exception", "cannot", "unable"]
        error_count = sum(1 for keyword in error_keywords 
                         if keyword in response_lower)
        
        if error_count > 0:
            confidence_score -= min(0.4, error_count * 0.15)
        
        # Enhanced assessment using AI model for complex cases
        if confidence_score < 0.8:
            try:
                confidence_score = await self._ai_confidence_assessment(response_content, config)
            except Exception as e:
                self.logger.warning(f"AI confidence assessment failed: {e}")
        
        return max(0.0, min(1.0, confidence_score))

    async def _ai_confidence_assessment(self, response_content: str, config: RunnableConfig) -> float:
        """
        Use AI model to assess confidence of the response
        
        Args:
            response_content: The response to assess
            config: Runtime config
            
        Returns:
            Confidence score from AI assessment
        """
        assessment_prompt = f"""
        Analyze the following AI response and assess its confidence level on a scale of 0.0 to 1.0.

        Consider:
        - Certainty of statements
        - Completeness of answer
        - Presence of uncertainty language
        - Factual accuracy indicators
        - Overall coherence

        Response to assess:
        "{response_content}"

        Provide only a numeric confidence score between 0.0 and 1.0.
        """
        
        try:
            assessment_response = await self.call_model(
                messages=[SystemMessage(content=assessment_prompt)],
                stream_tokens=False
            )
            
            # Extract numeric score from response
            assessment_text = str(assessment_response.content).strip()
            
            # Try to extract float from response
            import re
            score_match = re.search(r'([0-1]\.?\d*)', assessment_text)
            if score_match:
                return float(score_match.group(1))
            
        except Exception as e:
            self.logger.error(f"AI confidence assessment error: {e}")
        
        # Fallback to rule-based score
        return 0.5

    def _categorize_response_issues(self, response_content: str) -> str:
        """
        Categorize potential issues with the response
        
        Args:
            response_content: The AI response content
            
        Returns:
            Error category string or empty string if no issues
        """
        response_lower = response_content.lower()
        
        # Check for different types of issues
        if any(phrase in response_lower for phrase in [
            "don't know", "not sure", "uncertain", "can't tell"
        ]):
            return "UNCERTAINTY"
        
        if any(phrase in response_lower for phrase in [
            "not enough information", "insufficient", "more details needed"
        ]):
            return "INSUFFICIENT_INFO"
        
        if any(phrase in response_lower for phrase in [
            "ambiguous", "unclear", "multiple interpretations"
        ]):
            return "AMBIGUOUS_QUERY"
        
        if any(phrase in response_lower for phrase in [
            "tool failed", "execution failed", "error occurred"
        ]):
            return "TOOL_FAILURE"
        
        if any(phrase in response_lower for phrase in [
            "timeout", "timed out", "request expired"
        ]):
            return "TIMEOUT"
        
        if any(phrase in response_lower for phrase in [
            "technical limitation", "cannot process", "not capable"
        ]):
            return "TECHNICAL_LIMITATION"
        
        return ""

    async def _handle_low_confidence_response(
        self, 
        state: AgentState, 
        config: RunnableConfig,
        confidence_score: float,
        error_category: str,
        original_response: str
    ) -> AgentState:
        """
        Handle responses with low confidence or detected issues
        
        Args:
            state: Current agent state
            config: Runtime config
            confidence_score: Assessed confidence score
            error_category: Detected error category
            original_response: Original AI response
            
        Returns:
            Updated state with failsafe response
        """
        context = self.get_runtime_context(config)
        user_query = context.get('enhanced_query', context.get('original_query', ''))
        
        # Generate failsafe response based on issue type
        if error_category == "UNCERTAINTY":
            failsafe_response = await self._generate_uncertainty_response(user_query, original_response, config)
        elif error_category == "INSUFFICIENT_INFO":
            failsafe_response = await self._generate_info_request_response(user_query, config)
        elif error_category == "AMBIGUOUS_QUERY":
            failsafe_response = await self._generate_clarification_response(user_query, config)
        elif error_category in ["TOOL_FAILURE", "TIMEOUT", "TECHNICAL_LIMITATION"]:
            failsafe_response = await self._generate_technical_fallback_response(user_query, error_category, config)
        else:
            failsafe_response = await self._generate_general_fallback_response(user_query, confidence_score, config)
        
        # Create new AI message with failsafe response
        failsafe_message = AIMessage(content=failsafe_response)
        
        # Update state with failsafe response
        state["messages"] = [failsafe_message]
        state["failsafe_metadata"] = {
            "confidence_score": confidence_score,
            "error_category": error_category,
            "assessment": "FAILSAFE_TRIGGERED",
            "original_response_preview": original_response[:200] + "..." if len(original_response) > 200 else original_response,
            "timestamp": self._get_timestamp()
        }
        
        self.logger.info(f"Failsafe triggered - Category: {error_category}, Confidence: {confidence_score:.2f}")
        
        return state

    async def _generate_uncertainty_response(self, user_query: str, original_response: str, config: RunnableConfig) -> str:
        """Generate response for uncertainty cases"""
        prompt = f"""
        The AI expressed uncertainty about answering this query: "{user_query}"

        Original response showed uncertainty. Generate a helpful response that:
        1. Acknowledges the uncertainty honestly
        2. Provides what information can be confidently shared
        3. Suggests specific ways the user could get better help
        4. Maintains a helpful and professional tone

        Keep the response concise and actionable.
        """
        
        try:
            response = await self.call_model(
                messages=[SystemMessage(content=prompt)],
                stream_tokens=False
            )
            return str(response.content)
        except Exception:
            return f"I'm not entirely certain about the best answer to your question: '{user_query}'. To provide you with the most accurate information, could you provide more specific details or context about what you're trying to accomplish?"

    async def _generate_info_request_response(self, user_query: str, config: RunnableConfig) -> str:
        """Generate response requesting more information"""
        return f"I need more information to provide a complete answer to your question: '{user_query}'. Could you please provide additional details about your specific requirements, context, or constraints? This will help me give you a more accurate and useful response."

    async def _generate_clarification_response(self, user_query: str, config: RunnableConfig) -> str:
        """Generate response requesting clarification"""
        return f"Your question '{user_query}' could be interpreted in several ways. To provide the most helpful answer, could you please clarify which specific aspect you're most interested in? For example, are you looking for technical details, practical steps, or background information?"

    async def _generate_technical_fallback_response(self, user_query: str, error_category: str, config: RunnableConfig) -> str:
        """Generate response for technical issues"""
        error_explanations = {
            "TOOL_FAILURE": "I encountered a technical issue while processing your request",
            "TIMEOUT": "The request took longer than expected to process",
            "TECHNICAL_LIMITATION": "I have technical limitations that prevent me from fully addressing this request"
        }
        
        explanation = error_explanations.get(error_category, "I encountered a technical issue")
        
        return f"{explanation} regarding: '{user_query}'. I can try to help in a different way if you'd like to rephrase your question or break it down into smaller parts. Alternatively, you might want to try again in a moment."

    async def _generate_general_fallback_response(self, user_query: str, confidence_score: float, config: RunnableConfig) -> str:
        """Generate general fallback response"""
        return f"I want to provide you with the most accurate information possible for your question: '{user_query}'. However, I'm not fully confident in my response (confidence: {confidence_score:.1%}). Would you like me to try a different approach, or could you provide more context to help me give you a better answer?"

    def _handle_empty_response(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Handle cases where no response was generated"""
        context = self.get_runtime_context(config)
        user_query = context.get('enhanced_query', context.get('original_query', 'your request'))
        
        failsafe_message = AIMessage(
            content=f"I apologize, but I wasn't able to generate a response to {user_query}. This might be due to a technical issue. Could you please try rephrasing your question or asking about something more specific?"
        )
        
        state["messages"] = [failsafe_message]
        state["failsafe_metadata"] = {
            "confidence_score": 0.0,
            "error_category": "NO_RESPONSE",
            "assessment": "FAILSAFE_TRIGGERED",
            "timestamp": self._get_timestamp()
        }
        
        return state

    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # =============================================================================
    # UNIFIED MODE ENHANCEMENTS (Reactive/Collaborative/Proactive)
    # =============================================================================
    
    def _determine_failsafe_mode(self, state: AgentState, context: dict) -> str:
        """
        Determine failsafe mode based on state flags and prediction confidence
        
        Args:
            state: Current agent state
            context: Runtime context
            
        Returns:
            Failsafe mode: 'reactive', 'collaborative', or 'proactive'
        """
        try:
            # Check for proactive mode first (most advanced) - requires high confidence predictions
            if context.get('proactive_enabled', False) or state.get('proactive_enabled', False):
                predictions = state.get('proactive_predictions', {})
                if predictions and self._has_high_confidence_predictions(predictions):
                    return "proactive"
            
            # Check for collaborative mode
            if context.get('collaborative_enabled', False) or state.get('collaborative_enabled', False):
                return "collaborative"
            
            # Default to reactive mode (including low-confidence predictions)
            return "reactive"
            
        except Exception as e:
            self.logger.warning(f"Failed to determine failsafe mode: {e}")
            return "reactive"  # Safe default
    
    async def _apply_proactive_failsafe_enhancements(self, state: AgentState, context: dict) -> AgentState:
        """
        Apply proactive enhancements to failsafe assessment
        
        Args:
            state: Current agent state
            context: Runtime context with predictions
            
        Returns:
            Enhanced state with proactive failsafe capabilities
        """
        try:
            predictions = state.get('proactive_predictions', {})
            
            if not self._has_high_confidence_predictions(predictions):
                return state  # No high-confidence predictions
            
            # Extract confidence insights from predictions
            response_quality = predictions.get('response_quality', {})
            user_satisfaction = predictions.get('user_satisfaction', {})
            potential_issues = predictions.get('potential_issues', {})
            
            enhancements_applied = []
            
            # Adjust confidence threshold based on predictions
            if response_quality.get('confidence', 0) > 0.7:
                expected_quality = response_quality.get('quality_score', 0.5)
                if expected_quality < 0.6:
                    # Lower quality expected - raise confidence bar
                    original_threshold = self.confidence_threshold
                    self.confidence_threshold = min(0.9, original_threshold + 0.1)
                    enhancements_applied.append(f"Raised confidence threshold to {self.confidence_threshold:.2f} (expected low quality)")
                elif expected_quality > 0.8:
                    # High quality expected - relax threshold slightly
                    original_threshold = self.confidence_threshold
                    self.confidence_threshold = max(0.5, original_threshold - 0.05)
                    enhancements_applied.append(f"Adjusted confidence threshold to {self.confidence_threshold:.2f} (expected high quality)")
            
            # Prepare for predicted user dissatisfaction
            if user_satisfaction.get('confidence', 0) > 0.7:
                satisfaction_probability = user_satisfaction.get('satisfaction_probability', 0.5)
                if satisfaction_probability < 0.6:
                    # Low satisfaction expected - prepare enhanced fallback
                    state['enhanced_fallback_prepared'] = True
                    state['proactive_clarification_ready'] = True
                    enhancements_applied.append(f"Prepared enhanced fallback (satisfaction risk: {satisfaction_probability:.1%})")
            
            # Preemptive issue detection
            if potential_issues.get('confidence', 0) > 0.7:
                likely_issues = potential_issues.get('issue_categories', [])
                if likely_issues:
                    # Adjust error detection for predicted issue types
                    state['predicted_issue_categories'] = likely_issues[:3]  # Top 3 predicted issues
                    enhancements_applied.append(f"Enhanced detection for: {', '.join(likely_issues[:3])}")
            
            # Add metadata for monitoring
            state['proactive_failsafe_enhancements'] = {
                'enhancement_count': len(enhancements_applied),
                'applied_enhancements': enhancements_applied,
                'prediction_confidence_scores': {
                    'response_quality': response_quality.get('confidence', 0),
                    'user_satisfaction': user_satisfaction.get('confidence', 0),
                    'potential_issues': potential_issues.get('confidence', 0)
                },
                'adjusted_threshold': self.confidence_threshold
            }
            
            if enhancements_applied:
                self.logger.debug(f"Applied {len(enhancements_applied)} proactive failsafe enhancements")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to apply proactive failsafe enhancements: {e}")
            return state
    
    async def _apply_collaborative_failsafe_enhancements(self, state: AgentState, context: dict) -> AgentState:
        """
        Apply collaborative enhancements to failsafe assessment
        
        Args:
            state: Current agent state
            context: Runtime context
            
        Returns:
            Enhanced state with collaborative failsafe features
        """
        try:
            # Note: context parameter available for future enhancements
            _ = context  # Suppress unused parameter warning
            
            enhancements_applied = []
            
            # Enable transparent confidence reporting
            state['show_confidence_scores'] = True
            enhancements_applied.append("Enabled transparent confidence reporting")
            
            # Prepare collaborative failsafe responses
            state['collaborative_failsafe_enabled'] = True
            enhancements_applied.append("Enabled collaborative failsafe responses")
            
            # Lower threshold slightly for collaborative mode (human can catch issues)
            if hasattr(self, 'original_threshold'):
                original_threshold = self.original_threshold
            else:
                original_threshold = self.confidence_threshold
                self.original_threshold = original_threshold
            
            # Slightly more permissive in collaborative mode
            collaborative_threshold = max(0.5, original_threshold - 0.1)
            if collaborative_threshold != self.confidence_threshold:
                self.confidence_threshold = collaborative_threshold
                enhancements_applied.append(f"Adjusted threshold for collaboration: {collaborative_threshold:.2f}")
            
            # Enable human feedback collection on failsafe events
            state['collect_human_feedback_on_failsafe'] = True
            enhancements_applied.append("Enabled human feedback collection on failsafe events")
            
            # Prepare Co-comply integration
            state['failsafe_compliance_check'] = True
            enhancements_applied.append("Enabled compliance checking for failsafe responses")
            
            # Add metadata for monitoring
            state['collaborative_failsafe_enhancements'] = {
                'enhancement_count': len(enhancements_applied),
                'applied_enhancements': enhancements_applied,
                'collaboration_features': {
                    'transparent_confidence': state.get('show_confidence_scores', False),
                    'collaborative_responses': state.get('collaborative_failsafe_enabled', False),
                    'human_feedback': state.get('collect_human_feedback_on_failsafe', False),
                    'compliance_check': state.get('failsafe_compliance_check', False)
                },
                'adjusted_threshold': self.confidence_threshold
            }
            
            if enhancements_applied:
                self.logger.debug(f"Applied {len(enhancements_applied)} collaborative failsafe enhancements")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to apply collaborative failsafe enhancements: {e}")
            return state
    
    def _has_high_confidence_predictions(self, predictions: dict) -> bool:
        """Check if predictions have sufficient confidence for enhancement"""
        try:
            # Check all available prediction types for maximum confidence
            user_needs_confidence = predictions.get('user_needs', {}).get('confidence', 0.0)
            task_outcomes_confidence = predictions.get('task_outcomes', {}).get('confidence', 0.0)
            response_quality_confidence = predictions.get('response_quality', {}).get('confidence', 0.0)
            user_satisfaction_confidence = predictions.get('user_satisfaction', {}).get('confidence', 0.0)
            potential_issues_confidence = predictions.get('potential_issues', {}).get('confidence', 0.0)
            user_patterns_confidence = predictions.get('user_patterns', {}).get('confidence', 0.0)
            
            max_confidence = max(user_needs_confidence, task_outcomes_confidence, response_quality_confidence, user_satisfaction_confidence, potential_issues_confidence, user_patterns_confidence)
            return max_confidence > 0.7
        except:
            return False

    def _has_unhandled_tool_calls(self, messages) -> bool:
        """
        Check if there are unhandled tool calls in the message chain
        
        This prevents the failsafe node from calling the model when there are
        incomplete tool call chains, which would cause OpenAI API errors.
        
        Args:
            messages: List of messages to check
            
        Returns:
            True if there are unhandled tool calls
        """
        try:
            from langchain_core.messages import AIMessage, ToolMessage
            
            # Find all tool call IDs that were made
            tool_call_ids = set()
            tool_response_ids = set()
            
            for message in messages:
                if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
                    # Collect tool call IDs
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, 'id'):
                            tool_call_ids.add(tool_call.id)
                        elif isinstance(tool_call, dict) and 'id' in tool_call:
                            tool_call_ids.add(tool_call['id'])
                            
                elif isinstance(message, ToolMessage) and hasattr(message, 'tool_call_id'):
                    # Collect tool response IDs
                    tool_response_ids.add(message.tool_call_id)
            
            # Check if there are tool calls without responses
            unhandled_tool_calls = tool_call_ids - tool_response_ids
            
            if unhandled_tool_calls:
                self.logger.debug(f"Found {len(unhandled_tool_calls)} unhandled tool calls: {unhandled_tool_calls}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for unhandled tool calls: {e}")
            # Err on the side of caution
            return True