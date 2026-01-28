#!/usr/bin/env python3
"""
Response Node - Final response generation with LLM streaming

Clean response node that:
1. Gets default_response_prompt from MCP
2. Summarizes conversation messages
3. Calls model with token streaming via base_node
4. Adds final AIMessage to state
"""

import logging
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from .base_node import BaseNode
from isa_agent_sdk.utils.logger import agent_logger
from .utils.artifact_detector import detect_artifact

logger = agent_logger  # Use centralized logger for Loki integration


class ResponseNode(BaseNode):
    """Professional response generation node with MCP integration"""
    
    def __init__(self):
        super().__init__("ResponseNode")
    
    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Generate final response using MCP prompt and LLM streaming

        Args:
            state: Current agent state with conversation messages
            config: Runtime config with MCP service context

        Returns:
            Updated state with final AIMessage and end action
        """
        messages = state.get("messages", [])
        # Get session_id from config.configurable.thread_id (not from state!)
        session_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"

        # Log incoming state for debugging
        message_types = [type(msg).__name__ for msg in messages]
        output_format_from_config = config.get("configurable", {}).get("output_format") if config else None
        state_log = (
            f"[PHASE:NODE_RESPONSE] state_received | "
            f"session_id={session_id} | "
            f"messages_count={len(messages)} | "
            f"message_types={message_types} | "
            f"output_format_from_config={output_format_from_config}"
        )
        print(f"[RESPONSE_NODE] {state_log}", flush=True)
        self.logger.info(state_log)

        if not messages:
            self.logger.warning(
                f"[PHASE:NODE_RESPONSE] response_warning | "
                f"session_id={session_id} | "
                f"reason=no_messages"
            )
            return self._create_error_response(state, "No conversation messages available")
        
        # 1. Create conversation summary
        conversation_summary = self._build_conversation_summary(messages)
        
        # 2. Get response prompt from MCP with conversation summary
        # Check if JSON output is requested via config
        output_format = config.get("configurable", {}).get("output_format") if config else None
        
        if output_format == "json":
            # Use JSON-specific prompt arguments
            prompt_args = {
                "conversation_summary": conversation_summary,
                "output_format": "json"
            }
            # Try to get JSON-specific prompt, fallback to default if not available
            response_prompt = await self.mcp_get_prompt("default_response_json_prompt", prompt_args, config)
            
            if not response_prompt:
                # Fallback: Add JSON instruction to default prompt
                default_prompt = await self.mcp_get_prompt("default_response_prompt", {"conversation_summary": conversation_summary}, config)
                if default_prompt:
                    response_prompt = f"{default_prompt}\n\nIMPORTANT: Please provide your response in valid JSON format. Structure your response as a JSON object with relevant fields."
                else:
                    response_prompt = f"Please provide a helpful response based on this conversation in valid JSON format:\n\n{conversation_summary}\n\nResponse should be a properly formatted JSON object."
        else:
            # Normal text response
            prompt_args = {"conversation_summary": conversation_summary}
            response_prompt = await self.mcp_get_prompt("default_response_prompt", prompt_args, config)
            
            if not response_prompt:
                # Fallback to simple response prompt
                response_prompt = f"Please provide a helpful response based on this conversation:\n\n{conversation_summary}"
        
        try:
            # 3. Check output format and call model accordingly
            response_start = time.time()
            # Get output_format from config (more reliable than state)
            output_format = config.get("configurable", {}).get("output_format") if config else None

            # Log LLM input for debugging
            conversation_preview = conversation_summary[:300] if conversation_summary else 'N/A'
            stream_tokens_flag = (output_format != "json")

            self.logger.info(
                f"[PHASE:NODE_RESPONSE] llm_input | "
                f"session_id={session_id} | "
                f"prompt_name=default_response_prompt | "
                f"conversation_summary_len={len(conversation_summary)} | "
                f"conversation_preview='{conversation_preview}' | "
                f"output_format={output_format or 'text'} | "
                f"stream_tokens={stream_tokens_flag}"
            )
            
            # Log complete LLM input for session context debugging
            self.logger.info(
                f"[PHASE:NODE_RESPONSE] llm_complete_input | "
                f"session_id={session_id} | "
                f"response_prompt={response_prompt} | "
                f"conversation_summary={conversation_summary}"
            )

            print(f"[DEBUG] ResponseNode llm_input | session_id={session_id} | stream_tokens={stream_tokens_flag} | output_format={output_format} | state_keys={list(state.keys())}", flush=True)

            # Log the complete prompt being sent to the model
            print(f"[DEBUG] {'='*80}", flush=True)
            print(f"[DEBUG] COMPLETE PROMPT BEING SENT TO MODEL:", flush=True)
            print(f"[DEBUG] {'='*80}", flush=True)
            print(f"[DEBUG] {response_prompt}", flush=True)
            print(f"[DEBUG] {'='*80}", flush=True)

            self.logger.info(
                f"[RESPONSE_NODE] model_call_input | "
                f"session_id={session_id} | "
                f"output_format={output_format or 'text'} | "
                f"stream_tokens={stream_tokens_flag} | "
                f"prompt_length={len(response_prompt)} | "
                f"complete_prompt='{response_prompt}'"
            )

            # Use BaseNode's call_model
            if output_format == "json":
                # For JSON output, use OpenAI gpt-4.1-nano with response_format support
                print(f"[DEBUG] Calling model: gpt-4.1-nano (openai) with output_format=json", flush=True)
                response = await self.call_model(
                    messages=[HumanMessage(content=response_prompt)],
                    stream_tokens=False,  # Disable streaming for JSON
                    output_format="json",
                    provider="openai",
                    model="gpt-4.1-nano"
                )
            else:
                # For normal text output, use configured response model (gpt-5-nano by default)
                from isa_agent_sdk.core.config import settings
                print(f"[DEBUG] Calling model: {settings.response_model} ({settings.response_model_provider})", flush=True)
                response = await self.call_model(
                    messages=[HumanMessage(content=response_prompt)],
                    stream_tokens=stream_tokens_flag,
                    model=settings.response_model,
                    provider=settings.response_model_provider
                )

            print(f"[DEBUG] ResponseNode llm_output | session_id={session_id} | response_type={type(response).__name__} | has_content={hasattr(response, 'content')}", flush=True)

            response_duration = int((time.time() - response_start) * 1000)
            response_length = len(response.content) if hasattr(response, 'content') else 0
            response_preview = response.content[:300] if hasattr(response, 'content') else 'N/A'

            self.logger.info(
                f"[PHASE:NODE_RESPONSE] llm_output | "
                f"session_id={session_id} | "
                f"node=response | "
                f"duration_ms={response_duration} | "
                f"response_length={response_length} | "
                f"response_preview='{response_preview}' | "
                f"output_format={output_format or 'text'}"
            )

            # 5. Mark this as final assistant response (LangGraph best practice)
            # This differentiates it from internal reasoning messages
            if hasattr(response, 'name') or not response.name:
                response.name = "assistant"  # Standard assistant response name

            # Add metadata to help identify this as the final response
            if hasattr(response, 'additional_kwargs'):
                response.additional_kwargs.update({
                    "message_role": "final_response",
                    "is_internal": False,
                    "node": "response_node",
                    "output_format": output_format or "text"
                })

            self.logger.info(
                f"[PHASE:NODE_RESPONSE] marked_as_final | "
                f"session_id={session_id} | "
                f"name=assistant | "
                f"user_facing=True"
            )

            # 6. Detect artifact in response content
            response_content = response.content if hasattr(response, 'content') else ""
            artifact_data = detect_artifact(response_content)

            if artifact_data:
                # Emit artifact.generated event
                await self.emit_event(
                    event_type="artifact.generated",
                    content=f"Generated {artifact_data['type']} artifact: {artifact_data['title']}",
                    metadata={"artifact": artifact_data},
                    config=config
                )

                self.logger.info(
                    f"[PHASE:NODE_RESPONSE] artifact_detected | "
                    f"session_id={session_id} | "
                    f"artifact_id={artifact_data['id']} | "
                    f"artifact_type={artifact_data['type']} | "
                    f"artifact_title={artifact_data['title']}"
                )

            # 7. Add final response to conversation (add_messages reducer handles appending)
            # Note: Response prompt already used in model call, just add the final response
            return {
                "messages": [response],
                "next_action": "end"
            }

        except Exception as e:
            self.logger.error(
                f"response_error | "
                f"session_id={session_id} | "
                f"node=response | "
                f"error={type(e).__name__} | "
                f"message={str(e)[:200]}",
                exc_info=True
            )
            return self._create_error_response(state, f"Response error: {str(e)}")
    
    def _build_conversation_summary(self, messages) -> str:
        """
        Build conversation summary with FULL conversation history

        Filters out internal reasoning but keeps all user-facing messages
        to maintain complete conversation context for memory persistence.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted instruction with full conversation context
        """
        conversation_messages = []

        # Log all messages for debugging
        self.logger.info(f"[RESPONSE_NODE] _build_conversation_summary | total_messages={len(messages)}")

        for idx, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = getattr(msg, 'content', '')
            content_type = type(content).__name__
            content_length = len(str(content)) if content else 0

            # Get additional info
            has_additional_kwargs = hasattr(msg, 'additional_kwargs')
            is_internal = msg.additional_kwargs.get('is_internal', False) if has_additional_kwargs else False

            # Log each message in detail
            content_preview = str(content)[:200] if content else 'EMPTY'
            self.logger.info(
                f"[RESPONSE_NODE] message_{idx} | "
                f"type={msg_type} | "
                f"content_type={content_type} | "
                f"content_length={content_length} | "
                f"is_internal={is_internal} | "
                f"content_preview='{content_preview}'"
            )

            # Also print to console for immediate visibility
            print(f"[DEBUG] Message {idx}: type={msg_type}, content_type={content_type}, length={content_length}", flush=True)
            print(f"[DEBUG]   Content preview: {content_preview}", flush=True)

            # Skip internal reasoning messages (marked with is_internal=True)
            if has_additional_kwargs and is_internal:
                self.logger.info(f"[RESPONSE_NODE] message_{idx} | action=SKIPPED | reason=is_internal")
                print(f"[DEBUG]   -> SKIPPED (internal)", flush=True)
                continue

            # Skip messages with no content
            if not content:
                self.logger.info(f"[RESPONSE_NODE] message_{idx} | action=SKIPPED | reason=no_content")
                print(f"[DEBUG]   -> SKIPPED (no content)", flush=True)
                continue

            # Keep all user messages and assistant responses
            role = "User" if msg_type == "HumanMessage" else "Assistant"
            formatted_msg = f"{role}: {content}"
            conversation_messages.append(formatted_msg)

            self.logger.info(f"[RESPONSE_NODE] message_{idx} | action=INCLUDED | role={role}")
            print(f"[DEBUG]   -> INCLUDED as {role}", flush=True)

        # Use last 10 non-internal messages for context (prevents token overflow)
        conversation_text = "\n\n".join(conversation_messages[-10:])

        self.logger.info(
            f"[RESPONSE_NODE] conversation_summary | "
            f"total_included={len(conversation_messages)} | "
            f"used_last_10={min(10, len(conversation_messages))} | "
            f"summary_length={len(conversation_text)}"
        )

        summary = f"""Please provide a helpful response based on this conversation:

{conversation_text}

Respond naturally to the user's most recent question using all relevant context above."""

        # Log the complete summary
        self.logger.info(f"[RESPONSE_NODE] complete_summary | summary='{summary}'")
        print(f"[DEBUG] Complete conversation summary:", flush=True)
        print(f"[DEBUG] {summary}", flush=True)
        print(f"[DEBUG] {'='*80}", flush=True)

        return summary
    
    def _create_error_response(self, state: AgentState, error_message: str) -> AgentState:
        """Create error response and update state"""
        _ = state  # State parameter kept for API consistency
        error_response = AIMessage(content=f"I apologize, but I encountered an error: {error_message}")
        
        # Return state update using add_messages reducer
        return {
            "messages": [error_response],
            "next_action": "end"
        }