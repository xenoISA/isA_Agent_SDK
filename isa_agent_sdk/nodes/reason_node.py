#!/usr/bin/env python3
"""
Reason Node - Simple LLM reasoning and decision making

Clean reasoning node that:
1. Gets default_reason_prompt from context
2. Uses MCP get_prompt to get complete prompt
3. Calls model with token streaming via base_node
4. Determines next action based on response
"""

import logging
import time
from typing import Dict, Any
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from isa_agent_sdk.errors import (
    ISASDKError,
    ISAConnectionError,
    ISATimeoutError,
    ModelError,
    MCPError,
)
from .base_node import BaseNode
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger  # Use centralized logger for Loki integration


class ReasonNode(BaseNode):
    """Simple reasoning node for LLM interaction and decision making"""

    def __init__(self):
        super().__init__("ReasonNode")

    async def _build_system_prompt(
        self,
        config: RunnableConfig,
        context: Dict[str, Any],
        session_id: str
    ) -> str:
        """
        Build system prompt using MCP templates with user customization support.

        Priority:
        1. If system_prompt_config.replace is set, use that entirely
        2. Otherwise, call MCP to get template with user_instructions appended
        3. Fallback to hardcoded prompt if MCP fails

        Args:
            config: Runtime config containing system_prompt_config
            context: Runtime context with memory, tools, skills
            session_id: Session ID for logging

        Returns:
            Assembled system prompt string
        """
        # Get system prompt config from runtime config
        system_prompt_config = config.get("configurable", {}).get("system_prompt_config")

        # Extract context values
        memory_str = context.get('memory_context', '') or "None"
        tools_str = self._build_tools_string(context.get('available_tools', []))
        # Handle resources - can be dict (keys) or list (items)
        default_resources = context.get('default_resources')
        if default_resources:
            if isinstance(default_resources, dict):
                resources_str = ", ".join(default_resources.keys())
            elif isinstance(default_resources, list):
                resources_str = ", ".join(str(r) for r in default_resources)
            else:
                resources_str = str(default_resources)
        else:
            resources_str = "None"
        skill_injection = context.get('skill_injection', '')
        loaded_skills = context.get('loaded_skills', {})
        # Handle loaded_skills - can be dict (keys are skill names) or list (skill names)
        if isinstance(loaded_skills, dict):
            skill_names = list(loaded_skills.keys()) if loaded_skills else []
        elif isinstance(loaded_skills, list):
            skill_names = loaded_skills
        else:
            skill_names = []

        # Check for full replacement
        if system_prompt_config and system_prompt_config.is_replacement:
            self.logger.info(
                f"reason_node_using_replacement_prompt | "
                f"session_id={session_id} | "
                f"length={len(system_prompt_config.replace)}"
            )
            return system_prompt_config.replace

        # Get user instructions to append
        user_instructions = ""
        if system_prompt_config and system_prompt_config.append:
            user_instructions = system_prompt_config.append

        # Add project context (ISA.md/CLAUDE.md) if available
        project_context = config.get("configurable", {}).get("project_context", "")
        if project_context:
            # Prepend project context to user instructions
            project_section = f"""## PROJECT CONTEXT

The following is persistent project context that should inform all responses:

{project_context}

---
"""
            user_instructions = project_section + (user_instructions or "")
            self.logger.info(
                f"reason_node_project_context_injected | "
                f"session_id={session_id} | "
                f"context_length={len(project_context)}"
            )

        # Determine which MCP prompt to use
        prompt_name = "default_reason_prompt"
        if system_prompt_config and system_prompt_config.preset:
            prompt_name = system_prompt_config.get_mcp_prompt_name("reason")

        # Check if user has files (use RAG prompt if so)
        has_user_files = context.get('has_user_files', False)
        file_context = context.get('file_context', '')
        if has_user_files and prompt_name == "default_reason_prompt":
            prompt_name = "rag_reason_prompt"

        # Build prompt arguments for MCP
        prompt_args = {
            "memory": memory_str,
            "tools": tools_str,
            "resources": resources_str,
            "skills": skill_injection,
            "user_instructions": user_instructions,
        }

        # Add RAG-specific args if using RAG prompt
        if prompt_name == "rag_reason_prompt":
            prompt_args["file_context"] = file_context
            prompt_args["file_count"] = context.get('file_count', 0)
            prompt_args["file_types"] = ", ".join(
                set(f.get("content_type", "unknown") for f in context.get('relevant_files', []))
            ) if context.get('relevant_files') else ""

        # Try to get prompt from MCP
        self.logger.info(
            f"reason_node_fetching_mcp_prompt | "
            f"session_id={session_id} | "
            f"prompt_name={prompt_name} | "
            f"has_user_instructions={bool(user_instructions)}"
        )

        try:
            system_prompt = await self.mcp_get_prompt(prompt_name, prompt_args, config)

            if system_prompt:
                self.logger.info(
                    f"reason_node_mcp_prompt_success | "
                    f"session_id={session_id} | "
                    f"prompt_name={prompt_name} | "
                    f"length={len(system_prompt)}"
                )
                return system_prompt
        except Exception as e:
            self.logger.warning(
                f"reason_node_mcp_prompt_failed | "
                f"session_id={session_id} | "
                f"error={e}"
            )

        # Fallback: Build prompt locally (same as before but with user_instructions)
        self.logger.info(
            f"reason_node_using_fallback_prompt | "
            f"session_id={session_id}"
        )
        return self._build_fallback_prompt(
            memory_str, tools_str, resources_str,
            skill_names, skill_injection, user_instructions
        )

    def _build_tools_string(self, available_tools: list) -> str:
        """Build tools string from available tools list"""
        if not available_tools:
            return "None available"
        return ", ".join(t.get('name', 'unknown') for t in available_tools[:20])

    def _build_fallback_prompt(
        self,
        memory_str: str,
        tools_str: str,
        resources_str: str,
        skill_names: list,
        skill_injection: str,
        user_instructions: str = ""
    ) -> str:
        """Build fallback prompt when MCP is unavailable"""
        skills_section = ""
        if skill_injection:
            skills_section = f"""
## 🎯 LOADED SKILLS (Structured Workflows):
{skill_injection}

## SKILL-DRIVEN PLANNING (IMPORTANT):
When skills are loaded above, you MUST follow their defined workflow:

1. **Identify the skill to use** - Match user request to loaded skill
2. **Follow skill's workflow steps** - Skills define a structured process
3. **Pass skill workflow to create_execution_plan** - Include the skill's steps in the `guidance` parameter
4. **Map skill steps → concrete tasks** - Each skill step becomes a task in the plan

**⚠️ CRITICAL**: When a skill is loaded and user requests that workflow, you MUST include the skill's steps in your plan guidance.
"""

        user_section = ""
        if user_instructions:
            user_section = f"""
## CUSTOM INSTRUCTIONS (from user configuration):
{user_instructions}
"""

        return f"""You are an intelligent reasoning assistant in the THINKING PHASE. Your output shows your analytical process to the user.

## YOUR ROLE - Reasoning Layer:
You analyze requests and decide the best approach. Users see your thinking process.

## Your Capabilities:
- **Memory**: Previous conversations and preferences: {memory_str}
- **Tools**: {tools_str if tools_str else "None available"}
- **Resources**: {resources_str if resources_str else "None available"}
- **Skills**: {', '.join(skill_names) if skill_names else "None loaded"}
{skills_section}
## User Request:
(See conversation history below)

## YOUR THINKING PROCESS (user will see this):

### Step 1: Analyze the Request
- What does the user actually need?
- What type of task is this? (simple question, info gathering, complex task)
- What context is relevant from memory?
- **Is there a loaded skill that matches this request?** (Check skills above)

### Step 2: Evaluate Approach
Consider your options:
1. **Direct Answer** - I can answer this directly because...
2. **Use Single Tool** - I need to use [tool_name] to gather specific information...
3. **Skill-Driven Task** - A loaded skill matches this request, I'll follow its workflow...
4. **Complex Multi-Step Task** - This requires multiple steps and careful planning...

### Step 3: Decision & Action

**PRIORITY 1 - Skill-Driven Tasks (when skill is loaded):**
If a loaded skill matches the user's request:
➡️ **Call `create_execution_plan` with skill's workflow in guidance**
- Include the skill's defined steps in the guidance parameter
- Tasks should follow the skill's structured process
- Do NOT deviate from the skill's workflow

**PRIORITY 2 - Complex Multi-Step Tasks (no matching skill):**
If the request involves ANY of these:
- Multiple distinct steps or phases
- Research + Analysis + Report/Summary
- Multiple data sources or tools needed
- Sequential dependencies between tasks

➡️ **Call `create_execution_plan` tool** to break it into sub-tasks

**If using single tool:**
- Explain what information you need and why
- Call the appropriate tool

**If answering directly:**
- Provide concise analysis and conclusion
- Keep it brief - the response node will format the final reply

## OUTPUT GUIDELINES:
✅ DO:
- Show your analytical thinking
- Explain your reasoning clearly
- Be concise but thorough
- If skill loaded: mention you're following the skill's workflow
{user_section}
❌ DON'T:
- Write the full final response (that's the response node's job)
- Ignore loaded skills when they match the request"""
    
    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Execute LLM reasoning - simplified streamlined approach

        Args:
            state: Current agent state (may contain execution results)
            config: Runtime config with context (prompts, tools, memory, etc.)

        Returns:
            Updated state with LLM response and next_action
        """
        node_start = time.time()

        # Log incoming state
        messages = state.get("messages", [])
        message_types = [type(msg).__name__ for msg in messages]
        self.logger.debug(
            f"state_received | "
            f"messages_count={len(messages)} | "
            f"message_types={message_types} | "
            f"state_keys={list(state.keys())}"
        )

        # 1. Get context from runnable_config
        context_start = time.time()
        context = self.get_runtime_context(config)
        user_query = context.get('enhanced_query', context.get('original_query', ''))
        available_tools = context.get('available_tools', [])
        # Get session_id from config.configurable.thread_id (not from state!)
        session_id = config.get("configurable", {}).get("thread_id", "unknown") if config else "unknown"
        context_duration = int((time.time() - context_start) * 1000)

        # Debug: Log available_tools content
        self.logger.info(
            f"[DEBUG_REASON_NODE] session_id={session_id} | "
            f"context_keys={list(context.keys())} | "
            f"available_tools_count={len(available_tools)} | "
            f"tools={[t.get('name', 'unknown') for t in available_tools]}"
        )

        self.logger.debug(
            f"reason_node_context | "
            f"session_id={session_id} | "
            f"duration_ms={context_duration}"
        )
        self.logger.info(
            f"reason_node_context | "
            f"session_id={session_id} | "
            f"duration_ms={context_duration}"
        )

        # 2. Smart prompt selection based on execution results
        prompt_prep_start = time.time()
        has_execution_results = self._has_execution_results(state.get("messages", []))

        if has_execution_results:
            # Use review prompt for evaluating execution results
            prompt_name = "default_review_prompt"
            execution_results = self._extract_execution_results(state.get("messages", []))
            conversation_summary = self._build_conversation_summary(state.get("messages", []))

            # Convert tools list to comma-separated string for MCP
            tools_str = ", ".join([
                tool.get('name', str(tool)) if isinstance(tool, dict) else str(tool)
                for tool in available_tools
            ]) if available_tools else ""

            # Convert resources list to comma-separated string for MCP
            resources_str = ", ".join([
                resource.get('name', str(resource)) if isinstance(resource, dict) else str(resource)
                for resource in context.get('default_resources', [])
            ]) if context.get('default_resources') else ""

            prompt_args = {
                'user_message': user_query,
                'execution_results': execution_results,
                'conversation_summary': conversation_summary,
                'memory': context.get('memory_context', ''),
                'tools': tools_str,
                'resources': resources_str,
                'skills': context.get('skill_injection', '')  # Add skills for review context
            }

            self.logger.info(
                f"reason_decision | "
                f"session_id={session_id} | "
                f"mode=review | "
                f"tools_count={len(available_tools)} | "
                f"has_memory={bool(context.get('memory_context'))} | "
                f"has_skills={bool(context.get('skill_injection'))}"
            )
        else:
            # Use initial planning prompt
            prompt_name = "default_reason_prompt"
            # Convert tools list to comma-separated string for MCP
            tools_str = ", ".join([
                tool.get('name', str(tool)) if isinstance(tool, dict) else str(tool)
                for tool in available_tools
            ]) if available_tools else ""

            # Convert resources list to comma-separated string for MCP
            resources_str = ", ".join([
                resource.get('name', str(resource)) if isinstance(resource, dict) else str(resource)
                for resource in context.get('default_resources', [])
            ]) if context.get('default_resources') else ""

            # Add file information if user has uploaded files
            file_info_str = ""
            if context.get('has_user_files', False):
                user_file_info = context.get('user_file_info', {})
                file_count = user_file_info.get('total_files', 0)
                file_types = user_file_info.get('file_types', [])
                recent_files = user_file_info.get('recent_files', [])
                
                file_info_parts = [f"User has uploaded {file_count} files."]
                if file_types:
                    file_info_parts.append(f"File types: {', '.join(file_types[:3])}")
                if recent_files:
                    file_names = [f.get('file_name', 'Unknown') for f in recent_files[:3]]
                    file_info_parts.append(f"Recent files: {', '.join(file_names)}")
                file_info_parts.append("You can use data analysis tools (data_ingest, data_query, data_search) and knowledge tools (store_knowledge, search_knowledge) to process these files.")
                
                file_info_str = " ".join(file_info_parts)

            prompt_args = {
                'user_message': user_query,
                'tools': tools_str,
                'memory': context.get('memory_context', ''),
                'resources': resources_str,
                'file_info': file_info_str,
                'skills': context.get('skill_injection', '')  # Add skills from context
            }

            self.logger.info(
                f"reason_decision | "
                f"session_id={session_id} | "
                f"mode=initial | "
                f"tools_count={len(available_tools)} | "
                f"has_memory={bool(context.get('memory_context'))} | "
                f"has_skills={bool(context.get('skill_injection'))}"
            )
        
        prompt_prep_duration = int((time.time() - prompt_prep_start) * 1000)
        self.logger.debug(
            f"reason_node_prompt_prep | "
            f"session_id={session_id} | "
            f"duration_ms={prompt_prep_duration}"
        )

        # 3. Build system prompt using MCP templates with user customization
        prompt_lookup_start = time.time()
        system_prompt = await self._build_system_prompt(config, context, session_id)
        prompt_lookup_duration = int((time.time() - prompt_lookup_start) * 1000)

        self.logger.info(
            f"reason_node_prompt_built | "
            f"session_id={session_id} | "
            f"length={len(system_prompt)} | "
            f"duration_ms={prompt_lookup_duration}"
        )

        # 4. Construct messages for model call with conversation history
        messages_prep_start = time.time()
        conversation_messages = state.get("messages", [])

        # Clean up incomplete tool calls from conversation history
        conversation_messages = self._cleanup_incomplete_tool_calls(conversation_messages)

        # Build messages list with proper ordering
        messages = [SystemMessage(content=system_prompt)]
        self.logger.debug(f"SystemMessage created | content_length={len(messages[0].content)}")

        # Add conversation summary if exists (from auto-summarization)
        if state.get("summary"):
            summary_content = f"## Conversation Summary\n{state['summary']}\n\n(The above is a summary of earlier conversation. Recent messages follow below.)"
            messages.append(SystemMessage(content=summary_content))
            self.logger.info(
                f"reason_node_summary_included | "
                f"session_id={session_id} | "
                f"summary_length={len(state['summary'])}"
            )

        # Add full conversation history
        messages.extend(conversation_messages)
        messages_prep_duration = int((time.time() - messages_prep_start) * 1000)

        self.logger.debug(
            f"reason_node_messages_prep | "
            f"session_id={session_id} | "
            f"message_count={len(messages)} | "
            f"duration_ms={messages_prep_duration}"
        )

        try:
            # 4. Call model with complete conversation history and available tools using BaseNode
            model_start = time.time()
            self.logger.debug(
                f"reason_node_model_call_start | "
                f"session_id={session_id} | "
                f"tool_count={len(available_tools or [])} | "
                f"message_count={len(messages)}"
            )

            # Log LLM input for debugging
            tool_names_input = [t.get('name', 'unknown') for t in (available_tools or [])]
            last_user_msg = next((m.content for m in reversed(messages) if hasattr(m, 'type') and m.type == 'human'), 'N/A')[:200]
            self.logger.info(
                f"[PHASE:NODE_REASON] llm_input | "
                f"session_id={session_id} | "
                f"prompt_name={prompt_name} | "
                f"system_prompt_len={len(system_prompt)} | "
                f"tool_count={len(available_tools or [])} | "
                f"tools={tool_names_input} | "
                f"message_count={len(messages)} | "
                f"last_user_msg='{last_user_msg}'"
            )
            
            # Log complete LLM input for session context debugging
            complete_input_log = []
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content = getattr(msg, 'content', str(msg))
                # Truncate very long content but keep more for debugging
                if len(content) > 1000:
                    content = content[:1000] + "...[TRUNCATED]"
                complete_input_log.append(f"{i+1}. {msg_type}: {content[:200]}")

            self.logger.debug(
                f"reason_node_messages | "
                f"session_id={session_id} | "
                f"messages_count={len(messages)} | "
                f"types={[type(m).__name__ for m in messages]}"
            )

            self.logger.info(
                f"[PHASE:NODE_REASON] llm_complete_input | "
                f"session_id={session_id} | "
                f"messages_count={len(messages)} | "
                f"messages_preview={'; '.join(complete_input_log[:3])}"
            )

            # ReasonNode always uses streaming to show thinking process
            # isa-model 0.5.6+ supports tool_calls in streaming mode
            # output_format only affects ResponseNode for final output formatting

            # Use BaseNode's unified call_model method
            # Use DeepSeek-R1 for reasoning with thinking process visibility
            from isa_agent_sdk.core.config import settings
            call_model_start = time.time()
            response = await self.call_model(
                messages=messages,
                tools=available_tools or [],
                model=settings.reason_model,  # Override: Use deepseek-r1 for reasoning
                provider=settings.reason_model_provider,  # Override: Use yyds provider
                stream_tokens=True,  # Always enable streaming (0.5.6+ supports tool_calls in streaming)
                output_format=None,  # Force streaming mode, ignore JSON output format
                show_reasoning=True  # Enable thinking process visibility for DeepSeek-R1
            )
            call_model_duration = int((time.time() - call_model_start) * 1000)

            model_duration = int((time.time() - model_start) * 1000)

            self.logger.debug(
                f"reason_node_call_model | "
                f"session_id={session_id} | "
                f"duration_ms={call_model_duration}"
            )

            # 5. Determine next action
            has_tool_calls = hasattr(response, 'tool_calls') and bool(getattr(response, 'tool_calls', []))
            next_action = "call_tool" if has_tool_calls else "end"

            if has_tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
                tool_args_summary = []
                for tc in response.tool_calls[:3]:  # Log first 3 tool calls
                    args = tc.get('args', {})
                    args_str = str(args)[:100] if args else '{}'
                    tool_args_summary.append(f"{tc.get('name')}({args_str})")

                # Note: Tool call events are automatically handled by stream_processor's messages mode
                # when it processes AIMessage.tool_calls, so we don't send custom events here

                self.logger.info(
                    f"[PHASE:NODE_REASON] llm_output | "
                    f"session_id={session_id} | "
                    f"node=reason | "
                    f"duration_ms={model_duration} | "
                    f"decision=call_tool | "
                    f"tool_calls={len(response.tool_calls)} | "
                    f"tools={','.join(tool_names)} | "
                    f"tool_args={tool_args_summary}"
                )
            else:
                response_preview = response.content[:200] if hasattr(response, 'content') else 'N/A'
                self.logger.info(
                    f"[PHASE:NODE_REASON] llm_output | "
                    f"session_id={session_id} | "
                    f"node=reason | "
                    f"duration_ms={model_duration} | "
                    f"decision=end | "
                    f"response_length={len(response.content) if hasattr(response, 'content') else 0} | "
                    f"response_preview='{response_preview}'"
                )
            
            # 6. Send complete thinking content as event (non-streaming)
            # ALWAYS send thinking_complete to show reasoning process, even when calling tools
            # CRITICAL FIX: Send thinking_complete even if content is empty to prevent frontend hanging
            thinking_content = getattr(response, 'content', '')
            if has_tool_calls and not thinking_content:
                # If model only returned tool calls without reasoning text, provide minimal feedback
                thinking_content = "Planning to use tools..."

            self.stream_custom({
                "thinking_complete": thinking_content
            })
            self.logger.info(
                f"[PHASE:NODE_REASON] thinking_complete_sent | "
                f"session_id={session_id} | "
                f"has_tool_calls={has_tool_calls} | "
                f"content_length={len(thinking_content)} | "
                f"was_empty={len(thinking_content) == 0}"
            )

            # 7. Mark this as internal reasoning message (LangGraph best practice)
            # This allows filtering out reasoning from final conversation history
            if hasattr(response, 'name') or not response.name:
                # Set name to identify this as internal reasoning
                response.name = "internal_reasoning"

            # Add metadata to help identify this message type
            if hasattr(response, 'additional_kwargs'):
                # Preserve existing reasoning_content when updating
                existing_reasoning = response.additional_kwargs.get('reasoning_content') if response.additional_kwargs else None

                response.additional_kwargs.update({
                    "message_role": "reasoning",
                    "is_internal": True,
                    "node": "reason_node",
                    "prompt_used": prompt_name
                })

                # Ensure reasoning_content is preserved (DeepSeek-R1 requires it when tool_calls exist)
                if existing_reasoning:
                    response.additional_kwargs['reasoning_content'] = existing_reasoning
                elif has_tool_calls:
                    response.additional_kwargs['reasoning_content'] = ""

            self.logger.info(
                f"[PHASE:NODE_REASON] marked_as_internal | "
                f"session_id={session_id} | "
                f"name=internal_reasoning | "
                f"can_be_filtered=True"
            )

            # 8. Log output state and return
            output_msg_type = type(response).__name__
            self.logger.debug(
                f"state_output | "
                f"session_id={session_id} | "
                f"output_message_type={output_msg_type} | "
                f"message_name={response.name} | "
                f"next_action={next_action}"
            )

            # Return state update - let add_messages reducer handle appending
            return {
                "messages": [response],
                "next_action": next_action
            }

        except Exception as e:
            self.logger.error(
                f"reasoning_error | "
                f"session_id={session_id} | "
                f"node=reason | "
                f"error={type(e).__name__} | "
                f"message={str(e)[:200]}",
                exc_info=True
            )
            return self._create_error_response(f"Reasoning error: {str(e)}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response and update state"""
        error_response = AIMessage(content=f"I apologize, but I encountered an error: {error_message}")
        
        # Return state update - let add_messages reducer handle appending
        return {
            "messages": [error_response],
            "next_action": "end"
        }
    
    def _has_execution_results(self, messages) -> bool:
        """
        Check if messages contain execution results (tool results or agent executor results)
        
        Args:
            messages: List of conversation messages
            
        Returns:
            True if execution results are found
        """
        for message in messages:
            # Check for ToolMessage (indicates tool execution completed)
            if hasattr(message, 'tool_call_id') or type(message).__name__ == 'ToolMessage':
                return True
            
            if hasattr(message, 'content'):
                content = str(message.content)
                # Look for agent executor result patterns
                if any(pattern in content for pattern in [
                    "[TASK_RESULT]", 
                    "[AGENT_EXECUTOR]", 
                    "[PARALLEL_TASK_RESULT]",
                    "Autonomous Execution Complete"
                ]):
                    return True
        return False
    
    def _extract_execution_results(self, messages) -> str:
        """
        Extract execution results from messages (tool results and agent executor results)
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Formatted string of execution results
        """
        execution_messages = []
        
        for message in messages:
            # Extract ToolMessage results
            if hasattr(message, 'tool_call_id') or type(message).__name__ == 'ToolMessage':
                content = str(getattr(message, 'content', ''))
                if content:
                    execution_messages.append(f"[TOOL_RESULT]: {content}")
            
            # Extract agent executor results
            elif hasattr(message, 'content'):
                content = str(message.content)
                # Collect agent executor related messages
                if any(pattern in content for pattern in [
                    "[TASK_RESULT]", 
                    "[AGENT_EXECUTOR]", 
                    "[PARALLEL_TASK_RESULT]",
                    "[TASK_ERROR]",
                    "[PARALLEL_TIMEOUT]"
                ]):
                    execution_messages.append(content)
        
        return "\n\n".join(execution_messages) if execution_messages else "No execution results found"
    
    def _cleanup_incomplete_tool_calls(self, messages):
        """
        Remove incomplete tool calls from conversation history to prevent OpenAI API errors.

        An assistant message with tool_calls must be followed by ToolMessage for each tool_call_id.
        If we find orphaned tool_calls (without corresponding responses), we remove them.

        Args:
            messages: List of conversation messages

        Returns:
            Cleaned message list
        """
        if not messages:
            return messages

        cleaned_messages = []
        i = 0

        while i < len(messages):
            message = messages[i]

            # Check if this is an AIMessage with tool_calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call_ids = {tc.get('id') for tc in message.tool_calls if isinstance(tc, dict) and 'id' in tc}

                # Look ahead for corresponding ToolMessages
                j = i + 1
                found_tool_responses = set()

                while j < len(messages):
                    next_msg = messages[j]
                    if hasattr(next_msg, 'tool_call_id'):
                        found_tool_responses.add(next_msg.tool_call_id)
                        j += 1
                    else:
                        break

                # Check if all tool calls have responses
                if tool_call_ids and not tool_call_ids.issubset(found_tool_responses):
                    # Incomplete tool calls - skip this message and any partial responses
                    missing_ids = tool_call_ids - found_tool_responses
                    self.logger.warning(
                        f"Removing incomplete tool call message with missing responses for IDs: {missing_ids}"
                    )
                    i = j  # Skip to after any partial tool responses
                    continue

            cleaned_messages.append(message)
            i += 1

        return cleaned_messages

    def _build_conversation_summary(self, messages) -> str:
        """
        Build a summary of the complete conversation

        Args:
            messages: List of conversation messages

        Returns:
            Formatted conversation summary
        """
        summary_parts = []

        for i, message in enumerate(messages):
            if hasattr(message, 'content'):
                content = str(message.content)
                msg_type = type(message).__name__

                # Truncate very long messages for summary, but preserve tool results
                if msg_type == 'ToolMessage':
                    # Keep more content for tool results (they contain valuable data)
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                else:
                    # Regular messages get normal truncation
                    if len(content) > 500:
                        content = content[:500] + "..."

                summary_parts.append(f"{i+1}. {msg_type}: {content}")

        return "\n".join(summary_parts) if summary_parts else "No conversation history"
