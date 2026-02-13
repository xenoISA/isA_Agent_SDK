"""
Model Calling Mixin - Methods for calling LLM models
"""
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from isa_agent_sdk.services.trace.model_callback import trace_model_call


class ModelCallingMixin:
    """Mixin providing model calling methods for BaseNode"""

    @trace_model_call("BaseNode")
    async def call_model(
        self,
        messages,
        tools=None,
        model=None,
        provider=None,
        stream_tokens=True,
        output_format=None,
        show_reasoning=False
    ):
        """
        Call model with optional token streaming and output format

        Args:
            messages: List of LangChain messages
            tools: Optional tools for model
            model: Optional model override (e.g., "gpt-5", "gpt-oss-120b")
            provider: Optional provider override (e.g., "yyds", "cerebras", "openai")
            stream_tokens: Whether to stream tokens via stream_custom (default: True)
            output_format: "json" for structured JSON output, None for normal streaming
            show_reasoning: Enable reasoning visibility (for DeepSeek-R1)

        Returns:
            Final AIMessage response
        """
        call_start = time.time()

        # Log model input details
        self.logger.debug(
            f"model_input | node={self.node_name} | model={model} | provider={provider} | "
            f"stream={stream_tokens} | format={output_format} | tools={len(tools) if tools else 0} | messages={len(messages)}"
        )

        from isa_agent_sdk.clients.model_client import get_model_client

        service_start = time.time()
        model_service = await get_model_client()
        service_duration = int((time.time() - service_start) * 1000)

        self.logger.debug(
            f"base_node_get_model_service | "
            f"node={self.node_name} | "
            f"duration_ms={service_duration}"
        )

        # Handle JSON output mode
        self.logger.debug(f"call_model | output_format={output_format} | stream_tokens={stream_tokens}")
        if output_format == "json":
            self.logger.debug(f"JSON mode | model={model} | provider={provider}")
            try:
                # Use model_service for JSON output (it uses API mode AsyncISAModel)
                # Call model_service with response_format for JSON mode
                # Returns (AIMessage, billing_info) tuple
                response, billing_info = await model_service.call_model(
                    messages=messages,
                    tools=tools or [],
                    model=model,
                    provider=provider,
                    response_format={"type": "json_object"}  # Enable JSON mode
                )

                self.logger.debug(f"JSON mode response type: {type(response)}")
                return response

            except Exception as e:
                self.logger.error(f"JSON model call failed: {e}")
                return AIMessage(content=f"JSON output error: {str(e)}")

        # Normal streaming mode (existing logic)
        token_callback = None
        first_token_time = None

        if stream_tokens:
            self.logger.debug(f"stream_tokens=True | node={self.node_name}")

            def token_callback(data):
                nonlocal first_token_time
                if 'token' in data:
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = int((first_token_time - call_start) * 1000)
                        self.logger.debug(
                            f"base_node_first_token | "
                            f"node={self.node_name} | "
                            f"time_to_first_token_ms={ttft}"
                        )

                    # Check if this is a reasoning token (from DeepSeek-R1)
                    is_reasoning = data.get('type') == 'reasoning'

                    # Different chunk types based on node
                    if self.node_name == "ReasonNode":
                        if is_reasoning:
                            # Stream reasoning chunks separately
                            self.stream_custom({"reasoning_chunk": data['token']})
                        else:
                            self.stream_custom({"thinking_chunk": data['token']})
                    elif self.node_name == "ResponseNode":
                        self.stream_custom({"response_chunk": data['token']})
                    else:
                        # Fallback for other nodes
                        self.stream_custom({"custom_llm_chunk": data['token']})

        try:
            stream_start = time.time()
            self.logger.debug(
                f"base_node_call_start | "
                f"node={self.node_name} | "
                f"stream_tokens={stream_tokens} | "
                f"tools_count={len(tools) if tools else 0}"
            )

            # Choose between streaming and non-streaming based on stream_tokens
            if stream_tokens:
                # Streaming mode - use stream_tokens with token callback
                response, _ = await model_service.stream_tokens(
                    messages=messages,
                    token_callback=token_callback,
                    tools=tools,
                    model=model,
                    provider=provider,
                    timeout=120.0,
                    show_reasoning=show_reasoning
                )
            else:
                # Non-streaming mode - use call_model (fixes gpt-5 org verification issue)
                response, _ = await model_service.call_model(
                    messages=messages,
                    tools=tools,
                    model=model,
                    provider=provider,
                    timeout=120.0
                )

            stream_duration = int((time.time() - stream_start) * 1000)
            total_duration = int((time.time() - call_start) * 1000)

            response_content = getattr(response, 'content', '')
            response_type = type(response).__name__
            has_tool_calls = hasattr(response, 'tool_calls') and bool(getattr(response, 'tool_calls', []))

            self.logger.debug(
                f"model_output | "
                f"node={self.node_name} | "
                f"type={response_type} | "
                f"content_length={len(response_content)} | "
                f"has_tool_calls={has_tool_calls} | "
                f"stream_duration_ms={stream_duration} | "
                f"total_duration_ms={total_duration}"
            )

            return response
        except Exception as e:
            self.logger.error(f"Model call failed: {e}")
            return AIMessage(content=f"Model error: {str(e)}")

    def _messages_to_prompt(self, messages) -> str:
        """
        Convert LangChain messages to a prompt string for ISA client

        Args:
            messages: List of LangChain messages

        Returns:
            Combined prompt string
        """
        prompt_parts = []

        for msg in messages:
            content = getattr(msg, 'content', str(msg))
            msg_type = type(msg).__name__

            if msg_type == 'SystemMessage':
                prompt_parts.append(f"System: {content}")
            elif msg_type == 'HumanMessage':
                prompt_parts.append(f"Human: {content}")
            elif msg_type == 'AIMessage':
                prompt_parts.append(f"Assistant: {content}")
            elif msg_type == 'ToolMessage':
                prompt_parts.append(f"Tool Result: {content}")
            else:
                prompt_parts.append(f"{msg_type}: {content}")

        return "\n\n".join(prompt_parts)
