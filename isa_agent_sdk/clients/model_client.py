#!/usr/bin/env python3
"""
Model Client - Dual-mode model abstraction (Service/Direct)

Supports TWO modes:
1. SERVICE mode: Connect to isA_Model service via AsyncISAModel
2. DIRECT mode: Use AIFactory for direct API access (no service needed)

Core focus: Performance, Concurrency, Security, Stability, Flexibility
- OpenAI-compatible interface
- Connection pooling and reuse
- Async/concurrent processing
- Error handling and timeouts
- Format negotiation (LangChain, OpenAI dict, string)
- Automatic fallback from service to direct mode
"""

import logging
import asyncio
import json
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    """Model client operation mode"""
    SERVICE = "service"  # Connect to isA_Model service (AsyncISAModel)
    DIRECT = "direct"    # Direct API access via AIFactory (no service needed)
    AUTO = "auto"        # Try service first, fallback to direct


@dataclass
class ModelConfig:
    """Configuration for model client"""
    mode: ModelMode = ModelMode.AUTO

    # Service mode settings (AsyncISAModel → isA_Model service)
    service_url: Optional[str] = None  # e.g., http://localhost:8082

    # Direct mode settings (AIFactory → cloud APIs directly)
    provider: str = "openai"  # openai, ollama, cerebras, yyds, huggingface
    api_key: Optional[str] = None  # API key for direct mode

    # Common settings
    default_model: str = "gpt-4o-mini"
    timeout: float = 120.0
    fallback_to_direct: bool = True  # If service fails, try direct


class ModelClient:
    """
    Dual-mode model client supporting both service and direct API access.

    Usage:
        # Mode 1: Service mode (connects to isA_Model service)
        client = ModelClient(config=ModelConfig(mode=ModelMode.SERVICE, service_url="http://localhost:8082"))

        # Mode 2: Direct mode (uses AIFactory, no service needed)
        client = ModelClient(config=ModelConfig(mode=ModelMode.DIRECT, provider="openai"))

        # Mode 3: Auto mode (try service, fallback to direct)
        client = ModelClient(config=ModelConfig(mode=ModelMode.AUTO))
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        isa_url: str = None,  # Legacy parameter for backward compatibility
    ):
        """
        Initialize model client

        Args:
            config: ModelConfig with mode and settings
            isa_url: Legacy parameter - ISA Model API base URL (for backward compatibility)
        """
        # Handle legacy initialization - use settings for all defaults
        if config is None:
            config = ModelConfig()
            try:
                from isa_agent_sdk.core.config import settings
                # Set service URL from settings
                config.service_url = settings.resolved_isa_api_url
                # Set default model from settings (config → model flow)
                config.default_model = settings.ai_model
                config.provider = settings.ai_provider
            except Exception:
                pass
            # Override with explicit isa_url if provided
            if isa_url:
                config.mode = ModelMode.SERVICE
                config.service_url = isa_url

        self.config = config
        self._service_client = None  # AsyncISAModel for service mode
        self._direct_llm = None      # AIFactory LLM for direct mode
        self._active_mode: Optional[ModelMode] = None
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(f"ModelClient initialized with mode={config.mode.value}, provider={config.provider}")

    async def initialize(self) -> bool:
        """
        Initialize the appropriate backend based on mode.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        mode = self.config.mode

        if mode == ModelMode.SERVICE:
            return await self._init_service_mode()
        elif mode == ModelMode.DIRECT:
            return await self._init_direct_mode()
        elif mode == ModelMode.AUTO:
            # Try service first
            if self.config.service_url:
                if await self._init_service_mode():
                    return True

                if self.config.fallback_to_direct:
                    logger.warning("Service mode unavailable, falling back to direct mode")
                    return await self._init_direct_mode()
            else:
                # No service URL, use direct mode
                return await self._init_direct_mode()

        return False

    async def _init_service_mode(self) -> bool:
        """Initialize service mode with AsyncISAModel"""
        try:
            from isa_model.inference_client import AsyncISAModel

            service_url = self.config.service_url
            if not service_url:
                try:
                    from isa_agent_sdk.core.config import settings
                    service_url = settings.resolved_isa_api_url
                except Exception:
                    service_url = os.getenv("ISA_API_URL", "http://localhost:8082")

            self._service_client = AsyncISAModel(base_url=service_url)

            # Health check
            try:
                response = await asyncio.wait_for(
                    self._service_client.chat.completions.create(
                        model="gpt-5-nano",
                        messages=[{"role": "user", "content": "ping"}]
                    ),
                    timeout=5.0
                )
                if response.choices and len(response.choices) > 0:
                    self._active_mode = ModelMode.SERVICE
                    self._initialized = True
                    logger.info(f"ModelClient SERVICE mode initialized: {service_url}")
                    return True
            except Exception as e:
                logger.warning(f"Service health check failed: {e}")
                return False

        except ImportError:
            logger.warning("AsyncISAModel not available (isa_model package not installed)")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize service mode: {e}")
            return False

    async def _init_direct_mode(self) -> bool:
        """Initialize direct mode with AIFactory"""
        try:
            from isa_model.inference.ai_factory import AIFactory

            factory = AIFactory.get_instance()

            # Get LLM service based on provider
            self._direct_llm = factory.get_llm(
                provider=self.config.provider,
                model_name=self.config.default_model,
            )

            self._active_mode = ModelMode.DIRECT
            self._initialized = True
            logger.info(f"ModelClient DIRECT mode initialized: provider={self.config.provider}")
            return True

        except ImportError:
            logger.error("AIFactory not available (isa_model package not installed)")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize direct mode: {e}")
            return False

    @property
    def active_mode(self) -> Optional[ModelMode]:
        """Get currently active mode"""
        return self._active_mode

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for OpenAI-compatible API

        Args:
            tools: Tool definitions (may be in LangGraph or OpenAI format)

        Returns:
            Tools in OpenAI format
        """
        formatted_tools = []
        for tool in tools:
            if 'type' in tool and 'function' in tool:
                # Already in OpenAI format - but ensure parameters has type: "object"
                func_params = tool.get('function', {}).get('parameters', {})
                if isinstance(func_params, dict) and func_params.get('type') != 'object':
                    tool['function']['parameters'] = {
                        'type': 'object',
                        'properties': func_params.get('properties', {}),
                        'required': func_params.get('required', [])
                    }
                formatted_tools.append(tool)
            elif 'name' in tool:
                # Get parameters from inputSchema (MCP format) or parameters (OpenAI format)
                params = tool.get('inputSchema') or tool.get('parameters') or {}

                # Ensure params is a dict (safety check)
                if not isinstance(params, dict):
                    params = {}

                # CRITICAL: OpenAI API requires parameters to have type: "object"
                # MCP tools may not always set this, so we ensure it here
                if params and params.get('type') != 'object':
                    params = {
                        'type': 'object',
                        'properties': params.get('properties', {}),
                        'required': params.get('required', [])
                    }

                # Convert Pydantic 'title' to OpenAI 'description' for parameter fields
                if params and 'properties' in params and params['properties'] is not None:
                    for param_name, param_schema in params['properties'].items():
                        if 'title' in param_schema and 'description' not in param_schema:
                            param_schema['description'] = param_schema['title']

                # Convert from simple format to OpenAI format
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool['name'],
                        "description": tool.get('description', ''),
                        "parameters": params
                    }
                }
                formatted_tools.append(formatted_tool)
            else:
                formatted_tools.append(tool)

        return formatted_tools

    async def call_model(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        response_format: Optional[Dict[str, str]] = None
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """
        Call model with automatic mode selection.

        Args:
            messages: LangChain messages or OpenAI format
            tools: Optional tool definitions
            model: Optional model override
            provider: Optional provider override (for direct mode)
            timeout: Request timeout in seconds
            response_format: Optional response format

        Returns:
            (AIMessage, billing_info)
        """
        # Auto-initialize if needed
        if not self._initialized:
            await self.initialize()

        if self._active_mode == ModelMode.SERVICE:
            return await self._call_model_service(messages, tools, model, provider, timeout, response_format)
        elif self._active_mode == ModelMode.DIRECT:
            return await self._call_model_direct(messages, tools, model, provider, timeout, response_format)
        else:
            raise RuntimeError("ModelClient not initialized. Call initialize() first.")

    async def _call_model_service(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        response_format: Optional[Dict[str, str]] = None
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Call model via isA_Model service (AsyncISAModel)"""
        if not messages:
            raise ValueError("Messages required")

        args = {
            "model": model or self.config.default_model,
            "messages": messages,
            "stream": False
        }

        if provider:
            args["provider"] = provider

        if tools:
            args["tools"] = self._format_tools(tools)

        if response_format:
            args["response_format"] = response_format

        try:
            response = await asyncio.wait_for(
                self._service_client.chat.completions.create(**args),
                timeout=timeout
            )

            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                raw_tool_calls = getattr(message, 'tool_calls', None)

                # Convert ToolCall objects to dicts if needed (for LangChain compatibility)
                tool_calls_list = []
                if raw_tool_calls:
                    for tc in raw_tool_calls:
                        if isinstance(tc, dict):
                            tool_calls_list.append(tc)
                        elif hasattr(tc, 'function'):
                            # OpenAI ToolCall object format
                            tool_calls_list.append({
                                "name": tc.function.name,
                                "args": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                                "id": tc.id,
                                "type": getattr(tc, 'type', 'function')
                            })
                        else:
                            # Try to extract from object attributes
                            tool_calls_list.append({
                                "name": getattr(tc, 'name', 'unknown'),
                                "args": getattr(tc, 'args', {}),
                                "id": getattr(tc, 'id', f"call_{len(tool_calls_list)}"),
                                "type": getattr(tc, 'type', 'function')
                            })

                ai_message = AIMessage(
                    content=message.content or "",
                    tool_calls=tool_calls_list
                )

                billing = {}
                if hasattr(response, 'usage') and response.usage is not None:
                    billing = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

                logger.info(f"[SERVICE] Model call completed: {model or self.config.default_model}")
                return ai_message, billing
            else:
                raise RuntimeError("No choices in response")

        except asyncio.TimeoutError:
            raise RuntimeError(f"Model timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Model error: {str(e)}")

    async def _call_model_direct(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        response_format: Optional[Dict[str, str]] = None
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Call model directly via AIFactory"""
        if not messages:
            raise ValueError("Messages required")

        # Get or create LLM service
        llm = self._direct_llm

        # If provider or model changed, get new service
        if provider and provider != self.config.provider:
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            llm = factory.get_llm(provider=provider, model_name=model or self.config.default_model)
        elif model and model != self.config.default_model:
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            llm = factory.get_llm(provider=self.config.provider, model_name=model)

        # Bind tools if provided
        if tools:
            formatted_tools = self._format_tools(tools)
            llm = llm.bind_tools(formatted_tools)

        try:
            # Call with timeout
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout
            )

            # Handle response - could be string or AIMessage
            if isinstance(response, str):
                ai_message = AIMessage(content=response)
            elif hasattr(response, 'content'):
                ai_message = AIMessage(
                    content=response.content or "",
                    tool_calls=getattr(response, 'tool_calls', []) or []
                )
            else:
                ai_message = AIMessage(content=str(response))

            # Get billing from LLM if available
            billing = {}
            if hasattr(llm, 'last_token_usage'):
                billing = llm.last_token_usage

            logger.info(f"[DIRECT] Model call completed: provider={provider or self.config.provider}")
            return ai_message, billing

        except asyncio.TimeoutError:
            raise RuntimeError(f"Model timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Model error: {str(e)}")

    async def stream_tokens(
        self,
        messages: List[Any],
        token_callback,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        show_reasoning: bool = False
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """
        Stream tokens with callback for real-time processing.

        Args:
            messages: LangChain messages
            token_callback: Callback function for streaming tokens
            tools: Optional tool definitions
            model: Optional model override
            provider: Optional provider override
            timeout: Request timeout in seconds
            show_reasoning: Enable reasoning visibility (for DeepSeek-R1)

        Returns:
            (AIMessage, billing_info)
        """
        if not self._initialized:
            await self.initialize()

        # Auto-detect reasoning models and enable show_reasoning for them
        # This is required for DeepSeek-R1 to capture reasoning_content for tool call flows
        effective_model = model or self.config.default_model
        if not show_reasoning and self._is_reasoning_model(effective_model):
            show_reasoning = True
            logger.debug(f"Auto-enabled show_reasoning for reasoning model: {effective_model}")

        if self._active_mode == ModelMode.SERVICE:
            return await self._stream_tokens_service(messages, token_callback, tools, model, provider, timeout, show_reasoning)
        elif self._active_mode == ModelMode.DIRECT:
            return await self._stream_tokens_direct(messages, token_callback, tools, model, provider, timeout, show_reasoning)
        else:
            raise RuntimeError("ModelClient not initialized")

    def _is_reasoning_model(self, model_name: str) -> bool:
        """
        Check if a model is a reasoning model that requires show_reasoning=True.

        Reasoning models include:
        - DeepSeek-R1 and variants
        - OpenAI o-series (o1, o3, o4)
        - Gemini thinking models
        """
        if not model_name:
            return False
        model_lower = model_name.lower()
        return (
            model_lower.startswith("deepseek-r1") or
            "deepseek-reasoner" in model_lower or
            "r1" in model_lower or
            model_lower.startswith("o4-") or
            model_lower.startswith("o3-") or
            model_lower.startswith("o1-") or
            ":thinking" in model_lower
        )

    async def _stream_tokens_service(
        self,
        messages: List[Any],
        token_callback,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        show_reasoning: bool = False
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Stream tokens via isA_Model service"""
        if not messages:
            raise ValueError("Messages required for streaming")

        args = {
            "model": model or self.config.default_model,
            "messages": messages,
            "stream": True
        }

        if provider:
            args["provider"] = provider

        if tools:
            args["tools"] = self._format_tools(tools)

        if show_reasoning:
            args["show_reasoning"] = True

        try:
            stream = await asyncio.wait_for(
                self._service_client.chat.completions.create(**args),
                timeout=timeout
            )


            # If the response is not an async iterator but a direct response, handle it
            if hasattr(stream, 'choices') and not hasattr(stream, '__aiter__'):
                # This is a non-streaming response (direct ChatCompletion)
                print(f"[STREAM_EARLY_RETURN] Got direct response instead of stream, has_choices={hasattr(stream, 'choices')}, has_aiter={hasattr(stream, '__aiter__')}", flush=True)
                if stream.choices and len(stream.choices) > 0:
                    message = stream.choices[0].message
                    content = getattr(message, 'content', '') or ''
                    raw_tool_calls = getattr(message, 'tool_calls', None)
                    print(f"[STREAM_EARLY_RETURN] content_len={len(content)}, raw_tool_calls={raw_tool_calls is not None}, tool_calls_count={len(raw_tool_calls) if raw_tool_calls else 0}", flush=True)

                    tool_calls_list = []
                    if raw_tool_calls:
                        for tc in raw_tool_calls:
                            print(f"[STREAM_EARLY_RETURN] Processing tool_call: type={type(tc).__name__}, has_function={hasattr(tc, 'function')}", flush=True)
                            if hasattr(tc, 'function'):
                                tool_calls_list.append({
                                    "name": tc.function.name,
                                    "args": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                                    "id": tc.id,
                                    "type": getattr(tc, 'type', 'function')
                                })
                        print(f"[STREAM_EARLY_RETURN] Final tool_calls_list: {[tc['name'] for tc in tool_calls_list]}", flush=True)

                    if token_callback and content:
                        token_callback({"token": content})
                    return AIMessage(content=content, tool_calls=tool_calls_list), None

            return await self._process_stream(stream, token_callback)

        except asyncio.TimeoutError:
            raise RuntimeError(f"Streaming timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Streaming error: {str(e)}")

    async def _stream_tokens_direct(
        self,
        messages: List[Any],
        token_callback,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        show_reasoning: bool = False
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Stream tokens directly via AIFactory"""
        if not messages:
            raise ValueError("Messages required for streaming")

        # Get or create LLM service
        llm = self._direct_llm

        effective_provider = provider or self.config.provider
        effective_model = model or self.config.default_model

        if provider and provider != self.config.provider:
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            llm = factory.get_llm(provider=provider, model_name=effective_model)
        elif model and model != self.config.default_model:
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            llm = factory.get_llm(provider=effective_provider, model_name=model)

        print(f"[DIRECT_STREAM] Starting direct streaming | provider={effective_provider} | model={effective_model} | tools_count={len(tools) if tools else 0}", flush=True)

        if tools:
            formatted_tools = self._format_tools(tools)
            llm = llm.bind_tools(formatted_tools)
            print(f"[DIRECT_STREAM] Tools bound: {[t['function']['name'] for t in formatted_tools]}", flush=True)

        try:
            content_chunks = []
            tool_call_chunks = {}  # Accumulate tool call chunks by index
            chunk_count = 0

            async for chunk in llm.astream(messages):
                chunk_count += 1

                if chunk_count <= 3:
                    print(f"[DIRECT_CHUNK #{chunk_count}] type={type(chunk).__name__}", flush=True)

                if isinstance(chunk, str):
                    content_chunks.append(chunk)
                    if token_callback:
                        token_callback({"token": chunk})
                elif hasattr(chunk, 'content') and chunk.content:
                    content_chunks.append(chunk.content)
                    if token_callback:
                        token_callback({"token": chunk.content})

                # Handle tool calls - accumulate partial chunks
                if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                    # LangChain streaming format uses tool_call_chunks
                    for tc_chunk in chunk.tool_call_chunks:
                        idx = tc_chunk.get('index', 0)
                        if idx not in tool_call_chunks:
                            tool_call_chunks[idx] = {'name': '', 'args': '', 'id': '', 'type': 'function'}
                        if tc_chunk.get('name'):
                            tool_call_chunks[idx]['name'] = tc_chunk['name']
                        if tc_chunk.get('args'):
                            tool_call_chunks[idx]['args'] += tc_chunk['args']
                        if tc_chunk.get('id'):
                            tool_call_chunks[idx]['id'] = tc_chunk['id']
                    if chunk_count <= 5:
                        print(f"[DIRECT_CHUNK #{chunk_count}] tool_call_chunks detected, accumulated: {list(tool_call_chunks.keys())}", flush=True)
                elif hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    # Direct tool_calls format (complete tool calls)
                    for tc in chunk.tool_calls:
                        if isinstance(tc, dict):
                            idx = len(tool_call_chunks)
                            tool_call_chunks[idx] = tc
                        else:
                            # Convert object to dict
                            idx = len(tool_call_chunks)
                            tool_call_chunks[idx] = {
                                'name': getattr(tc, 'name', ''),
                                'args': getattr(tc, 'args', {}),
                                'id': getattr(tc, 'id', f'call_{idx}'),
                                'type': 'function'
                            }
                    if chunk_count <= 5:
                        print(f"[DIRECT_CHUNK #{chunk_count}] tool_calls detected: {[tc.get('name', '') for tc in tool_call_chunks.values()]}", flush=True)

            # Finalize tool calls - parse args if string
            final_tool_calls = []
            for idx in sorted(tool_call_chunks.keys()):
                tc = tool_call_chunks[idx]
                if tc.get('name'):  # Only include if we have a name
                    args = tc.get('args', {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except json.JSONDecodeError:
                            args = {}
                    final_tool_calls.append({
                        'name': tc['name'],
                        'args': args,
                        'id': tc.get('id', f'call_{idx}'),
                        'type': 'function'
                    })

            final_content = ''.join(content_chunks)
            print(f"[DIRECT_STREAM_COMPLETE] chunk_count={chunk_count} | content_len={len(final_content)} | tool_calls={len(final_tool_calls)}", flush=True)
            if final_tool_calls:
                print(f"[DIRECT_STREAM_COMPLETE] tool_calls: {[tc['name'] for tc in final_tool_calls]}", flush=True)

            ai_message = AIMessage(content=final_content, tool_calls=final_tool_calls)

            return ai_message, None

        except asyncio.TimeoutError:
            raise RuntimeError(f"Streaming timeout after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Streaming error: {str(e)}")

    async def _process_stream(self, stream, callback) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Process OpenAI-compatible stream with callbacks"""
        content_chunks = []
        reasoning_chunks = []  # Capture DeepSeek-R1 reasoning content
        tool_calls = []
        chunk_count = 0

        # Debug: Check stream type
        print(f"[STREAM_DEBUG] stream_type={type(stream).__name__}, has_aiter={hasattr(stream, '__aiter__')}", flush=True)

        try:
            async for chunk in stream:
                chunk_count += 1

                # Debug every chunk to understand the pattern
                if chunk_count <= 3:
                    print(f"[CHUNK_DEBUG #{chunk_count}] type: {type(chunk).__name__}", flush=True)
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        delta = getattr(choice, 'delta', None)
                        delta_content = getattr(delta, 'content', None) if delta else None
                        print(f"[CHUNK_DEBUG #{chunk_count}] delta.content: '{delta_content}'", flush=True)

                # Handle string chunks (from isA_Model service SSE)
                if isinstance(chunk, str):
                    # Check for reasoning content: [思考: token] format from DeepSeek-R1
                    if chunk.startswith('[思考:') and chunk.endswith(']'):
                        # Extract reasoning content from [思考: token] format
                        # [思考: = 4 chars, but we need to skip the space after colon too
                        reasoning_token = chunk[5:-1]  # Remove [思考:  (5 chars including space) and ]
                        reasoning_chunks.append(reasoning_token)
                        if chunk_count <= 3:
                            print(f"[REASONING_CAPTURE #{chunk_count}] captured: '{reasoning_token}'", flush=True)
                        if callback:
                            callback({"token": reasoning_token, "type": "reasoning"})
                    else:
                        # Regular content token
                        content_chunks.append(chunk)
                        if callback:
                            callback({"token": chunk})
                    continue

                # Handle dict chunks (from various API formats)
                if isinstance(chunk, dict):
                    # Format 1: Dict with 'result' key (DeepSeek-R1 and some proxied APIs)
                    if 'result' in chunk:
                        result = chunk.get('result', {})
                        if 'choices' in result and len(result['choices']) > 0:
                            choice = result['choices'][0]
                            message = choice.get('message') or choice.get('delta', {})
                            if message.get('content'):
                                content_chunks.append(message['content'])
                                if callback:
                                    callback({"token": message['content']})
                            if 'tool_calls' in message and message['tool_calls']:
                                for tc in message['tool_calls']:
                                    tool_call_dict = {
                                        "name": tc['function']['name'],
                                        "args": json.loads(tc['function']['arguments']) if isinstance(tc['function']['arguments'], str) else tc['function']['arguments'],
                                        "id": tc.get('id', f"call_{len(tool_calls)}"),
                                        "type": tc.get('type', 'function')
                                    }
                                    tool_calls.append(tool_call_dict)
                        continue

                    # Format 2: Dict with 'choices' directly
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        delta = choice.get('delta', {})
                        if delta.get('content'):
                            content_chunks.append(delta['content'])
                            if callback:
                                callback({"token": delta['content']})
                        # Also check for tool_calls in delta (streaming tool call format)
                        if delta.get('tool_calls'):
                            for tc in delta['tool_calls']:
                                if isinstance(tc, dict):
                                    # OpenAI streaming format: tool_calls come in chunks with function info
                                    func = tc.get('function', {})
                                    tool_call_dict = {
                                        "name": func.get('name', ''),
                                        "args": json.loads(func.get('arguments', '{}')) if isinstance(func.get('arguments'), str) and func.get('arguments') else func.get('arguments', {}),
                                        "id": tc.get('id', f"call_{len(tool_calls)}"),
                                        "type": tc.get('type', 'function')
                                    }
                                    # Only add if we have a name (skip partial chunks)
                                    if tool_call_dict['name']:
                                        tool_calls.append(tool_call_dict)
                        continue

                    # Format 3: Dict with 'tool_calls' directly (from isA_Model service)
                    if 'tool_calls' in chunk and chunk['tool_calls']:
                        for tc in chunk['tool_calls']:
                            if isinstance(tc, dict):
                                # Handle OpenAI format tool_call
                                if 'function' in tc:
                                    tool_call_dict = {
                                        "name": tc['function']['name'],
                                        "args": json.loads(tc['function']['arguments']) if isinstance(tc['function']['arguments'], str) else tc['function']['arguments'],
                                        "id": tc.get('id', f"call_{len(tool_calls)}"),
                                        "type": tc.get('type', 'function')
                                    }
                                else:
                                    # LangChain format
                                    tool_call_dict = {
                                        "name": tc.get('name', ''),
                                        "args": tc.get('args', {}),
                                        "id": tc.get('id', f"call_{len(tool_calls)}"),
                                        "type": tc.get('type', 'function')
                                    }
                                tool_calls.append(tool_call_dict)
                        continue

                    # Format 4: Other dict with content (fallback)
                    if 'content' in chunk and chunk['content']:
                        content_chunks.append(chunk['content'])
                        if callback:
                            callback({"token": chunk['content']})
                        continue

                # Handle OpenAI-compatible streaming objects (ChatCompletionChunk or ChatCompletion)
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = getattr(choice, 'delta', None)
                    message = getattr(choice, 'message', None)

                    # Check delta.content first (streaming format), then message.content (non-streaming)
                    content = None
                    if delta and hasattr(delta, 'content') and delta.content:
                        content = delta.content
                    elif message and hasattr(message, 'content') and message.content:
                        content = message.content

                    if content:
                        # Check if content is reasoning in [思考: token] format (from isA_Model service)
                        if isinstance(content, str) and content.startswith('[思考:') and content.endswith(']'):
                            # Extract reasoning token and add to reasoning_chunks instead of content
                            reasoning_token = content[4:-1]  # Remove [思考: (4 chars) and ] (1 char)
                            if reasoning_token.startswith(' '):
                                reasoning_token = reasoning_token[1:]  # Remove leading space after colon
                            reasoning_chunks.append(reasoning_token)
                            if callback:
                                callback({"token": reasoning_token, "type": "reasoning"})
                        else:
                            # Regular content
                            content_chunks.append(content)
                            if callback:
                                callback({"token": content})

                    # Capture DeepSeek-R1 reasoning_content (required for tool call conversations)
                    source = delta or message
                    if source:
                        # Check model_extra for reasoning_content (DeepSeek-R1 format)
                        if hasattr(source, 'model_extra') and source.model_extra:
                            reasoning = source.model_extra.get('reasoning_content')
                            if reasoning:
                                reasoning_chunks.append(reasoning)
                                if callback:
                                    callback({"token": reasoning, "type": "reasoning"})

                    # Check for tool calls in delta or message
                    if source:
                        source_tool_calls = getattr(source, 'tool_calls', None)
                        if source_tool_calls:
                            for tc in source_tool_calls:
                                if hasattr(tc, 'function'):
                                    tool_call_dict = {
                                        "name": tc.function.name,
                                        "args": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                                        "id": tc.id,
                                        "type": getattr(tc, 'type', 'function')
                                    }
                                    tool_calls.append(tool_call_dict)

        except Exception as e:
            logger.error(f"Stream iteration error: {type(e).__name__}: {e}")
            import traceback
            print(f"[STREAM_ERROR] {type(e).__name__}: {e}", flush=True)
            print(f"[STREAM_ERROR] Traceback: {traceback.format_exc()}", flush=True)

        final_content = ''.join(content_chunks)
        final_reasoning = ''.join(reasoning_chunks)

        print(f"[STREAM_COMPLETE] chunk_count={chunk_count}, content_chunks={len(content_chunks)}, final_content_len={len(final_content)}, tool_calls_count={len(tool_calls)}, reasoning_len={len(final_reasoning)}", flush=True)
        if final_content:
            print(f"[STREAM_COMPLETE] first_100_chars: '{final_content[:100]}'", flush=True)
        if tool_calls:
            print(f"[STREAM_COMPLETE] tool_calls: {[tc.get('name') for tc in tool_calls]}", flush=True)

        # Build additional_kwargs with reasoning_content if present (required for DeepSeek-R1 tool calls)
        additional_kwargs = {}
        if final_reasoning:
            additional_kwargs["reasoning_content"] = final_reasoning
            print(f"[REASONING_STORED] Storing reasoning_content in AIMessage, length={len(final_reasoning)}", flush=True)
            print(f"[REASONING_STORED] First 200 chars: '{final_reasoning[:200]}'", flush=True)
        else:
            print(f"[REASONING_STORED] No reasoning_content captured! reasoning_chunks={len(reasoning_chunks)}", flush=True)

        ai_message = AIMessage(
            content=final_content,
            tool_calls=tool_calls if tool_calls else [],
            additional_kwargs=additional_kwargs
        )

        print(f"[AIMESSAGE_CREATED] content_len={len(final_content)}, tool_calls={len(tool_calls)}, has_reasoning={bool(final_reasoning)}", flush=True)

        return ai_message, None

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for text"""
        if not texts:
            raise ValueError("Texts required for embeddings")

        if not self._initialized:
            await self.initialize()

        if self._active_mode == ModelMode.SERVICE:
            embedding = await self._service_client.embeddings.create(
                input=texts if len(texts) > 1 else texts[0],
                model=model or "text-embedding-3-small"
            )
            return [item.embedding for item in embedding.data]
        else:
            # Direct mode - use AIFactory
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            embed_service = factory.get_embed(model_name=model or "text-embedding-3-small")
            result = await embed_service.invoke(input_data=texts)
            return result if isinstance(result, list) else [result]

    async def health_check(self) -> bool:
        """Quick health check for service availability"""
        try:
            if not self._initialized:
                await self.initialize()

            if self._active_mode == ModelMode.SERVICE:
                response = await asyncio.wait_for(
                    self._service_client.chat.completions.create(
                        model="gpt-5-nano",
                        messages=[{"role": "user", "content": "ping"}]
                    ),
                    timeout=5.0
                )
                return response.choices and len(response.choices) > 0
            else:
                # Direct mode - just check if LLM is available
                return self._direct_llm is not None
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        if self._service_client:
            try:
                await self._service_client.close()
            except Exception as e:
                logger.warning(f"Error closing service client: {e}")
            self._service_client = None

        self._direct_llm = None
        self._initialized = False
        self._active_mode = None
        logger.info("ModelClient closed")


# Global singleton for performance (backward compatible)
_model_client: Optional[ModelClient] = None
_client_lock = asyncio.Lock()


async def get_model_client(
    isa_url: str = None,
    config: ModelConfig = None
) -> ModelClient:
    """
    Get thread-safe singleton model client.

    Args:
        isa_url: Legacy parameter for service URL
        config: ModelConfig for full configuration

    Returns:
        Initialized ModelClient

    Raises:
        RuntimeError: If model client initialization fails
    """
    global _model_client

    if _model_client is None:
        async with _client_lock:
            if _model_client is None:
                logger.info(f"Creating ModelClient singleton | isa_url={isa_url} | has_config={config is not None}")
                if config:
                    _model_client = ModelClient(config=config)
                else:
                    _model_client = ModelClient(isa_url=isa_url)

                success = await _model_client.initialize()

                if not success or _model_client.active_mode is None:
                    # Reset singleton so it can be retried
                    failed_client = _model_client
                    _model_client = None
                    raise RuntimeError(
                        f"ModelClient initialization failed. "
                        f"Mode attempted: {failed_client.config.mode.value}, "
                        f"Service URL: {failed_client.config.service_url}. "
                        f"Ensure isA_Model service is running at the configured URL."
                    )

                logger.info(f"Global ModelClient initialized: mode={_model_client.active_mode}")

    return _model_client


# Convenience function for direct mode
async def get_direct_model_client(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = None
) -> ModelClient:
    """
    Get a model client in direct mode (no service needed).

    Args:
        provider: LLM provider (openai, ollama, cerebras, yyds)
        model: Default model name
        api_key: API key for the provider

    Returns:
        ModelClient in direct mode
    """
    config = ModelConfig(
        mode=ModelMode.DIRECT,
        provider=provider,
        default_model=model,
        api_key=api_key,
    )
    client = ModelClient(config=config)
    await client.initialize()
    return client
