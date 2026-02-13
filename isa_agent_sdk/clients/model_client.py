#!/usr/bin/env python3
"""
Model Client - Dual-mode model abstraction (Service/Direct)

THIN WRAPPER that delegates to:
- SERVICE mode: AsyncISAModel (for isA_Model service)
- DIRECT mode: AIFactory (for direct API access)

All format conversion is handled by llm_adapter.py utilities:
- ResponseConverter: Converts any response format to AIMessage
- StreamingChunkProcessor: Processes streaming chunks

Core focus: Performance, Concurrency, Security, Stability, Flexibility
"""

import logging
import asyncio
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import AIMessage

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
    Thin wrapper dual-mode model client.

    Delegates to:
    - AsyncISAModel for SERVICE mode
    - AIFactory for DIRECT mode

    Uses llm_adapter.py utilities for all format conversion.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        isa_url: Optional[str] = None,  # Legacy parameter for backward compatibility
    ):
        # Handle legacy initialization
        if config is None:
            config = ModelConfig()
            try:
                from isa_agent_sdk.core.config import settings
                config.service_url = settings.resolved_isa_api_url
                config.default_model = settings.ai_model
                config.provider = settings.ai_provider
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not load settings: {e}")
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
        """Initialize the appropriate backend based on mode."""
        if self._initialized:
            return True

        mode = self.config.mode

        if mode == ModelMode.SERVICE:
            return await self._init_service_mode()
        elif mode == ModelMode.DIRECT:
            return await self._init_direct_mode()
        elif mode == ModelMode.AUTO:
            if self.config.service_url:
                if await self._init_service_mode():
                    return True
                if self.config.fallback_to_direct:
                    logger.warning("Service mode unavailable, falling back to direct mode")
                    return await self._init_direct_mode()
            else:
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
                except (ImportError, AttributeError):
                    service_url = os.getenv("ISA_API_URL", "http://localhost:8082")

            self._service_client = AsyncISAModel(base_url=service_url)

            # Health check (allow a bit longer; some providers are slow on cold start)
            try:
                response = await asyncio.wait_for(
                    self._service_client.chat.completions.create(
                        model="gpt-5-nano",
                        messages=[{"role": "user", "content": "ping"}]
                    ),
                    timeout=10.0
                )
                if response.choices and len(response.choices) > 0:
                    self._active_mode = ModelMode.SERVICE
                    self._initialized = True
                    logger.info(f"ModelClient SERVICE mode initialized: {service_url}")
                    return True
            except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                logger.warning(f"Service health check failed: {e}")
                return False

        except ImportError:
            logger.warning("AsyncISAModel not available (isa_model package not installed)")
            return False
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Failed to initialize service mode: {e}")
            return False

    async def _init_direct_mode(self) -> bool:
        """Initialize direct mode with AIFactory"""
        try:
            from isa_model.inference.ai_factory import AIFactory

            factory = AIFactory.get_instance()
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
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to initialize direct mode: {e}")
            return False

    @property
    def active_mode(self) -> Optional[ModelMode]:
        """Get currently active mode"""
        return self._active_mode

    async def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools using AdapterManager (single source of truth)."""
        try:
            from isa_model.inference.services.llm.helpers.llm_adapter import AdapterManager

            manager = AdapterManager()
            schemas, _ = await manager.convert_tools_to_schemas(tools)
            return schemas
        except (ImportError, AttributeError, KeyError, TypeError) as e:
            logger.warning(f"Tool format conversion failed: {e}, using fallback")
            # Fallback: manually add type: "function" wrapper if missing
            return self._format_tools_fallback(tools)

    def _format_tools_fallback(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback tool formatting when AdapterManager unavailable."""
        formatted = []
        for tool in tools:
            if isinstance(tool, dict):
                # Already has type: function wrapper
                if tool.get("type") == "function":
                    formatted.append(tool)
                # MCP format with inputSchema - convert to OpenAI format
                elif "inputSchema" in tool or "name" in tool:
                    formatted.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema", tool.get("parameters", {"type": "object", "properties": {}}))
                        }
                    })
                else:
                    formatted.append(tool)
            else:
                formatted.append(tool)
        return formatted

    async def call_model(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 120.0,
        response_format: Optional[Dict[str, str]] = None
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Call model with automatic mode selection."""
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
        """Call model via isA_Model service - uses ResponseConverter for response handling"""
        if not messages:
            raise ValueError("Messages required")

        # Import converter
        try:
            from isa_model.inference.services.llm.helpers.llm_adapter import ResponseConverter
        except ImportError:
            ResponseConverter = None

        requested_model = model or self.config.default_model
        # Route DeepSeek models through the yyds provider by default
        if provider is None and requested_model and "deepseek" in requested_model.lower():
            provider = "yyds"

        args = {
            "model": requested_model,
            "messages": messages,
            "stream": False
        }
        if provider:
            args["provider"] = provider
        if tools:
            args["tools"] = await self._format_tools(tools)
        if response_format:
            args["response_format"] = response_format

        try:
            response = await asyncio.wait_for(
                self._service_client.chat.completions.create(**args),
                timeout=timeout
            )

            # Use ResponseConverter when response is not a ChatCompletion
            if ResponseConverter and not hasattr(response, "choices"):
                ai_message = ResponseConverter.to_aimessage(response)
            else:
                ai_message = self._convert_response_to_aimessage(response)

            billing = {}
            if hasattr(response, 'usage') and response.usage:
                billing = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            logger.info(f"[SERVICE] Model call completed: {model or self.config.default_model}")
            return ai_message, billing

        except asyncio.TimeoutError:
            raise RuntimeError(f"Model timeout after {timeout}s")
        except (ConnectionError, OSError, ValueError) as e:
            # Fallback: if DeepSeek is unavailable, retry with GPT-5-nano
            requested_model = (model or self.config.default_model or "").lower()
            err = str(e)
            if "deepseek" in requested_model and (
                "model_not_found" in err
                or "does not exist" in err
                or "not found" in err
            ):
                fallback_args = dict(args)
                fallback_args["model"] = "gpt-5-nano"
                fallback_args["provider"] = "openai"
                try:
                    response = await asyncio.wait_for(
                        self._service_client.chat.completions.create(**fallback_args),
                        timeout=timeout
                    )
                    if ResponseConverter and not hasattr(response, "choices"):
                        ai_message = ResponseConverter.to_aimessage(response)
                    else:
                        ai_message = self._convert_response_to_aimessage(response)
                    billing = {}
                    if hasattr(response, 'usage') and response.usage:
                        billing = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    logger.warning("[SERVICE] DeepSeek unavailable, fell back to gpt-5-nano")
                    return ai_message, billing
                except (ConnectionError, OSError, ValueError, asyncio.TimeoutError) as fallback_error:
                    raise RuntimeError(f"Model error: {err}; fallback error: {fallback_error}")
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
        """Call model directly via AIFactory - uses ResponseConverter for response handling"""
        if not messages:
            raise ValueError("Messages required")

        # Import converter
        try:
            from isa_model.inference.services.llm.helpers.llm_adapter import ResponseConverter
        except ImportError:
            ResponseConverter = None

        # Get or create LLM service
        llm = self._direct_llm

        if provider and provider != self.config.provider:
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            llm = factory.get_llm(provider=provider, model_name=model or self.config.default_model)
        elif model and model != self.config.default_model:
            from isa_model.inference.ai_factory import AIFactory
            factory = AIFactory.get_instance()
            llm = factory.get_llm(provider=self.config.provider, model_name=model)

        if tools:
            formatted_tools = await self._format_tools(tools)
            llm = llm.bind_tools(formatted_tools)

        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)

            # Use ResponseConverter if available
            if ResponseConverter:
                ai_message = ResponseConverter.to_aimessage(response)
            else:
                ai_message = self._convert_response_to_aimessage(response)

            billing = {}
            if hasattr(llm, 'last_token_usage'):
                billing = llm.last_token_usage

            logger.info(f"[DIRECT] Model call completed: provider={provider or self.config.provider}")
            return ai_message, billing

        except asyncio.TimeoutError:
            raise RuntimeError(f"Model timeout after {timeout}s")
        except (ConnectionError, OSError, ValueError) as e:
            raise RuntimeError(f"Model error: {str(e)}")

    def _convert_response_to_aimessage(self, response: Any) -> AIMessage:
        """Fallback response conversion when ResponseConverter not available"""
        import json

        if isinstance(response, AIMessage):
            return response

        if isinstance(response, str):
            return AIMessage(content=response)

        # Dict response (OpenAI format)
        if isinstance(response, dict):
            content = ""
            tool_calls_list = []
            additional_kwargs = {}

            choices = response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '') or ''

                if message.get('reasoning_content'):
                    additional_kwargs['reasoning_content'] = message['reasoning_content']

                for tc in message.get('tool_calls', []):
                    func = tc.get('function', {})
                    args = func.get('arguments', '{}')
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls_list.append({
                        "name": func.get('name', ''),
                        "args": args,
                        "id": tc.get('id', f"call_{len(tool_calls_list)}"),
                        "type": 'function'
                    })

            return AIMessage(content=content, tool_calls=tool_calls_list, additional_kwargs=additional_kwargs)

        # ChatCompletion object (has .choices and .content property)
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            tool_calls_list = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                import json
                for tc in message.tool_calls:
                    if hasattr(tc, 'function'):
                        args = tc.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        tool_calls_list.append({
                            "name": tc.function.name,
                            "args": args,
                            "id": tc.id,
                            "type": 'function'
                        })
            return AIMessage(content=message.content or "", tool_calls=tool_calls_list)

        # Object with content attribute (AIMessage-like)
        if hasattr(response, 'content'):
            additional_kwargs = {}
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                additional_kwargs = response.additional_kwargs
            return AIMessage(
                content=response.content or "",
                tool_calls=getattr(response, 'tool_calls', []) or [],
                additional_kwargs=additional_kwargs
            )

        return AIMessage(content=str(response))

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
        """Stream tokens with callback for real-time processing."""
        if not self._initialized:
            await self.initialize()

        # Auto-detect reasoning models
        effective_model = model or self.config.default_model
        if not show_reasoning and self._is_reasoning_model(effective_model):
            show_reasoning = True

        if self._active_mode == ModelMode.SERVICE:
            response, meta = await self._stream_tokens_service(messages, token_callback, tools, model, provider, timeout, show_reasoning)
        elif self._active_mode == ModelMode.DIRECT:
            response, meta = await self._stream_tokens_direct(messages, token_callback, tools, model, provider, timeout, show_reasoning)
        else:
            raise RuntimeError("ModelClient not initialized")

        # DeepSeek-R1 compatibility: ensure reasoning_content is present for tool calls
        self._ensure_reasoning_content(response, effective_model)
        return response, meta

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if a model requires show_reasoning=True."""
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
        """Stream tokens via isA_Model service - uses StreamingChunkProcessor"""
        if not messages:
            raise ValueError("Messages required for streaming")

        # Import processor
        try:
            from isa_model.inference.services.llm.helpers.llm_adapter import (
                StreamingChunkProcessor, ResponseConverter
            )
            use_processor = True
        except ImportError:
            use_processor = False

        args = {
            "model": model or self.config.default_model,
            "messages": messages,
            "stream": True,
            # Hint AsyncChatCompletions to yield StreamEvent objects locally (not sent to server)
            "return_stream_events": True
        }
        if provider:
            args["provider"] = provider
        if tools:
            args["tools"] = await self._format_tools(tools)
        if show_reasoning:
            args["show_reasoning"] = True

        try:
            stream = await asyncio.wait_for(
                self._service_client.chat.completions.create(**args),
                timeout=timeout
            )

            # Handle non-streaming response
            if hasattr(stream, 'choices') and not hasattr(stream, '__aiter__'):
                if use_processor:
                    ai_message = ResponseConverter.to_aimessage(stream)
                else:
                    ai_message = self._convert_response_to_aimessage(stream)
                if token_callback and ai_message.content:
                    token_callback({"token": ai_message.content})
                return ai_message, None

            # Process stream
            if use_processor:
                processor = StreamingChunkProcessor()
                async for chunk in stream:
                    token = processor.process_chunk(chunk)
                    if token and token_callback:
                        token_callback({"token": token})
                return processor.get_final_response(), None
            else:
                return await self._process_stream_fallback(stream, token_callback)

        except asyncio.TimeoutError:
            raise RuntimeError(f"Streaming timeout after {timeout}s")
        except (ConnectionError, OSError, ValueError) as e:
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
        """Stream tokens directly via AIFactory - uses StreamingChunkProcessor"""
        if not messages:
            raise ValueError("Messages required for streaming")

        # Import processor
        try:
            from isa_model.inference.services.llm.helpers.llm_adapter import StreamingChunkProcessor
            use_processor = True
        except ImportError:
            use_processor = False

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

        if tools:
            formatted_tools = await self._format_tools(tools)
            llm = llm.bind_tools(formatted_tools)

        try:
            if use_processor:
                processor = StreamingChunkProcessor()
                async for chunk in llm.astream(messages):
                    token = processor.process_chunk(chunk)
                    if token and token_callback:
                        token_callback({"token": token})
                return processor.get_final_response(), None
            else:
                return await self._process_stream_direct_fallback(llm, messages, token_callback)

        except asyncio.TimeoutError:
            raise RuntimeError(f"Streaming timeout after {timeout}s")
        except (ConnectionError, OSError, ValueError) as e:
            raise RuntimeError(f"Streaming error: {str(e)}")

    async def _process_stream_fallback(self, stream, callback) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Fallback stream processing when StreamingChunkProcessor not available"""
        import json

        content_chunks = []
        tool_calls = []
        reasoning_content = None  # DeepSeek-R1 compatibility

        async for chunk in stream:
            if isinstance(chunk, str):
                content_chunks.append(chunk)
                if callback:
                    callback({"token": chunk})
            elif hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, 'delta', None) or getattr(choice, 'message', None)
                if delta:
                    content = getattr(delta, 'content', None)
                    if content:
                        content_chunks.append(content)
                        if callback:
                            callback({"token": content})
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tc in delta.tool_calls:
                            if hasattr(tc, 'function'):
                                args = tc.function.arguments
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        args = {}
                                tool_calls.append({
                                    "name": tc.function.name,
                                    "args": args,
                                    "id": tc.id,
                                    "type": 'function'
                                })
                # Extract reasoning_content if present (DeepSeek-R1 compatibility)
                if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                    reasoning_content = chunk.reasoning_content

        # Build additional_kwargs with reasoning_content if present
        additional_kwargs = {}
        if reasoning_content:
            additional_kwargs['reasoning_content'] = reasoning_content
        elif tool_calls:
            # DeepSeek-R1 requires reasoning_content when tool_calls exist
            additional_kwargs['reasoning_content'] = ""

        return AIMessage(content=''.join(content_chunks), tool_calls=tool_calls, additional_kwargs=additional_kwargs), None

    async def _process_stream_direct_fallback(self, llm, messages, callback) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """Fallback direct streaming when StreamingChunkProcessor not available"""
        import json

        content_chunks = []
        tool_calls = []
        reasoning_content = None

        async for chunk in llm.astream(messages):
            if isinstance(chunk, str):
                content_chunks.append(chunk)
                if callback:
                    callback({"token": chunk})
            elif hasattr(chunk, 'content') and chunk.content:
                content_chunks.append(chunk.content)
                if callback:
                    callback({"token": chunk.content})

            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tc in chunk.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
                    else:
                        tool_calls.append({
                            'name': getattr(tc, 'name', ''),
                            'args': getattr(tc, 'args', {}),
                            'id': getattr(tc, 'id', f'call_{len(tool_calls)}'),
                            'type': 'function'
                        })
            if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                reasoning_content = chunk.reasoning_content

        additional_kwargs = {}
        if reasoning_content:
            additional_kwargs['reasoning_content'] = reasoning_content
        elif tool_calls:
            additional_kwargs['reasoning_content'] = ""

        return AIMessage(content=''.join(content_chunks), tool_calls=tool_calls, additional_kwargs=additional_kwargs), None

    def _ensure_reasoning_content(self, message: AIMessage, model_name: Optional[str]) -> None:
        """Ensure reasoning_content exists for tool-call messages on reasoning models."""
        if not message or not hasattr(message, "tool_calls"):
            return
        if not message.tool_calls:
            return
        if not self._is_reasoning_model(model_name or ""):
            return
        additional_kwargs = message.additional_kwargs or {}
        if "reasoning_content" not in additional_kwargs:
            additional_kwargs["reasoning_content"] = ""
            message.additional_kwargs = additional_kwargs

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
                return self._direct_llm is not None
        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        if self._service_client:
            try:
                await self._service_client.close()
            except (ConnectionError, OSError, AttributeError) as e:
                logger.warning(f"Error closing service client: {e}")
            self._service_client = None

        self._direct_llm = None
        self._initialized = False
        self._active_mode = None
        logger.info("ModelClient closed")


# Global singleton for performance (backward compatible)
_model_client: Optional[ModelClient] = None
_client_lock: Optional[asyncio.Lock] = None


def _get_client_lock() -> asyncio.Lock:
    """Lazily create the asyncio lock on first use (requires running event loop)."""
    global _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    return _client_lock


async def get_model_client(
    isa_url: str = None,
    config: ModelConfig = None
) -> ModelClient:
    """Get thread-safe singleton model client."""
    global _model_client

    if _model_client is None:
        async with _get_client_lock():
            if _model_client is None:
                logger.info(f"Creating ModelClient singleton | isa_url={isa_url} | has_config={config is not None}")
                if config:
                    _model_client = ModelClient(config=config)
                else:
                    _model_client = ModelClient(isa_url=isa_url)

                success = await _model_client.initialize()

                if not success or _model_client.active_mode is None:
                    failed_client = _model_client
                    _model_client = None
                    raise RuntimeError(
                        f"ModelClient initialization failed. "
                        f"Mode attempted: {failed_client.config.mode.value}, "
                        f"Service URL: {failed_client.config.service_url}."
                    )

                logger.info(f"Global ModelClient initialized: mode={_model_client.active_mode}")

    return _model_client


async def get_direct_model_client(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = None
) -> ModelClient:
    """Get a model client in direct mode (no service needed)."""
    config = ModelConfig(
        mode=ModelMode.DIRECT,
        provider=provider,
        default_model=model,
        api_key=api_key,
    )
    client = ModelClient(config=config)
    await client.initialize()
    return client
