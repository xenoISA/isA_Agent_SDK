#!/usr/bin/env python3
"""
Model Client - High-performance ISA model abstraction using AsyncISAModel

Core focus: Performance, Concurrency, Security, Stability
- OpenAI-compatible interface (AsyncISAModel)
- Connection pooling and reuse
- Async/concurrent processing
- Error handling and timeouts
- Format negotiation (LangChain, OpenAI dict, string)
- Minimal overhead design

Migrated from model_service.py to use AsyncISAModel client directly.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import async model client (OpenAI-compatible interface)
from isa_model.inference_client import AsyncISAModel

logger = logging.getLogger(__name__)


class ModelClient:
    """
    High-performance model client with OpenAI-compatible interface

    Uses AsyncISAModel for direct access to ISA Model API with format negotiation.
    Supports LangChain messages, OpenAI dict format, and string prompts.
    """

    def __init__(self, isa_url: str = None):
        """
        Initialize model client

        Args:
            isa_url: ISA Model API base URL (e.g., http://localhost:8082)
        """
        if isa_url is None:
            from isa_agent_sdk.core.config import settings
            isa_url = settings.resolved_isa_api_url

        self.isa_url = isa_url
        self._client: Optional[AsyncISAModel] = None
        self._lock = asyncio.Lock()

        logger.info(f"ModelClient initialized with ISA URL: {isa_url}")

    async def _get_client(self) -> AsyncISAModel:
        """Thread-safe client singleton with connection reuse"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    # Initialize AsyncISAModel with base_url
                    self._client = AsyncISAModel(base_url=self.isa_url)
                    logger.info(f"AsyncISAModel client initialized: {self.isa_url}")
        return self._client

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
                # Already in OpenAI format
                formatted_tools.append(tool)
            elif 'name' in tool:
                # DEBUG: Log original tool BEFORE formatting
                if tool.get('name') == 'get_weather':
                    import json
                    print(f"\n[DEBUG] Original MCP tool keys: {list(tool.keys())}", flush=True)
                    print(f"[DEBUG] Has 'inputSchema': {'inputSchema' in tool}", flush=True)
                    print(f"[DEBUG] Has 'parameters': {'parameters' in tool}", flush=True)
                    if 'inputSchema' in tool:
                        print(f"[DEBUG] inputSchema type: {type(tool['inputSchema'])}", flush=True)
                        print(f"[DEBUG] inputSchema: {json.dumps(tool['inputSchema'], indent=2)}", flush=True)

                # Get parameters from inputSchema (MCP format) or parameters (OpenAI format)
                # Handle None inputSchema - some MCP tools have inputSchema: null
                params = tool.get('inputSchema') or tool.get('parameters') or {}

                # Ensure params is a dict (safety check)
                if not isinstance(params, dict):
                    params = {}

                # CRITICAL FIX: Convert Pydantic 'title' to OpenAI 'description' for parameter fields
                # Pydantic generates 'title' in JSON schema, but OpenAI expects 'description'
                if params and 'properties' in params and params['properties'] is not None:
                    for param_name, param_schema in params['properties'].items():
                        # If parameter has 'title' but no 'description', use title as description
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

                # DEBUG: Log get_weather tool schema specifically
                if tool.get('name') == 'get_weather':
                    import json
                    print(f"\n[DEBUG TOOL FORMAT] get_weather tool schema (AFTER FIX):", flush=True)
                    print(json.dumps(formatted_tool, indent=2), flush=True)
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
        High-performance model call with timeout and error handling

        Uses AsyncISAModel with OpenAI-compatible chat.completions.create() API.
        Supports format negotiation: LangChain messages, OpenAI format, or strings.

        Args:
            messages: LangChain messages (format negotiation - accepts as-is!)
            tools: Optional tool definitions
            model: Optional model override (default: gpt-4o-mini)
            provider: Optional provider override
            timeout: Request timeout in seconds
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            (AIMessage, billing_info)
        """
        if not messages:
            raise ValueError("Messages required")

        client = await self._get_client()

        # Build args for chat.completions.create()
        # AsyncISAModel supports format negotiation - pass messages as-is!
        args = {
            "model": model or "gpt-4o-mini",
            "messages": messages,  # Format negotiation - LangChain works!
            "stream": False
        }

        if provider:
            args["provider"] = provider

        if tools:
            args["tools"] = self._format_tools(tools)

        if response_format:
            args["response_format"] = response_format

        try:
            # Call with timeout protection using OpenAI-compatible API
            response = await asyncio.wait_for(
                client.chat.completions.create(**args),
                timeout=timeout
            )

            # Extract result from OpenAI-compatible response
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message

                # Debug logging for tool_calls
                raw_tool_calls = getattr(message, 'tool_calls', None)
                logger.info(f"[MODEL_RESPONSE] content_length={len(message.content or '')} | has_tool_calls_attr={hasattr(message, 'tool_calls')} | tool_calls_value={raw_tool_calls}")

                # Convert to AIMessage
                ai_message = AIMessage(
                    content=message.content or "",
                    tool_calls=raw_tool_calls or []
                )

                logger.info(f"[AIMESSAGE_CREATED] content_length={len(ai_message.content)} | tool_calls_count={len(ai_message.tool_calls)}")

                # Extract billing info from response metadata
                billing = {}
                if hasattr(response, 'usage') and response.usage is not None:
                    billing = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

                logger.info(
                    f"Model call completed: {model or 'gpt-4o-mini'}, "
                    f"tokens={billing.get('total_tokens', 0)}"
                )

                return ai_message, billing
            else:
                raise RuntimeError("No choices in response")

        except asyncio.TimeoutError:
            logger.error(f"Model call timeout after {timeout}s")
            raise RuntimeError(f"Model timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Model call failed: {e}")
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
        Stream tokens with callback for real-time processing

        Uses AsyncISAModel streaming with OpenAI-compatible API.
        Format negotiation: LangChain messages work directly!

        Args:
            messages: LangChain messages (format negotiation)
            token_callback: Callback function for streaming tokens
            tools: Optional tool definitions
            model: Optional model override
            provider: Optional provider override
            timeout: Request timeout in seconds
            show_reasoning: Enable reasoning visibility (for DeepSeek-R1)

        Returns:
            (AIMessage, billing_info)
        """
        if not messages:
            raise ValueError("Messages required for streaming")

        client = await self._get_client()

        # Build args for streaming chat.completions.create()
        args = {
            "model": model or "gpt-4o-mini",
            "messages": messages,  # Format negotiation
            "stream": True
        }

        if provider:
            args["provider"] = provider

        if tools:
            args["tools"] = self._format_tools(tools)

        # Add show_reasoning for DeepSeek-R1
        if show_reasoning:
            args["show_reasoning"] = True
            logger.info(f"[REASONING] Enabled reasoning visibility for {model}")

        try:
            import time as time_module
            invoke_start = time_module.time()
            logger.info(f"[TIMING] model_client_stream_start | time={invoke_start}")

            # Get the stream
            stream = await asyncio.wait_for(
                client.chat.completions.create(**args),
                timeout=timeout
            )

            invoke_duration = int((time_module.time() - invoke_start) * 1000)
            logger.info(f"[TIMING] model_client_stream_ready | duration_ms={invoke_duration}")

            # Process stream with callback
            result = await self._stream_with_callback(stream, token_callback)

            return result

        except asyncio.TimeoutError:
            logger.error(f"Streaming timeout after {timeout}s")
            raise RuntimeError(f"Streaming timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise RuntimeError(f"Streaming error: {str(e)}")

    async def _stream_with_callback(
        self,
        stream,
        callback
    ) -> Tuple[AIMessage, Optional[Dict[str, Any]]]:
        """
        Process OpenAI-compatible stream with token-level callbacks

        Streams all tokens through callback for real-time display.
        Compatible with AsyncISAModel streaming format.
        """
        import time as time_module
        stream_start = time_module.time()

        content_chunks = []
        tool_calls = []
        first_token_received = False
        token_count = 0

        try:
            logger.info(f"[STREAM_DEBUG] Starting OpenAI stream iteration")
            chunk_index = 0

            async for chunk in stream:
                chunk_index += 1
                logger.debug(f"[STREAM_DEBUG] Chunk #{chunk_index} received: {chunk}")

                # Handle dict chunks (DeepSeek-R1 tool calls come as complete dict at the end)
                if isinstance(chunk, dict) and 'result' in chunk:
                    logger.info(f"[STREAM_DEBUG] Received dict chunk with 'result' key (likely DeepSeek-R1 tool call)")
                    result = chunk.get('result', {})
                    if 'choices' in result and len(result['choices']) > 0:
                        message = result['choices'][0].get('message', {})
                        if 'tool_calls' in message and message['tool_calls']:
                            logger.info(f"[STREAM_DEBUG] DeepSeek-R1 tool calls detected in dict chunk | count={len(message['tool_calls'])}")
                            for tc in message['tool_calls']:
                                tool_call_dict = {
                                    "name": tc['function']['name'],
                                    "args": json.loads(tc['function']['arguments']) if isinstance(tc['function']['arguments'], str) else tc['function']['arguments'],
                                    "id": tc.get('id', f"call_{len(tool_calls)}"),
                                    "type": tc.get('type', 'function')
                                }
                                tool_calls.append(tool_call_dict)
                                logger.info(f"[STREAM_DEBUG] DeepSeek-R1 tool call added | name={tc['function']['name']} | args={tc['function']['arguments']}")
                    continue  # Skip to next chunk

                # Handle OpenAI-compatible streaming chunks
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    logger.debug(f"[STREAM_DEBUG] Chunk #{chunk_index} delta: {delta}")
                    logger.debug(f"[STREAM_DEBUG] Chunk #{chunk_index} has content attr: {hasattr(delta, 'content')}")
                    if hasattr(delta, 'content'):
                        logger.debug(f"[STREAM_DEBUG] Chunk #{chunk_index} delta.content value: '{delta.content}'")

                    # Log timing for first chunk
                    if not first_token_received:
                        first_chunk_time = int((time_module.time() - stream_start) * 1000)
                        logger.info(f"[PERF] First chunk received: {first_chunk_time}ms")
                        first_token_received = True

                    # Handle content delta
                    if delta.content:
                        token_count += 1
                        # Check if this is a reasoning chunk ([思考: xxx])
                        is_reasoning = delta.content.startswith('[思考:') and delta.content.endswith(']')

                        if is_reasoning:
                            # Extract reasoning content (remove [思考: and ])
                            reasoning_content = delta.content[4:-1]
                            # Stream reasoning token with type marker
                            if callback:
                                callback({"token": reasoning_content, "type": "reasoning"})
                        else:
                            # Normal content token
                            if callback:
                                callback({"token": delta.content})

                        content_chunks.append(delta.content)
                        logger.debug(f"[STREAM_DEBUG] Token #{token_count} appended: '{delta.content}' | Total chunks: {len(content_chunks)}")

                        # Log progress every 10 tokens
                        if token_count % 10 == 0:
                            logger.debug(f"[STREAM_DEBUG] Streamed {token_count} tokens")

                    # DEBUG: Log delta object details
                    logger.info(f"[STREAM_DEBUG] Chunk #{chunk_index} delta type: {type(delta)}")
                    logger.info(f"[STREAM_DEBUG] Chunk #{chunk_index} delta dir: {[attr for attr in dir(delta) if not attr.startswith('_')]}")
                    logger.info(f"[STREAM_DEBUG] Chunk #{chunk_index} has tool_calls attr: {hasattr(delta, 'tool_calls')}")
                    if hasattr(delta, 'tool_calls'):
                        logger.info(f"[STREAM_DEBUG] Chunk #{chunk_index} delta.tool_calls value: {delta.tool_calls}")
                        logger.info(f"[STREAM_DEBUG] Chunk #{chunk_index} delta.tool_calls type: {type(delta.tool_calls)}")

                    # Handle tool calls delta (isa-model 0.5.6+: complete tool_calls in delta)
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        logger.info(f"[STREAM_DEBUG] Received tool_calls in stream | count={len(delta.tool_calls)}")
                        # tool_calls come as complete structures, not incremental deltas
                        # Convert to LangChain format
                        for tc in delta.tool_calls:
                            tool_call_dict = {
                                "name": tc.function.name,
                                "args": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                                "id": tc.id,
                                "type": tc.type
                            }
                            tool_calls.append(tool_call_dict)
                            logger.info(f"[STREAM_DEBUG] Tool call added | name={tc.function.name} | args={tc.function.arguments}")
                    else:
                        logger.info(f"[STREAM_DEBUG] Chunk #{chunk_index} NO tool_calls detected")
                else:
                    logger.debug(f"[STREAM_DEBUG] Chunk #{chunk_index} has no choices or empty choices")

            stream_duration = int((time_module.time() - stream_start) * 1000)
            logger.info(f"[STREAM_DEBUG] Stream complete: {token_count} tokens, {len(content_chunks)} chunks in {stream_duration}ms")
            logger.info(f"[STREAM_DEBUG] Final content_chunks: {content_chunks[:10]}... (first 10)" if len(content_chunks) > 10 else f"[STREAM_DEBUG] Final content_chunks: {content_chunks}")

            # Build final AIMessage
            final_content = ''.join(content_chunks) if content_chunks else ""
            logger.info(f"[STREAM_DEBUG] Building AIMessage with content length: {len(final_content)}, tool_calls count: {len(tool_calls)}")
            logger.debug(f"[STREAM_DEBUG] Final content preview: '{final_content[:200]}...'" if len(final_content) > 200 else f"[STREAM_DEBUG] Final content: '{final_content}'")

            ai_message = AIMessage(
                content=final_content,
                tool_calls=tool_calls if tool_calls else []
            )

            # Note: Billing info not available in streaming mode from AsyncISAModel
            billing = None

            logger.info(f"[STREAM_DEBUG] Returning AIMessage: content={len(ai_message.content)} chars, tool_calls={len(ai_message.tool_calls)}")
            return ai_message, billing

        except Exception as e:
            logger.error(f"[STREAM_DEBUG] OpenAI streaming error: {e}", exc_info=True)
            return AIMessage(content=f"Stream error: {str(e)}"), None

    async def transcribe_audio(
        self,
        audio_file_path: str,
        model: Optional[str] = None,
        timeout: float = 60.0,
        enable_diarization: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using ISA model service

        Args:
            audio_file_path: Path to audio file
            model: Model to use (default: gpt-4o-mini-transcribe)
            timeout: Request timeout in seconds
            enable_diarization: Enable speaker diarization

        Returns:
            Dictionary with transcription result
        """
        if not audio_file_path:
            raise ValueError("Audio file path required")

        client = await self._get_client()

        try:
            # Use OpenAI-compatible audio.transcriptions.create() interface
            transcription_args = {
                "file": audio_file_path,
                "model": model or "gpt-4o-mini-transcribe"
            }

            if enable_diarization:
                transcription_args["response_format"] = "diarized_json"
                transcription_args["enable_diarization"] = True

            transcription = await asyncio.wait_for(
                client.audio.transcriptions.create(**transcription_args),
                timeout=timeout
            )

            # Convert to dict response
            result = {
                "success": True,
                "text": transcription.text,
                "language": getattr(transcription, 'language', None),
                "duration": getattr(transcription, 'duration', None),
                "usage": getattr(transcription, 'usage', None)
            }

            if enable_diarization and hasattr(transcription, 'segments'):
                result["segments"] = transcription.segments

            logger.info(f"Audio transcription completed: {len(result['text'])} chars")

            return result

        except asyncio.TimeoutError:
            logger.error(f"Audio transcription timeout after {timeout}s")
            raise RuntimeError(f"Audio transcription timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise RuntimeError(f"Audio transcription error: {str(e)}")

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for text using ISA model service

        Args:
            texts: List of text strings to embed
            model: Model to use (default: text-embedding-3-small)

        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Texts required for embeddings")

        client = await self._get_client()

        try:
            # Use OpenAI-compatible embeddings.create() interface
            embedding = await client.embeddings.create(
                input=texts if len(texts) > 1 else texts[0],
                model=model or "text-embedding-3-small"
            )

            # Extract embeddings
            embeddings = [item.embedding for item in embedding.data]

            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Embedding error: {str(e)}")

    async def health_check(self) -> bool:
        """Quick health check for service availability"""
        try:
            client = await self._get_client()

            # Try a simple chat call with minimal timeout
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": "ping"}]
                ),
                timeout=5.0
            )

            return response.choices and len(response.choices) > 0
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
            self._client = None
            logger.info("Model client closed")


# Global singleton for performance
_model_client: Optional[ModelClient] = None
_client_lock = asyncio.Lock()


async def get_model_client(isa_url: str = None) -> ModelClient:
    """Get thread-safe singleton model client"""
    global _model_client

    if _model_client is None:
        async with _client_lock:
            if _model_client is None:
                _model_client = ModelClient(isa_url)
                logger.info(f"Global ModelClient initialized")

    return _model_client
