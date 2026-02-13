#!/usr/bin/env python3
"""
isA Agent SDK - Query Function
==============================

Main entry point for the isA Agent SDK.
Provides Claude Agent SDK-compatible `query()` function plus advanced features.

Example:
    from isa_agent_sdk import query, ISAAgentOptions

    # Simple usage
    async for msg in query("Hello, world!"):
        print(msg.content)

    # With options
    async for msg in query(
        prompt="Fix the bug in auth.py",
        options=ISAAgentOptions(
            allowed_tools=["Read", "Edit", "Bash"],
            execution_mode="collaborative"
        )
    ):
        if msg.is_text:
            print(msg.content, end="")
        elif msg.is_checkpoint:
            await msg.respond({"continue": True})
"""

import asyncio
import uuid
import time
import logging
from typing import AsyncIterator, Iterator, Optional, Dict, Any, List, Union, cast

from .options import ISAAgentOptions, ExecutionMode, ExecutionEnv, ToolDiscoveryMode, OutputFormat, OutputFormatType, SystemPromptConfig
from ._messages import (
    AgentMessage,
    ConversationHistory,
    EventData,
    EventEmitter,
    ISAEventType,
    ResultSubtype,
)
from ._structured import (
    StructuredOutputParser,
    parse_structured_output,
)

# Setup logging
logger = logging.getLogger(__name__)


async def query(
    prompt: str,
    options: Optional[ISAAgentOptions] = None,
    *,
    # Convenience parameters (override options)
    allowed_tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    cwd: Optional[str] = None,
    resume: Optional[str] = None,
    output_format: Optional[Union[OutputFormat, Dict[str, Any]]] = None,
) -> AsyncIterator[AgentMessage]:
    """
    Execute an agent query and stream messages.

    This is the main entry point for the isA Agent SDK, providing
    Claude Agent SDK-compatible interface with advanced features.

    Args:
        prompt: The user's request/question
        options: Full configuration options (ISAAgentOptions)

        # Convenience overrides (take precedence over options):
        allowed_tools: List of tool names to allow
        model: Model to use for reasoning
        system_prompt: Custom system prompt
        session_id: Session identifier (auto-generated if not provided)
        user_id: User identifier
        cwd: Working directory for file operations
        resume: Session ID to resume from
        output_format: Structured output format (JSON schema or Pydantic model)

    Yields:
        AgentMessage objects for each event (text, tool_use, etc.)

    Example:
        # Simple query
        async for msg in query("What files are in this directory?"):
            print(msg.content)

        # With tools
        async for msg in query(
            "Read the README.md file",
            allowed_tools=["Read", "Glob"]
        ):
            if msg.is_text:
                print(msg.content)
            elif msg.is_tool_use:
                print(f"[Tool: {msg.tool_name}]")

        # With structured output
        from pydantic import BaseModel

        class Recipe(BaseModel):
            name: str
            ingredients: list[str]

        async for msg in query(
            "Find a cookie recipe",
            output_format=OutputFormat.from_pydantic(Recipe)
        ):
            if msg.has_structured_output:
                recipe = msg.parse(Recipe)
                print(f"Recipe: {recipe.name}")
    """
    # Create or merge options
    opts = options or ISAAgentOptions()

    # Apply convenience overrides
    if allowed_tools is not None:
        opts.allowed_tools = allowed_tools
    if model is not None:
        opts.model = model
    if system_prompt is not None:
        opts.system_prompt = system_prompt
    if session_id is not None:
        opts.session_id = session_id
    if user_id is not None:
        opts.user_id = user_id
    if cwd is not None:
        opts.cwd = cwd
    if resume is not None:
        opts.resume = resume
    if output_format is not None:
        opts.output_format = output_format

    # Generate session ID if not provided
    if not opts.session_id:
        opts.session_id = f"sdk_{uuid.uuid4().hex[:12]}"

    # Create executor based on options
    executor = _create_executor(opts)

    # Execute and stream
    async for msg in executor.execute(prompt, opts):
        yield msg


def query_sync(
    prompt: str,
    options: Optional[ISAAgentOptions] = None,
    **kwargs
) -> Iterator[AgentMessage]:
    """
    Synchronous version of query().

    Runs the async query in an event loop. Use this when not in an async context.

    Args:
        prompt: The user's request/question
        options: Full configuration options
        **kwargs: Convenience parameters (same as query())

    Yields:
        AgentMessage objects

    Example:
        for msg in query_sync("Hello"):
            print(msg.content)
    """
    # Always create a fresh event loop for sync execution
    # This avoids issues with closed loops from previous asyncio.run() calls
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Create async generator
        async_gen = query(prompt, options, **kwargs)

        # Iterate synchronously
        while True:
            try:
                msg = loop.run_until_complete(async_gen.__anext__())
                yield msg
            except StopAsyncIteration:
                break
    finally:
        # Clean up the loop properly
        try:
            # Cancel any pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Run until all tasks are cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


class QueryExecutor:
    """
    Internal executor that handles the actual query execution.

    This class bridges the SDK interface to the LangGraph execution.
    Follows the same patterns as chat_service.py for reliable execution.
    """

    def __init__(self, options: ISAAgentOptions):
        self.options = options
        self._graph = None
        self._graph_builder = None
        self._checkpoint_callback = None
        self._initialized = False

        # Structured output configuration
        self._output_format: Optional[OutputFormat] = None
        self._structured_parser: Optional[StructuredOutputParser] = None
        self._setup_structured_output(options)

    def _setup_structured_output(self, options: ISAAgentOptions) -> None:
        """Configure structured output handling"""
        if not options.output_format:
            return

        # Normalize to OutputFormat
        if isinstance(options.output_format, dict):
            self._output_format = OutputFormat(**options.output_format)
        else:
            self._output_format = options.output_format

        # Create parser if using json_schema
        if self._output_format.type == OutputFormatType.JSON_SCHEMA:
            self._structured_parser = StructuredOutputParser(
                schema=self._output_format.schema,
                pydantic_model=self._output_format._pydantic_model
            )

    async def _ensure_initialized(self) -> bool:
        """Initialize graph builder and graph if not already done"""
        if self._initialized and self._graph:
            return True

        try:
            from .graphs.smart_agent_graph import SmartAgentGraphBuilder

            # Build graph with options
            graph_config = self.options.to_graph_config()
            self._graph_builder = SmartAgentGraphBuilder(config=graph_config)
            self._graph = self._graph_builder.build_graph()
            self._initialized = True
            logger.info("QueryExecutor initialized with SmartAgentGraph")
            return True

        except ImportError as e:
            logger.warning(f"LangGraph components not available: {e}")
            return False

    async def execute(
        self,
        prompt: str,
        options: ISAAgentOptions
    ) -> AsyncIterator[AgentMessage]:
        """Execute the query and yield messages"""
        execute_start = time.time()
        session_id = options.session_id or "unknown"

        # Emit session start
        yield AgentMessage.session_start(session_id)

        try:
            # Build and run the graph
            async for msg in self._run_graph(prompt, options):
                yield msg

        except Exception as e:
            logger.error(f"Query execution error: {e}", exc_info=True)
            # Emit error
            yield AgentMessage.error(str(e), session_id=session_id)

        finally:
            # Emit session end with timing
            execute_duration = int((time.time() - execute_start) * 1000)
            yield AgentMessage.session_end(
                session_id,
                duration_ms=execute_duration
            )

    async def _run_graph(
        self,
        prompt: str,
        options: ISAAgentOptions
    ) -> AsyncIterator[AgentMessage]:
        """
        Run the LangGraph and convert events to AgentMessages.

        This method handles the integration with isA's SmartAgentGraph,
        including proper context preparation with MCP tools and skills.
        Uses the same proven patterns as chat_service.py.
        """
        session_id = options.session_id or "unknown"
        user_id = options.user_id or "sdk_user"

        # Log execution environment (routing happens at tool execution level in BaseNode.mcp_call_tool)
        if options.env == ExecutionEnv.DESKTOP:
            logger.info(f"[SDK] Desktop execution mode | session_id={session_id} | user_id={user_id} | tools will route via Pool Manager")

        # Try to initialize LangGraph components
        graph_available = await self._ensure_initialized()

        if not graph_available:
            # LangGraph not available - use fallback
            yield AgentMessage.error(
                "LangGraph not available. Using mock execution.",
                session_id=session_id
            )
            async for msg in self._mock_execution(prompt, options):
                yield msg
            return

        try:
            from .graphs.utils.context_init import (
                prepare_runtime_context,
                create_initial_state
            )
            from .agent_types.event_types import EventEmitter as ISAEventEmitter
            from langchain_core.runnables import RunnableConfig

            context_start = time.time()

            # Prepare runtime context with SDK options (tools, skills)
            # This integrates with MCP to load tools and skills
            runtime_context = await prepare_runtime_context(
                user_id=user_id,
                thread_id=session_id,
                user_query=prompt,
                allowed_tools=options.allowed_tools,
                skills=options.skills,
                enable_file_context=True,
                mcp_url=options.isa_mcp_url  # Allow custom MCP URL from options
            )

            # Add SDK tool definitions to available_tools
            # This ensures LLM knows about @tool decorated functions
            if options.mcp_servers:
                from ._tools import SDKMCPServer
                sdk_tools_added = []
                available_tools = runtime_context.get('available_tools', [])

                for server_name, server in options.mcp_servers.items():
                    if isinstance(server, SDKMCPServer):
                        # Get tool definitions from SDK server
                        for tool_def in server.list_tools():
                            # Create full tool name with MCP prefix
                            full_tool_name = f"mcp__{server_name}__{tool_def['name']}"
                            # Add to available tools with full name
                            tool_entry = {
                                "name": full_tool_name,
                                "description": tool_def.get('description', ''),
                                "inputSchema": tool_def.get('inputSchema', {})
                            }
                            available_tools.append(tool_entry)
                            sdk_tools_added.append(full_tool_name)

                runtime_context['available_tools'] = available_tools

                if sdk_tools_added:
                    logger.info(
                        f"[SDK] SDK tools added to available_tools | "
                        f"session_id={session_id} | "
                        f"sdk_tools={sdk_tools_added}"
                    )

            context_duration = int((time.time() - context_start) * 1000)
            logger.info(
                f"[SDK] Context prepared | "
                f"session_id={session_id} | "
                f"tools={len(runtime_context.get('available_tools', []))} | "
                f"duration_ms={context_duration}"
            )

            # Initialize model client (with custom URL if provided)
            # This must happen before graph execution to ensure correct service URL
            from .clients.model_client import get_model_client
            model_url = options.isa_model_url  # May be None, which uses defaults from settings
            try:
                await get_model_client(isa_url=model_url)
                logger.info(f"[SDK] Model client initialized | custom_url={model_url}")
            except RuntimeError as e:
                logger.error(f"[SDK] Model client initialization failed: {e}")
                yield AgentMessage.error(
                    f"Model service unavailable: {e}. Please ensure isA_Model service is running.",
                    session_id=session_id
                )
                return

            # Emit context ready events
            tools_count = len(runtime_context.get('available_tools', []))
            prompts_count = len(runtime_context.get('default_prompts', {}))
            memory_length = len(runtime_context.get('memory_context', ''))

            yield AgentMessage(
                type="system",
                content=f"Context ready: {tools_count} tools, {prompts_count} prompts",
                session_id=session_id,
                metadata={
                    "event": "context.complete",
                    "tools_count": tools_count,
                    "prompts_count": prompts_count,
                    "memory_length": memory_length,
                    "duration_ms": context_duration
                }
            )

            # Create initial state using the official pattern
            # This ensures proper add_messages reducer behavior with checkpointer
            initial_state = create_initial_state(prompt)

            # Build runtime config with context (same pattern as chat_service)
            # Include model and other options in configurable for nodes to access
            runtime_config: RunnableConfig = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id,  # For desktop routing
                    "env": options.env.value if hasattr(options.env, 'value') else str(options.env),  # desktop, cloud_pool, cloud_shared
                    "model": options.model,  # Pass model to nodes
                    "allowed_tools": options.allowed_tools,
                    "execution_mode": options.execution_mode.value if hasattr(options.execution_mode, 'value') else options.execution_mode,
                    "cwd": options.cwd,
                    **runtime_context  # Spread context for nodes to access
                },
                "recursion_limit": options.max_iterations or 50
            }

            # Add structured output configuration
            if self._output_format:
                runtime_config["configurable"]["output_format"] = self._output_format.to_model_format()
                runtime_config["configurable"]["output_schema"] = self._output_format.schema

            # Add system prompt configuration
            if options.system_prompt:
                if isinstance(options.system_prompt, str):
                    # Simple string - convert to config (treated as append)
                    system_prompt_config = SystemPromptConfig.from_string(options.system_prompt)
                else:
                    system_prompt_config = options.system_prompt
                runtime_config["configurable"]["system_prompt_config"] = system_prompt_config
                logger.info(
                    f"[SDK] System prompt configured | "
                    f"session_id={session_id} | "
                    f"preset={system_prompt_config.preset} | "
                    f"has_append={system_prompt_config.append is not None} | "
                    f"is_replacement={system_prompt_config.is_replacement}"
                )

            # Load project context (ISA.md/CLAUDE.md support)
            if options.project_context:
                from .utils.project_context import load_project_context, format_project_context_for_prompt
                project_content = load_project_context(
                    source=options.project_context,
                    start_dir=options.cwd
                )
                if project_content:
                    runtime_config["configurable"]["project_context"] = project_content
                    logger.info(
                        f"[SDK] Project context loaded | "
                        f"session_id={session_id} | "
                        f"source={options.project_context[:50] if len(options.project_context) > 50 else options.project_context} | "
                        f"content_length={len(project_content)}"
                    )

            # Add SDK MCP servers (project-based @tool decorator) to config
            if options.mcp_servers:
                from ._tools import SDKMCPServer
                sdk_servers = {
                    name: server
                    for name, server in options.mcp_servers.items()
                    if isinstance(server, SDKMCPServer)
                }
                if sdk_servers:
                    runtime_config["configurable"]["sdk_mcp_servers"] = sdk_servers
                    logger.info(
                        f"[SDK] SDK MCP servers configured | "
                        f"session_id={session_id} | "
                        f"servers={list(sdk_servers.keys())}"
                    )

            # Initialize graph async components (e.g., EventTriggerManager)
            # This is required for event triggers to work properly
            if self._graph_builder and hasattr(self._graph_builder, 'initialize'):
                init_success = await self._graph_builder.initialize(runtime_context)
                if init_success:
                    logger.info(f"[SDK] Graph async components initialized | session_id={session_id}")
                else:
                    logger.warning(f"[SDK] Graph initialization had failures | session_id={session_id}")

            # Stream graph execution using StreamProcessor (same as chat_service.py)
            # This properly captures custom stream events from stream_custom()
            from .services.utils.stream_processor import StreamProcessor
            from .graphs.utils.context_update import update_context_after_chat

            logger.info(f"[SDK] Starting graph stream via StreamProcessor | session_id={session_id}")
            stream_start = time.time()
            first_event_received = False
            event_counts = {
                'thinking': 0,
                'token': 0,
                'tool_use': 0,
                'tool_result': 0,
                'complete': 0
            }

            # Track response content for memory storage
            response_content_parts = []

            stream_processor = StreamProcessor(logger=logger)

            async for event in stream_processor.process_stream(
                self._graph,
                initial_state,
                cast(RunnableConfig, runtime_config),
                session_id,
                stream_context="sdk"
            ):
                # Log first event timing
                if not first_event_received:
                    first_event_time = int((time.time() - stream_start) * 1000)
                    logger.info(
                        f"[SDK] First graph event | "
                        f"session_id={session_id} | "
                        f"event_type={event.get('type')} | "
                        f"time_ms={first_event_time}"
                    )
                    first_event_received = True

                # Convert StreamProcessor events to AgentMessages
                msg = self._convert_stream_event(event, session_id)
                if msg:
                    # Track event counts
                    if msg.is_thinking:
                        event_counts['thinking'] += 1
                    elif msg.is_text:
                        event_counts['token'] += 1
                        # Collect text content for memory storage
                        if msg.content:
                            response_content_parts.append(msg.content)
                    elif msg.is_tool_use:
                        event_counts['tool_use'] += 1
                    elif msg.is_tool_result:
                        event_counts['tool_result'] += 1
                    elif msg.is_complete:
                        event_counts['complete'] += 1
                        # Complete message often contains final response
                        if msg.content:
                            response_content_parts.append(msg.content)

                    yield msg

            # Log completion
            stream_duration = int((time.time() - stream_start) * 1000)
            logger.info(
                f"[SDK] Graph stream complete | "
                f"session_id={session_id} | "
                f"events={event_counts} | "
                f"duration_ms={stream_duration}"
            )

            # Store memories after conversation (same as chat_service.py)
            # This enables cross-session memory recall
            if response_content_parts and user_id:
                try:
                    from .clients.mcp_client import MCPClient
                    from .core.config import settings
                    from langchain_core.messages import HumanMessage, AIMessage

                    ai_response = "".join(response_content_parts)
                    mcp_url = options.isa_mcp_url or settings.resolved_mcp_server_url

                    # Create proper LangChain message objects for extraction
                    messages = [
                        HumanMessage(content=prompt),
                        AIMessage(content=ai_response)
                    ]

                    # Create MCP client for memory storage
                    mcp = MCPClient(mcp_url)
                    await mcp.initialize()

                    try:
                        memory_result = await update_context_after_chat(
                            session_id=session_id,
                            user_id=user_id,
                            final_state={"messages": messages},
                            mcp_service=mcp,
                            conversation_complete=True
                        )

                        if memory_result.get("memories_stored", 0) > 0:
                            logger.info(
                                f"[SDK] Memory stored | "
                                f"session_id={session_id} | "
                                f"user_id={user_id} | "
                                f"memories={memory_result.get('memories_stored', 0)}"
                            )
                    finally:
                        await mcp.close()

                except Exception as mem_err:
                    logger.warning(f"[SDK] Memory storage failed (non-blocking): {mem_err}")

        except ImportError as e:
            logger.error(f"Import error during graph execution: {e}")
            yield AgentMessage.error(
                f"Import error: {e}",
                session_id=session_id
            )
            async for msg in self._mock_execution(prompt, options):
                yield msg

        except Exception as e:
            logger.error(f"Graph execution error: {e}", exc_info=True)
            yield AgentMessage.error(
                f"Execution error: {str(e)}",
                session_id=session_id
            )

    def _convert_graph_event(
        self,
        event: Dict[str, Any],
        session_id: str
    ) -> Optional[AgentMessage]:
        """
        Convert a LangGraph event to an AgentMessage.

        Handles all LangGraph v2 event types and maps them to SDK message types.
        """
        event_type = event.get("event", "")
        data = event.get("data", {})
        name = event.get("name", "")
        tags = event.get("tags", [])

        # === Content Streaming Events ===

        if event_type == "on_chat_model_stream":
            # Streaming token from LLM
            chunk = data.get("chunk")
            print(f"[STREAM_EVENT] on_chat_model_stream | chunk_type={type(chunk).__name__} | has_content={hasattr(chunk, 'content') if chunk else False}", flush=True)
            if chunk and hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                print(f"[STREAM_EVENT] token: '{content}'", flush=True)

                # Check for thinking vs response content
                # DeepSeek R1 and similar models may have reasoning_content
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        return AgentMessage.thinking(reasoning, session_id)

                # Regular response token
                return AgentMessage.text(content, session_id)

        elif event_type == "on_chat_model_start":
            # Model starting - emit thinking indicator
            model_name = data.get("name", "model")
            return AgentMessage(
                type="thinking",
                content="Reasoning...",
                session_id=session_id,
                metadata={"model": model_name, "event": "model_start"}
            )

        elif event_type == "on_chat_model_end":
            # Model finished - check for complete response
            output = data.get("output")
            if output and hasattr(output, "content") and output.content:
                # Check if this is the final response (not intermediate)
                # We typically get individual tokens via streaming, so this
                # is often empty or redundant
                pass

        # === Tool Events ===

        elif event_type == "on_tool_start":
            # Tool execution starting
            tool_input = data.get("input", {})
            tool_name = name or data.get("name", "unknown")
            return AgentMessage.tool_use(
                tool_name=tool_name,
                args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
                session_id=session_id,
                tool_use_id=event.get("run_id")
            )

        elif event_type == "on_tool_end":
            # Tool execution completed
            tool_output = data.get("output", "")
            tool_name = name or data.get("name", "unknown")

            # Handle different output formats
            if hasattr(tool_output, "content"):
                result = tool_output.content
            elif isinstance(tool_output, dict):
                result = tool_output
            else:
                result = str(tool_output) if tool_output else ""

            return AgentMessage.tool_result(
                tool_name=tool_name,
                result=result,
                session_id=session_id,
                tool_use_id=event.get("run_id")
            )

        elif event_type == "on_tool_error":
            # Tool execution failed
            error = data.get("error", "Unknown tool error")
            tool_name = name or "unknown"
            return AgentMessage.tool_result(
                tool_name=tool_name,
                result=None,
                error=str(error),
                session_id=session_id
            )

        # === Chain/Graph Events ===

        elif event_type == "on_chain_start":
            # Chain/node starting
            chain_name = name or data.get("name", "")
            # Only emit for significant nodes
            if chain_name and chain_name not in ["RunnableSequence", "RunnableLambda"]:
                return AgentMessage(
                    type="node_enter",
                    content=f"Entering {chain_name}",
                    session_id=session_id,
                    metadata={"node": chain_name, "event": "node.enter"}
                )

        elif event_type == "on_chain_end":
            # Chain/node completed
            output = data.get("output", {})
            chain_name = name or ""

            # Check for final output with messages
            if isinstance(output, dict) and "messages" in output:
                messages = output["messages"]
                if messages:
                    last_msg = messages[-1]
                    # Check if this is a final AI response
                    if hasattr(last_msg, "content") and last_msg.content:
                        # Check metadata to see if this is the final response node
                        metadata = getattr(last_msg, "additional_kwargs", {})
                        node = metadata.get("node", "")
                        is_internal = metadata.get("is_internal", True)

                        if not is_internal or node == "format_response":
                            return AgentMessage.result(
                                content=last_msg.content,
                                session_id=session_id,
                                metadata={"node": node}
                            )

            # Node exit event for significant nodes
            if chain_name and chain_name not in ["RunnableSequence", "RunnableLambda"]:
                return AgentMessage(
                    type="node_exit",
                    content=f"Exiting {chain_name}",
                    session_id=session_id,
                    metadata={"node": chain_name, "event": "node.exit"}
                )

        # === Custom Events (from StreamProcessor) ===

        elif event_type == "on_custom_event":
            # Custom events from our graph nodes
            custom_name = name or data.get("name", "")
            custom_data = data.get("data", data)

            if custom_name == "hil_request" or "interrupt" in custom_name.lower():
                # Human-in-the-loop request
                return AgentMessage.hil_request(
                    question=custom_data.get("question", "Approval required"),
                    request_type=custom_data.get("request_type", "approval"),
                    options=custom_data.get("options"),
                    session_id=session_id
                )

            elif custom_name == "progress" or "task" in custom_name.lower():
                # Progress update
                return AgentMessage.progress(
                    step=custom_data.get("step", str(custom_data)),
                    percent=custom_data.get("percent"),
                    session_id=session_id
                )

        # === Retriever Events ===

        elif event_type == "on_retriever_start":
            query = data.get("query", "")
            return AgentMessage(
                type="progress",
                content=f"Searching: {query[:100]}...",
                session_id=session_id,
                metadata={"event": "retriever.start", "query": query}
            )

        elif event_type == "on_retriever_end":
            docs = data.get("documents", [])
            return AgentMessage(
                type="progress",
                content=f"Found {len(docs)} documents",
                session_id=session_id,
                metadata={"event": "retriever.end", "doc_count": len(docs)}
            )

        return None

    def _convert_stream_event(
        self,
        event: Dict[str, Any],
        session_id: str
    ) -> Optional[AgentMessage]:
        """
        Convert a StreamProcessor event to an AgentMessage.

        StreamProcessor events use the isA EventType format (e.g., "content.token", "tool.call").
        """
        event_type = event.get("type", "")
        content = event.get("content", "")
        metadata = event.get("metadata", {})

        # === Content Events ===

        if event_type == "content.token":
            # Response token streaming
            return AgentMessage.text(content, session_id)

        elif event_type == "content.thinking":
            # Thinking/reasoning token
            return AgentMessage.thinking(content, session_id)

        elif event_type == "content.complete":
            # Complete content (non-streaming final response)
            # Try to parse as structured output if configured
            structured_output = None
            subtype = ResultSubtype.SUCCESS.value

            if self._structured_parser and content:
                parse_result = self._structured_parser.parse(content)
                if parse_result.success:
                    structured_output = parse_result.data
                    # If parsed to Pydantic model, convert back to dict
                    if hasattr(structured_output, 'model_dump'):
                        structured_output = structured_output.model_dump()
                else:
                    # Structured output parsing failed
                    subtype = ResultSubtype.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES.value
                    logger.warning(f"Structured output parsing failed: {parse_result.error}")

            return AgentMessage.result(
                content,
                session_id,
                structured_output=structured_output,
                subtype=subtype,
                pydantic_model=self._output_format._pydantic_model if self._output_format else None
            )

        # === Tool Events ===

        elif event_type == "tool.call":
            tool_name = metadata.get("tool", "unknown")
            tool_args = metadata.get("args", {})
            return AgentMessage.tool_use(
                tool_name=tool_name,
                args=tool_args,
                session_id=session_id
            )

        elif event_type == "tool.result":
            tool_name = metadata.get("tool", "unknown")
            result = content or metadata.get("result", "")
            return AgentMessage.tool_result(
                tool_name=tool_name,
                result=result,
                session_id=session_id
            )

        elif event_type == "tool.executing":
            # Tool executing progress
            return AgentMessage(
                type="progress",
                content=content,
                session_id=session_id,
                metadata=metadata
            )

        # === Session Events ===

        elif event_type == "session.start":
            return AgentMessage.session_start(session_id)

        elif event_type == "session.end":
            return AgentMessage.session_end(session_id)

        elif event_type == "session.paused":
            # HIL interrupt
            return AgentMessage.hil_request(
                question=content or "Approval required",
                request_type=metadata.get("request_type", "approval"),
                options=metadata.get("options"),
                session_id=session_id
            )

        # === Node Events ===

        elif event_type == "node.enter":
            return AgentMessage(
                type="node_enter",
                content=content,
                session_id=session_id,
                metadata=metadata
            )

        elif event_type == "node.exit":
            return AgentMessage(
                type="node_exit",
                content=content,
                session_id=session_id,
                metadata=metadata
            )

        # === System Events ===

        elif event_type in ("system.ready", "system.info"):
            return AgentMessage(
                type="system",
                content=content,
                session_id=session_id,
                metadata=metadata
            )

        elif event_type == "system.error":
            return AgentMessage.error(content, session_id=session_id)

        # === Task Events ===

        elif event_type.startswith("task."):
            return AgentMessage(
                type="progress",
                content=content,
                session_id=session_id,
                metadata=metadata
            )

        # === Context Events ===

        elif event_type.startswith("context."):
            return AgentMessage(
                type="system",
                content=content,
                session_id=session_id,
                metadata=metadata
            )

        # Skip unknown events
        return None

    async def _mock_execution(
        self,
        prompt: str,
        options: ISAAgentOptions
    ) -> AsyncIterator[AgentMessage]:
        """
        Mock execution for when LangGraph is not available.

        This allows basic testing of the SDK without full dependencies.
        """
        session_id = options.session_id or "unknown"

        yield AgentMessage(
            type="thinking",
            content=f"Processing: {prompt}",
            session_id=session_id,
            metadata={"mock": True}
        )

        await asyncio.sleep(0.1)  # Simulate processing

        yield AgentMessage.result(
            content=f"[Mock Response] Received: {prompt}\n\n"
                    f"Note: LangGraph dependencies are not available. "
                    f"Install the full isA Agent package for real execution.",
            session_id=session_id,
            metadata={"mock": True}
        )


def _create_executor(options: ISAAgentOptions) -> QueryExecutor:
    """Create the appropriate executor based on options"""
    # For now, just return the standard executor
    # In the future, this could return different executors based on
    # execution_env (cloud_pool, cloud_shared, desktop)
    return QueryExecutor(options)


# Convenience functions for common use cases

async def ask(prompt: str, **kwargs) -> str:
    """
    Simple query that returns just the final text response.

    Args:
        prompt: The question/request
        **kwargs: Options passed to query()

    Returns:
        The final text response as a string
    """
    result = []
    async for msg in query(prompt, **kwargs):
        if msg.is_text or msg.is_complete:
            if msg.content:
                result.append(msg.content)
    return "".join(result)


def ask_sync(prompt: str, **kwargs) -> str:
    """Synchronous version of ask()"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(ask(prompt, **kwargs))
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


async def resume(
    session_id: str,
    resume_value: Optional[Dict[str, Any]] = None,
    options: Optional[ISAAgentOptions] = None,
) -> AsyncIterator[AgentMessage]:
    """
    Resume an interrupted session.

    Use this to continue execution after a checkpoint/HIL interrupt.

    Args:
        session_id: Session ID to resume
        resume_value: Value to pass for resumption (e.g., {"authorized": True})
        options: Optional options (will use defaults if not provided)

    Yields:
        AgentMessage objects from resumed execution

    Example:
        # After receiving a checkpoint message
        async for msg in resume(session_id, {"continue": True}):
            print(msg.content)
    """
    opts = options or ISAAgentOptions()
    opts.session_id = session_id
    opts.resume = session_id

    try:
        from .graphs.smart_agent_graph import SmartAgentGraphBuilder
        from .graphs.utils.context_init import prepare_runtime_context
        from langgraph.types import Command

        # Build graph
        graph_config = opts.to_graph_config()
        builder = SmartAgentGraphBuilder(config=graph_config)
        graph = builder.build_graph()

        user_id = opts.user_id or "sdk_user"

        # Prepare runtime context
        runtime_context = await prepare_runtime_context(
            user_id=user_id,
            thread_id=session_id,
            user_query="",  # No new query for resume
            mcp_url=opts.isa_mcp_url  # Allow custom MCP URL from options
        )

        # Configure with same thread_id to resume from checkpoint
        config = {
            "configurable": {
                "thread_id": session_id,
                **runtime_context
            },
            "recursion_limit": opts.max_iterations or 50
        }

        # Resume with Command if we have a value
        if resume_value is not None:
            initial_input = Command(resume=resume_value)
        else:
            initial_input = None

        yield AgentMessage.session_start(session_id, resumed=True)

        # Stream resumed execution
        async for event in graph.astream_events(
            initial_input,
            config=config,
            version="v2"
        ):
            executor = QueryExecutor(opts)
            msg = executor._convert_graph_event(event, session_id)
            if msg:
                yield msg

        yield AgentMessage.session_end(session_id, resumed=True)

    except ImportError as e:
        yield AgentMessage.error(
            f"Resume not available: {e}",
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Resume error: {e}", exc_info=True)
        yield AgentMessage.error(str(e), session_id=session_id)


def resume_sync(
    session_id: str,
    resume_value: Optional[Dict[str, Any]] = None,
    options: Optional[ISAAgentOptions] = None,
) -> Iterator[AgentMessage]:
    """Synchronous version of resume()"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async_gen = resume(session_id, resume_value, options)

        while True:
            try:
                msg = loop.run_until_complete(async_gen.__anext__())
                yield msg
            except StopAsyncIteration:
                break
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> AgentMessage:
    """
    Execute a single tool directly via MCP.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        session_id: Optional session ID
        user_id: Optional user ID for context

    Returns:
        AgentMessage with tool result

    Example:
        result = await execute_tool("web_search", {"query": "latest news"})
        print(result.tool_result_value)
    """
    session_id = session_id or f"tool_{uuid.uuid4().hex[:8]}"

    try:
        # Try to use MCP client for tool execution
        from .clients.mcp_client import MCPClient
        from .core.config import settings

        # Initialize MCP client
        mcp_url = settings.resolved_mcp_server_url
        mcp = MCPClient(mcp_url)
        await mcp.initialize()

        try:
            result = await mcp.call_tool(tool_name, tool_args)

            return AgentMessage.tool_result(
                tool_name=tool_name,
                result=result,
                session_id=session_id
            )
        finally:
            await mcp.close()

    except ImportError as e:
        logger.warning(f"MCP client import failed: {e}")
        return AgentMessage.error(
            f"MCP client not available for tool execution: {e}",
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return AgentMessage.tool_result(
            tool_name=tool_name,
            result=None,
            error=str(e),
            session_id=session_id
        )


async def get_available_tools(
    user_query: Optional[str] = None,
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Get list of available tools, optionally filtered by query relevance.

    Args:
        user_query: Optional query to search for relevant tools
        max_results: Maximum number of tools to return

    Returns:
        List of tool definitions with name, description, and inputSchema
    """
    try:
        from .graphs.utils.context_init import get_runtime_helper

        helper = await get_runtime_helper()

        if user_query:
            return await helper.search_relevant_tools(user_query, max_results)
        else:
            # Return default tools
            return helper._default_tools[:max_results]

    except ImportError as e:
        logger.warning(f"Cannot get tools: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        return []


async def get_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current state of a session from checkpointer.

    Args:
        session_id: Session ID to get state for

    Returns:
        Session state dict or None if not found
    """
    try:
        from .graphs.smart_agent_graph import SmartAgentGraphBuilder
        from .services.persistence import durable_service

        # Build graph to access checkpointer
        builder = SmartAgentGraphBuilder()
        graph = builder.build_graph()

        checkpointer = getattr(graph, 'checkpointer', None)
        if not checkpointer:
            return None

        thread_config = {"configurable": {"thread_id": session_id}}
        state_snapshot = await checkpointer.aget_tuple(thread_config)

        if state_snapshot:
            checkpoint_data = state_snapshot.checkpoint
            if 'channel_values' in checkpoint_data:
                state = checkpoint_data['channel_values']
                return {
                    "session_id": session_id,
                    "messages_count": len(state.get('messages', [])),
                    "has_summary": bool(state.get('summary')),
                    "next_action": state.get('next_action'),
                    "checkpoint_id": state_snapshot.config.get('configurable', {}).get('checkpoint_id')
                }

        return None

    except Exception as e:
        logger.error(f"Error getting session state: {e}")
        return None


__all__ = [
    # Main entry points
    "query",
    "query_sync",
    "ask",
    "ask_sync",

    # Resume functionality
    "resume",
    "resume_sync",

    # Tool execution
    "execute_tool",
    "get_available_tools",

    # Session management
    "get_session_state",

    # Executor class
    "QueryExecutor",
]
