#!/usr/bin/env python3
"""
ReactAgent - LangGraph-based autonomous task execution agent

A professional implementation of the ReAct pattern using LangGraph for
structured reasoning and tool execution with proper error handling.
"""

import json
from typing import List, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from isa_agent_sdk.clients.model_client import get_model_client
from isa_agent_sdk.utils.logger import agent_logger


class ReactAgent:
    """
    ReAct Agent implementation using LangGraph

    Implements the Reasoning and Acting (ReAct) pattern for autonomous task execution
    with native tool calling support and proper message flow handling.
    """

    def __init__(self, tools: List[Dict[str, Any]], mcp_manager):
        """
        Initialize ReactAgent with tools and MCP manager

        Args:
            tools: List of available tool definitions
            mcp_manager: MCP service manager for tool execution
        """
        self.logger = agent_logger  # Use centralized logger for Loki integration
        self.tools = tools
        self.mcp_manager = mcp_manager
        self.get_model_client = get_model_client
        
        self.logger.debug(f"ReactAgent initialized with {len(tools)} tools")
        
        try:
            self.graph = self._create_graph()
        except Exception as e:
            self.logger.error(f"Failed to create ReactAgent graph: {e}")
            raise
    
    def _create_graph(self) -> StateGraph:
        """
        Create the ReAct agent execution graph
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._call_tools)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    async def _call_model(self, state: MessagesState) -> Dict[str, List[AIMessage]]:
        """
        Call the LLM with conversation history and available tools
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with model response
        """
        messages = state["messages"]
        
        # Build system prompt with tool descriptions
        tool_descriptions = [
            f"- {tool.get('name', '')}: {tool.get('description', '')}"
            for tool in self.tools
        ]
        
        system_prompt = f"""You are a helpful AI assistant with access to the following tools:

{chr(10).join(tool_descriptions)}

IMPORTANT: When you need to gather information or perform actions, you MUST use the available tools.

To use a tool, respond with this exact format:
TOOL: tool_name WITH ARGS: {{"arg1": "value1", "arg2": "value2"}}

For example:
- To search: TOOL: web_search WITH ARGS: {{"query": "AI trends 2024", "count": 5}}
- To crawl: TOOL: web_crawl WITH ARGS: {{"url": "https://example.com", "analysis_request": "extract key points"}}

If you can answer directly without tools, provide a complete response.
If you need information that requires tools, use them proactively."""

        # Prepare messages with system context
        messages_with_system = [HumanMessage(content=system_prompt)] + messages
        
        self.logger.debug(f"Calling model with {len(self.tools)} tools")
        
        try:
            # Always use normal streaming model client call (ignore output_format)
            # ReactAgent should always stream regardless of final response format requirements
            model_client = await self.get_model_client()
            response, _ = await model_client.stream_tokens(
                messages=messages_with_system,
                tools=self.tools,
                token_callback=None  # No streaming for ReactAgent internal processing
            )
            
            # Log tool calls for debugging
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.debug(f"Model generated {len(response.tool_calls)} tool calls")
            
            return {"messages": [response]}
            
        except Exception as e:
            self.logger.error(f"Model call failed: {e}")
            error_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
            return {"messages": [error_response]}
    
    def _should_continue(self, state: MessagesState) -> str:
        """
        Determine next action based on model response
        
        Args:
            state: Current conversation state
            
        Returns:
            Next node name or END
        """
        last_message = state["messages"][-1]
        
        # Check for native tool calls (preferred LangChain format)
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            self.logger.debug(f"Found {len(last_message.tool_calls)} tool calls - executing tools")
            return "tools"
            
        # Fallback: check for custom TOOL: format
        elif hasattr(last_message, 'content') and "TOOL:" in last_message.content:
            self.logger.debug("Found custom TOOL: format - executing tools")
            return "tools"
            
        else:
            self.logger.debug("No tools required - ending conversation")
            return END
    
    async def _call_tools(self, state: MessagesState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tools requested by the model
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with tool execution results
        """
        last_message = state["messages"][-1]
        
        try:
            # Handle native tool calls (preferred LangChain format)
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return await self._execute_native_tool_calls(last_message.tool_calls)
                
            # Fallback: Handle custom TOOL: format
            elif hasattr(last_message, 'content') and "TOOL:" in last_message.content:
                return await self._execute_custom_tool_format(last_message.content)
                
            else:
                self.logger.warning("Tool execution requested but no tools found")
                return {"messages": []}
                
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            error_message = AIMessage(content=f"Tool execution failed: {str(e)}")
            return {"messages": [error_message]}
    
    async def _execute_native_tool_calls(self, tool_calls: List[Dict]) -> Dict[str, List[ToolMessage]]:
        """Execute native LangChain tool calls"""
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_call_id = tool_call['id']
            
            self.logger.debug(f"Executing tool: {tool_name}")
            
            try:
                result = await self.mcp_manager.call_tool(tool_name, tool_args)
                tool_message = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id
                )
                tool_messages.append(tool_message)
                self.logger.debug(f"Tool {tool_name} executed successfully")
                
            except Exception as e:
                self.logger.error(f"Tool {tool_name} failed: {e}")
                error_message = ToolMessage(
                    content=f"Tool execution failed: {str(e)}",
                    tool_call_id=tool_call_id
                )
                tool_messages.append(error_message)
        
        return {"messages": tool_messages}
    
    async def _execute_custom_tool_format(self, content: str) -> Dict[str, List[AIMessage]]:
        """Execute custom TOOL: format (fallback)"""
        try:
            tool_part = content.split("TOOL:")[1].split("WITH ARGS:")[0].strip()
            args_part = content.split("WITH ARGS:")[1].strip()
            
            try:
                args = json.loads(args_part)
            except json.JSONDecodeError:
                args = {}
            
            result = await self.mcp_manager.call_tool(tool_part, args)
            result_message = AIMessage(
                content=f"Tool {tool_part} executed successfully. Result: {result}"
            )
            
            return {"messages": [result_message]}
            
        except Exception as e:
            self.logger.error(f"Custom tool format execution failed: {e}")
            error_message = AIMessage(content=f"Tool execution failed: {str(e)}")
            return {"messages": [error_message]}
    
    def _messages_to_prompt(self, messages) -> str:
        """
        Convert LangChain messages to a prompt string for ISA client
        
        Args:
            messages: List of LangChain messages
            
        Returns:
            Combined prompt string
        """
        prompt_parts = []
        
        for message in messages:
            content = getattr(message, 'content', str(message))
            msg_type = type(message).__name__
            
            if msg_type == 'HumanMessage':
                prompt_parts.append(f"Human: {content}")
            elif msg_type == 'AIMessage':
                prompt_parts.append(f"Assistant: {content}")
            elif msg_type == 'SystemMessage':
                prompt_parts.append(f"System: {content}")
            else:
                prompt_parts.append(f"{msg_type}: {content}")
        
        return "\n\n".join(prompt_parts)
    
    async def execute(self, task_prompt: str) -> str:
        """
        Execute a task using the ReAct agent
        
        Args:
            task_prompt: User task description
            
        Returns:
            Final response content
        """
        try:
            initial_state = {"messages": [HumanMessage(content=task_prompt)]}
            result = await self.graph.ainvoke(initial_state)
            
            if result["messages"]:
                return result["messages"][-1].content
            else:
                return "No response generated"
                
        except Exception as e:
            self.logger.error(f"ReactAgent execution failed: {e}")
            return f"ReactAgent execution failed: {str(e)}"


def create_react_agent(tools: List[Dict[str, Any]], mcp_manager) -> ReactAgent:
    """
    Factory function to create a ReactAgent instance
    
    Args:
        tools: List of available tool definitions
        mcp_manager: MCP service manager for tool execution
        
    Returns:
        Configured ReactAgent instance
    """
    return ReactAgent(tools, mcp_manager)