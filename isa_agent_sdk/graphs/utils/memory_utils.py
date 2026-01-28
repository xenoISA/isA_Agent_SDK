"""
Memory aggregation utilities for context collection

This module provides intelligent memory search and aggregation using MCP Memory Tools.
It takes user_id + session_id and returns consolidated memory text for context building.

MCP Memory Tools Integration (15 tools):
  Storage Tools (6):
    - store_factual_memory: AI-powered fact extraction and storage
    - store_episodic_memory: Event memory extraction
    - store_semantic_memory: Concept memory extraction
    - store_procedural_memory: Procedure extraction
    - store_working_memory: Short-term memory with TTL
    - store_session_message: Conversation message tracking

  Search Tools (4):
    - search_memories: Universal search across all memory types
    - search_facts_by_subject: Subject-based factual memory search
    - search_episodes_by_event_type: Event type-based episodic search
    - search_concepts_by_category: Category-based semantic concept search

  Retrieval Tools (3):
    - get_session_context: Get comprehensive session data
    - summarize_session: AI-powered session summarization
    - get_active_working_memories: Retrieve active working memories

  Utility Tools (2):
    - get_memory_statistics: User memory statistics
    - memory_health_check: Service health status

Core functionality:
- Session context retrieval with summaries and recent messages
- Cross-memory-type search using semantic similarity
- Active working memory retrieval for current tasks
- Memory consolidation and formatting for context injection

Design principles:
- Intelligent aggregation: combines multiple memory types into coherent context
- Relevance-based filtering: prioritizes important and recent memories
- Context-aware formatting: structures memory for optimal LLM consumption
- Error resilience: graceful degradation when memory services unavailable
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MemoryAggregator:
    """
    Intelligent memory aggregation using MCP Memory Tools
    
    Features:
    - Multi-source memory retrieval (session, factual, episodic, working, semantic)
    - Intelligent relevance scoring and filtering
    - Context-optimized formatting for LLM consumption
    - Graceful error handling and fallback mechanisms
    """
    
    def __init__(self, mcp_service, max_context_length: int = 2000):
        """
        Initialize memory aggregator
        
        Args:
            mcp_service: MCP service instance for tool execution
            max_context_length: Maximum character length for aggregated context
        """
        self.mcp_service = mcp_service
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(f"{__name__}.MemoryAggregator")
    
    async def get_aggregated_memory(
        self, 
        user_id: str, 
        session_id: str,
        query_context: Optional[str] = None,
        include_session: bool = True,
        include_working: bool = True,
        include_semantic: bool = True,
        include_episodic: bool = True,
        include_factual: bool = True
    ) -> str:
        """
        Get aggregated memory context for user and session
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query_context: Optional query context for semantic search
            include_*: Control which memory types to include
            
        Returns:
            Formatted memory context string
        """
        try:
            self.logger.info(f"Aggregating memory for user {user_id}, session {session_id}")
            
            # Collect memory from multiple sources concurrently
            memory_parts = []
            
            # 1. Session context (conversation history and summaries)
            if include_session:
                session_context = await self._get_session_context(user_id, session_id)
                if session_context:
                    memory_parts.append(("Session Context", session_context))
            
            # 2. Active working memories (current tasks)
            if include_working:
                working_memories = await self._get_working_memories(user_id)
                if working_memories:
                    memory_parts.append(("Current Tasks", working_memories))
            
            # 3. Semantic search across all memory types (if query context provided)
            if query_context and any([include_semantic, include_episodic, include_factual]):
                semantic_memories = await self._search_relevant_memories(
                    user_id, query_context, include_semantic, include_episodic, include_factual
                )
                if semantic_memories:
                    memory_parts.append(("Relevant Context", semantic_memories))
            
            # 4. Recent factual memories (user preferences, info)
            if include_factual and not query_context:
                factual_memories = await self._get_recent_factual_memories(user_id)
                if factual_memories:
                    memory_parts.append(("User Information", factual_memories))
            
            # Format and consolidate memory context
            return self._format_memory_context(memory_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate memory: {e}")
            return self._get_fallback_context(user_id, session_id)
    
    async def _get_session_context(self, user_id: str, session_id: str) -> Optional[str]:
        """Get session context including summaries and recent messages"""
        try:
            result = await self.mcp_service.call_tool("get_session_context", {
                "user_id": user_id,
                "session_id": session_id,
                "include_summaries": "true",  # MCP requires string, not bool
                "max_recent_messages": 5
            })
            
            # Parse the MCP response
            parsed = self._parse_mcp_response(result)
            if parsed.get('status') == 'success' and parsed.get('data', {}).get('session_found'):
                return self._extract_session_content(parsed)
            
            self.logger.debug(f"No session context found for {session_id}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get session context: {e}")
            return None
    
    async def _get_working_memories(self, user_id: str) -> Optional[str]:
        """Get active working memories (current tasks)"""
        try:
            result = await self.mcp_service.call_tool("get_active_working_memories", {
                "user_id": user_id
            })
            
            # Parse the MCP response
            parsed = self._parse_mcp_response(result)
            if parsed.get('status') == 'success':
                return self._extract_working_content(parsed)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get working memories: {e}")
            return None
    
    async def _search_relevant_memories(
        self, 
        user_id: str, 
        query: str,
        include_semantic: bool,
        include_episodic: bool, 
        include_factual: bool
    ) -> Optional[str]:
        """Search for relevant memories using semantic similarity"""
        try:
            # Build memory types filter
            memory_types = []
            if include_semantic:
                memory_types.append("SEMANTIC")
            if include_episodic:
                memory_types.append("EPISODIC")
            if include_factual:
                memory_types.append("FACTUAL")
            
            if not memory_types:
                return None
            
            # Use list directly as expected by MCP tool
            
            result = await self.mcp_service.call_tool("search_memories", {
                "user_id": user_id,
                "query": query,
                "memory_types": memory_types,
                "top_k": 8
            })
            
            # Parse the MCP response
            parsed = self._parse_mcp_response(result)
            if parsed.get('status') == 'success':
                return self._extract_search_content(parsed)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to search relevant memories: {e}")
            return None
    
    async def _get_recent_factual_memories(self, user_id: str) -> Optional[str]:
        """Get recent factual memories about user preferences and information"""
        try:
            # Search factual memories by subject using the correct MCP tool
            # Note: search_facts_by_subject searches by subject keyword, not by fact type
            results = []
            for subject in ["personal", "preference", "skill"]:
                result = await self.mcp_service.call_tool("search_facts_by_subject", {
                    "user_id": user_id,
                    "subject": subject,
                    "limit": 3
                })

                parsed = self._parse_mcp_response(result)
                if parsed.get('status') == 'success':
                    results.extend(parsed.get('data', {}).get('results', []))

            if results:
                return self._extract_factual_content({"data": {"results": results}})

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get factual memories: {e}")
            return None
    
    def _parse_mcp_response(self, raw_response) -> dict:
        """Parse MCP response (handles both event stream, direct JSON, and dict)"""
        try:
            # If already a dict, return it directly
            if isinstance(raw_response, dict):
                return raw_response

            # Check for event stream format first (must be string)
            if isinstance(raw_response, str) and 'event: message' in raw_response and 'data:' in raw_response:
                # Extract the JSON part after 'data:'
                lines = raw_response.split('\n')
                for line in lines:
                    if line.startswith('data:'):
                        json_str = line[5:].strip()  # Remove 'data:' prefix
                        data = json.loads(json_str)

                        # Extract the actual content from MCP response
                        if 'result' in data and 'content' in data['result']:
                            content = data['result']['content'][0]['text']
                            # Parse the inner JSON content
                            return json.loads(content)

            # Fallback: try to parse as direct JSON string
            if isinstance(raw_response, str):
                return json.loads(raw_response)

            # If not string or dict, return error
            return {"status": "error", "error": f"Unexpected response type: {type(raw_response).__name__}"}

        except Exception as e:
            self.logger.error(f"Failed to parse MCP response: {e}")
            return {"status": "error", "error": "Failed to parse response"}
    
    def _extract_session_content(self, parsed_response: dict) -> str:
        """Extract session content from parsed MCP response"""
        data = parsed_response.get('data', {})
        result = []
        
        # Extract conversation state
        conv_state = data.get('conversation_state', {})
        topics = conv_state.get('topics', [])
        content = conv_state.get('content', '')
        summary = data.get('conversation_summary', '')
        
        if summary:
            result.append(f"Summary: {summary}")
        if topics:
            result.append(f"Topics: {', '.join(topics)}")
        if content:
            result.append(f"Recent: {content}")
        
        return '\n'.join(result) if result else "Session found but no content"
    
    def _extract_working_content(self, parsed_response: dict) -> str:
        """Extract working memory content from parsed MCP response"""
        data = parsed_response.get('data', {})
        results = data.get('results', [])
        
        if results:
            working_info = []
            for item in results:
                content = item.get('content', '')
                priority = item.get('priority', 'medium')
                working_info.append(f"Task: {content} (Priority: {priority})")
            return '\n'.join(working_info)
        else:
            return "No active working memories"
    
    def _extract_search_content(self, parsed_response: dict) -> str:
        """Extract search results from parsed MCP response"""
        data = parsed_response.get('data', {})
        results = data.get('results', [])
        
        if results:
            search_info = []
            for item in results:
                content = item.get('content', '')
                memory_type = item.get('memory_type', 'unknown')
                score = item.get('similarity_score', 0)
                search_info.append(f"{memory_type.title()}: {content} (Score: {score:.2f})")
            return '\n'.join(search_info)
        else:
            return "No memories found"
    
    def _extract_factual_content(self, parsed_response: dict) -> str:
        """Extract factual information from parsed MCP response"""
        data = parsed_response.get('data', {})
        results = data.get('results', [])
        
        if results:
            factual_info = []
            for item in results:
                if 'content' in item:
                    factual_info.append(item['content'])
                elif 'subject' in item and 'object_value' in item:
                    fact = f"{item['subject']}: {item['object_value']}"
                    factual_info.append(fact)
            return '\n'.join(factual_info)
        else:
            return "No factual information found"
    
    def _format_memory_context(self, memory_parts: List[tuple]) -> str:
        """Format memory parts into coherent context string"""
        if not memory_parts:
            return ""
        
        formatted_parts = []
        current_length = 0
        
        for section_name, content in memory_parts:
            if not content:
                continue
            
            # Format section with clear headers
            section = f"## {section_name}\n{content.strip()}\n"
            
            # Check if adding this section would exceed max length
            if current_length + len(section) > self.max_context_length:
                # Truncate this section to fit
                remaining_space = self.max_context_length - current_length - len(f"## {section_name}\n\n")
                if remaining_space > 50:  # Only add if there's meaningful space
                    truncated_content = content.strip()[:remaining_space] + "..."
                    section = f"## {section_name}\n{truncated_content}\n"
                    formatted_parts.append(section)
                break
            
            formatted_parts.append(section)
            current_length += len(section)
        
        result = '\n'.join(formatted_parts).strip()
        
        # Add header if we have content
        if result:
            result = "# Memory Context\n\n" + result
        
        return result
    
    def _get_fallback_context(self, user_id: str, session_id: str) -> str:
        """Return minimal fallback context when memory retrieval fails"""
        return f"# Memory Context\n\n## Session Information\nUser: {user_id}\nSession: {session_id}\nNote: Full memory context unavailable"


# Convenience functions for easy integration

async def get_user_memory_context(
    mcp_service,
    user_id: str,
    session_id: str,
    query_context: Optional[str] = None,
    max_length: int = 2000
) -> str:
    """
    Convenience function to get aggregated memory context
    
    Args:
        mcp_service: MCP service instance
        user_id: User identifier
        session_id: Session identifier  
        query_context: Optional query for semantic search
        max_length: Maximum context length
        
    Returns:
        Formatted memory context string
    """
    aggregator = MemoryAggregator(mcp_service, max_length)
    return await aggregator.get_aggregated_memory(
        user_id, session_id, query_context
    )


async def get_session_summary_only(
    mcp_service,
    user_id: str,
    session_id: str
) -> str:
    """
    Get only session context (conversation summaries and recent messages)
    
    Args:
        mcp_service: MCP service instance
        user_id: User identifier
        session_id: Session identifier
        
    Returns:
        Session context string
    """
    aggregator = MemoryAggregator(mcp_service, 1000)
    return await aggregator.get_aggregated_memory(
        user_id=user_id,
        session_id=session_id,
        include_session=True,
        include_working=False,
        include_semantic=False,
        include_episodic=False,
        include_factual=False
    )


async def get_working_memory_only(
    mcp_service,
    user_id: str
) -> str:
    """
    Get only active working memories (current tasks)
    
    Args:
        mcp_service: MCP service instance
        user_id: User identifier
        
    Returns:
        Working memory context string
    """
    aggregator = MemoryAggregator(mcp_service, 800)
    return await aggregator.get_aggregated_memory(
        user_id=user_id,
        session_id="temp",  # Not used when only getting working memory
        include_session=False,
        include_working=True,
        include_semantic=False,
        include_episodic=False,
        include_factual=False
    )


# ==================== MEMORY STORAGE FUNCTIONS ====================

async def store_conversation_memories(
    mcp_service,
    user_id: str,
    session_id: str,
    user_message: str,
    ai_response: str
) -> Dict[str, Any]:
    """
    Store conversation memories using MCP intelligent dialog processing
    
    Args:
        mcp_service: MCP service instance
        user_id: User identifier
        session_id: Session identifier
        user_message: First user input message
        ai_response: Final AI response message
        
    Returns:
        Storage results dictionary
    """
    try:
        # Create dialog content from user input and AI response
        dialog_content = f"Human: {user_message}\n\nAI: {ai_response}"
        
        # Store conversation using intelligent processing tools
        storage_results = []
        
        # 1. Store session message for conversation tracking
        session_result = await mcp_service.call_tool("store_session_message", {
            "user_id": user_id,
            "session_id": session_id,
            "message_content": dialog_content,
            "message_type": "conversation",
            "role": "system",
            "importance_score": 0.7
        })
        storage_results.append({"tool": "store_session_message", "result": session_result})
        
        # 2. Store factual memory from dialog
        factual_result = await mcp_service.call_tool("store_factual_memory", {
            "user_id": user_id,
            "dialog_content": dialog_content,
            "importance_score": 0.6
        })
        storage_results.append({"tool": "store_factual_memory", "result": factual_result})
        
        # 3. Store episodic memory from dialog
        episodic_result = await mcp_service.call_tool("store_episodic_memory", {
            "user_id": user_id,
            "dialog_content": dialog_content,
            "importance_score": 0.5
        })
        storage_results.append({"tool": "store_episodic_memory", "result": episodic_result})
        
        # 4. Store semantic memory from dialog
        semantic_result = await mcp_service.call_tool("store_semantic_memory", {
            "user_id": user_id,
            "dialog_content": dialog_content,
            "importance_score": 0.6
        })
        storage_results.append({"tool": "store_semantic_memory", "result": semantic_result})
        
        # 5. Store procedural memory from dialog
        procedural_result = await mcp_service.call_tool("store_procedural_memory", {
            "user_id": user_id,
            "dialog_content": dialog_content,
            "importance_score": 0.7
        })
        storage_results.append({"tool": "store_procedural_memory", "result": procedural_result})
        
        # 6. Store working memory from dialog
        working_result = await mcp_service.call_tool("store_working_memory", {
            "user_id": user_id,
            "dialog_content": dialog_content,
            "ttl_seconds": 86400,
            "importance_score": 0.5
        })
        storage_results.append({"tool": "store_working_memory", "result": working_result})
        
        # Helper function to parse MCP responses
        def parse_mcp_result(raw_response: str) -> dict:
            try:
                # Check for event stream format first
                if 'event: message' in raw_response and 'data:' in raw_response:
                    lines = raw_response.split('\n')
                    for line in lines:
                        if line.startswith('data:'):
                            json_str = line[5:].strip()
                            data = json.loads(json_str)
                            
                            # Extract content from MCP response structure
                            if 'result' in data and 'content' in data['result']:
                                content = data['result']['content'][0]['text']
                                # Parse the inner JSON content
                                try:
                                    return json.loads(content)
                                except json.JSONDecodeError:
                                    # If content is not JSON, treat as success message
                                    return {"status": "success", "message": content}
                            
                # Try to parse as direct JSON
                if raw_response.strip().startswith('{'):
                    return json.loads(raw_response)
                
                # If not JSON, treat as success if no error keywords
                if "error" not in raw_response.lower():
                    return {"status": "success", "message": raw_response}
                else:
                    return {"status": "error", "message": raw_response}
                    
            except Exception as e:
                # Check if response indicates success despite parsing failure
                if "success" in raw_response.lower() or "stored" in raw_response.lower():
                    return {"status": "success", "message": raw_response}
                return {"status": "error", "error": str(e)}
        
        # Count successful storage operations
        successful_tools = []
        for result in storage_results:
            try:
                # Parse the MCP response to check for success
                parsed = parse_mcp_result(result["result"])
                if parsed.get('status') == 'success':
                    successful_tools.append(result["tool"])
                elif isinstance(result["result"], str) and "success" in result["result"].lower():
                    # Fallback for simple string responses
                    successful_tools.append(result["tool"])
            except:
                pass
        
        logger.info(f"Stored conversation memories for user {user_id}, session {session_id}")
        
        return {
            "memories_stored": len(successful_tools),
            "tools_used": successful_tools,
            "storage_results": storage_results,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to store conversation memories: {e}")
        return {
            "memories_stored": 0,
            "tools_used": [],
            "error": str(e),
            "success": False
        }