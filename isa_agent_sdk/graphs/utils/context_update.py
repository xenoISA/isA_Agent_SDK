#!/usr/bin/env python3
"""
Context Update - Memory storage after chat completion

Simple utility to store conversation memories using MCP intelligent dialog processing.
Called by chat_service after graph execution completes.

Enhanced with Co-memorize collaborative memory curation.
"""

import logging
from typing import Dict, Any, List
from .memory_utils import store_conversation_memories
# from .memory_curation_utils import check_memory_curation_opportunity, execute_memory_curation  # Removed - no longer needed

logger = logging.getLogger(__name__)


async def update_context_after_chat(
    session_id: str,
    user_id: str,
    final_state: Dict[str, Any],
    mcp_service,
    conversation_complete: bool = True
) -> Dict[str, Any]:
    """
    Update user context and memory after chat completion with Co-memorize integration

    Args:
        session_id: Chat session ID
        user_id: User identifier
        final_state: Final state from LangGraph execution
        mcp_service: MCP service for memory operations
        conversation_complete: Whether this is a completed conversation

    Returns:
        Update results and status
    """
    update_results = {
        "memory_updated": False,
        "memories_stored": 0,
        "tools_used": [],
        "curation_performed": False,
        "curation_results": {},
        "errors": []
    }

    try:
        # Defensive check: ensure final_state is a dictionary
        if not isinstance(final_state, dict):
            logger.warning(
                f"final_state is not a dict (type: {type(final_state).__name__}), "
                f"skipping memory update for session {session_id}"
            )
            update_results["errors"].append(f"Invalid final_state type: {type(final_state).__name__}")
            return update_results

        messages = final_state.get("messages", [])
        
        if len(messages) < 2:
            logger.info("Not enough messages for memory storage")
            return update_results
        
        # Get first user message and last AI response
        user_message = _extract_user_message(messages)
        ai_response = _extract_ai_response(messages)
        
        if not user_message or not ai_response:
            logger.info("Could not extract user message and AI response")
            return update_results
        
        # 1. Store conversation memories using intelligent MCP tools
        storage_result = await store_conversation_memories(
            mcp_service, user_id, session_id, user_message, ai_response
        )
        
        if storage_result["success"]:
            update_results["memory_updated"] = True
            update_results["memories_stored"] = storage_result["memories_stored"]
            update_results["tools_used"] = storage_result["tools_used"]
            
            logger.info(f"Context updated successfully for session {session_id}")
        else:
            update_results["errors"].append(storage_result.get("error", "Unknown storage error"))
        
        # 2. Co-memorize: Memory curation removed - no longer needed
        # Memory curation functionality has been removed from the system
        logger.debug(f"Memory curation disabled for user {user_id}")
        
    except Exception as e:
        error_msg = f"Context update failed: {str(e)}"
        logger.error(error_msg)
        update_results["errors"].append(error_msg)
    
    return update_results


def _extract_user_message(messages: List[Any]) -> str:
    """Extract the first meaningful user message"""
    for msg in messages:
        if hasattr(msg, 'content'):
            content = getattr(msg, 'content', '')
            # Look for HumanMessage or user content
            if 'human' in type(msg).__name__.lower() and content:
                return content
    return ""


def _extract_ai_response(messages: List[Any]) -> str:
    """Extract the last AI response message"""
    for msg in reversed(messages):
        if hasattr(msg, 'content'):
            content = getattr(msg, 'content', '')
            # Look for AIMessage content
            if 'ai' in type(msg).__name__.lower() and content:
                return content
    return ""