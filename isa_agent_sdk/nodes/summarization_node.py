#!/usr/bin/env python3
"""
Summarization Node - Official LangGraph Pattern

Automatically summarizes old conversation messages to manage context window.
Uses official LangGraph RemoveMessage pattern.

Official docs: https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/
"""

from typing import Dict, Any, List
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from .base_node import BaseNode
from isa_agent_sdk.agent_types.agent_state import AgentState
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger


class SummarizationNode(BaseNode):
    """
    Official LangGraph pattern for conversation summarization

    Inherits from BaseNode to access model service and logging.

    Features:
    - Compresses old messages into a summary
    - Preserves recent N messages (default: 5)
    - Uses RemoveMessage to delete old messages
    - Token-aware compression
    """

    def __init__(
        self,
        preserve_last_n: int = 5,
        summary_model: str = "gpt-4o-mini"
    ):
        """
        Initialize summarization node

        Args:
            preserve_last_n: Number of recent messages to keep in full
            summary_model: Model to use for summarization (fast & cheap)
        """
        super().__init__(node_name="SummarizationNode")
        self.preserve_last_n = preserve_last_n
        self.summary_model = summary_model

    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Execute summarization (Official LangGraph pattern)

        Args:
            state: Current agent state

        Returns:
            Updated state with summary and removed old messages
        """
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")

        logger.info(
            f"[SUMMARIZATION] Starting conversation summarization | "
            f"total_messages={len(messages)} | "
            f"preserve_last={self.preserve_last_n} | "
            f"has_existing_summary={bool(existing_summary)}"
        )

        # Check if we have enough messages to summarize
        if len(messages) <= self.preserve_last_n:
            logger.info(
                f"[SUMMARIZATION] Skipping - not enough messages | "
                f"messages={len(messages)} | "
                f"threshold={self.preserve_last_n}"
            )
            return {}  # No changes

        # Split messages: old (to summarize) and recent (to keep)
        old_messages = messages[:-self.preserve_last_n]
        recent_messages = messages[-self.preserve_last_n:]

        logger.info(
            f"[SUMMARIZATION] Message split | "
            f"old_messages={len(old_messages)} | "
            f"recent_messages={len(recent_messages)}"
        )

        # Generate summary of old messages
        try:
            new_summary = await self._generate_summary(
                old_messages=old_messages,
                existing_summary=existing_summary
            )

            logger.info(
                f"[SUMMARIZATION] Summary generated | "
                f"summary_length={len(new_summary)} | "
                f"old_messages_compressed={len(old_messages)}"
            )

        except Exception as e:
            logger.error(
                f"[SUMMARIZATION] Summary generation failed | "
                f"error={type(e).__name__} | "
                f"message={str(e)[:200]}",
                exc_info=True
            )
            # Don't fail the whole conversation, just skip summarization
            return {}

        # Use RemoveMessage to delete old messages (Official LangGraph pattern)
        messages_to_delete = [RemoveMessage(id=m.id) for m in old_messages]

        logger.info(
            f"[SUMMARIZATION] Complete | "
            f"messages_removed={len(messages_to_delete)} | "
            f"messages_remaining={len(recent_messages)} | "
            f"summary_updated=True"
        )

        # Calculate token savings (estimated)
        old_tokens_estimate = sum(len(str(m.content)) // 4 for m in old_messages)
        summary_tokens_estimate = len(new_summary) // 4
        tokens_saved = old_tokens_estimate - summary_tokens_estimate
        compression_ratio = (tokens_saved / old_tokens_estimate * 100) if old_tokens_estimate > 0 else 0

        logger.info(
            f"[SUMMARIZATION] Token savings (estimated) | "
            f"old_tokens={old_tokens_estimate} | "
            f"summary_tokens={summary_tokens_estimate} | "
            f"tokens_saved={tokens_saved} | "
            f"compression={compression_ratio:.1f}%"
        )

        return {
            "summary": new_summary,
            "messages": messages_to_delete
        }

    async def _generate_summary(
        self,
        old_messages: List,
        existing_summary: str
    ) -> str:
        """
        Generate summary of old messages using LLM

        Args:
            old_messages: Messages to summarize
            existing_summary: Previous summary (if any)

        Returns:
            New summary text
        """
        # Format messages for summarization
        messages_text = self._format_messages_for_summary(old_messages)

        # Build summarization prompt
        if existing_summary:
            # Update existing summary
            prompt = f"""You are summarizing a conversation to manage context window size.

Previous Summary:
{existing_summary}

New Messages to Add:
{messages_text}

Please provide an updated summary that:
1. Combines the previous summary with new information
2. Preserves key decisions, actions, and important context
3. Removes redundant or less important details
4. Keeps the summary concise (2-4 paragraphs max)
5. Uses past tense and third person

Updated Summary:"""
        else:
            # Create first summary
            prompt = f"""You are summarizing a conversation to manage context window size.

Conversation to Summarize:
{messages_text}

Please provide a concise summary that:
1. Captures key topics discussed
2. Preserves important decisions and actions
3. Maintains essential context for continuing the conversation
4. Keeps the summary concise (2-4 paragraphs max)
5. Uses past tense and third person

Summary:"""

        # Generate summary using BaseNode's call_model (includes billing tracking)
        summary_response = await self.call_model(
            messages=[HumanMessage(content=prompt)],
            model=self.summary_model,
            stream_tokens=False  # No streaming for summarization
        )

        # Extract summary text (summary_response is an AIMessage)
        summary_text = summary_response.content

        return summary_text.strip()

    def _format_messages_for_summary(self, messages: List) -> str:
        """
        Format messages into readable text for summarization

        Args:
            messages: List of messages

        Returns:
            Formatted text string
        """
        formatted_lines = []

        for msg in messages:
            # Get role and content
            if hasattr(msg, 'type'):
                role = msg.type
            elif hasattr(msg, '__class__'):
                role = msg.__class__.__name__.replace('Message', '').lower()
            else:
                role = 'unknown'

            content = getattr(msg, 'content', str(msg))

            # Skip empty messages
            if not content or not str(content).strip():
                continue

            # Format based on role
            if role in ['human', 'user']:
                formatted_lines.append(f"User: {content}")
            elif role in ['ai', 'assistant']:
                formatted_lines.append(f"Assistant: {content}")
            elif role == 'system':
                formatted_lines.append(f"System: {content}")
            elif role == 'tool':
                # Tool messages can be verbose, truncate if needed
                content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else content
                formatted_lines.append(f"Tool Result: {content_preview}")
            else:
                formatted_lines.append(f"{role.capitalize()}: {content}")

        return "\n\n".join(formatted_lines)


# Helper function for conditional edge
def should_summarize(
    state: AgentState,
    message_threshold: int = 20,
    enable_summarization: bool = True
) -> str:
    """
    Decide if conversation should be summarized (for conditional edge)

    Args:
        state: Current agent state
        message_threshold: Number of messages before triggering summarization
        enable_summarization: Global flag to enable/disable summarization

    Returns:
        "summarize" if should summarize, "continue" otherwise
    """
    if not enable_summarization:
        return "continue"

    messages = state.get("messages", [])
    message_count = len(messages)

    if message_count > message_threshold:
        logger.info(
            f"[SUMMARIZATION] Trigger condition met | "
            f"messages={message_count} | "
            f"threshold={message_threshold} | "
            f"action=summarize"
        )
        return "summarize"

    logger.debug(
        f"[SUMMARIZATION] No summarization needed | "
        f"messages={message_count} | "
        f"threshold={message_threshold}"
    )
    return "continue"


# Convenience function for graph integration
async def summarize_conversation(state: AgentState) -> Dict[str, Any]:
    """
    Convenience function to use in graph workflows

    Usage in graph:
        workflow.add_node("summarize", summarize_conversation)

    Args:
        state: Agent state

    Returns:
        Updated state with summary
    """
    node = SummarizationNode(
        preserve_last_n=5,
        summary_model="gpt-4o-mini"
    )
    return await node.execute(state)
