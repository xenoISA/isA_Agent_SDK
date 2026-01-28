"""
Feedback Collectors - MVP Implementation
File: app/services/feedback/collectors.py

Collects explicit and implicit feedback signals.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import (
    FeedbackEvent,
    FeedbackType,
    ImplicitSignals,
    FeedbackScore
)
from isa_agent_sdk.utils.logger import api_logger


class FeedbackCollector:
    """
    Collects feedback from various sources
    
    Example:
        collector = FeedbackCollector()
        
        # Collect explicit feedback
        event = collector.collect_explicit_feedback(
            session_id="session_123",
            user_input="Thanks, that was helpful!",
            ai_response="Glad I could help!",
            turn_id=5
        )
    """
    
    def __init__(self):
        self.feedback_events: List[FeedbackEvent] = []
        self.session_timings: Dict[str, Dict[str, Any]] = {}
    
    def collect_explicit_feedback(
        self,
        session_id: str,
        user_input: str,
        ai_response: Optional[str] = None,
        turn_id: int = 0,
        user_id: str = "default",
        explicit_rating: Optional[float] = None,
        user_comment: Optional[str] = None
    ) -> FeedbackEvent:
        """
        Collect explicit feedback event
        
        Args:
            session_id: Session identifier
            user_input: User's message
            ai_response: AI's response
            turn_id: Turn number in conversation
            user_id: User identifier
            explicit_rating: Optional explicit rating (0.0-1.0)
            user_comment: Optional user comment
            
        Returns:
            FeedbackEvent object
        """
        event = FeedbackEvent(
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            feedback_type=FeedbackType.RATING if explicit_rating else FeedbackType.COMMENT,
            user_message=user_input,
            ai_response=ai_response,
            explicit_rating=explicit_rating,
            user_comment=user_comment or user_input
        )
        
        self.feedback_events.append(event)
        api_logger.debug(f"Collected explicit feedback for session {session_id}, turn {turn_id}")
        
        return event
    
    def collect_implicit_signals(
        self,
        session_id: str,
        turn_id: int,
        regeneration_requested: bool = False,
        content_edited: bool = False,
        content_copied: bool = False,
        response_time: Optional[float] = None
    ) -> ImplicitSignals:
        """
        Collect implicit feedback signals
        
        Args:
            session_id: Session identifier
            turn_id: Turn number
            regeneration_requested: Did user request regeneration
            content_edited: Did user edit the response
            content_copied: Did user copy the response
            response_time: Time taken to respond (seconds)
            
        Returns:
            ImplicitSignals object
        """
        # Initialize session tracking if needed
        if session_id not in self.session_timings:
            self.session_timings[session_id] = {
                "start_time": datetime.now(),
                "last_message_time": datetime.now(),
                "regeneration_count": 0,
                "edit_count": 0,
                "copy_count": 0
            }
        
        session_data = self.session_timings[session_id]
        
        # Update counts
        if regeneration_requested:
            session_data["regeneration_count"] += 1
        if content_edited:
            session_data["edit_count"] += 1
        if content_copied:
            session_data["copy_count"] += 1
        
        # Calculate time between messages
        now = datetime.now()
        time_between = (now - session_data["last_message_time"]).total_seconds()
        session_data["last_message_time"] = now
        
        # Calculate session duration
        session_duration = (now - session_data["start_time"]).total_seconds()
        
        signals = ImplicitSignals(
            time_to_first_response=response_time,
            time_between_messages=time_between,
            session_duration=session_duration,
            regeneration_count=session_data["regeneration_count"],
            edit_count=session_data["edit_count"],
            copy_count=session_data["copy_count"]
        )
        
        api_logger.debug(f"Collected implicit signals for session {session_id}, turn {turn_id}")
        
        return signals
    
    def get_session_events(self, session_id: str) -> List[FeedbackEvent]:
        """Get all feedback events for a session"""
        return [e for e in self.feedback_events if e.session_id == session_id]
    
    def clear_session(self, session_id: str):
        """Clear feedback data for a session"""
        self.feedback_events = [e for e in self.feedback_events if e.session_id != session_id]
        if session_id in self.session_timings:
            del self.session_timings[session_id]
        api_logger.info(f"Cleared feedback data for session {session_id}")

