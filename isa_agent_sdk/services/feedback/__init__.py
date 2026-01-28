"""
Feedback Service Package - MVP Implementation
File: app/services/feedback/__init__.py

Professional feedback collection and analysis for AI agents.
Based on industry best practices from Claude, ChatGPT, Gemini, and Grok.

Example Usage:
    from isa_agent_sdk.services.feedback import get_feedback_service
    
    # Get service
    service = get_feedback_service()
    
    # Process feedback
    event = await service.process_user_message(
        session_id="session_123",
        user_input="Thanks, that was helpful!",
        ai_response="Glad I could help!",
        turn_id=5
    )
    
    # Get metrics
    metrics = service.get_session_metrics("session_123")
    print(f"Quality: {metrics.quality_grade}")
"""

# Main service
from .feedback_service import (
    FeedbackService,
    get_feedback_service,
    reset_feedback_service
)

# Data models
from .models import (
    FeedbackType,
    FeedbackDimension,
    SentimentPolarity,
    UserIntent,
    FeedbackScore,
    ImplicitSignals,
    GroundingFeedback,
    FeedbackEvent,
    SessionMetrics,
    AggregatedMetrics
)

# Components (for advanced usage)
from .collectors import FeedbackCollector
from .analyzers import PatternAnalyzer, SemanticAnalyzer
from .aggregator import FeedbackAggregator

# Package metadata
__version__ = "2.0.0-MVP"
__author__ = "isA Agent Team"

# Public API
__all__ = [
    # Main service
    "FeedbackService",
    "get_feedback_service",
    "reset_feedback_service",
    
    # Data models
    "FeedbackType",
    "FeedbackDimension",
    "SentimentPolarity",
    "UserIntent",
    "FeedbackScore",
    "ImplicitSignals",
    "GroundingFeedback",
    "FeedbackEvent",
    "SessionMetrics",
    "AggregatedMetrics",
    
    # Components (advanced)
    "FeedbackCollector",
    "PatternAnalyzer",
    "SemanticAnalyzer",
    "FeedbackAggregator"
]

