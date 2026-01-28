"""
Feedback Service Data Models
File: app/services/feedback/models.py

Defines all data structures for feedback collection and analysis.
Inspired by industry best practices from Claude, ChatGPT, Gemini, and Grok.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class FeedbackType(Enum):
    """Types of feedback (explicit + implicit)"""
    # Explicit feedback
    RATING = "rating"                    # Star rating or thumbs up/down
    CORRECTION = "correction"            # User corrects AI response
    COMPLAINT = "complaint"              # Issue report
    APPRECIATION = "appreciation"        # Positive feedback / thanks
    COMMENT = "comment"                  # Free-form text feedback
    
    # Implicit feedback
    REGENERATION = "regeneration"        # User requests new response
    CLARIFICATION = "clarification"      # Follow-up question
    ABANDONMENT = "abandonment"          # User leaves mid-conversation
    EDIT = "edit"                        # User edits AI response
    CONTINUATION = "continuation"        # User continues conversation
    COPY = "copy"                        # User copies AI response


class FeedbackDimension(Enum):
    """Multi-dimensional evaluation criteria"""
    # Primary dimensions (inspired by Claude's Constitutional AI)
    HELPFULNESS = "helpfulness"          # How helpful was the response
    ACCURACY = "accuracy"                # Factual correctness
    CLARITY = "clarity"                  # Clear and understandable
    SAFETY = "safety"                    # Harmless and appropriate
    
    # Secondary dimensions
    COMPLETENESS = "completeness"        # Addressed all aspects
    EFFICIENCY = "efficiency"            # Concise and to-the-point
    ENGAGEMENT = "engagement"            # Interesting and engaging (Grok-inspired)
    RELEVANCE = "relevance"              # On-topic and contextual


class SentimentPolarity(Enum):
    """Sentiment analysis results"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class UserIntent(Enum):
    """User intent classification"""
    CONFIRMATION = "confirmation"        # Confirming understanding
    QUESTION = "question"                # Asking follow-up
    CORRECTION = "correction"            # Correcting AI
    APPRECIATION = "appreciation"        # Expressing thanks
    COMPLAINT = "complaint"              # Expressing dissatisfaction
    NEUTRAL = "neutral"                  # Neutral statement


@dataclass
class FeedbackScore:
    """
    Multi-dimensional feedback scores
    
    Example:
        score = FeedbackScore(
            helpfulness=0.9,
            accuracy=0.85,
            clarity=0.92,
            safety=1.0
        )
    """
    # Primary dimensions (0.0-1.0)
    helpfulness: float = 0.5
    accuracy: float = 0.5
    clarity: float = 0.5
    safety: float = 0.5
    
    # Secondary dimensions (0.0-1.0)
    completeness: float = 0.5
    efficiency: float = 0.5
    engagement: float = 0.5
    relevance: float = 0.5
    
    # Meta
    confidence: float = 0.5              # Confidence in the scores
    timestamp: datetime = field(default_factory=datetime.now)
    
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            'helpfulness': 0.25,
            'accuracy': 0.25,
            'clarity': 0.15,
            'safety': 0.15,
            'completeness': 0.10,
            'efficiency': 0.05,
            'engagement': 0.03,
            'relevance': 0.02
        }
        
        return (
            self.helpfulness * weights['helpfulness'] +
            self.accuracy * weights['accuracy'] +
            self.clarity * weights['clarity'] +
            self.safety * weights['safety'] +
            self.completeness * weights['completeness'] +
            self.efficiency * weights['efficiency'] +
            self.engagement * weights['engagement'] +
            self.relevance * weights['relevance']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['overall_score'] = self.overall_score()
        return result


@dataclass
class ImplicitSignals:
    """
    Implicit feedback signals (ChatGPT-style)
    
    Example:
        signals = ImplicitSignals(
            time_to_first_response=2.5,
            regeneration_count=1,
            abandonment=False
        )
    """
    # Timing signals (seconds)
    time_to_first_response: Optional[float] = None
    time_between_messages: Optional[float] = None
    session_duration: Optional[float] = None
    
    # Interaction signals (counts)
    regeneration_count: int = 0
    edit_count: int = 0
    copy_count: int = 0
    follow_up_count: int = 0
    
    # Navigation signals
    scroll_depth: Optional[float] = None      # 0.0-1.0
    revisit_count: int = 0
    abandonment: bool = False
    
    # Engagement signals
    message_length_change: Optional[float] = None  # User msg length trend
    response_engagement: Optional[float] = None    # 0.0-1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class GroundingFeedback:
    """
    Grounding and factual accuracy feedback (Gemini-style)
    
    Example:
        grounding = GroundingFeedback(
            claims_made=10,
            claims_verified=8,
            sources_cited=3,
            factual_accuracy=0.85
        )
    """
    claims_made: int = 0
    claims_verified: int = 0
    sources_cited: int = 0
    factual_accuracy: float = 0.5        # 0.0-1.0
    citation_quality: float = 0.5        # 0.0-1.0
    hallucination_detected: bool = False
    
    def verification_rate(self) -> float:
        """Calculate claim verification rate"""
        if self.claims_made == 0:
            return 1.0
        return self.claims_verified / self.claims_made
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['verification_rate'] = self.verification_rate()
        return result


@dataclass
class FeedbackEvent:
    """
    Single feedback event
    
    Example:
        event = FeedbackEvent(
            session_id="session_123",
            turn_id=5,
            feedback_type=FeedbackType.RATING,
            explicit_rating=0.9,
            user_comment="Very helpful!"
        )
    """
    # Identifiers
    event_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    session_id: str = ""
    user_id: str = "default"
    turn_id: int = 0
    
    # Feedback content
    feedback_type: FeedbackType = FeedbackType.RATING
    explicit_rating: Optional[float] = None      # 0.0-1.0 or 1-5
    user_comment: Optional[str] = None
    
    # AI response context
    ai_response: Optional[str] = None
    user_message: Optional[str] = None
    
    # Scores and signals
    scores: Optional[FeedbackScore] = None
    implicit_signals: Optional[ImplicitSignals] = None
    grounding: Optional[GroundingFeedback] = None
    
    # Analysis results
    sentiment: Optional[SentimentPolarity] = None
    intent: Optional[UserIntent] = None
    issues_identified: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    analyzed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "turn_id": self.turn_id,
            "feedback_type": self.feedback_type.value,
            "explicit_rating": self.explicit_rating,
            "user_comment": self.user_comment,
            "sentiment": self.sentiment.value if self.sentiment else None,
            "intent": self.intent.value if self.intent else None,
            "issues_identified": self.issues_identified,
            "timestamp": self.timestamp.isoformat(),
            "analyzed": self.analyzed
        }
        
        if self.scores:
            result["scores"] = self.scores.to_dict()
        if self.implicit_signals:
            result["implicit_signals"] = self.implicit_signals.to_dict()
        if self.grounding:
            result["grounding"] = self.grounding.to_dict()
            
        return result


@dataclass
class SessionMetrics:
    """
    Aggregated metrics for a conversation session
    
    Example:
        metrics = SessionMetrics(
            session_id="session_123",
            total_turns=10,
            average_score=0.85,
            quality_grade="excellent"
        )
    """
    session_id: str
    user_id: str = "default"
    
    # Turn statistics
    total_turns: int = 0
    user_turns: int = 0
    ai_turns: int = 0
    feedback_events: int = 0
    
    # Score aggregation
    average_score: float = 0.5
    dimension_averages: Dict[str, float] = field(default_factory=dict)
    score_trend: str = "stable"              # improving, declining, stable
    
    # Quality assessment
    quality_grade: str = "fair"              # excellent, good, fair, poor
    satisfaction_rate: float = 0.5           # 0.0-1.0
    
    # Implicit signals summary
    regeneration_rate: float = 0.0           # Regenerations per turn
    abandonment_rate: float = 0.0            # % of abandoned sessions
    avg_response_time: float = 0.0           # Average time between turns
    
    # Issues
    common_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None         # Seconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class AggregatedMetrics:
    """
    Metrics aggregated across multiple sessions
    
    Example:
        metrics = AggregatedMetrics(
            time_window="daily",
            total_sessions=150,
            average_satisfaction=0.82
        )
    """
    # Time window
    time_window: str                         # hourly, daily, weekly, monthly
    start_time: datetime
    end_time: datetime
    
    # Session statistics
    total_sessions: int = 0
    total_feedback_events: int = 0
    total_turns: int = 0
    
    # Quality metrics
    average_satisfaction: float = 0.5
    quality_distribution: Dict[str, int] = field(default_factory=dict)  # {excellent: 50, good: 30, ...}
    
    # Dimension scores
    dimension_averages: Dict[str, float] = field(default_factory=dict)
    
    # Issue analysis
    top_issues: List[tuple] = field(default_factory=list)  # [(issue, count), ...]
    issue_resolution_rate: float = 0.0
    
    # Trends
    satisfaction_trend: str = "stable"
    most_improved_dimension: Optional[str] = None
    most_declined_dimension: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        result['timestamp'] = self.timestamp.isoformat()
        return result

