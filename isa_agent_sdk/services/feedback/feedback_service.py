"""
Feedback Service - MVP Implementation
File: app/services/feedback/feedback_service.py

Main orchestrating service for feedback collection and analysis.
Combines pattern-based (fast) and semantic (deep) analysis.

Example Usage:
    from isa_agent_sdk.services.feedback import get_feedback_service
    
    # Get service instance
    feedback_service = get_feedback_service()
    
    # Collect and analyze feedback
    await feedback_service.process_user_message(
        session_id="session_123",
        user_input="Thanks, that was very helpful!",
        ai_response="Glad I could help!",
        turn_id=5
    )
    
    # Get session metrics
    metrics = feedback_service.get_session_metrics("session_123")
    print(f"Quality: {metrics.quality_grade}")
    print(f"Score: {metrics.average_score}")
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import (
    FeedbackEvent,
    FeedbackScore,
    SessionMetrics,
    FeedbackType
)
from .collectors import FeedbackCollector
from .analyzers import PatternAnalyzer, SemanticAnalyzer
from .aggregator import FeedbackAggregator
from isa_agent_sdk.utils.logger import api_logger


class FeedbackService:
    """
    Main feedback service with hybrid analysis
    
    Architecture:
    - Pattern Analyzer: Fast, rule-based analysis (always runs)
    - Semantic Analyzer: Deep, AI-driven analysis (runs when needed)
    - Strategy: Start with patterns, escalate to AI for low-confidence or complex cases
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feedback service
        
        Args:
            config: Optional configuration
                - use_semantic_analysis: Enable AI analysis (default: True)
                - semantic_threshold: Confidence threshold for AI (default: 0.5)
                - analysis_model: Model for semantic analysis (default: "gpt-4")
        """
        self.config = config or {}
        
        # Components
        self.collector = FeedbackCollector()
        self.pattern_analyzer = PatternAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer(config) if self.config.get("use_semantic_analysis", True) else None
        self.aggregator = FeedbackAggregator()
        
        # Configuration
        self.use_semantic = self.config.get("use_semantic_analysis", True)
        self.semantic_threshold = self.config.get("semantic_threshold", 0.5)
        
        # Storage (in-memory for MVP)
        self.sessions: Dict[str, List[FeedbackEvent]] = {}
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        api_logger.info("🔍 FeedbackService initialized (hybrid: pattern + semantic)")
    
    async def process_user_message(
        self,
        session_id: str,
        user_input: str,
        ai_response: Optional[str] = None,
        turn_id: int = 0,
        user_id: str = "default",
        explicit_rating: Optional[float] = None
    ) -> FeedbackEvent:
        """
        Process user message and analyze feedback
        
        Args:
            session_id: Session identifier
            user_input: User's message
            ai_response: AI's response (if available)
            turn_id: Turn number in conversation
            user_id: User identifier
            explicit_rating: Optional explicit rating (0.0-1.0)
            
        Returns:
            Analyzed FeedbackEvent
            
        Example:
            event = await service.process_user_message(
                session_id="session_123",
                user_input="Thanks, that was helpful!",
                ai_response="Glad I could help!",
                turn_id=5
            )
            
            print(f"Score: {event.scores.overall_score()}")
        """
        # Collect explicit feedback
        event = self.collector.collect_explicit_feedback(
            session_id=session_id,
            user_input=user_input,
            ai_response=ai_response,
            turn_id=turn_id,
            user_id=user_id,
            explicit_rating=explicit_rating
        )
        
        # Store conversation history
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        if ai_response:
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })
        
        self.conversation_history[session_id].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze feedback (hybrid approach)
        await self._analyze_feedback(event)
        
        # Store event
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(event)
        
        api_logger.info(f"Processed feedback for session {session_id}, turn {turn_id}: score={event.scores.overall_score():.2f}")
        
        return event
    
    async def _analyze_feedback(self, event: FeedbackEvent):
        """
        Analyze feedback using hybrid approach
        
        Strategy:
        1. Always run pattern analysis (fast)
        2. If confidence is low OR explicit rating provided, run semantic analysis
        3. Use semantic result if available and confident
        """
        # Step 1: Pattern analysis (always)
        pattern_score = self.pattern_analyzer.analyze(event)
        
        # Step 2: Decide if semantic analysis is needed
        needs_semantic = (
            self.use_semantic and
            self.semantic_analyzer and
            (
                pattern_score.confidence < self.semantic_threshold or
                event.explicit_rating is not None
            )
        )
        
        if needs_semantic:
            try:
                # Get conversation history
                history = self.conversation_history.get(event.session_id, [])
                
                # Run semantic analysis
                semantic_score = await self.semantic_analyzer.analyze(event, history)
                
                # Use semantic result if more confident
                if semantic_score.confidence > pattern_score.confidence:
                    api_logger.debug(f"Using semantic analysis (confidence: {semantic_score.confidence:.2f})")
                else:
                    api_logger.debug(f"Using pattern analysis (confidence: {pattern_score.confidence:.2f})")
                    event.scores = pattern_score
                    
            except Exception as e:
                api_logger.warning(f"Semantic analysis failed, using pattern analysis: {e}")
                event.scores = pattern_score
        else:
            # Use pattern analysis result
            api_logger.debug(f"Using pattern analysis (confidence: {pattern_score.confidence:.2f})")
            event.scores = pattern_score
    
    def get_session_metrics(self, session_id: str) -> SessionMetrics:
        """
        Get aggregated metrics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionMetrics object
            
        Example:
            metrics = service.get_session_metrics("session_123")
            print(f"Quality: {metrics.quality_grade}")
            print(f"Average score: {metrics.average_score:.2f}")
            print(f"Recommendations: {metrics.recommendations}")
        """
        events = self.sessions.get(session_id, [])
        user_id = events[0].user_id if events else "default"
        
        metrics = self.aggregator.aggregate_session(session_id, events, user_id)
        
        return metrics
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session summary
        
        Returns:
            Dict with metrics, events, and insights
        """
        if session_id not in self.sessions:
            return {"status": "not_found", "session_id": session_id}
        
        metrics = self.get_session_metrics(session_id)
        events = self.sessions[session_id]
        
        return {
            "status": "analyzed",
            "session_id": session_id,
            "metrics": metrics.to_dict(),
            "total_events": len(events),
            "analyzed_events": len([e for e in events if e.analyzed]),
            "latest_score": events[-1].scores.overall_score() if events and events[-1].scores else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear feedback data for a session"""
        cleared = False
        
        if session_id in self.sessions:
            del self.sessions[session_id]
            cleared = True
        
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            cleared = True
        
        self.collector.clear_session(session_id)
        
        if cleared:
            api_logger.info(f"Cleared feedback data for session {session_id}")
        
        return cleared
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.sessions.keys())
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service-level statistics"""
        total_sessions = len(self.sessions)
        total_events = sum(len(events) for events in self.sessions.values())
        
        # Quality distribution
        quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for session_id in self.sessions:
            metrics = self.get_session_metrics(session_id)
            quality_dist[metrics.quality_grade] += 1
        
        return {
            "service_name": "FeedbackService",
            "version": "2.0.0-MVP",
            "architecture": "hybrid (pattern + semantic)",
            "active_sessions": total_sessions,
            "total_feedback_events": total_events,
            "quality_distribution": quality_dist,
            "configuration": {
                "use_semantic_analysis": self.use_semantic,
                "semantic_threshold": self.semantic_threshold,
                "pattern_categories": 5
            },
            "timestamp": datetime.now().isoformat()
        }


# Global service instance
_feedback_service_instance: Optional[FeedbackService] = None


def get_feedback_service(config: Optional[Dict[str, Any]] = None) -> FeedbackService:
    """
    Get or create feedback service singleton instance
    
    Args:
        config: Optional configuration
        
    Returns:
        FeedbackService instance
        
    Example:
        # Get default instance
        service = get_feedback_service()
        
        # Get instance with custom config
        service = get_feedback_service({
            "use_semantic_analysis": True,
            "semantic_threshold": 0.6,
            "analysis_model": "gpt-4"
        })
    """
    global _feedback_service_instance
    
    if _feedback_service_instance is None:
        _feedback_service_instance = FeedbackService(config)
        api_logger.info("🔍 Global FeedbackService initialized")
    
    return _feedback_service_instance


def reset_feedback_service():
    """
    Reset the global feedback service instance (useful for testing)
    
    Example:
        # In test teardown
        reset_feedback_service()
    """
    global _feedback_service_instance
    _feedback_service_instance = None

