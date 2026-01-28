"""
Feedback Aggregator
File: app/services/feedback/aggregator.py

Aggregates feedback events into session and temporal metrics.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import (
    FeedbackEvent,
    SessionMetrics,
    FeedbackScore
)
from isa_agent_sdk.utils.logger import api_logger


class FeedbackAggregator:
    """
    Aggregates feedback into meaningful metrics
    
    Example:
        aggregator = FeedbackAggregator()
        
        # Aggregate session
        metrics = aggregator.aggregate_session(
            session_id="session_123",
            events=[event1, event2, event3]
        )
        
        print(f"Quality: {metrics.quality_grade}")
        print(f"Avg Score: {metrics.average_score}")
    """
    
    def aggregate_session(
        self,
        session_id: str,
        events: List[FeedbackEvent],
        user_id: str = "default"
    ) -> SessionMetrics:
        """
        Aggregate feedback for a single session
        
        Args:
            session_id: Session identifier
            events: List of feedback events
            user_id: User identifier
            
        Returns:
            SessionMetrics object
        """
        if not events:
            return self._get_empty_metrics(session_id, user_id)
        
        # Count turns
        user_turns = len([e for e in events if e.user_message])
        ai_turns = len([e for e in events if e.ai_response])
        total_turns = max(user_turns, ai_turns)
        
        # Calculate scores
        analyzed_events = [e for e in events if e.analyzed and e.scores]
        
        if not analyzed_events:
            return self._get_empty_metrics(session_id, user_id)
        
        # Average overall score
        average_score = sum(e.scores.overall_score() for e in analyzed_events) / len(analyzed_events)
        
        # Dimension averages
        dimension_averages = self._calculate_dimension_averages(analyzed_events)
        
        # Quality grade
        quality_grade = self._calculate_quality_grade(average_score)
        
        # Satisfaction rate (% of positive feedback)
        satisfaction_rate = self._calculate_satisfaction_rate(analyzed_events)
        
        # Score trend
        score_trend = self._calculate_trend([e.scores.overall_score() for e in analyzed_events])
        
        # Implicit signals summary
        regeneration_rate = 0.0
        abandonment_rate = 0.0
        avg_response_time = 0.0
        
        events_with_signals = [e for e in events if e.implicit_signals]
        if events_with_signals:
            total_regenerations = sum(
                e.implicit_signals.regeneration_count
                for e in events_with_signals
            )
            regeneration_rate = total_regenerations / max(total_turns, 1)
            
            abandonment_count = sum(
                1 for e in events_with_signals
                if e.implicit_signals.abandonment
            )
            abandonment_rate = abandonment_count / len(events_with_signals)
            
            response_times = [
                e.implicit_signals.time_between_messages
                for e in events_with_signals
                if e.implicit_signals.time_between_messages
            ]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        # Common issues
        common_issues = self._get_common_issues(analyzed_events)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            average_score,
            quality_grade,
            dimension_averages,
            common_issues
        )
        
        # Time tracking
        start_time = min(e.timestamp for e in events)
        end_time = max(e.timestamp for e in events)
        duration = (end_time - start_time).total_seconds()
        
        metrics = SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            total_turns=total_turns,
            user_turns=user_turns,
            ai_turns=ai_turns,
            feedback_events=len(events),
            average_score=average_score,
            dimension_averages=dimension_averages,
            score_trend=score_trend,
            quality_grade=quality_grade,
            satisfaction_rate=satisfaction_rate,
            regeneration_rate=regeneration_rate,
            abandonment_rate=abandonment_rate,
            avg_response_time=avg_response_time,
            common_issues=common_issues,
            recommendations=recommendations,
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
        
        api_logger.info(f"Aggregated session {session_id}: quality={quality_grade}, avg_score={average_score:.2f}")
        
        return metrics
    
    def _calculate_dimension_averages(self, events: List[FeedbackEvent]) -> Dict[str, float]:
        """Calculate average score for each dimension"""
        dimensions = {}
        
        if not events:
            return dimensions
        
        # Sum all dimension scores
        dimension_sums = {
            'helpfulness': 0.0,
            'accuracy': 0.0,
            'clarity': 0.0,
            'safety': 0.0,
            'completeness': 0.0,
            'efficiency': 0.0,
            'engagement': 0.0,
            'relevance': 0.0
        }
        
        for event in events:
            if event.scores:
                dimension_sums['helpfulness'] += event.scores.helpfulness
                dimension_sums['accuracy'] += event.scores.accuracy
                dimension_sums['clarity'] += event.scores.clarity
                dimension_sums['safety'] += event.scores.safety
                dimension_sums['completeness'] += event.scores.completeness
                dimension_sums['efficiency'] += event.scores.efficiency
                dimension_sums['engagement'] += event.scores.engagement
                dimension_sums['relevance'] += event.scores.relevance
        
        # Calculate averages
        count = len(events)
        for dim, total in dimension_sums.items():
            dimensions[dim] = total / count
        
        return dimensions
    
    def _calculate_quality_grade(self, average_score: float) -> str:
        """Calculate quality grade from average score"""
        if average_score >= 0.8:
            return "excellent"
        elif average_score >= 0.6:
            return "good"
        elif average_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_satisfaction_rate(self, events: List[FeedbackEvent]) -> float:
        """Calculate satisfaction rate (% positive feedback)"""
        if not events:
            return 0.5
        
        positive_count = sum(
            1 for e in events
            if e.scores and e.scores.overall_score() >= 0.6
        )
        
        return positive_count / len(events)
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate score trend (improving, declining, stable)"""
        if len(scores) < 3:
            return "stable"
        
        # Compare recent vs earlier
        recent_avg = sum(scores[-3:]) / 3
        earlier_avg = sum(scores[:-3]) / max(1, len(scores) - 3)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _get_common_issues(self, events: List[FeedbackEvent]) -> List[str]:
        """Get common issues from feedback events"""
        issue_counts: Dict[str, int] = {}
        
        for event in events:
            for issue in event.issues_identified:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return issues that appear 2+ times, sorted by frequency
        common = [
            issue for issue, count in issue_counts.items()
            if count >= 2
        ]
        
        return sorted(common, key=lambda x: issue_counts[x], reverse=True)[:5]
    
    def _generate_recommendations(
        self,
        average_score: float,
        quality_grade: str,
        dimension_averages: Dict[str, float],
        common_issues: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall quality recommendations
        if quality_grade == "poor":
            recommendations.append("⚠️ Overall quality is poor - immediate attention needed")
        elif quality_grade == "fair":
            recommendations.append("📊 Room for improvement in response quality")
        elif quality_grade == "excellent":
            recommendations.append("✅ Excellent quality maintained - continue current approach")
        
        # Dimension-specific recommendations
        if dimension_averages:
            lowest_dim = min(dimension_averages.items(), key=lambda x: x[1])
            if lowest_dim[1] < 0.5:
                recommendations.append(f"🎯 Focus on improving {lowest_dim[0]} (score: {lowest_dim[1]:.2f})")
            
            highest_dim = max(dimension_averages.items(), key=lambda x: x[1])
            if highest_dim[1] >= 0.8:
                recommendations.append(f"⭐ {highest_dim[0].capitalize()} is a strength (score: {highest_dim[1]:.2f})")
        
        # Issue-specific recommendations
        if common_issues:
            recommendations.append(f"🔍 Address common issues: {', '.join(common_issues[:3])}")
        
        return recommendations
    
    def _get_empty_metrics(self, session_id: str, user_id: str) -> SessionMetrics:
        """Get empty metrics for session with no data"""
        return SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            total_turns=0,
            user_turns=0,
            ai_turns=0,
            feedback_events=0,
            average_score=0.5,
            dimension_averages={},
            score_trend="stable",
            quality_grade="fair",
            satisfaction_rate=0.5,
            regeneration_rate=0.0,
            abandonment_rate=0.0,
            avg_response_time=0.0,
            common_issues=[],
            recommendations=["No feedback data available yet"]
        )

