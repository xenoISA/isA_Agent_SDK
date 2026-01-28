"""
Pattern-based Feedback Analyzer
File: app/services/feedback/analyzers/pattern_analyzer.py

Fast, rule-based feedback analysis using keyword patterns.
Migrated from feedback_service.py with enhancements.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import (
    FeedbackEvent,
    FeedbackScore,
    SentimentPolarity,
    UserIntent
)
from isa_agent_sdk.utils.logger import api_logger


class FeedbackPatterns:
    """Centralized feedback pattern definitions"""
    
    SATISFACTION = [
        'thanks', 'thank you', 'perfect', 'exactly', 'great', 'awesome',
        'helpful', 'works', 'correct', 'good', 'nice', 'excellent', 
        'wonderful', 'amazing', 'brilliant', 'fantastic', 'superb',
        'appreciate', 'love', 'impressive'
    ]
    
    DISSATISFACTION = [
        'wrong', 'incorrect', 'not helpful', 'unclear', 'confusing',
        'bad', 'terrible', 'useless', 'doesn\'t work', 'failed',
        'disappointing', 'frustrating', 'awful', 'horrible', 'poor',
        'broken', 'error', 'bug'
    ]
    
    CLARIFICATION = [
        'what', 'how', 'why', 'can you explain', 'clarify', 'what do you mean',
        'i don\'t understand', 'unclear', 'confused', 'elaborate',
        'can you help me understand', 'what does that mean', 'please explain'
    ]
    
    FOLLOW_UP = [
        'also', 'additionally', 'furthermore', 'and', 'plus', 'next',
        'what about', 'how about', 'can you also', 'another question',
        'one more thing', 'while we\'re at it'
    ]
    
    CORRECTION = [
        'actually', 'no', 'not quite', 'incorrect', 'wrong', 'fix',
        'should be', 'meant to', 'correction', 'that\'s not right'
    ]


class PatternAnalyzer:
    """
    Fast pattern-based feedback analyzer
    
    Example:
        analyzer = PatternAnalyzer()
        
        event = FeedbackEvent(...)
        score = analyzer.analyze(event)
        
        print(f"Helpfulness: {score.helpfulness}")
        print(f"Overall: {score.overall_score()}")
    """
    
    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize pattern analyzer
        
        Args:
            custom_patterns: Optional custom pattern dictionaries
        """
        self.patterns = {
            'satisfaction': FeedbackPatterns.SATISFACTION,
            'dissatisfaction': FeedbackPatterns.DISSATISFACTION,
            'clarification': FeedbackPatterns.CLARIFICATION,
            'follow_up': FeedbackPatterns.FOLLOW_UP,
            'correction': FeedbackPatterns.CORRECTION
        }
        
        # Merge custom patterns
        if custom_patterns:
            for key, patterns in custom_patterns.items():
                if key in self.patterns:
                    self.patterns[key].extend(patterns)
                else:
                    self.patterns[key] = patterns
        
        api_logger.info("🔍 PatternAnalyzer initialized with pattern-based analysis")
    
    def analyze(self, event: FeedbackEvent) -> FeedbackScore:
        """
        Analyze feedback event using patterns
        
        Args:
            event: FeedbackEvent to analyze
            
        Returns:
            FeedbackScore with multi-dimensional scores
        """
        if not event.user_message:
            return self._get_default_score()
        
        user_text = event.user_message.lower()
        
        # Count pattern matches
        pattern_counts = self._count_patterns(user_text)
        
        # Calculate dimension scores
        helpfulness = self._calculate_helpfulness(pattern_counts, user_text)
        accuracy = self._calculate_accuracy(pattern_counts, user_text)
        clarity = self._calculate_clarity(pattern_counts, user_text)
        safety = 1.0  # Pattern-based can't determine safety well
        completeness = self._calculate_completeness(pattern_counts, user_text)
        efficiency = self._calculate_efficiency(pattern_counts, user_text)
        engagement = self._calculate_engagement(pattern_counts, user_text)
        relevance = self._calculate_relevance(pattern_counts, user_text)
        
        # Calculate confidence based on pattern strength
        confidence = self._calculate_confidence(pattern_counts)
        
        # Determine sentiment and intent
        event.sentiment = self._determine_sentiment(pattern_counts)
        event.intent = self._determine_intent(pattern_counts)
        
        # Identify issues
        event.issues_identified = self._identify_issues(pattern_counts)
        
        score = FeedbackScore(
            helpfulness=helpfulness,
            accuracy=accuracy,
            clarity=clarity,
            safety=safety,
            completeness=completeness,
            efficiency=efficiency,
            engagement=engagement,
            relevance=relevance,
            confidence=confidence
        )
        
        event.scores = score
        event.analyzed = True
        
        api_logger.debug(f"Pattern analysis complete: overall={score.overall_score():.2f}, confidence={confidence:.2f}")
        
        return score
    
    def _count_patterns(self, text: str) -> Dict[str, int]:
        """Count pattern matches in text"""
        counts = {}
        
        for pattern_type, patterns in self.patterns.items():
            count = sum(1 for pattern in patterns if pattern in text)
            counts[pattern_type] = min(count, 3)  # Cap at 3
        
        # Count questions
        counts['questions'] = text.count('?')
        
        return counts
    
    def _calculate_helpfulness(self, counts: Dict[str, int], text: str) -> float:
        """Calculate helpfulness score (0.0-1.0)"""
        base = 0.5
        
        # Positive indicators
        if counts['satisfaction'] > 0:
            base += 0.3 * min(counts['satisfaction'] / 2, 1.0)
        
        # Negative indicators
        if counts['dissatisfaction'] > 0:
            base -= 0.4 * min(counts['dissatisfaction'] / 2, 1.0)
        
        # Clarification needed (slightly negative)
        if counts['clarification'] > 0 and counts['dissatisfaction'] == 0:
            base -= 0.1 * min(counts['clarification'] / 2, 1.0)
        
        return max(0.0, min(1.0, base))
    
    def _calculate_accuracy(self, counts: Dict[str, int], text: str) -> float:
        """Calculate accuracy score (0.0-1.0)"""
        base = 0.5
        
        # Corrections indicate inaccuracy
        if counts['correction'] > 0:
            base -= 0.5 * min(counts['correction'] / 2, 1.0)
        
        # Dissatisfaction might indicate inaccuracy
        if counts['dissatisfaction'] > 0:
            base -= 0.2 * min(counts['dissatisfaction'] / 2, 1.0)
        
        # Satisfaction indicates accuracy
        if counts['satisfaction'] > 0:
            base += 0.2 * min(counts['satisfaction'] / 2, 1.0)
        
        return max(0.0, min(1.0, base))
    
    def _calculate_clarity(self, counts: Dict[str, int], text: str) -> float:
        """Calculate clarity score (0.0-1.0)"""
        base = 0.5
        
        # Clarification requests indicate unclear response
        if counts['clarification'] > 0:
            base -= 0.3 * min(counts['clarification'] / 2, 1.0)
        
        # Satisfaction often correlates with clarity
        if counts['satisfaction'] > 0:
            base += 0.2 * min(counts['satisfaction'] / 2, 1.0)
        
        return max(0.0, min(1.0, base))
    
    def _calculate_completeness(self, counts: Dict[str, int], text: str) -> float:
        """Calculate completeness score (0.0-1.0)"""
        base = 0.5
        
        # Follow-up questions might indicate incompleteness
        if counts['follow_up'] > 0:
            base -= 0.15 * min(counts['follow_up'] / 2, 1.0)
        
        # Multiple questions might indicate incompleteness
        if counts['questions'] > 1:
            base -= 0.1 * min((counts['questions'] - 1) / 2, 1.0)
        
        # Satisfaction suggests completeness
        if counts['satisfaction'] > 0:
            base += 0.15 * min(counts['satisfaction'] / 2, 1.0)
        
        return max(0.0, min(1.0, base))
    
    def _calculate_efficiency(self, counts: Dict[str, int], text: str) -> float:
        """Calculate efficiency score (0.0-1.0)"""
        base = 0.6  # Assume generally efficient
        
        # Long clarification sequences indicate inefficiency
        if counts['clarification'] > 1:
            base -= 0.2 * min(counts['clarification'] / 3, 1.0)
        
        # Quick satisfaction indicates efficiency
        if counts['satisfaction'] > 0 and len(text.split()) < 10:
            base += 0.2
        
        return max(0.0, min(1.0, base))
    
    def _calculate_engagement(self, counts: Dict[str, int], text: str) -> float:
        """Calculate engagement score (0.0-1.0) - Grok-inspired"""
        base = 0.5
        
        # Follow-ups indicate engagement
        if counts['follow_up'] > 0:
            base += 0.2 * min(counts['follow_up'] / 2, 1.0)
        
        # Satisfaction indicates engagement
        if counts['satisfaction'] > 0:
            base += 0.2 * min(counts['satisfaction'] / 2, 1.0)
        
        # Message length as engagement signal
        word_count = len(text.split())
        if word_count > 20:
            base += 0.1
        
        return max(0.0, min(1.0, base))
    
    def _calculate_relevance(self, counts: Dict[str, int], text: str) -> float:
        """Calculate relevance score (0.0-1.0)"""
        base = 0.6  # Assume generally relevant
        
        # Satisfaction indicates relevance
        if counts['satisfaction'] > 0:
            base += 0.2 * min(counts['satisfaction'] / 2, 1.0)
        
        # Dissatisfaction might indicate irrelevance
        if counts['dissatisfaction'] > 0:
            base -= 0.2 * min(counts['dissatisfaction'] / 2, 1.0)
        
        return max(0.0, min(1.0, base))
    
    def _calculate_confidence(self, counts: Dict[str, int]) -> float:
        """Calculate confidence in analysis (0.0-1.0)"""
        # More pattern matches = higher confidence
        total_matches = sum(counts.values())
        
        if total_matches == 0:
            return 0.3  # Low confidence with no patterns
        elif total_matches >= 5:
            return 0.9  # High confidence with many patterns
        else:
            return 0.5 + (total_matches * 0.08)  # Scale between
    
    def _determine_sentiment(self, counts: Dict[str, int]) -> SentimentPolarity:
        """Determine sentiment polarity"""
        sat_score = counts['satisfaction']
        dissat_score = counts['dissatisfaction']
        
        if sat_score > dissat_score:
            return SentimentPolarity.POSITIVE
        elif dissat_score > sat_score:
            return SentimentPolarity.NEGATIVE
        elif sat_score > 0 and dissat_score > 0:
            return SentimentPolarity.MIXED
        else:
            return SentimentPolarity.NEUTRAL
    
    def _determine_intent(self, counts: Dict[str, int]) -> UserIntent:
        """Determine user intent"""
        # Find dominant pattern
        intent_scores = {
            UserIntent.APPRECIATION: counts['satisfaction'],
            UserIntent.COMPLAINT: counts['dissatisfaction'],
            UserIntent.QUESTION: counts['clarification'] + counts['questions'],
            UserIntent.CORRECTION: counts['correction'],
            UserIntent.CONFIRMATION: counts['follow_up']
        }
        
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if max_intent[1] == 0:
            return UserIntent.NEUTRAL
        else:
            return max_intent[0]
    
    def _identify_issues(self, counts: Dict[str, int]) -> List[str]:
        """Identify specific issues from patterns"""
        issues = []
        
        if counts['dissatisfaction'] >= 2:
            issues.append("User expressed dissatisfaction")
        
        if counts['clarification'] >= 2:
            issues.append("Response unclear, multiple clarifications needed")
        
        if counts['correction'] >= 1:
            issues.append("User corrected AI response")
        
        if counts['questions'] >= 3:
            issues.append("Many follow-up questions")
        
        return issues
    
    def _get_default_score(self) -> FeedbackScore:
        """Get default score when no input"""
        return FeedbackScore(
            helpfulness=0.5,
            accuracy=0.5,
            clarity=0.5,
            safety=1.0,
            completeness=0.5,
            efficiency=0.5,
            engagement=0.5,
            relevance=0.5,
            confidence=0.1
        )

