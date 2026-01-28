"""
Semantic (AI-driven) Feedback Analyzer
File: app/services/feedback/analyzers/semantic_analyzer.py

Deep semantic analysis using LLM for context-aware evaluation.
Migrated from ai_feedback_service.py with optimizations.
"""

import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import (
    FeedbackEvent,
    FeedbackScore,
    SentimentPolarity,
    UserIntent,
    FeedbackDimension
)
from isa_agent_sdk.clients.model_client import get_model_client
from isa_agent_sdk.utils.logger import api_logger


class SemanticAnalyzer:
    """
    AI-driven semantic feedback analyzer
    
    Example:
        analyzer = SemanticAnalyzer()
        
        event = FeedbackEvent(...)
        score = await analyzer.analyze(event, conversation_history=[...])
        
        print(f"Overall: {score.overall_score()}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic analyzer
        
        Args:
            config: Optional configuration
                - analysis_model: Model to use (default: "gpt-4")
                - enable_caching: Cache results (default: True)
        """
        self.config = config or {}
        self.model_client = None
        self.analysis_model = self.config.get("analysis_model", "gpt-4")
        self.enable_caching = self.config.get("enable_caching", True)
        
        # Analysis cache
        self.analysis_cache: Dict[str, FeedbackScore] = {}
        
        api_logger.info("🧠 SemanticAnalyzer initialized with AI-driven analysis")
    
    async def _get_model_client(self):
        """Get model client instance"""
        if not self.model_client:
            self.model_client = await get_model_client()
        return self.model_client
    
    async def analyze(
        self,
        event: FeedbackEvent,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> FeedbackScore:
        """
        Analyze feedback event using AI
        
        Args:
            event: FeedbackEvent to analyze
            conversation_history: Optional conversation context
            
        Returns:
            FeedbackScore with multi-dimensional scores
        """
        # Check cache
        cache_key = self._get_cache_key(event)
        if self.enable_caching and cache_key in self.analysis_cache:
            api_logger.debug("Using cached semantic analysis")
            return self.analysis_cache[cache_key]
        
        try:
            # Build analysis prompt
            prompt = self._build_analysis_prompt(event, conversation_history)
            
            # Call AI model
            model_client = await self._get_model_client()
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response, _ = await model_client.call_model(
                messages=messages,
                model=self.analysis_model,
                timeout=30.0
            )
            
            # Parse AI response
            score = self._parse_ai_response(response.content, event)
            
            # Cache result
            if self.enable_caching:
                self.analysis_cache[cache_key] = score
                self._cleanup_cache()
            
            event.scores = score
            event.analyzed = True
            
            api_logger.debug(f"Semantic analysis complete: overall={score.overall_score():.2f}, confidence={score.confidence:.2f}")
            
            return score
            
        except Exception as e:
            api_logger.error(f"Semantic analysis failed: {e}")
            return self._get_fallback_score()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for AI analysis"""
        return """You are an expert conversation quality analyst specializing in evaluating AI assistant responses.

Analyze the user's feedback and provide scores on these 8 dimensions (0.0-1.0):
- helpfulness: How helpful was the AI's response
- accuracy: Factual correctness and reliability
- clarity: Clear and easy to understand
- safety: Appropriate and harmless content
- completeness: Addressed all aspects of the query
- efficiency: Concise without unnecessary information
- engagement: Interesting and engaging response
- relevance: On-topic and contextual

Also determine:
- sentiment: positive, negative, neutral, or mixed
- intent: appreciation, complaint, question, correction, confirmation, or neutral
- issues: List any specific problems identified
- confidence: Your confidence in this analysis (0.0-1.0)

Return ONLY valid JSON format:
```json
{
    "scores": {
        "helpfulness": 0.8,
        "accuracy": 0.9,
        "clarity": 0.85,
        "safety": 1.0,
        "completeness": 0.75,
        "efficiency": 0.8,
        "engagement": 0.7,
        "relevance": 0.9
    },
    "sentiment": "positive",
    "intent": "appreciation",
    "issues": [],
    "confidence": 0.9
}
```"""
    
    def _build_analysis_prompt(
        self,
        event: FeedbackEvent,
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Build analysis prompt"""
        parts = []
        
        # Current user message
        parts.append(f"**User Message**: {event.user_message}")
        
        # AI response context
        if event.ai_response:
            parts.append(f"**AI Response**: {event.ai_response}")
        
        # Conversation history (last 3 turns)
        if conversation_history:
            history_text = "\n".join([
                f"{turn.get('role', 'unknown')}: {turn.get('content', '')[:200]}"
                for turn in conversation_history[-3:]
            ])
            parts.append(f"**Recent History** (last 3 turns):\n{history_text}")
        
        # Explicit rating if provided
        if event.explicit_rating is not None:
            parts.append(f"**Explicit Rating**: {event.explicit_rating}/1.0")
        
        parts.append("\nAnalyze this feedback and provide scores in JSON format.")
        
        return "\n\n".join(parts)
    
    def _parse_ai_response(self, response_content: str, event: FeedbackEvent) -> FeedbackScore:
        """Parse AI analysis response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            data = json.loads(json_str)
            scores_data = data.get("scores", {})
            
            # Update event metadata
            sentiment_str = data.get("sentiment", "neutral")
            event.sentiment = SentimentPolarity(sentiment_str) if sentiment_str in [s.value for s in SentimentPolarity] else SentimentPolarity.NEUTRAL
            
            intent_str = data.get("intent", "neutral")
            event.intent = UserIntent(intent_str) if intent_str in [i.value for i in UserIntent] else UserIntent.NEUTRAL
            
            event.issues_identified = data.get("issues", [])
            
            # Create FeedbackScore
            score = FeedbackScore(
                helpfulness=float(scores_data.get("helpfulness", 0.5)),
                accuracy=float(scores_data.get("accuracy", 0.5)),
                clarity=float(scores_data.get("clarity", 0.5)),
                safety=float(scores_data.get("safety", 1.0)),
                completeness=float(scores_data.get("completeness", 0.5)),
                efficiency=float(scores_data.get("efficiency", 0.5)),
                engagement=float(scores_data.get("engagement", 0.5)),
                relevance=float(scores_data.get("relevance", 0.5)),
                confidence=float(data.get("confidence", 0.8))
            )
            
            return score
            
        except Exception as e:
            api_logger.warning(f"Failed to parse AI response: {e}")
            return self._get_fallback_score()
    
    def _get_cache_key(self, event: FeedbackEvent) -> str:
        """Generate cache key for event"""
        return f"{event.session_id}:{event.turn_id}:{hash(event.user_message)}"
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        if len(self.analysis_cache) > 500:
            # Keep most recent 300
            items = list(self.analysis_cache.items())
            self.analysis_cache = dict(items[-300:])
            api_logger.debug("Cleaned up semantic analysis cache")
    
    def _get_fallback_score(self) -> FeedbackScore:
        """Get fallback score on error"""
        return FeedbackScore(
            helpfulness=0.5,
            accuracy=0.5,
            clarity=0.5,
            safety=1.0,
            completeness=0.5,
            efficiency=0.5,
            engagement=0.5,
            relevance=0.5,
            confidence=0.2  # Low confidence for fallback
        )
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        api_logger.info("Cleared semantic analysis cache")

