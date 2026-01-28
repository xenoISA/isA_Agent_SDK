"""
MVP Test Script for Feedback Service
File: app/services/feedback/tests/test_feedback_service_mvp.py

Simple test to verify MVP works correctly.

Usage:
    python app/services/feedback/tests/test_feedback_service_mvp.py
"""

import sys
import asyncio
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/Users/xenodennis/Documents/Fun/isA_Agent')

from isa_agent_sdk.services.feedback import (
    get_feedback_service,
    reset_feedback_service,
    FeedbackType,
    SentimentPolarity
)


async def test_basic_feedback_collection():
    """Test basic feedback collection"""
    print("\n" + "="*60)
    print("Test 1: Basic Feedback Collection")
    print("="*60)
    
    reset_feedback_service()
    service = get_feedback_service({"use_semantic_analysis": False})  # Pattern-only for speed
    
    # Collect positive feedback
    event = await service.process_user_message(
        session_id="test_session_1",
        user_input="Thanks, that was really helpful!",
        ai_response="Glad I could help!",
        turn_id=1
    )
    
    assert event.analyzed, "Event should be analyzed"
    assert event.scores is not None, "Scores should be calculated"
    print(f"✓ Positive feedback collected")
    print(f"  - Overall score: {event.scores.overall_score():.2f}")
    print(f"  - Helpfulness: {event.scores.helpfulness:.2f}")
    print(f"  - Sentiment: {event.sentiment.value if event.sentiment else 'N/A'}")


async def test_negative_feedback():
    """Test negative feedback detection"""
    print("\n" + "="*60)
    print("Test 2: Negative Feedback Detection")
    print("="*60)
    
    service = get_feedback_service()
    
    event = await service.process_user_message(
        session_id="test_session_2",
        user_input="That's wrong and not helpful at all",
        ai_response="Let me try again...",
        turn_id=1
    )
    
    assert event.scores.overall_score() < 0.5, "Negative feedback should have low score"
    print(f"✓ Negative feedback detected")
    print(f"  - Overall score: {event.scores.overall_score():.2f}")
    print(f"  - Helpfulness: {event.scores.helpfulness:.2f}")
    print(f"  - Issues: {event.issues_identified}")


async def test_session_metrics():
    """Test session metrics aggregation"""
    print("\n" + "="*60)
    print("Test 3: Session Metrics Aggregation")
    print("="*60)
    
    service = get_feedback_service()
    
    # Multiple feedback events
    session_id = "test_session_3"
    
    messages = [
        ("How do I install Python?", "To install Python, visit python.org and download..."),
        ("Thanks, that's helpful!", "You're welcome!"),
        ("What about pip?", "Pip is Python's package manager..."),
        ("Perfect, got it!", "Great! Anything else?")
    ]
    
    for i, (user_msg, ai_msg) in enumerate(messages):
        await service.process_user_message(
            session_id=session_id,
            user_input=user_msg,
            ai_response=ai_msg,
            turn_id=i+1
        )
    
    # Get metrics
    metrics = service.get_session_metrics(session_id)
    
    assert metrics.total_turns > 0, "Should have turns"
    assert metrics.average_score > 0, "Should have average score"
    print(f"✓ Session metrics calculated")
    print(f"  - Total turns: {metrics.total_turns}")
    print(f"  - Average score: {metrics.average_score:.2f}")
    print(f"  - Quality grade: {metrics.quality_grade}")
    print(f"  - Recommendations: {metrics.recommendations[:2]}")


async def test_explicit_rating():
    """Test explicit rating support"""
    print("\n" + "="*60)
    print("Test 4: Explicit Rating Support")
    print("="*60)
    
    service = get_feedback_service()
    
    event = await service.process_user_message(
        session_id="test_session_4",
        user_input="User gave 5 stars",
        ai_response="Thank you for the rating!",
        turn_id=1,
        explicit_rating=1.0  # 5/5 stars = 1.0
    )
    
    assert event.explicit_rating == 1.0, "Explicit rating should be stored"
    print(f"✓ Explicit rating processed")
    print(f"  - Explicit rating: {event.explicit_rating}")
    print(f"  - Overall score: {event.scores.overall_score():.2f}")


async def test_service_stats():
    """Test service statistics"""
    print("\n" + "="*60)
    print("Test 5: Service Statistics")
    print("="*60)
    
    service = get_feedback_service()
    
    stats = service.get_service_stats()
    
    assert "service_name" in stats, "Should have service name"
    assert "active_sessions" in stats, "Should have active sessions count"
    print(f"✓ Service stats retrieved")
    print(f"  - Service: {stats['service_name']} v{stats['version']}")
    print(f"  - Architecture: {stats['architecture']}")
    print(f"  - Active sessions: {stats['active_sessions']}")
    print(f"  - Total events: {stats['total_feedback_events']}")
    print(f"  - Quality distribution: {stats['quality_distribution']}")


async def test_multi_dimensional_scoring():
    """Test multi-dimensional scoring"""
    print("\n" + "="*60)
    print("Test 6: Multi-Dimensional Scoring")
    print("="*60)
    
    service = get_feedback_service()
    
    event = await service.process_user_message(
        session_id="test_session_6",
        user_input="Thanks! Very clear and complete explanation.",
        ai_response="Happy to help!",
        turn_id=1
    )
    
    scores = event.scores
    print(f"✓ Multi-dimensional scores calculated")
    print(f"  - Helpfulness: {scores.helpfulness:.2f}")
    print(f"  - Accuracy: {scores.accuracy:.2f}")
    print(f"  - Clarity: {scores.clarity:.2f}")
    print(f"  - Completeness: {scores.completeness:.2f}")
    print(f"  - Engagement: {scores.engagement:.2f}")
    print(f"  - Overall: {scores.overall_score():.2f}")
    print(f"  - Confidence: {scores.confidence:.2f}")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Feedback Service MVP - Test Suite")
    print("="*60)
    
    try:
        await test_basic_feedback_collection()
        await test_negative_feedback()
        await test_session_metrics()
        await test_explicit_rating()
        await test_service_stats()
        await test_multi_dimensional_scoring()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
        print("\nSummary:")
        print("  - Basic feedback collection: ✓")
        print("  - Negative feedback detection: ✓")
        print("  - Session metrics aggregation: ✓")
        print("  - Explicit rating support: ✓")
        print("  - Service statistics: ✓")
        print("  - Multi-dimensional scoring: ✓")
        print("\nThe MVP feedback service is working correctly!")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

