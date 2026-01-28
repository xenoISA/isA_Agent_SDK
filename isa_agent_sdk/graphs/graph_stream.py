#!/usr/bin/env python3
"""
ChatService Testing Script

Comprehensive testing script for the SmartAgent system through ChatService.
Tests various scenarios including simple chat, tool calls, and autonomous planning.
This approach ensures complete end-to-end testing including prompt parameter handling.
"""

import asyncio
import sys
import os
from pprint import pprint

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from isa_agent_sdk.services.chat_service import ChatService
from isa_agent_sdk.components import SessionService

async def main():
    print("🚀 Testing ChatService End-to-End with Mock HIL")
    print("=" * 80)
    
    # Enable Mock HIL Service for testing
    hil_enabled = enable_mock_hil_for_testing()
    if hil_enabled:
        print("🤖 Mock HIL Service enabled - all HIL requests will be auto-approved")
    else:
        print("⚠️ Failed to enable Mock HIL Service - tests may fail on interrupts")
    print()
    
    # Test scenarios covering all functionality - 全面测试所有可能的事件类型
    test_scenarios = {
        "simple": "what is 2+2",
        "simple2": "What is the capital of France?",
        "web_search": "search for latest AI news",
        "image_generation": "generate a image of a cute cat playing with a ball",
        "weather": "what's the weather in beijing",
        "complex_tool_chain": "search for Python optimization techniques, then generate an image showing code optimization",
        "autonomous": "Create an execution plan to research AI trends, generate a summary report, and search for additional technical details",
        "storytelling": "Write a story about a robot discovering emotions",
        "error_trigger": "use invalid tool parameters to trigger error handling",  # 故意触发错误
        "memory_test": "Remember this: I am testing the chat system memory functionality",
        "billing_test": "perform multiple operations to test billing events",
        "mixed_operations": "first search for machine learning tutorials, then create an image of a neural network diagram, and finally write a summary",
        "summarization_test": "Tell me about AI"  # Test auto-summarization with long conversation
    }
    
    print("\n📋 Comprehensive Test Scenarios:")
    for key, query in test_scenarios.items():
        print(f"  {key}: {query}")
    
    # 运行所有场景或选择特定场景
    import sys
    if len(sys.argv) > 1:
        selected_scenario = sys.argv[1]
        if selected_scenario not in test_scenarios:
            print(f"❌ Unknown scenario: {selected_scenario}")
            print(f"Available scenarios: {', '.join(test_scenarios.keys())}")
            return
    else:
        selected_scenario = "simple"  # 默认测试简单对话
    
    # 使用测试场景中的查询
    selected_query = test_scenarios[selected_scenario]
    
    # Add prompt template parameters for testing prompt enhancement
    prompt_name = None
    prompt_args = None
    
    if selected_scenario in ["image_generation", "complex_tool_chain", "mixed_operations"]:
        prompt_name = "text_to_image_prompt"
        prompt_args = {
            "prompt": "cute cat playing with a colorful ball",
            "style_preset": "photorealistic",
            "quality": "high"
        }
    elif selected_scenario == "storytelling":
        prompt_name = "storytelling_prompt"
        prompt_args = {
            "subject": "a robot discovering emotions for the first time",
            "depth": "deep", 
            "reference_text": "Focus on the internal journey and transformation"
        }
    
    print(f"\n🎯 Running {selected_scenario} test: {selected_query}")
    if prompt_name:
        print(f"📝 Using prompt template: {prompt_name}")
        print(f"📋 Prompt args: {prompt_args}")
    print("=" * 80)
    
    # Initialize required services
    session_service = SessionService()
    
    # Initialize ChatService - this is the proper entry point
    chat_service = ChatService(session_service=session_service)
    await chat_service.service_init()
    
    # Test parameters
    import uuid
    user_id = str(uuid.uuid4())
    thread_id = "test_thread"
    session_id = "test_session"
    
    print(f"👤 User ID: {user_id}")
    print(f"🧵 Thread ID: {thread_id}")
    print(f"📝 Session ID: {session_id}")
    print()
    
    print("📤 Sending chat request through ChatService...")
    print("📊 Monitoring complete end-to-end execution including prompt enhancement...")
    print()
    
    # Execute chat through ChatService and monitor results
    response_count = 0
    all_events = []  # 记录所有事件
    unique_event_types = set()  # 记录唯一事件类型
    thinking_count = 0
    token_count = 0

    try:
        async for response_chunk in chat_service.execute(
            user_input=selected_query,
            session_id=session_id,
            user_id=user_id,
            prompt_name=prompt_name,
            prompt_args=prompt_args
        ):
            response_count += 1

            # 记录事件
            if isinstance(response_chunk, dict) and "type" in response_chunk:
                event_type = response_chunk.get("type", "unknown")
                unique_event_types.add(event_type)

                # Count thinking and token events
                if event_type == "thinking":
                    thinking_count += 1
                elif event_type == "token":
                    token_count += 1

                all_events.append({
                    "index": response_count,
                    "type": event_type,
                    "timestamp": response_chunk.get("timestamp", ""),
                    "content_preview": str(response_chunk.get("content", ""))[:100]
                })
            
            # Handle different types of streaming responses
            if isinstance(response_chunk, dict):
                if "type" in response_chunk:
                    response_type = response_chunk["type"]
                    
                    if response_type == "message":
                        content = response_chunk.get("content", "")
                        print(f"💬 Message [{response_count}]: {content[:100]}{'...' if len(content) > 100 else ''}")
                        
                    elif response_type == "tool_call":
                        tool_name = response_chunk.get("tool_name", "unknown")
                        tool_args = response_chunk.get("args", {})
                        print(f"🔧 Tool Call [{response_count}]: {tool_name}")
                        print(f"   Args: {str(tool_args)[:80]}{'...' if len(str(tool_args)) > 80 else ''}")
                        
                    elif response_type == "tool_result":
                        tool_name = response_chunk.get("tool_name", "unknown")
                        result = response_chunk.get("result", "")
                        print(f"✅ Tool Result [{response_count}]: {tool_name}")
                        print(f"   Result: {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
                        
                    elif response_type == "node_update":
                        node_name = response_chunk.get("node", "unknown")
                        next_action = response_chunk.get("next_action", "")
                        credits = response_chunk.get("credits_used", 0)
                        print(f"📊 Node Update [{response_count}]: {node_name}")
                        if next_action:
                            print(f"   Next: {next_action}")
                        if credits:
                            print(f"   Credits: {credits}")
                            
                    elif response_type == "final_response":
                        final_content = response_chunk.get("content", "")
                        total_cost = response_chunk.get("cost", 0)
                        total_credits = response_chunk.get("credits_used", 0)
                        print(f"🎯 Final Response [{response_count}]:")
                        print(f"   Content: {final_content[:150]}{'...' if len(final_content) > 150 else ''}")
                        print(f"   Cost: ${total_cost:.4f}")
                        print(f"   Credits: {total_credits}")
                        
                    elif response_type == "error":
                        error_msg = response_chunk.get("error", "Unknown error")
                        print(f"❌ Error [{response_count}]: {error_msg}")
                        
                    else:
                        print(f"🔄 Other [{response_count}]: {response_type} - {str(response_chunk)[:80]}...")
                        
                else:
                    # Handle chunks without explicit type
                    print(f"📦 Chunk [{response_count}]: {str(response_chunk)[:100]}...")
            else:
                # Handle non-dict responses
                print(f"📄 Response [{response_count}]: {str(response_chunk)[:100]}...")
                
    except Exception as e:
        print(f"❌ ChatService execution failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()[:300]}...")
    
    print(f"\n✅ ChatService test completed")
    print(f"📊 Total responses processed: {response_count}")
    print(f"💭 Thinking events: {thinking_count}")
    print(f"🔤 Token events: {token_count}")

    # Critical diagnostic
    if thinking_count > 0 and token_count == 0:
        print(f"\n⚠️  WARNING: Found {thinking_count} thinking events but 0 token events!")
        print(f"    This indicates ResponseNode is not streaming tokens properly.")
    elif thinking_count > 0 and token_count > 0:
        print(f"\n✅ GOOD: Both thinking ({thinking_count}) and token ({token_count}) events present.")
    
    # Get HIL interaction summary
    mock_hil = get_mock_hil_service()
    hil_summary = mock_hil.get_interaction_summary()
    
    print(f"\n🤖 Mock HIL Service Summary:")
    print(f"   Total HIL interactions: {hil_summary['total_interactions']}")
    if hil_summary['total_interactions'] > 0:
        print(f"   Question patterns: {hil_summary['question_patterns']}")
        print(f"   Node sources: {hil_summary['node_sources']}")
        print(f"   Last interaction: {hil_summary['interactions'][-1]['question'][:50]}..." if hil_summary['interactions'] else "   None")
    
    # 输出事件统计
    print(f"\n📈 Event Type Analysis:")
    print(f"   Unique event types found: {len(unique_event_types)}")
    for event_type in sorted(unique_event_types):
        count = sum(1 for e in all_events if e["type"] == event_type)
        print(f"   - {event_type}: {count} occurrences")
    
    # 输出详细事件日志
    print(f"\n📋 Detailed Event Log:")
    for event in all_events:
        print(f"   [{event['index']:3d}] {event['type']:15s} | {event['content_preview']}")
    
    # Clear HIL history for next test
    mock_hil.clear_history()
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())