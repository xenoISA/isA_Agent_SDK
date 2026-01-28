#!/usr/bin/env python3
"""
Automated Plan Tools HIL Integration Test (Non-Interactive)
This version can run in CI/CD without user input
"""

import requests
import json
import time

# Configuration
API_BASE = "http://localhost:8080"
TEST_USER_ID = "test_plan_hil_auto"
API_KEY = "authenticated"


def test_agent_health():
    """Test 1: Check if agent service is healthy"""
    print("\n" + "="*80)
    print("TEST 1: Agent Service Health Check")
    print("="*80)

    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Agent service is healthy")
            print(f"   Response: {response.text[:100]}")
            return True
        else:
            print(f"❌ Agent health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to agent: {e}")
        return False


def test_mcp_connectivity():
    """Test 2: Check MCP server connectivity from agent container"""
    print("\n" + "="*80)
    print("TEST 2: MCP Server Connectivity")
    print("="*80)

    try:
        # Try from host
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code == 200:
            print("✅ MCP server accessible from host")
            data = response.json()
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Tools: {data.get('capabilities', {}).get('tools', 0)}")
            return True
        else:
            print(f"❌ MCP health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to MCP server: {e}")
        return False


def test_simple_chat():
    """Test 3: Send a simple chat message without HIL"""
    print("\n" + "="*80)
    print("TEST 3: Simple Chat (No HIL)")
    print("="*80)

    url = f"{API_BASE}/api/v1/agents/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "message": "Hello! What can you help me with?",
        "user_id": TEST_USER_ID,
        "session_id": f"test_simple_{int(time.time())}",
        "stream": False  # Non-streaming for simplicity
    }

    print(f"📤 Sending: {payload['message']}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            print("✅ Chat request successful")
            data = response.json()
            print(f"   Response preview: {str(data)[:200]}...")
            return True
        else:
            print(f"❌ Chat request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat request error: {e}")
        return False


def test_plan_creation_detection():
    """Test 4: Send plan creation request and detect HIL trigger"""
    print("\n" + "="*80)
    print("TEST 4: Plan Creation HIL Detection")
    print("="*80)

    url = f"{API_BASE}/api/v1/agents/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    message = """Create a simple execution plan with 2 tasks:
1. Check code quality
2. Run tests

Please use create_execution_plan tool."""

    payload = {
        "message": message,
        "user_id": TEST_USER_ID,
        "session_id": f"test_plan_{int(time.time())}",
        "stream": True
    }

    print(f"📤 Sending: {message[:80]}...")
    print("📡 Monitoring for HIL request_review event...")

    hil_detected = False
    hil_type = None

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)

        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            return False

        # Monitor SSE stream for HIL event
        event_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                event = json.loads(data_str)
                event_type = event.get("type", "")
                event_count += 1

                # Look for HIL request
                if event_type == "hil.request":
                    hil_detected = True
                    metadata = event.get("metadata", {})
                    interrupt_data = metadata.get("interrupt_data", {})
                    hil_method = interrupt_data.get("method", "")
                    content_type = interrupt_data.get("content_type", "")
                    hil_type = f"{hil_method}/{content_type}"

                    print(f"\n🔔 HIL DETECTED!")
                    print(f"   Method: {hil_method}")
                    print(f"   Content Type: {content_type}")

                    # Show plan details
                    plan_content = interrupt_data.get("content", {})
                    if isinstance(plan_content, dict):
                        print(f"   Plan ID: {plan_content.get('plan_id', 'N/A')}")
                        print(f"   Total Tasks: {plan_content.get('total_tasks', 'N/A')}")
                        print(f"   Execution Mode: {plan_content.get('execution_mode', 'N/A')}")

                        tasks = plan_content.get("tasks", [])
                        if tasks:
                            print(f"   Tasks:")
                            for i, task in enumerate(tasks, 1):
                                print(f"      {i}. {task.get('title', 'Untitled')}")

                # Print event types for debugging
                if event_count <= 10:
                    print(f"   Event {event_count}: {event_type}")

            except json.JSONDecodeError:
                pass

        print(f"\n📊 Total events received: {event_count}")

        if hil_detected:
            print(f"✅ HIL request_review triggered successfully!")
            print(f"   Type: {hil_type}")
            return True
        else:
            print("⚠️  HIL was not triggered (may need different prompt)")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all automated tests"""
    print("\n" + "🧪"*40)
    print("  AUTOMATED Plan Tools HIL Integration Tests")
    print("  (Non-Interactive Version for CI/CD)")
    print("🧪"*40)

    print(f"\n📋 Configuration:")
    print(f"   API Base: {API_BASE}")
    print(f"   User ID: {TEST_USER_ID}")

    results = []

    # Run tests
    tests = [
        ("Agent Health", test_agent_health),
        ("MCP Connectivity", test_mcp_connectivity),
        ("Simple Chat", test_simple_chat),
        ("Plan HIL Detection", test_plan_creation_detection),
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

        time.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "📊"*40)
    print("  TEST SUMMARY")
    print("📊"*40)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} - {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n  Total: {passed_count}/{total_count} passed")
    print("="*80 + "\n")

    # Exit code for CI/CD
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
