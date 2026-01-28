#!/usr/bin/env python3
"""
Debug script to test failed HIL scenarios directly
"""

import requests
import json
import time

API_BASE = "http://localhost:8080"
TEST_USER_ID = "test_user_001"
API_KEY = "authenticated"


def test_scenario(scenario_name: str, message: str, expected_hil: bool):
    """Test a single scenario and print detailed results"""

    print("\n" + "=" * 70)
    print(f"Testing: {scenario_name}")
    print("=" * 70)

    session_id = f"debug_test_{int(time.time())}"

    url = f"{API_BASE}/api/v1/agents/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "message": message,
        "user_id": TEST_USER_ID,
        "session_id": session_id,
        "stream": True
    }

    print(f"📤 Message: {message}")
    print(f"📋 Session: {session_id}")
    print(f"⚙️  Expected HIL: {expected_hil}\n")

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)

        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}\n")
            return False

        hil_detected = False
        hil_details = []
        tool_results = []

        # Parse SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                event = json.loads(data_str)
                event_type = event.get("type", "")

                # Capture HIL events
                if event_type == "hil.request":
                    hil_detected = True
                    interrupt_data = event.get("metadata", {}).get("interrupt_data", {})
                    hil_details.append({
                        "type": interrupt_data.get("type"),
                        "method": interrupt_data.get("method"),
                        "question": interrupt_data.get("question", "")[:80]
                    })

                # Capture tool results
                if event_type == "tool.result":
                    content = event.get("content", "")
                    tool_results.append(content[:200])

            except json.JSONDecodeError:
                pass

        # Print results
        print("\n" + "-" * 70)
        print("RESULTS:")
        print("-" * 70)

        if expected_hil:
            if hil_detected:
                print("✅ HIL DETECTED (as expected)")
                for detail in hil_details:
                    print(f"   Type: {detail['type']}")
                    print(f"   Method: {detail['method']}")
                    print(f"   Question: {detail['question']}")
            else:
                print("❌ HIL NOT DETECTED (but expected)")
                print("\nTool Results:")
                for i, result in enumerate(tool_results, 1):
                    print(f"   [{i}] {result}")
        else:
            if hil_detected:
                print("❌ HIL DETECTED (but not expected)")
            else:
                print("✅ NO HIL (as expected)")

        print("-" * 70 + "\n")

        return hil_detected == expected_hil

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return False


if __name__ == "__main__":
    print("\n🔍 DEBUG: Testing Failed HIL Scenarios\n")

    results = []

    # Test Scenario 6: Input Augmentation
    results.append((
        "Scenario 6: Input Augmentation",
        test_scenario(
            "Scenario 6: Input Augmentation",
            "Use test_input_augmentation tool to add more project requirements",
            expected_hil=True
        )
    ))

    time.sleep(2)

    # Test Scenario 8: Review Generated Code
    results.append((
        "Scenario 8: Review Generated Code",
        test_scenario(
            "Scenario 8: Review Generated Code",
            "Use test_review_generated_code tool to review payment processing code",
            expected_hil=True
        )
    ))

    time.sleep(2)

    # Test Scenario 12: List All (should NOT trigger HIL)
    results.append((
        "Scenario 12: List All HIL Scenarios",
        test_scenario(
            "Scenario 12: List All HIL Scenarios",
            "Use test_all_hil_scenarios tool to list all available HIL test scenarios",
            expected_hil=False
        )
    ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    passed_count = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {passed_count}/{len(results)} passed\n")
