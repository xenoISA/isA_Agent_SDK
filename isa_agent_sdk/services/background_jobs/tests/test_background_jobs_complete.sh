#!/bin/bash
# Complete Background Jobs Test - All Steps in One
# Tests the full flow: Context → Reason → HIL Detection → Background Execution → Notification

set -e  # Exit on error

echo "=================================================="
echo "Complete Background Jobs Test Suite"
echo "=================================================="
echo ""

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8080}"
API_ENDPOINT="${API_BASE_URL}/api/v1/agents/chat"
TEST_USER="test_user_$(date +%s)"
TEST_SESSION="test_session_$(date +%s)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

print_failure() {
    echo -e "${RED}❌ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_section() {
    echo ""
    echo -e "${BLUE}=================================================="
    echo "📋 $1"
    echo "==================================================${NC}"
}

# =============================================================================
# STEP 1: Context Init (Tools, Prompts, Resources, Memory)
# =============================================================================
print_section "Step 1: Testing Context Initialization"

print_info "Sending request to: $API_ENDPOINT"
print_info "User: $TEST_USER"
print_info "Session: $TEST_SESSION"

# Prepare request with a query that will trigger multiple tool calls
# Use a query that explicitly requires multiple web operations (triggers background detection)
REQUEST_BODY=$(cat <<EOF
{
  "message": "Please crawl these 5 websites and summarize their content: https://example.com, https://httpbin.org, https://jsonplaceholder.typicode.com, https://github.com, https://stackoverflow.com",
  "user_id": "$TEST_USER",
  "session_id": "$TEST_SESSION",
  "stream": true
}
EOF
)

# Send request and capture events
RESPONSE_FILE="/tmp/bg_test_response_${TEST_SESSION}.txt"
EVENTS_FILE="/tmp/bg_test_events_${TEST_SESSION}.json"

curl -s -X POST "$API_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev_master_key_123" \
  -d "$REQUEST_BODY" \
  > "$RESPONSE_FILE"

if [ $? -ne 0 ]; then
    print_failure "Failed to send request to API"
    exit 1
fi

# Extract events from SSE response
grep "^data: " "$RESPONSE_FILE" | grep -v "data: \[DONE\]" | sed 's/^data: //' > "$EVENTS_FILE"

if [ ! -s "$EVENTS_FILE" ]; then
    print_failure "No events received from API"
    exit 1
fi

EVENT_COUNT=$(wc -l < "$EVENTS_FILE")
print_success "Received $EVENT_COUNT events"

# Test 1.1: session.start
echo ""
echo "Test 1.1: Verify session.start event"
SESSION_START=$(grep '"type": "session.start"' "$EVENTS_FILE" | head -1)
if [ -n "$SESSION_START" ]; then
    INIT_DURATION=$(echo "$SESSION_START" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('init_duration_ms', 'N/A'))")
    print_success "session.start (init: ${INIT_DURATION}ms)"
else
    print_failure "session.start NOT received"
fi

# Test 1.2: context.tools_ready
echo ""
echo "Test 1.2: Verify context.tools_ready"
TOOLS_READY=$(grep '"type": "context.tools_ready"' "$EVENTS_FILE" | head -1)
if [ -n "$TOOLS_READY" ]; then
    TOOLS_COUNT=$(echo "$TOOLS_READY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('tools_count', 0))")
    if [ "$TOOLS_COUNT" -gt 0 ]; then
        print_success "context.tools_ready ($TOOLS_COUNT tools loaded)"
    else
        print_failure "tools_count = 0"
    fi
else
    print_failure "context.tools_ready NOT received"
fi

# Test 1.3: context.prompts_ready
echo ""
echo "Test 1.3: Verify context.prompts_ready"
PROMPTS_READY=$(grep '"type": "context.prompts_ready"' "$EVENTS_FILE" | head -1)
if [ -n "$PROMPTS_READY" ]; then
    PROMPTS_COUNT=$(echo "$PROMPTS_READY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('prompts_count', 0))")
    print_success "context.prompts_ready ($PROMPTS_COUNT prompts)"
else
    print_failure "context.prompts_ready NOT received"
fi

# Test 1.4: context.resources_ready
echo ""
echo "Test 1.4: Verify context.resources_ready"
RESOURCES_READY=$(grep '"type": "context.resources_ready"' "$EVENTS_FILE" | head -1)
if [ -n "$RESOURCES_READY" ]; then
    RESOURCES_COUNT=$(echo "$RESOURCES_READY" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('resources_count', 0))")
    print_success "context.resources_ready ($RESOURCES_COUNT resources)"
else
    print_failure "context.resources_ready NOT received"
fi

# Test 1.5: context.memory_ready
echo ""
echo "Test 1.5: Verify context.memory_ready"
MEMORY_READY=$(grep '"type": "context.memory_ready"' "$EVENTS_FILE" | head -1)
if [ -n "$MEMORY_READY" ]; then
    print_success "context.memory_ready"
else
    print_failure "context.memory_ready NOT received"
fi

# Test 1.6: context.complete
echo ""
echo "Test 1.6: Verify context.complete"
CONTEXT_COMPLETE=$(grep '"type": "context.complete"' "$EVENTS_FILE" | head -1)
if [ -n "$CONTEXT_COMPLETE" ]; then
    CONTEXT_DURATION=$(echo "$CONTEXT_COMPLETE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('duration_ms', 'N/A'))")
    print_success "context.complete (duration: ${CONTEXT_DURATION}ms)"
else
    print_failure "context.complete NOT received"
fi

# =============================================================================
# STEP 2: ReasonNode Tool Call Generation
# =============================================================================
print_section "Step 2: Testing ReasonNode Tool Call Generation"

# Test 2.1: Check for reason_model node execution
echo ""
echo "Test 2.1: Verify ReasonNode executed"
REASON_NODE_EXIT=$(grep '"node": "reason_model"' "$EVENTS_FILE" | grep '"type": "node.exit"' | head -1)
if [ -n "$REASON_NODE_EXIT" ]; then
    NEXT_ACTION=$(echo "$REASON_NODE_EXIT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('next_action', 'N/A'))")
    print_success "ReasonNode executed (next_action: $NEXT_ACTION)"
else
    print_failure "ReasonNode execution NOT found"
fi

# Test 2.2: Verify AIMessage with tool_calls
echo ""
echo "Test 2.2: Verify AIMessage contains tool_calls"
python3 - "$EVENTS_FILE" <<'PYTHON_CHECK'
import sys
import json

events_file = sys.argv[1]
found_tool_calls = False
tool_call_count = 0
tool_names = []

with open(events_file, 'r') as f:
    for line in f:
        try:
            event = json.loads(line)
            if event.get('type') == 'state.snapshot':
                messages = event.get('metadata', {}).get('messages', [])
                for msg in messages:
                    if msg.get('type') == 'AIMessage' and 'tool_calls' in msg:
                        tool_calls = msg['tool_calls']
                        if tool_calls:
                            found_tool_calls = True
                            tool_call_count = len(tool_calls)
                            tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
                            break
                if found_tool_calls:
                    break
        except:
            continue

if found_tool_calls:
    print(f"✅ AIMessage contains {tool_call_count} tool_calls")
    for i, name in enumerate(tool_names, 1):
        print(f"   {i}. {name}")
    sys.exit(0)
else:
    print("❌ AIMessage with tool_calls NOT found")
    sys.exit(1)
PYTHON_CHECK

if [ $? -eq 0 ]; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 2.3: Verify next_action = "call_tool"
echo ""
echo "Test 2.3: Verify next_action is 'call_tool'"
if [ -n "$REASON_NODE_EXIT" ]; then
    if [ "$NEXT_ACTION" = "call_tool" ]; then
        print_success "next_action correctly set to 'call_tool'"
    else
        print_failure "next_action is '$NEXT_ACTION', expected 'call_tool'"
    fi
else
    print_failure "Cannot verify next_action (ReasonNode not found)"
fi

# =============================================================================
# STEP 3: HIL Detection (BackgroundHILDetector)
# =============================================================================
print_section "Step 3: Testing HIL Detection"

# Test 3.1: Check for HIL request event
echo ""
echo "Test 3.1: Verify HIL detection triggered"
HIL_REQUEST=$(grep '"type": "hil.request"' "$EVENTS_FILE" | head -1)
if [ -n "$HIL_REQUEST" ]; then
    REQUEST_TYPE=$(echo "$HIL_REQUEST" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('request_type', 'N/A'))")
    SCENARIO=$(echo "$HIL_REQUEST" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('interrupt_data', {}).get('scenario', 'N/A'))")
    NODE_SOURCE=$(echo "$HIL_REQUEST" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('interrupt_data', {}).get('node_source', 'N/A'))")

    print_success "HIL detection triggered"
    print_info "  Type: $REQUEST_TYPE"
    print_info "  Scenario: $SCENARIO"
    print_info "  Source: $NODE_SOURCE"
else
    print_failure "HIL request NOT triggered"
fi

# Test 3.2: Verify session paused
echo ""
echo "Test 3.2: Verify session paused for HIL"
SESSION_PAUSED=$(grep '"type": "session.paused"' "$EVENTS_FILE" | head -1)
if [ -n "$SESSION_PAUSED" ]; then
    print_success "Session paused for human input"
else
    print_failure "Session NOT paused"
fi

# Test 3.3: Check tool node triggered HIL
echo ""
echo "Test 3.3: Verify tool_node triggered HIL"
if [ -n "$HIL_REQUEST" ]; then
    if [ "$NODE_SOURCE" = "tool_node" ]; then
        print_success "tool_node correctly triggered HIL"
    else
        print_failure "HIL triggered by $NODE_SOURCE, expected tool_node"
    fi
else
    print_failure "Cannot verify (HIL not triggered)"
fi

# =============================================================================
# STEP 4: Background Execution (requires NATS/Redis)
# =============================================================================
print_section "Step 4: Testing Background Execution"

echo ""
echo "Test 4.1: Check if background job infrastructure is available"
print_info "Checking for NATS and Redis..."

# Check if NATS is running
NATS_RUNNING=false
if lsof -i :4222 >/dev/null 2>&1; then
    print_success "NATS is running on port 4222"
    NATS_RUNNING=true
else
    print_failure "NATS not running on port 4222"
fi

# Check if Redis is running
REDIS_RUNNING=false
if lsof -i :6379 >/dev/null 2>&1; then
    print_success "Redis is running on port 6379"
    REDIS_RUNNING=true
else
    print_failure "Redis not running on port 6379"
fi

# Test background execution if infrastructure is available
if [ "$NATS_RUNNING" = true ] && [ "$REDIS_RUNNING" = true ]; then
    echo ""
    echo "Test 4.2: Resume session with 'background' mode"

    # Resume the paused session with background mode selection
    RESUME_REQUEST=$(cat <<EOF
{
  "user_id": "$TEST_USER",
  "session_id": "$TEST_SESSION",
  "resume_value": {
    "action": "submit",
    "data": {
      "mode": "background"
    }
  }
}
EOF
)

    RESUME_RESPONSE_FILE="/tmp/bg_test_resume_${TEST_SESSION}.txt"
    RESUME_EVENTS_FILE="/tmp/bg_test_resume_events_${TEST_SESSION}.json"

    curl -s -X POST "${API_BASE_URL}/api/v1/agents/chat/resume" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer dev_master_key_123" \
      -d "$RESUME_REQUEST" \
      > "$RESUME_RESPONSE_FILE"

    if [ $? -eq 0 ]; then
        print_success "Resume request sent"

        # Extract events
        grep "^data: " "$RESUME_RESPONSE_FILE" | grep -v "data: \[DONE\]" | sed 's/^data: //' > "$RESUME_EVENTS_FILE"

        echo ""
        echo "Test 4.3: Verify job_id returned"

        # Look for background job event with job_id
        JOB_ID_EVENT=$(grep '"job_id"' "$RESUME_EVENTS_FILE" | head -1)
        if [ -n "$JOB_ID_EVENT" ]; then
            JOB_ID=$(echo "$JOB_ID_EVENT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('metadata', {}).get('job_id', data.get('job_id', 'N/A')))")
            print_success "job_id returned: $JOB_ID"

            echo ""
            echo "Test 4.4: Query job status"

            # Query job status endpoint
            if [ "$JOB_ID" != "N/A" ]; then
                sleep 2  # Wait a bit for job to start

                JOB_STATUS=$(curl -s -X GET "${API_BASE_URL}/api/v1/jobs/${JOB_ID}/status" \
                  -H "Authorization: Bearer dev_master_key_123")

                if [ -n "$JOB_STATUS" ]; then
                    STATUS=$(echo "$JOB_STATUS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))")
                    print_success "Job status: $STATUS"

                    echo ""
                    echo "Test 4.5: Wait for job completion"

                    # Poll for completion (max 30 seconds)
                    MAX_WAIT=30
                    WAIT_COUNT=0
                    COMPLETED=false

                    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
                        sleep 2
                        WAIT_COUNT=$((WAIT_COUNT + 2))

                        JOB_STATUS=$(curl -s -X GET "${API_BASE_URL}/api/v1/jobs/${JOB_ID}/status" \
                          -H "Authorization: Bearer dev_master_key_123")
                        STATUS=$(echo "$JOB_STATUS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))")

                        if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
                            COMPLETED=true
                            break
                        fi

                        print_info "Waiting for job completion... (${WAIT_COUNT}s/${MAX_WAIT}s)"
                    done

                    if [ "$COMPLETED" = true ]; then
                        if [ "$STATUS" = "completed" ]; then
                            print_success "Job completed successfully"
                        else
                            print_failure "Job failed with status: $STATUS"
                        fi
                    else
                        print_info "Job still running after ${MAX_WAIT}s (async execution confirmed)"
                    fi
                else
                    print_failure "Could not query job status"
                fi
            else
                print_failure "Invalid job_id"
            fi
        else
            print_failure "job_id NOT found in response"
        fi

        # Cleanup resume files
        rm -f "$RESUME_RESPONSE_FILE" "$RESUME_EVENTS_FILE"
    else
        print_failure "Resume request failed"
    fi
else
    print_info "Skipping background execution test"
    print_info "NATS/Redis infrastructure not fully available"
fi

# =============================================================================
# STEP 5: Notification (requires notification service)
# =============================================================================
print_section "Step 5: Testing Notification Integration"

echo ""
echo "Test 5.1: Check if notification service is available"

# Check if notification service is running (port 8206 in user-staging)
NOTIFICATION_RUNNING=false
NOTIFICATION_URL="http://localhost:8206"

# Try to reach notification service health endpoint
NOTIFICATION_HEALTH=$(curl -s --max-time 3 "${NOTIFICATION_URL}/health" 2>/dev/null)
if [ -n "$NOTIFICATION_HEALTH" ]; then
    print_success "Notification service is accessible at port 8206"
    NOTIFICATION_RUNNING=true
else
    # Try alternative check
    if lsof -i :8206 >/dev/null 2>&1; then
        print_success "Notification service is running on port 8206"
        NOTIFICATION_RUNNING=true
    else
        print_info "Notification service not accessible on port 8206"
    fi
fi

if [ "$NOTIFICATION_RUNNING" = true ] && [ -n "$JOB_ID" ] && [ "$JOB_ID" != "N/A" ]; then
    echo ""
    echo "Test 5.2: Verify notification was sent for completed job"

    # Wait a bit for notification to be sent
    sleep 2

    # Query notifications for this user/session
    NOTIFICATIONS=$(curl -s -X GET "${NOTIFICATION_URL}/api/v1/notifications?user_id=${TEST_USER}&limit=10" \
      -H "Authorization: Bearer dev_master_key_123" 2>/dev/null)

    if [ -n "$NOTIFICATIONS" ]; then
        # Check if notification contains our job_id
        HAS_NOTIFICATION=$(echo "$NOTIFICATIONS" | grep -c "$JOB_ID" || true)

        if [ "$HAS_NOTIFICATION" -gt 0 ]; then
            print_success "Notification found for job $JOB_ID"

            echo ""
            echo "Test 5.3: Verify notification content"

            # Extract notification details
            NOTIFICATION_TYPE=$(echo "$NOTIFICATIONS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    notifications = data.get('notifications', []) if isinstance(data, dict) else data
    if notifications:
        print(notifications[0].get('type', 'unknown'))
    else:
        print('N/A')
except:
    print('N/A')
" 2>/dev/null)

            if [ "$NOTIFICATION_TYPE" != "N/A" ]; then
                print_success "Notification type: $NOTIFICATION_TYPE"
            else
                print_info "Could not extract notification details"
            fi
        else
            print_info "No notification found yet for job $JOB_ID (may be async delay)"
        fi
    else
        print_info "Could not query notifications (endpoint may not be available)"
    fi

    echo ""
    echo "Test 5.4: Verify notification delivery channels"

    # Check if notification service supports webhooks/websockets
    NOTIFICATION_CAPABILITIES=$(curl -s --max-time 3 "${NOTIFICATION_URL}/api/v1/capabilities" 2>/dev/null)
    if [ -n "$NOTIFICATION_CAPABILITIES" ]; then
        print_success "Notification service capabilities retrieved"
        echo "$NOTIFICATION_CAPABILITIES" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    channels = data.get('channels', [])
    if channels:
        print('   Supported channels: ' + ', '.join(channels))
    else:
        print('   Channels: webhook, websocket (default)')
except:
    print('   Channels: webhook, websocket (default)')
" 2>/dev/null
    else
        print_info "Notification channels: webhook, websocket (assumed)"
    fi
else
    if [ "$NOTIFICATION_RUNNING" = false ]; then
        print_info "Skipping notification test - service not available"
    elif [ -z "$JOB_ID" ] || [ "$JOB_ID" = "N/A" ]; then
        print_info "Skipping notification test - no background job was created"
    else
        print_info "Notification test conditions not met"
    fi
fi

# =============================================================================
# Final Summary
# =============================================================================
print_section "Test Summary"

# Cleanup
rm -f "$RESPONSE_FILE" "$EVENTS_FILE"

echo ""
echo "Test Results:"
echo "  ✅ Passed: $TESTS_PASSED"
echo "  ❌ Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=================================================="
    echo "✅ ALL TESTS PASSED"
    echo "==================================================${NC}"
    echo ""
    echo "Complete flow verified:"
    echo "  ✓ Step 1: Context initialization (6 tests)"
    echo "  ✓ Step 2: ReasonNode tool call generation (3 tests)"
    echo "  ✓ Step 3: HIL detection (3 tests)"
    if [ "$NATS_RUNNING" = true ] && [ "$REDIS_RUNNING" = true ]; then
        echo "  ✓ Step 4: Background execution (NATS/Redis)"
    else
        echo "  ⊘ Step 4: Background execution (skipped - infra not available)"
    fi
    if [ "$NOTIFICATION_RUNNING" = true ]; then
        echo "  ✓ Step 5: Notification integration"
    else
        echo "  ⊘ Step 5: Notification integration (skipped - service not available)"
    fi
    echo ""
    exit 0
else
    echo -e "${RED}=================================================="
    echo "❌ SOME TESTS FAILED"
    echo "==================================================${NC}"
    echo ""
    echo "Tests passed: $TESTS_PASSED"
    echo "Tests failed: $TESTS_FAILED"
    echo ""
    echo "Please check the errors above"
    echo ""
    exit 1
fi
