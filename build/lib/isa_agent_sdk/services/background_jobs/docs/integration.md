🎯 Background Task Trigger Rules

  Rule 1: 3+ Web Crawls (Primary Trigger)

  # From tool_node.py:1037
  if len(web_crawls) >= 3:
      estimated_time = len(web_crawls) * 12  # ~36+ seconds
      # Trigger HIL choice

  Example: User asks "研究 5 篇 AI 论文"
  - ReasonNode generates 5 web_crawl tool calls
  - ToolNode detects: 5 >= 3 → Long task detected!

  Rule 2: 5+ Web Searches (Secondary Trigger)

  # From tool_node.py:1048
  if len(web_searches) >= 5:
      estimated_time = len(web_searches) * 3  # ~15+ seconds
      # Trigger HIL choice

  Example: User asks "搜索最新的 8 个科技新闻"
  - ReasonNode generates 8 web_search tool calls
  - ToolNode detects: 8 >= 5 → Long task detected!

  ---
  ✅ YES! Fully Integrated with HIL Service

  From tool_node.py:1059-1101, here's the complete integration:

  async def _offer_execution_choice(self, task_info: dict, config: RunnableConfig) -> str:
      """Offer user choice via HIL service"""
      from ..services.hil_service import hil_service  # ✅ HIL imported

      question = f"""🕐 Long-running task detected: {tool_count} web crawls 
  (~{estimated_time}s)
      
  Choose execution mode:
  • Type 'quick' - Fast response (3 sources, ~30s)
  • Type 'comprehensive' - Wait for all {tool_count} sources (~{estimated_time}s)
  • Type 'background' - Run in background, get job_id immediately

  Your choice:"""

      # ✅ Uses HIL interrupt - pauses graph execution!
      response = hil_service.ask_human_with_interrupt(
          question=question,
          context=json.dumps(task_info, indent=2),
          node_source="tool_node"
      )

      choice = str(response).lower().strip()

      # ✅ Parse user choice
      if choice in ["quick", "q"]:
          return "quick"
      elif choice in ["background", "bg", "b"]:
          return "background"  # → Goes to NATS queue!
      else:
          return "comprehensive"

  ---
  🔄 Complete Execution Flow

  User: "帮我研究 10 篇 AI 论文的最新进展"
     │
     ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 1: ReasonNode Generates Tools                      │
  │  → 10 x web_crawl tool calls                             │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 2: ToolNode Detects Long Task                      │
  │  → _detect_long_running_task()                           │
  │  → Detects: 10 web_crawls >= 3 ✅                        │
  │  → Estimated time: 10 * 12s = 120 seconds                │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 3: HIL Service Interrupts (GRAPH PAUSES!)          │
  │  → hil_service.ask_human_with_interrupt()                │
  │  → LangGraph interrupt() called                          │
  │  → Graph execution FROZEN, waiting for user              │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 4: User Sees Question                              │
  │                                                           │
  │  🕐 Long-running task detected: 10 web crawls (~120s)    │
  │                                                           │
  │  Choose execution mode:                                   │
  │  • Type 'quick' - Fast (3 sources, ~30s)                 │
  │  • Type 'comprehensive' - Wait for all 10 (~120s)        │
  │  • Type 'background' - Run in background, get job_id     │
  │                                                           │
  │  Your choice: _____                                       │
  └──────────────┬───────────────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
     User types:      "background"
         │                │
         ▼                ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 5: Graph Resumes with User's Choice                │
  │  → choice = "background"                                  │
  │  → _queue_background_job() called                        │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 6: Task Queued to NATS                             │
  │  → submit_tool_execution_task()                          │
  │  → NATS JetStream: ISA_AGENT_TASKS                       │
  │  → Priority: high                                         │
  │  → Redis: Initial status stored                          │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 7: User Gets Immediate Response                    │
  │                                                           │
  │  {                                                        │
  │    "status": "queued",                                    │
  │    "job_id": "job_abc123",                                │
  │    "poll_url": "/api/v1/jobs/job_abc123",                │
  │    "sse_url": "/api/v1/jobs/job_abc123/stream",          │
  │    "estimated_completion": "120s"                         │
  │  }                                                        │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 8: Worker Picks Up Task (Background)               │
  │  → Worker pulls from NATS: ISA_AGENT_TASKS               │
  │  → Executes 10 web_crawls one by one                     │
  │  → Updates Redis: 10%, 20%, 30%... 100%                  │
  │  → Pub/Sub broadcasts progress events                    │
  └──────────────┬───────────────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Step 9: User Monitors Progress                          │
  │  Option A: Poll                                           │
  │    GET /api/v1/jobs/job_abc123                           │
  │    → {"progress_percent": 30, "completed_tools": 3}      │
  │                                                           │
  │  Option B: Stream (SSE)                                   │
  │    GET /api/v1/jobs/job_abc123/stream                    │
  │    → event: tool_complete                                 │
  │    → data: {"tool_name": "web_crawl", "progress": 30}    │
  └───────────────────────────────────────────────────────────┘

  ---
  📊 Trigger Rules Summary

  | Trigger Condition | Threshold | Estimated Time | Action           |
  |-------------------|-----------|----------------|------------------|
  | web_crawl         | ≥ 3 calls | ~36+ seconds   | HIL Choice       |
  | web_search        | ≥ 5 calls | ~15+ seconds   | HIL Choice       |
  | Other tools       | N/A       | No detection   | Execute normally |

  HIL Choices & Outcomes

  | User Choice   | Result          | Description                    |
  |---------------|-----------------|--------------------------------|
  | quick         | Sync execution  | Limits to 3 web_crawls (~30s)  |
  | comprehensive | Sync execution  | Waits for all tools (~120s)    |
  | background    | Async execution | Queues to NATS, returns job_id |

  ---
  🎯 Key Integration Points

  1. HIL Service Usage ✅

  # tool_node.py:1081
  response = hil_service.ask_human_with_interrupt(
      question=question,
      context=json.dumps(task_info, indent=2),
      node_source="tool_node"
  )

  - ✅ Uses ask_human_with_interrupt() from HIL service
  - ✅ Triggers LangGraph interrupt() - pauses execution
  - ✅ Graph waits for user response via /api/chat/resume

  2. LangGraph Interrupt Flow ✅

  # hil_service.py:414
  human_response = interrupt(interrupt_data)  # Pauses graph!

  - ✅ Graph execution FROZEN until user responds
  - ✅ User responds via /api/v1/agents/chat/resume
  - ✅ Graph resumes with user's choice

  3. Background Job Integration ✅

  # tool_node.py:1153
  task_result = await submit_tool_execution_task(
      task_data={...},
      priority="high"
  )

  - ✅ Submits to NATS queue
  - ✅ Returns job_id to user
  - ✅ Worker processes asynchronously

  ---
  🎊 Summary

  ✅ YES, Fully Integrated with HIL!

  1. Detection: Automatic (3+ web_crawls or 5+ web_searches)
  2. HIL Interrupt: Graph pauses, user sees question
  3. User Choice: quick / comprehensive / background
  4. Background Execution: If "background" chosen → NATS queue
  5. Monitoring: Poll or SSE streaming

  🚀 Ready to Use Right Now!

  The system will automatically detect long tasks and ask the user via HIL service. If user
  chooses "background", it goes to the Worker queue!