Feature Comparison                                                                                               
  ┌───────────────────┬───────────────────────────────────┬───────────────────────────────────┬───────────────────┐
  │      Feature      │            Claude SDK             │           isA Agent SDK           │      Status       │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Hooks             │ Lifecycle hooks (before/after     │ HookMatcher in options.py         │ ⚠️ Needs          │
  │                   │ tool execution)                   │                                   │ verification      │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Structured        │ JSON schema validated outputs     │ OutputFormat + Pydantic           │ ✅ Implemented    │
  │ Outputs           │                                   │ integration in options.py         │                   │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ System Prompts    │ Customizable prompts              │ SystemPromptConfig with           │ ✅ Implemented    │
  │                   │ (preset + append)                 │ preset/append/replace + MCP       │                   │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Sessions          │ Persistence & resumption          │ DurableService + resume()         │ ✅ Present        │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Subagents         │ Task delegation                   │ AgentDefinition + Task tool       │ ✅ Present        │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ MCP               │ Model Context Protocol            │ Full MCPClient integration        │ ✅ Present        │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Permissions       │ Tool permissions                  │ HIL system + PermissionMode       │ ✅ Present        │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Custom Tools      │ Define/use tools                  │ Via MCP + execute_tool()          │ ✅ Present        │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ User Input (Ask   │ Interactive input                 │ HIL: collect_input(),             │ ✅ Present        │
  │ User)             │                                   │ collect_selection()               │                   │
  ├───────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Streaming         │ Real-time tokens                  │ Async generators + SSE            │ ✅ Present        │
  └───────────────────┴───────────────────────────────────┴───────────────────────────────────┴───────────────────┘