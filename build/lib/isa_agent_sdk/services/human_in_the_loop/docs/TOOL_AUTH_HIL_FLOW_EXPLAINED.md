# HIL (Human-in-the-Loop) 完整数据流程解析

## 🎯 核心问题

**为什么测试中没有触发 HIL 事件？**

答案：**工具安全级别默认是 LOW，不会触发授权请求！**

---

## 📊 完整数据流程

### 第 1 步：用户发送请求
```
用户 → 前端 → POST /api/v1/agents/chat
{
  "message": "请搜索今天的天气",
  "session_id": "...",
  "user_id": "..."
}
```

### 第 2 步：Agent 处理消息
```
ChatService → ReasonNode → 生成工具调用
{
  "tool_calls": [{
    "name": "get_weather",
    "args": {...}
  }]
}
```

### 第 3 步：ToolNode 批量检查授权 ⭐️ **关键步骤**

**代码位置**: `src/nodes/tool_node.py:88-102`

```python
# 1. 提取所有工具名称
tool_names = ["get_weather"]

# 2. 批量检查安全级别（通过 MCP）
authorization_results = await self._batch_check_tool_authorization(tool_names, config)
# 返回: {"get_weather": ("LOW", False)}
#         工具名 → (安全级别, 是否需要授权)

# 3. 筛选需要授权的高安全工具
high_security_tools = [
    (tool_name, security_level)
    for tool_name, (security_level, needs_auth) in authorization_results.items()
    if needs_auth  # 只有 HIGH 和 CRITICAL 才是 True
]

# 4. 如果有高安全工具，触发批量授权
if high_security_tools:
    await self._request_batch_authorization(high_security_tools, config)
    # 👆 这里会调用 interrupt()
```

### 第 4 步：触发 HIL (如果需要授权)

**代码位置**: `src/nodes/tool_node.py:642-674`

```python
async def _request_batch_authorization(self, high_security_tools, config):
    # 创建授权请求
    authorization_request = {
        "type": "batch_tool_authorization",
        "tools": high_security_tools,  # [("web_search", "HIGH")]
        "user_id": user_id,
        "message": "Multiple tools require authorization..."
    }

    # ⭐️ 关键：调用 LangGraph 的 interrupt()
    interrupt(authorization_request)
```

### 第 5 步：LangGraph Interrupt 机制

LangGraph 的 `interrupt()` 做了什么？

```python
from langgraph.types import interrupt

# 调用 interrupt() 会：
# 1. 暂停当前图的执行
# 2. 将 interrupt_data 保存到图的 state
# 3. 等待外部调用 resume() 继续执行
```

### 第 6 步：ChatService 处理 Interrupt ⭐️ **问题所在**

**代码位置**: `src/services/chat_service.py`

```python
# ChatService 需要：
# 1. 检测到图执行被 interrupt
# 2. 提取 interrupt_data
# 3. 发送 SSE 事件到前端

# SSE 事件格式
{
    "type": "hil.request",
    "content": "Authorization required",
    "metadata": {
        "interrupt_data": {
            "type": "batch_tool_authorization",
            "tools": [["web_search", "HIGH"]],
            "user_id": "...",
            "message": "..."
        }
    }
}
```

### 第 7 步：前端接收并显示

```typescript
// 前端监听 SSE
const event = JSON.parse(data)

if (event.type === 'hil.request') {
    const hilData = event.metadata.interrupt_data
    setHilInterruptData(hilData)
    setShowHILModal(true)
}
```

### 第 8 步：用户操作 Modal

```
用户点击 Approve → 前端发送 Resume 请求
```

### 第 9 步：Resume 执行

```
POST /api/v1/agents/chat/resume
{
  "session_id": "...",
  "user_id": "...",
  "resume_value": {
    "action": "approve",
    "approved": true,
    "message": "User approved"
  }
}

→ ChatService.resume_execution()
→ LangGraph 继续执行
→ ToolNode 执行工具
→ 返回结果
```

---

## 🔍 为什么测试中没有触发 HIL？

### 原因分析

```python
# 第 3 步检查安全级别时：
authorization_results = {
    "get_weather": ("LOW", False)  # ← 安全级别是 LOW
}

# 第 4 步筛选：
high_security_tools = []  # ← 没有高安全工具！

# 第 5 步判断：
if high_security_tools:  # ← False，不进入
    await self._request_batch_authorization(...)
```

**结论**: `get_weather` 工具的安全级别是 `LOW`，不会触发授权请求！

---

## 🛠️ 如何让工具触发 HIL？

### 方法 1: 使用高安全级别的工具

MCP 服务器需要将工具标记为 HIGH 或 CRITICAL：

```python
# 在 MCP 服务器中
tools = [
    {
        "name": "web_search",
        "security_level": "HIGH",  # ← 设置为 HIGH
        "description": "Search the web"
    }
]
```

### 方法 2: 修改安全级别检查逻辑

临时测试方案：

```python
# 在 tool_node.py 的 _batch_check_tool_authorization 中
# 强制返回 HIGH 级别进行测试
def _batch_check_tool_authorization(self, tool_names, config):
    results = {}
    for tool_name in tool_names:
        # 临时：强制所有工具为 HIGH
        results[tool_name] = ("HIGH", True)
    return results
```

### 方法 3: 使用测试工具

创建一个专门用于测试的高安全工具：

```python
# 添加测试工具
test_tools = {
    "test_dangerous_operation": {
        "security_level": "CRITICAL",
        "description": "Test high-security operation"
    }
}
```

---

## 📋 完整的 HIL 触发条件

| 条件 | 说明 |
|------|------|
| 工具安全级别 | 必须是 `HIGH` 或 `CRITICAL` |
| MCP 集成 | MCP 服务器需要返回安全级别 |
| ChatService 支持 | 需要处理 interrupt 并发送 SSE |
| 前端集成 | 需要监听 `hil.request` 事件 |

---

## 🧪 测试方案

### 选项 A: 修改 MCP 返回的安全级别

```python
# 在 MCP 服务器配置中
{
    "get_weather": {
        "security_level": "HIGH"  # ← 改为 HIGH
    }
}
```

### 选项 B: 临时修改 ToolNode 逻辑

```python
# src/nodes/tool_node.py:564
async def _batch_check_tool_authorization(self, tool_names, config):
    results = {}
    for tool_name in tool_names:
        # 🧪 测试：强制所有工具需要授权
        results[tool_name] = ("HIGH", True)
    return results
```

### 选项 C: 使用已有的高安全工具

如果 MCP 中有 `web_search`、`file_write` 等工具：

```python
# 测试消息
"请搜索网络信息并保存到文件"
# → 触发 web_search (HIGH) 和 file_write (HIGH)
```

---

## 🎯 验证清单

要让 HIL 工作，需要确保：

- [ ] MCP 服务器配置了工具安全级别
- [ ] 至少有一个工具的级别是 HIGH/CRITICAL
- [ ] ToolNode 能正确查询安全级别
- [ ] ChatService 能检测并处理 interrupt
- [ ] ChatService 发送 `hil.request` SSE 事件
- [ ] 前端监听并显示 HIL Modal
- [ ] Resume API 正确配置

---

## 🔧 快速修复方案

### 1. 查看当前 MCP 工具配置

```bash
# 查询 MCP 服务器的工具列表
curl http://localhost:8080/api/v1/mcp/tools
```

### 2. 临时修改 ToolNode（仅用于测试）

```python
# src/nodes/tool_node.py:564
async def _batch_check_tool_authorization(self, tool_names, config):
    results = {}
    for tool_name in tool_names:
        # 🧪 临时：强制触发授权测试
        if tool_name in ["get_weather", "web_search"]:
            results[tool_name] = ("HIGH", True)  # ← 强制 HIGH
        else:
            results[tool_name] = ("LOW", False)
    return results
```

### 3. 重启后端并测试

```bash
python main.py

# 然后运行测试
python tests/test_hil_scenarios.py
```

---

## 📚 相关代码位置

| 文件 | 行号 | 说明 |
|------|------|------|
| `src/nodes/tool_node.py` | 88-102 | 批量授权检查 |
| `src/nodes/tool_node.py` | 542-641 | 安全级别查询 |
| `src/nodes/tool_node.py` | 642-674 | 批量授权请求 (interrupt) |
| `src/services/chat_service.py` | - | 处理 interrupt 并发送 SSE |
| `src/services/human_in_the_loop/` | - | HIL 服务实现 |

---

## 🎉 总结

**HIL 数据流程**:
```
1. 用户请求
2. Agent 生成工具调用
3. ToolNode 检查安全级别 (通过 MCP)
4. 如果 HIGH/CRITICAL → interrupt()
5. ChatService 检测 interrupt
6. 发送 SSE hil.request 事件
7. 前端显示 Modal
8. 用户操作
9. Resume API
10. 继续执行
```

**关键问题**:
- 默认工具安全级别太低（LOW）
- 需要 MCP 配置或临时修改代码
- ChatService 需要正确处理 interrupt

**下一步**:
1. 检查 MCP 工具配置
2. 或临时修改 ToolNode 强制 HIGH
3. 验证 ChatService 的 interrupt 处理
4. 运行测试
