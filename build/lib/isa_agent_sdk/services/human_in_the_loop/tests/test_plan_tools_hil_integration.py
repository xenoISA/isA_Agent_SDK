#!/usr/bin/env python3
"""
Plan Tools HIL Integration E2E Test

测试完整的 Plan Tools 集成流程:
1. context_init.py - 匹配 create_execution_plan 工具
2. reason_node.py - 调用 create_execution_plan 返回 tool_call
3. plan_tools.py - 触发 HIL request_review (line 256-270)
4. 前端交互 - 用户 Approve/Edit/Reject
5. agent_executor_node.py - 执行每个任务
6. react_agent.py - 使用 ReactAgent 执行具体任务
7. reason_node.py - 重新评估结果
8. response_node.py - 返回最终响应

流程图:
用户请求 → context_init (匹配工具) → reason_node (tool_call) →
plan_tools (HIL request_review) → 前端确认 → agent_executor →
react_agent (执行任务) → reason_node (评估) → response_node (完成)

前置条件:
1. MCP 服务器运行中: cd isA_MCP && python server.py
2. isA_Agent 运行中: cd isA_Agent && python main.py
3. 前端运行中: cd frontend_mini && npm run dev
4. 浏览器访问: http://localhost:5173
"""

import requests
import json
import time
from typing import Any, Optional

# 配置
API_BASE = "http://localhost:8080"
TEST_USER_ID = "test_plan_hil_user"
API_KEY = "authenticated"

# 全局变量
current_test_session = None
hil_detected = False
hil_type_detected = None
hil_resumed = False
plan_execution_started = False
plan_execution_completed = False


def print_banner(title: str):
    """打印测试场景标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_step(step: str):
    """打印流程步骤"""
    print(f"\n🔹 {step}")
    print("-" * 80)


def send_chat_and_monitor_flow(message: str):
    """
    发送聊天消息并监控完整 HIL 流程

    监控事件:
    1. hil.request (plan_tools.py:256-270 触发的 request_review)
    2. content.token/content.complete (用户 approve 后的响应)
    3. agent_execution (AgentExecutorNode 开始执行)
    4. task_step_start/task_step_complete (ReactAgent 执行任务)
    5. session.end (最终完成)
    """
    global current_test_session, hil_detected, hil_type_detected, hil_resumed
    global plan_execution_started, plan_execution_completed

    # 重置状态
    current_test_session = f"plan_hil_test_{int(time.time())}"
    hil_detected = False
    hil_type_detected = None
    hil_resumed = False
    plan_execution_started = False
    plan_execution_completed = False

    url = f"{API_BASE}/api/v1/agents/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "message": message,
        "user_id": TEST_USER_ID,
        "session_id": current_test_session,
        "stream": True
    }

    print_banner("Plan Tools HIL 集成测试")
    print(f"📤 发送请求: {message}")
    print(f"   Session ID: {current_test_session}")
    print(f"   User ID: {TEST_USER_ID}\n")

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=300)

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   Response: {response.text}\n")
            return False

        print("✅ 聊天请求成功")
        print("📡 开始监听 SSE 流...\n")
        print("-" * 80)

        # 监听 SSE 流
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]

                # 检查是否结束
                if data_str.strip() == "[DONE]":
                    print("\n" + "-" * 80)
                    print("✅ SSE 流结束")

                    # 验证完整流程
                    if hil_detected and hil_resumed and plan_execution_started and plan_execution_completed:
                        print("\n" + "🎉" * 40)
                        print("🎉  完整 HIL Plan Tools 流程测试成功!")
                        print("🎉" * 40)
                        print("\n✅ 流程验证:")
                        print(f"   1. ✅ HIL request_review 触发 (类型: {hil_type_detected})")
                        print("   2. ✅ 用户在前端 Approve")
                        print("   3. ✅ AgentExecutorNode 开始执行")
                        print("   4. ✅ ReactAgent 执行任务")
                        print("   5. ✅ 执行完成\n")
                        return True
                    else:
                        print("\n⚠️  流程未完全完成:")
                        print(f"   HIL 触发: {'✅' if hil_detected else '❌'}")
                        print(f"   用户 Resume: {'✅' if hil_resumed else '❌'}")
                        print(f"   计划执行开始: {'✅' if plan_execution_started else '❌'}")
                        print(f"   计划执行完成: {'✅' if plan_execution_completed else '❌'}")
                        return False

                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")
                    content = event.get("content", "")
                    metadata = event.get("metadata", {})

                    # ========================================
                    # 阶段 1: 检测 HIL request_review
                    # ========================================
                    if event_type == "hil.request":
                        hil_detected = True
                        interrupt_data = metadata.get("interrupt_data", {})
                        hil_method = interrupt_data.get("method", "unknown")
                        content_type = interrupt_data.get("content_type", "unknown")

                        # 记录 HIL 类型
                        hil_type_detected = f"{hil_method}/{content_type}"

                        print("\n" + "🔔" * 40)
                        print_step("阶段 1: HIL Request Review 触发 ✅")
                        print("🔔" * 40)
                        print(f"\n📋 HIL 方法: {hil_method}")
                        print(f"📋 内容类型: {content_type}")

                        # 验证是否是 execution_plan 类型
                        if hil_method == "request_review" and content_type == "execution_plan":
                            print("✅ 正确触发 request_review for execution_plan!")
                        else:
                            print(f"⚠️  HIL 类型不匹配 (期望: request_review/execution_plan)")

                        print("\n📝 HIL 数据预览:")
                        plan_content = interrupt_data.get("content", {})
                        if isinstance(plan_content, dict):
                            print(f"   Plan ID: {plan_content.get('plan_id', 'N/A')}")
                            print(f"   总任务数: {plan_content.get('total_tasks', 'N/A')}")
                            print(f"   执行模式: {plan_content.get('execution_mode', 'N/A')}")
                            print(f"   假设: {plan_content.get('solution_hypothesis', 'N/A')[:60]}...")

                            # 显示任务列表
                            tasks = plan_content.get("tasks", [])
                            if tasks:
                                print(f"\n   任务列表:")
                                for i, task in enumerate(tasks[:3], 1):
                                    print(f"      {i}. {task.get('title', 'Untitled')}")
                                if len(tasks) > 3:
                                    print(f"      ... 还有 {len(tasks) - 3} 个任务")

                        print("\n" + "⏳" * 40)
                        print_step("等待用户在前端操作 HIL Modal")
                        print("⏳  请在前端 (http://localhost:5173) 操作:")
                        print("⏳  - Approve: 确认计划，开始自动执行")
                        print("⏳  - Edit: 修改任务、添加/删除步骤")
                        print("⏳  - Reject: 取消计划，重新开始")
                        print("⏳" * 40 + "\n")

                    # ========================================
                    # 阶段 2: 检测用户 Resume 后的响应
                    # ========================================
                    elif event_type in ["content.token", "content.complete"] and hil_detected and not hil_resumed:
                        hil_resumed = True
                        print("\n" + "✅" * 40)
                        print_step("阶段 2: 用户 Resume 确认 ✅")
                        print("✅  检测到用户在前端完成操作!")
                        print("✅  计划已批准，准备开始执行...")
                        print("✅" * 40 + "\n")

                    # ========================================
                    # 阶段 3: 检测 AgentExecutorNode 开始执行
                    # ========================================
                    elif event_type == "agent_execution" and not plan_execution_started:
                        plan_execution_started = True
                        exec_status = content if isinstance(content, dict) else {}

                        print("\n" + "🚀" * 40)
                        print_step("阶段 3: AgentExecutorNode 开始执行 ✅")
                        print(f"🚀  执行状态: {exec_status.get('status', 'unknown')}")
                        print(f"🚀  剩余步骤: {exec_status.get('remaining_steps', 'N/A')}")
                        print("🚀" * 40 + "\n")

                    # ========================================
                    # 阶段 4: 检测任务执行 (ReactAgent)
                    # ========================================
                    elif event_type == "task_step_start":
                        task_info = content if isinstance(content, dict) else {}
                        print(f"\n⏳ 任务开始: {task_info.get('task_title', 'Unknown')} " +
                              f"({task_info.get('task_index', '?')}/{task_info.get('total_tasks', '?')})")

                    elif event_type == "task_step_complete":
                        task_info = content if isinstance(content, dict) else {}
                        task_status = task_info.get('status', 'unknown')
                        emoji = "✅" if task_status == "success" else "❌"
                        print(f"{emoji} 任务完成: {task_info.get('task_title', 'Unknown')} " +
                              f"(状态: {task_status})")

                    # ========================================
                    # 阶段 5: 检测执行完成
                    # ========================================
                    elif "agent_execution" in event_type and hil_resumed:
                        exec_info = content if isinstance(content, dict) else {}
                        exec_status = exec_info.get('status', '')

                        if exec_status in ["all_completed", "completed"]:
                            plan_execution_completed = True
                            print("\n" + "🎯" * 40)
                            print_step("阶段 5: 计划执行完成 ✅")
                            print(f"🎯  总任务数: {exec_info.get('total_tasks', 'N/A')}")
                            print(f"🎯  已完成: {exec_info.get('completed', 'N/A')}")
                            print(f"🎯  失败: {exec_info.get('failed', 'N/A')}")
                            print("🎯" * 40 + "\n")

                    # 打印关键事件 (简化日志)
                    if event_type not in ["content.token"]:  # token 太多不打印
                        if event_type == "hil.request":
                            status = " 🔔 HIL!"
                        elif event_type in ["content.complete", "content.token"] and hil_detected and not hil_resumed:
                            status = " ✅ Resumed!"
                        elif event_type == "agent_execution":
                            status = " 🚀 Executing!"
                        elif event_type in ["task_step_start", "task_step_complete"]:
                            status = " 🔧 Task!"
                        else:
                            status = ""

                        content_preview = str(content)[:80] if content else ''
                        print(f"📨 {event_type}{status}: {content_preview}")

                except json.JSONDecodeError:
                    pass

        print("-" * 80 + "\n")

        # 如果所有阶段都完成，返回成功
        return hil_detected and hil_resumed and plan_execution_started and plan_execution_completed

    except requests.exceptions.Timeout:
        print("\n❌ 请求超时（300秒）")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_plan_creation():
    """
    测试场景 1: 简单计划创建

    触发流程:
    1. 用户请求创建执行计划
    2. context_init 匹配到 create_execution_plan 工具
    3. reason_node 调用 create_execution_plan
    4. plan_tools.py:256-270 触发 request_review HIL
    5. 前端显示 execution_plan HIL modal
    6. 用户 Approve
    7. AgentExecutorNode 执行任务列表
    8. ReactAgent 执行每个任务
    9. 返回最终结果
    """
    print_banner("测试场景 1: 简单计划创建 + HIL + 执行")

    print("📝 测试场景:")
    print("   - 创建简单的 2-3 任务执行计划")
    print("   - 触发 HIL request_review")
    print("   - 用户 Approve 后自动执行")
    print("   - ReactAgent 执行每个任务\n")

    print("📋 测试流程:")
    print("   1. 发送计划创建请求")
    print("   2. 等待 HIL request_review 触发")
    print("   3. 在前端 Approve 计划")
    print("   4. 观察 AgentExecutorNode 执行")
    print("   5. 验证任务完成\n")

    message = """
创建一个简单的执行计划来分析 Python 项目的代码质量。

可用工具: code_scanner, dependency_checker, test_runner

请创建一个包含 2-3 个任务的计划,使用 create_execution_plan 工具。
"""

    print("💡 期待行为:")
    print("   - reason_node 识别需要调用 create_execution_plan")
    print("   - plan_tools.py 返回 request_review HIL 响应")
    print("   - 前端显示 execution_plan 类型的 HIL modal")
    print("   - 用户 Approve 后,AgentExecutorNode 执行任务")
    print("   - ReactAgent 使用相关工具执行每个任务\n")

    input("按 Enter 开始测试...")

    return send_chat_and_monitor_flow(message)


def test_complex_plan_with_dependencies():
    """
    测试场景 2: 复杂计划(带依赖关系)

    测试更复杂的场景:
    - 多个任务 (4-6个)
    - 任务间有依赖关系
    - 可能触发 sequential 执行模式
    """
    print_banner("测试场景 2: 复杂计划 + 依赖关系")

    print("📝 测试场景:")
    print("   - 创建复杂执行计划 (4-6 任务)")
    print("   - 任务间存在依赖关系")
    print("   - 测试 sequential 执行模式")
    print("   - 测试任务状态更新\n")

    message = """
创建一个全面的安全审计执行计划。

任务要求:
1. 扫描代码漏洞
2. 检查依赖安全性
3. 运行安全测试
4. 生成审计报告

可用工具: code_scanner, dependency_checker, security_analyzer, test_runner, report_generator

请使用 create_execution_plan 创建一个详细的、有依赖关系的执行计划。
"""

    print("💡 期待行为:")
    print("   - 生成 4-6 个任务")
    print("   - 任务按依赖顺序执行 (sequential mode)")
    print("   - 每个任务完成后更新状态")
    print("   - AgentExecutorNode 按顺序执行\n")

    input("按 Enter 开始测试...")

    return send_chat_and_monitor_flow(message)


def test_plan_edit_workflow():
    """
    测试场景 3: 计划编辑工作流

    测试用户编辑计划的场景:
    - 用户在 HIL modal 中选择 Edit
    - 修改任务列表 (添加/删除/修改)
    - 保存后开始执行修改后的计划
    """
    print_banner("测试场景 3: 计划编辑工作流")

    print("📝 测试场景:")
    print("   - 触发 HIL request_review")
    print("   - 用户点击 Edit 编辑计划")
    print("   - 修改任务列表")
    print("   - 保存并执行修改后的计划\n")

    message = """
创建一个数据分析执行计划。

可用工具: data_ingest, data_query, data_search, visualization

请使用 create_execution_plan 创建计划。
"""

    print("💡 操作步骤:")
    print("   1. 等待 HIL modal 显示")
    print("   2. 点击 'Edit' 按钮")
    print("   3. 修改任务:")
    print("      - 添加新任务")
    print("      - 删除不需要的任务")
    print("      - 修改任务描述")
    print("   4. 保存修改")
    print("   5. 观察执行修改后的计划\n")

    input("按 Enter 开始测试...")

    return send_chat_and_monitor_flow(message)


def test_plan_rejection():
    """
    测试场景 4: 计划拒绝流程

    测试用户拒绝计划的场景:
    - 用户在 HIL modal 中选择 Reject
    - 系统返回拒绝消息
    - 用户可以重新请求创建计划
    """
    print_banner("测试场景 4: 计划拒绝流程")

    print("📝 测试场景:")
    print("   - 触发 HIL request_review")
    print("   - 用户点击 Reject 拒绝计划")
    print("   - 系统处理拒绝")
    print("   - 不执行任何任务\n")

    message = """
创建一个测试执行计划。

可用工具: test_runner, coverage_checker

请使用 create_execution_plan 创建计划。
"""

    print("💡 操作步骤:")
    print("   1. 等待 HIL modal 显示")
    print("   2. 点击 'Reject' 按钮")
    print("   3. 确认拒绝")
    print("   4. 观察系统响应\n")

    print("💡 期待行为:")
    print("   - 计划不会被执行")
    print("   - AgentExecutorNode 不会被调用")
    print("   - 系统返回拒绝确认消息\n")

    input("按 Enter 开始测试...")

    return send_chat_and_monitor_flow(message)


def main():
    """主测试流程"""
    print("\n" + "🧪" * 40)
    print("  Plan Tools HIL 集成 E2E 测试套件")
    print("  完整流程: context_init → reason → plan_tools (HIL) → ")
    print("           executor → react_agent → reason → response")
    print("🧪" * 40)

    print("\n📋 测试环境:")
    print(f"   API Base: {API_BASE}")
    print(f"   User ID: {TEST_USER_ID}")

    print("\n⚠️  测试流程:")
    print("   1. 发送请求 → context_init 匹配工具")
    print("   2. reason_node 调用 create_execution_plan")
    print("   3. plan_tools.py:256-270 触发 HIL request_review")
    print("   4. 前端显示 HIL modal")
    print("   5. 👉 你在前端手动操作 (Approve/Edit/Reject)")
    print("   6. AgentExecutorNode 执行任务")
    print("   7. ReactAgent 执行每个具体任务")
    print("   8. reason_node 评估结果")
    print("   9. response_node 返回最终响应")

    print("\n📍 前置条件:")
    print("   ✅ MCP 服务器已启动: cd isA_MCP && python server.py")
    print("   ✅ isA_Agent 已启动: cd isA_Agent && python main.py")
    print("   ✅ 前端已启动: cd frontend_mini && npm run dev")
    print("   ✅ 浏览器已打开: http://localhost:5173")

    print("\n🎯 关键验证点:")
    print("   📋 plan_tools.py:256-270 的 request_review 是否触发")
    print("   📋 HIL modal 是否正确显示 execution_plan 类型")
    print("   📋 用户 Approve 后是否进入 AgentExecutorNode")
    print("   📋 ReactAgent 是否正确执行每个任务")
    print("   📋 任务完成后是否返回 reason_node 评估")

    print("\n💡 推荐测试顺序:")
    print("   1️⃣  场景 1 - 简单计划创建 (最基础)")
    print("   2️⃣  场景 2 - 复杂计划 + 依赖 (测试完整性)")
    print("   3️⃣  场景 3 - 计划编辑 (测试交互)")
    print("   4️⃣  场景 4 - 计划拒绝 (测试取消流程)")

    input("\n按 Enter 开始测试...")

    scenarios = [
        ("1", "简单计划创建 + HIL + 执行", test_simple_plan_creation),
        ("2", "复杂计划 + 依赖关系", test_complex_plan_with_dependencies),
        ("3", "计划编辑工作流", test_plan_edit_workflow),
        ("4", "计划拒绝流程", test_plan_rejection),
    ]

    test_results = []

    while True:
        print("\n" + "=" * 80)
        print("  选择测试场景:")
        print("=" * 80)
        for num, name, _ in scenarios:
            result = next((r for r in test_results if r["num"] == num), None)
            status = ""
            if result:
                status = " ✅" if result["passed"] else " ❌"
            print(f"  [{num}] {name}{status}")
        print("  [A] 运行所有场景")
        print("  [0] 退出并显示测试报告")
        print("=" * 80)

        choice = input("\n请输入选择: ").strip().upper()

        if choice == "0":
            break
        elif choice == "A":
            print("\n开始运行所有测试场景...\n")
            for num, name, func in scenarios:
                print(f"\n{'='*80}")
                print(f"运行场景 {num}: {name}")
                print(f"{'='*80}")
                passed = func()
                test_results.append({"num": num, "name": name, "passed": passed})
                time.sleep(2)
        else:
            for num, name, func in scenarios:
                if choice == num:
                    passed = func()
                    # 更新或添加结果
                    existing = next((r for r in test_results if r["num"] == num), None)
                    if existing:
                        existing["passed"] = passed
                    else:
                        test_results.append({"num": num, "name": name, "passed": passed})
                    break
            else:
                print("❌ 无效选择,请重试")

    # 显示测试报告
    if test_results:
        print("\n" + "📊" * 40)
        print("  测试报告")
        print("📊" * 40 + "\n")

        passed_count = sum(1 for r in test_results if r["passed"])
        total_count = len(test_results)

        for result in test_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"  {status} - 场景 {result['num']}: {result['name']}")

        print(f"\n  总计: {passed_count}/{total_count} 通过")
        print("=" * 80 + "\n")

    print("\n👋 测试结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 测试被中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
