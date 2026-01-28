#!/usr/bin/env python3
"""
Background Task Approval 端到端测试
流程：发送请求 → 触发 task.approval → 等待用户在前端手动操作 → 验证结果
"""

import requests
import json
import time

# 配置
API_BASE = "http://localhost:8080"
TEST_USER_ID = "test_user_001"
API_KEY = "authenticated"

# 全局变量
current_test_session = None
task_approval_detected = False
task_approved = False


def print_banner(title: str):
    """打印测试场景标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def send_chat_and_wait_approval(message: str, expected_tools: list):
    """
    发送聊天消息，触发后台任务确认，等待用户在前端手动操作
    """
    global current_test_session, task_approval_detected, task_approved

    # 生成新的 session ID
    current_test_session = f"task_approval_test_{int(time.time())}"
    task_approval_detected = False
    task_approved = False

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

    print(f"📤 发送聊天消息: {message}")
    print(f"   Session ID: {current_test_session}")
    print(f"   User ID: {TEST_USER_ID}")
    print(f"   期待工具: {', '.join(expected_tools)}\n")

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)

        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"   Response: {response.text}\n")
            return False

        print("✅ 聊天请求成功")
        print("📡 开始监听 SSE 流...\n")
        print("-" * 70)

        # 监听 SSE 流
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:]

                # 检查是否结束
                if data_str.strip() == "[DONE]":
                    print("\n" + "-" * 70)
                    print("✅ SSE 流结束")
                    if task_approval_detected and task_approved:
                        print("✅ ✅ ✅ 测试成功！Task Approval 已触发并已确认")
                    elif task_approval_detected:
                        print("⚠️  Task Approval 已触发，但未检测到确认响应")
                    else:
                        print("❌ 未检测到 Task Approval 事件")
                    break

                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")
                    content = event.get("content", "")

                    # 检测 Task Approval 请求
                    if event_type == "task.approval":
                        task_approval_detected = True
                        metadata = event.get("metadata", {})
                        job_id = metadata.get("job_id", "unknown")
                        task_data = metadata.get("task", {})
                        tools = task_data.get("tools", [])

                        print("\n" + "⚙️ " * 35)
                        print("⚙️   Task Approval 请求已触发！")
                        print("⚙️ " * 35)
                        print(f"\n📋 Job ID: {job_id}")
                        print(f"📋 Tools Count: {len(tools)}")

                        print("\n📝 Tools:")
                        for i, tool in enumerate(tools, 1):
                            tool_name = tool.get("tool_name", "unknown")
                            print(f"   {i}. {tool_name}")

                        print("\n📝 完整数据:")
                        print(json.dumps(metadata, indent=2, ensure_ascii=False))

                        print("\n" + "⏳" * 35)
                        print("⏳  请在前端 (http://localhost:5173) 操作 Task Approval Modal")
                        print("⏳  点击 [APPROVE & QUEUE TASK] 或 [REJECT TASK]")
                        print("⏳  等待你的操作...")
                        print("⏳" * 35 + "\n")

                    # 检测确认后的响应
                    elif event_type in ["task.queued", "task.rejected", "content.token", "content.complete"]:
                        if task_approval_detected and not task_approved:
                            print("\n" + "✅" * 35)
                            print("✅  检测到确认后的响应！")
                            print("✅  用户已在前端完成操作")
                            print("✅" * 35 + "\n")
                            task_approved = True

                    # 打印事件（简化）
                    if event_type not in ["content.token"]:  # token 太多不打印
                        status = ""
                        if event_type == "task.approval":
                            status = " ⚙️ Task Approval!"
                        elif task_approval_detected and not task_approved and event_type in ["task.queued", "content.complete"]:
                            status = " ✅ Confirmed!"
                        print(f"📨 {event_type}{status}: {content[:80] if content else ''}")

                except json.JSONDecodeError:
                    pass

        print("-" * 70 + "\n")
        return task_approval_detected and task_approved

    except requests.exceptions.Timeout:
        print("\n❌ 请求超时（120秒）")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_1_web_crawl():
    """测试场景 1: Web 爬虫后台任务"""
    print_banner("场景 1: Web Crawl Background Job")

    print("📝 测试场景:")
    print("   - 请求执行多个网页爬取任务")
    print("   - Agent 将任务提交给后台队列前，请求用户确认")
    print("   - 前端显示 TaskApprovalModal，列出所有工具")
    print("   - 用户可以批准、拒绝、或修改优先级\n")

    message = "请帮我爬取以下网站的内容：example.com, test.com, demo.com"

    print("💡 提示: 这需要 Agent 配置在提交后台任务前触发 task.approval")
    print("   预期工具: web_crawl (多次调用)\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_approval(message, ["web_crawl"])


def test_scenario_2_data_processing():
    """测试场景 2: 数据处理后台任务"""
    print_banner("场景 2: Data Processing Background Job")

    print("📝 测试场景:")
    print("   - 请求执行耗时的数据处理任务")
    print("   - 包含多个步骤：读取、处理、分析、保存")
    print("   - 用户确认是否在后台执行\n")

    message = "请处理 data.csv 文件：清洗数据、统计分析、生成报告"

    print("💡 提示: 预期工具: file_read, data_clean, analyze, generate_report\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_approval(message, ["file_read", "data_clean", "analyze", "generate_report"])


def test_scenario_3_api_calls():
    """测试场景 3: 批量 API 调用后台任务"""
    print_banner("场景 3: Batch API Calls Background Job")

    print("📝 测试场景:")
    print("   - 批量调用外部 API")
    print("   - 每个 API 调用作为独立工具")
    print("   - 用户确认资源消耗和优先级\n")

    message = "请调用天气 API 获取北京、上海、广州、深圳的天气"

    print("💡 提示: 预期工具: api_call (多次调用)\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_approval(message, ["api_call"])


def test_scenario_4_priority_modification():
    """测试场景 4: 修改任务优先级"""
    print_banner("场景 4: Modify Task Priority")

    print("📝 测试场景:")
    print("   - 提交高优先级后台任务")
    print("   - 用户可以在前端降低优先级为 normal 或 low")
    print("   - 测试优先级修改功能\n")

    message = "紧急任务：请立即执行完整系统备份"

    print("💡 提示: 在前端将优先级从 HIGH 改为 NORMAL\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_approval(message, ["system_backup"])


def main():
    """主测试流程"""
    print("\n" + "🧪" * 35)
    print("  Background Task Approval 端到端测试")
    print("🧪" * 35)

    print("\n📋 测试环境:")
    print(f"   API Base: {API_BASE}")
    print(f"   User ID: {TEST_USER_ID}")

    print("\n⚠️  测试流程:")
    print("   1. 脚本发送聊天消息，触发后台任务请求")
    print("   2. Agent 发送 task.approval 事件到前端")
    print("   3. 前端显示 TaskApprovalModal")
    print("   4. 👉 你在前端手动操作（批准/拒绝/修改优先级）")
    print("   5. 脚本验证是否收到确认响应")

    print("\n📍 前置条件:")
    print("   ✅ 后端已启动: python main.py")
    print("   ✅ 前端已启动: cd frontend_mini && npm run dev")
    print("   ✅ 浏览器已打开: http://localhost:5173")
    print("   ✅ Agent 已配置 task.approval 支持")
    print("   ✅ NATS + Redis 服务已启动（Docker）")

    print("\n💡 提示:")
    print("   - 需要 Agent 在提交后台任务前触发 task.approval 事件")
    print("   - 事件格式: {type: 'task.approval', metadata: {job_id, task: {...}}}")
    print("   - 前端会自动显示 TaskApprovalModal")

    input("\n按 Enter 开始测试...")

    scenarios = [
        ("1", "Web Crawl Background Job", test_scenario_1_web_crawl),
        ("2", "Data Processing Background Job", test_scenario_2_data_processing),
        ("3", "Batch API Calls Background Job", test_scenario_3_api_calls),
        ("4", "Modify Task Priority", test_scenario_4_priority_modification),
    ]

    test_results = []

    while True:
        print("\n" + "=" * 70)
        print("  选择测试场景:")
        print("=" * 70)
        for num, name, _ in scenarios:
            result = next((r for r in test_results if r["num"] == num), None)
            status = ""
            if result:
                status = " ✅" if result["passed"] else " ❌"
            print(f"  [{num}] {name}{status}")
        print("  [A] 运行所有场景")
        print("  [0] 退出并显示测试报告")
        print("=" * 70)

        choice = input("\n请输入选择: ").strip().upper()

        if choice == "0":
            break
        elif choice == "A":
            print("\n开始运行所有测试场景...\n")
            for num, name, func in scenarios:
                print(f"\n{'='*70}")
                print(f"运行场景 {num}: {name}")
                print(f"{'='*70}")
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
                print("❌ 无效选择，请重试")

    # 显示测试报告
    if test_results:
        print("\n" + "📊" * 35)
        print("  测试报告")
        print("📊" * 35 + "\n")

        passed_count = sum(1 for r in test_results if r["passed"])
        total_count = len(test_results)

        for result in test_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"  {status} - 场景 {result['num']}: {result['name']}")

        print(f"\n  总计: {passed_count}/{total_count} 通过")
        print("=" * 70 + "\n")

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
