#!/usr/bin/env python3
"""
HIL (Human-in-the-Loop) E2E Test Suite
流程：发送请求 → 触发 HIL → 等待用户在前端手动操作 → 验证结果

========================================================
✨ 4 CORE HIL METHODS - Aligned with MCP HIL Example Tools
========================================================

本测试文件与 MCP 服务器的 HIL Example Tools 完全对齐，测试以下 4 种核心 HIL 交互方法：

📋 HIL Method 1: request_authorization()
   工具授权 - Approve/Reject operations
   测试工具: test_authorization_low_risk, test_authorization_high_risk, test_authorization_critical_risk
   用户操作: Approve / Reject

📋 HIL Method 2: request_input()
   收集用户输入 - Collect user input or augment data
   测试工具: test_input_credentials, test_input_selection, test_input_augmentation
   用户操作: Submit / Skip / Cancel

📋 HIL Method 3: request_review()
   审查和编辑 - Review and optionally edit content
   测试工具: test_review_execution_plan, test_review_generated_code, test_review_config_readonly
   用户操作: Approve / Edit / Reject

📋 HIL Method 4: request_input_with_authorization()
   输入+授权 - Input data and authorize action
   测试工具: test_input_with_auth_payment, test_input_with_auth_deployment
   用户操作: Approve with Input / Cancel

========================================================
前置条件 (Prerequisites)
========================================================

1. ✅ MCP 服务器已启动并加载 hil_example_tools
   cd /Users/xenodennis/Documents/Fun/isA_MCP
   python server.py

2. ✅ isA_Agent 后端服务正在运行
   cd /Users/xenodennis/Documents/Fun/isA_Agent
   python main.py

3. ✅ 前端正在运行
   cd frontend_mini && npm run dev
   浏览器访问: http://localhost:5173

4. ✅ MCP 工具已在 Agent 中正确注册

========================================================
测试流程 (Test Flow)
========================================================

1. 运行此测试脚本: python tests/test_hil_scenarios.py
2. 选择测试场景（推荐从场景 1 开始）
3. 观察终端输出，等待 HIL 事件触发
4. 在浏览器中操作 HIL Modal
5. 返回终端查看测试结果

========================================================
期待行为 (Expected Behavior)
========================================================

1. Agent 调用 MCP 工具（例如 test_authorization_low_risk）
2. MCP 工具内部调用 self.request_authorization() 返回 HIL 响应
3. ToolNode 检测到 HIL 响应并调用 interrupt()
4. ChatService 检测到 interrupt 并发送 SSE hil.request 事件
5. 前端显示 HIL Modal，显示对应的 HIL 数据
6. 用户操作后，前端调用 /resume API
7. 执行继续，工具返回结果
8. Agent 继续处理或返回最终响应
"""

import requests
import json
import time
import threading
from typing import Any, Optional

# 配置
API_BASE = "http://localhost:8080"
TEST_USER_ID = "test_user_001"
API_KEY = "authenticated"

# 全局变量
current_test_session = None
hil_detected = False
hil_resumed = False


def print_banner(title: str):
    """打印测试场景标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def send_chat_and_wait_hil(message: str, expected_hil_type: str):
    """
    发送聊天消息，触发 HIL，等待用户在前端手动操作
    """
    global current_test_session, hil_detected, hil_resumed

    # 生成新的 session ID
    current_test_session = f"hil_test_{int(time.time())}"
    hil_detected = False
    hil_resumed = False

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
    print(f"   期待 HIL 类型: {expected_hil_type}\n")

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
                    if hil_detected and hil_resumed:
                        print("✅ ✅ ✅ 测试成功！HIL 已触发并已 Resume")
                    elif hil_detected:
                        print("⚠️  HIL 已触发，但未检测到 Resume")
                    else:
                        print("❌ 未检测到 HIL 事件")
                    break

                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")
                    content = event.get("content", "")

                    # 检测 HIL 请求
                    if event_type == "hil.request":
                        hil_detected = True
                        interrupt_data = event.get("metadata", {}).get("interrupt_data", {})
                        hil_type = interrupt_data.get("type", "unknown")
                        hil_method = interrupt_data.get("method", "unknown")

                        print("\n" + "🔔" * 35)
                        print("🔔  HIL 请求已触发！")
                        print("🔔" * 35)
                        print(f"\n📋 HIL 类型: {hil_type}")
                        print(f"📋 HIL 方法: {hil_method}")
                        print(f"📋 期待类型: {expected_hil_type}")

                        # 更灵活的类型匹配：支持类型或方法匹配
                        if expected_hil_type == "none" or hil_type == expected_hil_type or hil_method == expected_hil_type:
                            print("✅ HIL 类型/方法匹配！\n")
                        else:
                            print(f"⚠️  HIL 类型不匹配（但可能仍然正常）\n")

                        print("📝 HIL 数据:")
                        print(json.dumps(interrupt_data, indent=2, ensure_ascii=False))

                        print("\n" + "⏳" * 35)
                        print("⏳  请在前端 (http://localhost:5173) 操作 HIL Modal")
                        print("⏳  点击 Approve/Reject/输入文本后提交")
                        print("⏳  等待你的操作...")
                        print("⏳" * 35 + "\n")

                    # 检测 Resume 后的响应
                    elif event_type in ["content.token", "content.complete", "session.end"]:
                        if hil_detected and not hil_resumed:
                            print("\n" + "✅" * 35)
                            print("✅  检测到 Resume 后的响应！")
                            print("✅  用户已在前端完成操作")
                            print("✅" * 35 + "\n")
                            hil_resumed = True

                    # 打印事件（简化）
                    if event_type not in ["content.token"]:  # token 太多不打印
                        status = ""
                        if event_type == "hil.request":
                            status = " 🔔 HIL!"
                        elif hil_detected and not hil_resumed and event_type in ["content.complete", "session.end"]:
                            status = " ✅ Resumed!"
                        print(f"📨 {event_type}{status}: {content[:80] if content else ''}")

                except json.JSONDecodeError:
                    pass

        print("-" * 70 + "\n")
        return hil_detected and hil_resumed

    except requests.exceptions.Timeout:
        print("\n❌ 请求超时（120秒）")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return False


def scenario_1_authorization_low_risk():
    """场景 1: 低风险授权 - LOW Risk Authorization"""
    print_banner("场景 1: Authorization - LOW Risk (低风险操作授权)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_authorization_low_risk")
    print("   - HIL 方法: request_authorization()")
    print("   - 操作: Update cache TTL configuration")
    print("   - 风险级别: LOW")
    print("   - 用户操作: Approve / Reject\n")

    print("📋 测试内容:")
    print("   - 修改缓存 TTL 配置从 5 分钟改为 10 分钟")
    print("   - 影响服务: api-gateway, web-frontend")
    print("   - 低风险操作，测试基本授权流程\n")

    message = "Use test_authorization_low_risk tool to test LOW risk authorization"

    print("💡 提示:")
    print("   - Agent 会调用 MCP 工具 test_authorization_low_risk")
    print("   - 工具内部调用 request_authorization() 返回 HIL 数据")
    print("   - 前端显示授权请求，包含操作描述和风险级别")
    print("   - 你需要在前端点击 Approve 或 Reject\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "authorization")


def scenario_2_authorization_high_risk():
    """场景 2: 高风险授权 - HIGH Risk Authorization"""
    print_banner("场景 2: Authorization - HIGH Risk (高风险操作授权)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_authorization_high_risk")
    print("   - HIL 方法: request_authorization()")
    print("   - 操作: Process $5000 payment to vendor")
    print("   - 风险级别: HIGH")
    print("   - 用户操作: Approve / Reject\n")

    print("📋 测试内容:")
    print("   - 向供应商 Acme Corp 支付 $5000")
    print("   - 发票号: INV-2024-001")
    print("   - 支付方式: Stripe")
    print("   - 高风险操作，需要明确授权\n")

    message = "Use test_authorization_high_risk tool to test HIGH risk payment authorization"

    print("💡 提示:")
    print("   - 涉及金额支付，属于高风险操作")
    print("   - 前端应显示详细的支付信息和警告")
    print("   - 用户必须仔细审查后再决定\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "authorization")


def scenario_3_authorization_critical_risk():
    """场景 3: 关键风险授权 - CRITICAL Risk Authorization"""
    print_banner("场景 3: Authorization - CRITICAL Risk (关键风险操作授权)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_authorization_critical_risk")
    print("   - HIL 方法: request_authorization()")
    print("   - 操作: Delete production database table")
    print("   - 风险级别: CRITICAL")
    print("   - 用户操作: Approve / Reject\n")

    print("📋 测试内容:")
    print("   - 删除生产数据库表 'legacy_customers'")
    print("   - 数据量: 50,000 行，250MB")
    print("   - ⚠️ 不可逆操作")
    print("   - 关键风险，需要极其谨慎\n")

    message = "Use test_authorization_critical_risk tool to test CRITICAL risk database deletion"

    print("💡 提示:")
    print("   - 这是最高风险级别的操作")
    print("   - 前端应显示明显的警告标记")
    print("   - 用户必须完全理解后果才能批准\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "authorization")


def scenario_4_input_credentials():
    """场景 4: 输入凭证 - Credentials Input"""
    print_banner("场景 4: Input - Credentials Collection (凭证收集)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_input_credentials")
    print("   - HIL 方法: request_input()")
    print("   - 输入类型: credentials")
    print("   - 用户操作: Submit / Skip / Cancel\n")

    print("📋 测试内容:")
    print("   - 收集 OpenAI API Key")
    print("   - 格式要求: sk-开头，51 字符")
    print("   - 包含输入验证规则")
    print("   - 安全存储，不记录日志\n")

    message = "Use test_input_credentials tool to collect OpenAI API key"

    print("💡 提示:")
    print("   - 前端应显示密码输入框（遮蔽字符）")
    print("   - 显示格式要求和说明")
    print("   - 可以选择 Submit、Skip 或 Cancel\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "input")


def scenario_5_input_selection():
    """场景 5: 输入选择 - Selection Input"""
    print_banner("场景 5: Input - Environment Selection (环境选择)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_input_selection")
    print("   - HIL 方法: request_input()")
    print("   - 输入类型: selection")
    print("   - 用户操作: Submit / Skip / Cancel\n")

    print("📋 测试内容:")
    print("   - 选择部署环境")
    print("   - 选项: development / staging / production")
    print("   - 每个选项都有说明")
    print("   - 测试选择型输入\n")

    message = "Use test_input_selection tool to choose deployment environment"

    print("💡 提示:")
    print("   - 前端应显示下拉选择框或单选按钮")
    print("   - 显示每个选项的说明")
    print("   - 用户选择后 Submit\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "input")


def scenario_6_input_augmentation():
    """场景 6: 输入增强 - Augmentation Input"""
    print_banner("场景 6: Input - Requirements Augmentation (需求增强)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_input_augmentation")
    print("   - HIL 方法: request_input()")
    print("   - 输入类型: augmentation")
    print("   - 用户操作: Submit / Skip / Cancel\n")

    print("📋 测试内容:")
    print("   - AI 已生成初始需求")
    print("   - 请求用户补充遗漏的内容")
    print("   - 提供建议项供参考")
    print("   - 测试数据增强场景\n")

    message = "Use test_input_augmentation tool to add more project requirements"

    print("💡 提示:")
    print("   - 前端显示当前需求列表")
    print("   - 显示建议的补充项")
    print("   - 用户添加额外需求后 Submit\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "input")


def scenario_7_review_execution_plan():
    """场景 7: 审查执行计划 - Review Execution Plan"""
    print_banner("场景 7: Review - Execution Plan (可编辑)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_review_execution_plan")
    print("   - HIL 方法: request_review()")
    print("   - 内容类型: execution_plan")
    print("   - 用户操作: Approve / Edit / Reject\n")

    print("📋 测试内容:")
    print("   - 审查部署执行计划")
    print("   - 包含 4 个任务步骤")
    print("   - 可编辑（添加/删除/修改任务）")
    print("   - 批准后开始自动执行\n")

    message = "Use test_review_execution_plan tool to review deployment plan"

    print("💡 提示:")
    print("   - 前端显示完整的执行计划")
    print("   - 用户可以点击 Edit 修改计划")
    print("   - 或直接 Approve/Reject\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "review")


def scenario_8_review_generated_code():
    """场景 8: 审查生成代码 - Review Generated Code"""
    print_banner("场景 8: Review - Generated Code (可编辑)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_review_generated_code")
    print("   - HIL 方法: request_review()")
    print("   - 内容类型: code")
    print("   - 用户操作: Approve / Edit / Reject\n")

    print("📋 测试内容:")
    print("   - 审查生成的支付处理代码")
    print("   - 包含安全性检查要点")
    print("   - 可直接编辑代码")
    print("   - 测试代码审查场景\n")

    message = "Use test_review_generated_code tool to review payment processing code"

    print("💡 提示:")
    print("   - 前端显示代码高亮")
    print("   - 提供编辑器修改代码")
    print("   - 显示安全检查清单\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "review")


def scenario_9_review_config_readonly():
    """场景 9: 审查配置（只读）- Review Config (Read-only)"""
    print_banner("场景 9: Review - Configuration (只读)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_review_config_readonly")
    print("   - HIL 方法: request_review()")
    print("   - 内容类型: config")
    print("   - 用户操作: Approve / Reject (不可编辑)\n")

    print("📋 测试内容:")
    print("   - 审查生产配置")
    print("   - 包含数据库、缓存、API 配置")
    print("   - 只读模式，不可编辑")
    print("   - 测试只读审查场景\n")

    message = "Use test_review_config_readonly tool to review production configuration"

    print("💡 提示:")
    print("   - 前端显示配置内容，但不可编辑")
    print("   - 只能 Approve 或 Reject")
    print("   - 如需修改，需要 Reject 后单独处理\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "review")


def scenario_10_input_with_auth_payment():
    """场景 10: 输入+授权（支付）- Input with Authorization (Payment)"""
    print_banner("场景 10: Input + Authorization - Payment (组合流程)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_input_with_auth_payment")
    print("   - HIL 方法: request_input_with_authorization()")
    print("   - 输入类型: number")
    print("   - 用户操作: Approve with Input / Cancel\n")

    print("📋 测试内容:")
    print("   - 第一步：输入支付金额（$0.01 - $10,000）")
    print("   - 第二步：授权支付操作（HIGH 风险）")
    print("   - 两步合一，测试组合流程")
    print("   - 涉及真实支付场景\n")

    message = "Use test_input_with_auth_payment tool to process vendor payment"

    print("💡 提示:")
    print("   - 前端首先显示金额输入框")
    print("   - 用户输入后，显示授权确认")
    print("   - 可以在任一步骤取消\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "input_with_authorization")


def scenario_11_input_with_auth_deployment():
    """场景 11: 输入+授权（部署）- Input with Authorization (Deployment)"""
    print_banner("场景 11: Input + Authorization - Deployment (组合流程)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_input_with_auth_deployment")
    print("   - HIL 方法: request_input_with_authorization()")
    print("   - 输入类型: text (JSON)")
    print("   - 用户操作: Approve with Input / Cancel\n")

    print("📋 测试内容:")
    print("   - 第一步：输入生产部署配置（JSON 格式）")
    print("   - 第二步：授权部署操作（CRITICAL 风险）")
    print("   - ⚠️ 影响生产环境")
    print("   - 测试最高风险的组合流程\n")

    message = "Use test_input_with_auth_deployment tool to deploy to production"

    print("💡 提示:")
    print("   - 需要输入完整的环境变量配置")
    print("   - CRITICAL 级别，需要极其谨慎")
    print("   - 测试最复杂的 HIL 场景\n")

    input("按 Enter 开始测试...")

    return send_chat_and_wait_hil(message, "input_with_authorization")


def scenario_12_list_all_scenarios():
    """场景 12: 列出所有 HIL 场景"""
    print_banner("场景 12: List All HIL Scenarios (列出所有场景)")

    print("📝 测试场景:")
    print("   - MCP 工具: test_all_hil_scenarios")
    print("   - 返回所有 12 个 HIL 测试工具的信息")
    print("   - 包含 4 种核心 HIL 方法说明")
    print("   - 用于快速了解可用的测试场景\n")

    message = "Use test_all_hil_scenarios tool to list all available HIL test scenarios"

    print("💡 提示:")
    print("   - 这个工具不会触发 HIL，只是返回信息")
    print("   - 可以看到所有 HIL 方法和对应的测试工具")
    print("   - 用于验证 MCP 工具注册是否成功\n")

    input("按 Enter 开始测试...")

    # 这个不会触发 HIL，但我们仍然发送消息
    return send_chat_and_wait_hil(message, "none")


def main():
    """主测试流程"""
    print("\n" + "🧪" * 35)
    print("  HIL (Human-in-the-Loop) E2E 测试套件")
    print("  基于 MCP HIL Example Tools")
    print("🧪" * 35)

    print("\n📋 测试环境:")
    print(f"   API Base: {API_BASE}")
    print(f"   User ID: {TEST_USER_ID}")

    print("\n⚠️  测试流程:")
    print("   1. 脚本发送聊天消息，Agent 调用 MCP 工具")
    print("   2. MCP 工具触发 HIL 响应")
    print("   3. 前端显示 HIL Modal")
    print("   4. 👉 你在前端手动操作（点击按钮/输入文本）")
    print("   5. 脚本验证是否收到 Resume 响应")

    print("\n📍 前置条件:")
    print("   ✅ MCP 服务器已启动: cd isA_MCP && python server.py")
    print("   ✅ isA_Agent 已启动: cd isA_Agent && python main.py")
    print("   ✅ 前端已启动: cd frontend_mini && npm run dev")
    print("   ✅ 浏览器已打开: http://localhost:5173")

    print("\n🎯 4 种核心 HIL 方法:")
    print("   📋 Method 1: request_authorization() - 场景 1-3")
    print("   📋 Method 2: request_input() - 场景 4-6")
    print("   📋 Method 3: request_review() - 场景 7-9")
    print("   📋 Method 4: request_input_with_authorization() - 场景 10-11")

    print("\n💡 推荐测试顺序:")
    print("   1️⃣  场景 1 - LOW Risk Authorization (最简单)")
    print("   2️⃣  场景 4 - Credentials Input")
    print("   3️⃣  场景 7 - Execution Plan Review")
    print("   4️⃣  场景 10 - Payment Input + Authorization (组合)")
    print("   5️⃣  场景 12 - List All Scenarios (查看所有工具)")

    input("\n按 Enter 开始测试...")

    scenarios = [
        ("1", "Authorization - LOW Risk", scenario_1_authorization_low_risk),
        ("2", "Authorization - HIGH Risk", scenario_2_authorization_high_risk),
        ("3", "Authorization - CRITICAL Risk", scenario_3_authorization_critical_risk),
        ("4", "Input - Credentials", scenario_4_input_credentials),
        ("5", "Input - Selection", scenario_5_input_selection),
        ("6", "Input - Augmentation", scenario_6_input_augmentation),
        ("7", "Review - Execution Plan", scenario_7_review_execution_plan),
        ("8", "Review - Generated Code", scenario_8_review_generated_code),
        ("9", "Review - Config (Read-only)", scenario_9_review_config_readonly),
        ("10", "Input+Auth - Payment", scenario_10_input_with_auth_payment),
        ("11", "Input+Auth - Deployment", scenario_11_input_with_auth_deployment),
        ("12", "List All HIL Scenarios", scenario_12_list_all_scenarios),
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
