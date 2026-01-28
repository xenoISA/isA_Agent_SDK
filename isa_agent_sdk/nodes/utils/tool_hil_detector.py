#!/usr/bin/env python3
"""
Tool HIL Detector - 检测MCP工具返回的HIL响应

简单检测MCP返回的HIL请求，根据MCP HIL Integration Guide规范。
"""

import json
from typing import Optional, Dict, Any
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger


class ToolHILDetector:
    """检测MCP工具返回的HIL响应"""

    @staticmethod
    def is_hil_response(result) -> bool:
        """
        检测是否是HIL响应

        支持两种MCP返回格式:
        1. 字符串JSON: '{"hil_required": true, ...}'
        2. MCP完整响应: {'result': {'content': [{'type': 'text', 'text': '...'}]}}

        Args:
            result: MCP工具返回的结果 (str 或 dict)

        Returns:
            True if HIL required
        """
        try:
            # Case 1: Result is a dict (MCP full response format)
            if isinstance(result, dict):
                # Check if it's MCP response format: {'result': {'content': [...]}}
                if 'result' in result:
                    result_data = result['result']

                    # Extract text from MCP content format
                    if isinstance(result_data, dict) and 'content' in result_data:
                        content_list = result_data['content']
                        if isinstance(content_list, list) and len(content_list) > 0:
                            first_content = content_list[0]
                            if isinstance(first_content, dict) and 'text' in first_content:
                                # Parse the embedded JSON string
                                text = first_content['text']
                                if isinstance(text, str):
                                    parsed = json.loads(text)
                                    if isinstance(parsed, dict):
                                        return parsed.get("hil_required") is True

                # Direct dict format (less common)
                return result.get("hil_required") is True

            # Case 2: Result is a JSON string
            if isinstance(result, str) and result.startswith('{'):
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed.get("hil_required") is True

        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            logger.debug(f"[HILDetector] Not a HIL response: {e}")
            pass
        return False

    @staticmethod
    def detect_hil_type(result) -> Optional[str]:
        """
        检测HIL类型 (4种核心类型)

        根据MCP返回的 status 和 hil_type 字段判断:
        - authorization
        - input
        - review
        - input_with_authorization

        Args:
            result: MCP工具返回的结果 (str 或 dict)

        Returns:
            HIL类型或None
        """
        try:
            # Extract parsed JSON from MCP response
            parsed = ToolHILDetector._extract_json_from_result(result)
            if not parsed:
                return None

            status = parsed.get("status")
            hil_type = parsed.get("hil_type")

            # 根据MCP标准映射
            if status == "authorization_requested":
                if hil_type == "input_with_authorization":
                    return "input_with_authorization"
                return "authorization"
            elif status == "human_input_requested":
                if hil_type == "review":
                    return "review"
                return "input"

        except (TypeError, KeyError) as e:
            logger.debug(f"[HILDetector] Failed to detect HIL type: {e}")
            pass
        return None

    @staticmethod
    def extract_hil_data(result) -> Dict[str, Any]:
        """
        提取HIL数据

        提取MCP返回的完整HIL数据结构

        Args:
            result: MCP工具返回的结果 (str 或 dict)

        Returns:
            HIL数据字典
        """
        try:
            # Extract parsed JSON from MCP response
            parsed = ToolHILDetector._extract_json_from_result(result)
            if not parsed:
                return {}

            return {
                "status": parsed.get("status"),
                "hil_type": parsed.get("hil_type"),
                "action": parsed.get("action"),
                "question": parsed.get("question"),
                "message": parsed.get("message"),
                "context": parsed.get("context", {}),
                "options": parsed.get("options", []),
                "timeout": parsed.get("timeout", 300),
                "data": parsed.get("data", {}),
                "original": parsed
            }
        except (TypeError, KeyError) as e:
            logger.error(f"[HILDetector] Failed to extract HIL data: {e}")
            return {}

    @staticmethod
    def _extract_json_from_result(result) -> Optional[Dict[str, Any]]:
        """
        从MCP响应中提取JSON数据

        支持格式:
        1. MCP完整响应: {'result': {'content': [{'type': 'text', 'text': '...'}]}}
        2. 字符串JSON: '{"hil_required": true, ...}'
        3. 直接dict: {"hil_required": true, ...}

        Args:
            result: MCP工具返回的结果

        Returns:
            解析后的JSON字典，或None
        """
        try:
            # Case 1: Dict with MCP format
            if isinstance(result, dict):
                # MCP response format: {'result': {'content': [...]}}
                if 'result' in result:
                    result_data = result['result']
                    if isinstance(result_data, dict) and 'content' in result_data:
                        content_list = result_data['content']
                        if isinstance(content_list, list) and len(content_list) > 0:
                            first_content = content_list[0]
                            if isinstance(first_content, dict) and 'text' in first_content:
                                text = first_content['text']
                                if isinstance(text, str):
                                    return json.loads(text)

                # Direct dict format
                return result

            # Case 2: JSON string
            if isinstance(result, str) and result.startswith('{'):
                return json.loads(result)

        except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
            logger.debug(f"[HILDetector] Failed to parse result: {e}")
            pass

        return None
