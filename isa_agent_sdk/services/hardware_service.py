"""
硬件服务 - 处理硬件设备请求和上下文
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from isa_agent_sdk.agent_types.hardware_types import (
    DeviceContext, HardwareRequest, HardwareResponse, 
    SensorData, DeviceCommand, DeviceType
)

logger = logging.getLogger(__name__)


class HardwareService:
    """
    硬件服务 - 管理设备上下文和处理硬件相关请求
    
    功能:
    - 识别和验证硬件设备请求
    - 管理设备状态和上下文
    - 处理传感器数据和自动化触发
    - 为不同设备类型生成适配的响应格式
    """
    
    def __init__(self):
        self.registered_devices: Dict[str, DeviceContext] = {}
        self.device_last_seen: Dict[str, float] = {}
        self.automation_rules: List[Dict[str, Any]] = []
        
    async def process_hardware_request(
        self, 
        message: str, 
        user_id: str, 
        request_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        处理硬件设备请求
        
        Args:
            message: 原始消息内容
            user_id: 用户ID (通常是设备ID)
            request_data: 请求数据，包含device_context等
            
        Returns:
            处理后的请求数据，如果不是硬件请求则返回None
        """
        device_context = request_data.get("device_context")
        
        if not device_context:
            return None
            
        try:
            # 创建设备上下文对象
            if isinstance(device_context, dict):
                context = DeviceContext(**device_context)
            else:
                context = device_context
            
            # 注册/更新设备信息
            await self._register_or_update_device(context)
            
            # 检查是否为传感器事件触发
            sensor_data = request_data.get("sensor_data")
            trigger_type = request_data.get("trigger_type", "user_request")
            
            if sensor_data and trigger_type == "sensor_event":
                # 处理传感器自动化
                automation_triggered = await self._check_automation_rules(sensor_data, context)
                request_data["automation_triggered"] = automation_triggered
            
            # 添加设备特定的处理逻辑
            request_data["device_context"] = context
            request_data["is_hardware_request"] = True
            
            logger.info(f"Hardware request processed: {context.device_type} - {context.device_id}")
            return request_data
            
        except Exception as e:
            logger.error(f"Failed to process hardware request: {e}")
            return None
    
    async def format_response_for_device(
        self, 
        response_content: str,
        device_context: DeviceContext,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> HardwareResponse:
        """
        为特定设备格式化响应
        
        Args:
            response_content: Agent生成的响应内容
            device_context: 设备上下文
            additional_data: 额外数据
            
        Returns:
            格式化的硬件响应
        """
        hardware_response = HardwareResponse(
            text_response=response_content,
            metadata=additional_data or {}
        )
        
        # 根据设备类型和能力定制响应
        if device_context.device_type == DeviceType.SMART_FRAME.value:
            hardware_response = await self._format_smart_frame_response(
                hardware_response, device_context, response_content
            )
        elif device_context.device_type == DeviceType.ARDUINO.value:
            hardware_response = await self._format_arduino_response(
                hardware_response, device_context, response_content
            )
        
        return hardware_response
    
    async def _register_or_update_device(self, device_context: DeviceContext):
        """注册或更新设备信息"""
        device_id = device_context.device_id
        current_time = datetime.now().timestamp()
        
        # 更新设备注册信息
        self.registered_devices[device_id] = device_context
        self.device_last_seen[device_id] = current_time
        
        logger.info(f"Device registered/updated: {device_context.device_type} - {device_id}")
    
    async def _check_automation_rules(
        self, 
        sensor_data: Dict[str, Any], 
        device_context: DeviceContext
    ) -> bool:
        """
        检查传感器数据是否触发自动化规则
        
        Args:
            sensor_data: 传感器数据
            device_context: 设备上下文
            
        Returns:
            是否触发了自动化规则
        """
        try:
            sensor_obj = SensorData(**sensor_data) if isinstance(sensor_data, dict) else sensor_data
            
            # 示例自动化规则：温度过高报警
            if sensor_obj.sensor_type == "temperature" and isinstance(sensor_obj.value, (int, float)):
                if sensor_obj.value > 30:  # 温度超过30度
                    logger.info(f"Temperature alert triggered: {sensor_obj.value}°C from {device_context.device_id}")
                    return True
            
            # 可以在这里添加更多自动化规则
            return False
            
        except Exception as e:
            logger.error(f"Failed to check automation rules: {e}")
            return False
    
    async def _format_smart_frame_response(
        self, 
        response: HardwareResponse,
        device_context: DeviceContext,
        content: str
    ) -> HardwareResponse:
        """格式化智能相框响应"""
        # 智能相框需要音频和显示数据
        if "audio_output" in device_context.capabilities:
            # 生成TTS音频URL (这里是模拟，实际需要调用TTS服务)
            response.audio_url = f"/api/tts/generate?text={content[:100]}&device_id={device_context.device_id}"
        
        if "display" in device_context.capabilities:
            # 准备显示数据
            response.display_data = {
                "type": "text",
                "content": content,
                "display_time": 10,  # 显示10秒
                "font_size": "medium",
                "background_color": "#000000",
                "text_color": "#FFFFFF"
            }
            
            # 如果内容包含结构化信息，提取显示元素
            if "天气" in content:
                response.display_data["type"] = "weather_widget"
            elif "日程" in content or "计划" in content:
                response.display_data["type"] = "schedule_widget"
        
        return response
    
    async def _format_arduino_response(
        self, 
        response: HardwareResponse,
        device_context: DeviceContext,
        content: str
    ) -> HardwareResponse:
        """格式化Arduino响应"""
        # Arduino主要通过GPIO控制，提取控制指令
        device_commands = []
        
        # 解析内容中的控制指令
        if "开启LED" in content or "打开灯" in content:
            device_commands.append(DeviceCommand(
                device_id=device_context.device_id,
                command="gpio_write",
                parameters={"pin": 13, "value": 1},  # 假设LED连接到13号引脚
                require_confirmation=True
            ))
        
        if "关闭LED" in content or "关灯" in content:
            device_commands.append(DeviceCommand(
                device_id=device_context.device_id,
                command="gpio_write", 
                parameters={"pin": 13, "value": 0},
                require_confirmation=True
            ))
        
        if "读取温度" in content:
            device_commands.append(DeviceCommand(
                device_id=device_context.device_id,
                command="sensor_read",
                parameters={"sensor_type": "temperature", "pin": "A0"}
            ))
        
        response.device_commands = device_commands
        return response
    
    def get_registered_devices(self) -> Dict[str, DeviceContext]:
        """获取所有注册的设备"""
        return self.registered_devices.copy()
    
    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """获取设备状态"""
        if device_id not in self.registered_devices:
            return None
            
        device = self.registered_devices[device_id]
        last_seen = self.device_last_seen.get(device_id, 0)
        
        # 判断设备是否在线 (5分钟内有活动)
        is_online = (datetime.now().timestamp() - last_seen) < 300
        
        return {
            "device_id": device_id,
            "device_type": device.device_type,
            "device_name": device.device_name,
            "status": "online" if is_online else "offline",
            "last_seen": last_seen,
            "capabilities": device.capabilities,
            "location": device.location
        }
    
    def add_automation_rule(self, rule: Dict[str, Any]):
        """添加自动化规则"""
        self.automation_rules.append(rule)
        logger.info(f"Automation rule added: {rule.get('name', 'unnamed')}")
    
    def get_device_summary(self) -> Dict[str, Any]:
        """获取设备摘要统计"""
        total_devices = len(self.registered_devices)
        device_types = {}
        online_count = 0
        
        current_time = datetime.now().timestamp()
        
        for device_id, device in self.registered_devices.items():
            # 统计设备类型
            device_type = device.device_type
            device_types[device_type] = device_types.get(device_type, 0) + 1
            
            # 统计在线设备
            last_seen = self.device_last_seen.get(device_id, 0)
            if (current_time - last_seen) < 300:  # 5分钟内活跃
                online_count += 1
        
        return {
            "total_devices": total_devices,
            "online_devices": online_count,
            "offline_devices": total_devices - online_count,
            "device_types": device_types,
            "automation_rules": len(self.automation_rules)
        }


# 全局硬件服务实例
_hardware_service: Optional[HardwareService] = None


def get_hardware_service() -> HardwareService:
    """获取硬件服务实例"""
    global _hardware_service
    
    if _hardware_service is None:
        _hardware_service = HardwareService()
    
    return _hardware_service