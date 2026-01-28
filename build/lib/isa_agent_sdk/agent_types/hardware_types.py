"""
硬件相关数据类型定义
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time


class DeviceType(Enum):
    """设备类型枚举"""
    SMART_FRAME = "smart_frame"
    ARDUINO = "arduino"
    RASPBERRY_PI = "raspberry_pi"
    TEMPERATURE_SENSOR = "temperature_sensor"
    CAMERA = "camera"
    SPEAKER = "speaker"
    LED_STRIP = "led_strip"
    MOTOR = "motor"
    RELAY = "relay"
    UNKNOWN = "unknown"


class ConnectionProtocol(Enum):
    """连接协议枚举"""
    SERIAL = "serial"
    TCP = "tcp"
    UDP = "udp"
    BLUETOOTH = "bluetooth"
    USB = "usb"
    HTTP = "http"
    WEBSOCKET = "websocket"


class DeviceCapability(Enum):
    """设备能力枚举"""
    DISPLAY = "display"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    VOICE_INPUT = "voice_input"
    CAMERA = "camera"
    SENSOR_READ = "sensor_read"
    GPIO_CONTROL = "gpio_control"
    MOTOR_CONTROL = "motor_control"
    LED_CONTROL = "led_control"
    RELAY_CONTROL = "relay_control"


@dataclass
class DeviceContext:
    """设备上下文信息"""
    device_id: str
    device_type: str
    device_name: Optional[str] = None
    capabilities: List[str] = None
    location: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    firmware_version: Optional[str] = None
    connection_protocol: Optional[str] = None
    connection_info: Optional[Dict[str, Any]] = None
    status: str = "online"
    last_seen: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.last_seen is None:
            self.last_seen = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SensorData:
    """传感器数据"""
    device_id: str
    sensor_type: str
    value: Union[float, int, str, Dict[str, Any]]
    unit: Optional[str] = None
    timestamp: Optional[float] = None
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DeviceCommand:
    """设备命令"""
    device_id: str
    command: str
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=低, 2=中, 3=高
    timeout: Optional[float] = None
    require_confirmation: bool = False
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class DeviceResponse:
    """设备响应格式"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    device_id: Optional[str] = None
    command: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HardwareRequest:
    """硬件请求数据结构"""
    message: str
    user_id: str
    device_context: DeviceContext
    media_files: Optional[List[Dict[str, Any]]] = None
    sensor_data: Optional[SensorData] = None
    trigger_type: str = "user_request"  # user_request, sensor_event, automation
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.media_files is None:
            self.media_files = []
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class HardwareResponse:
    """硬件响应数据结构"""
    text_response: str
    audio_url: Optional[str] = None
    display_data: Optional[Dict[str, Any]] = None
    device_commands: Optional[List[DeviceCommand]] = None
    automation_triggered: bool = False
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.device_commands is None:
            self.device_commands = []
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


# 常用设备配置模板
DEVICE_TEMPLATES = {
    DeviceType.SMART_FRAME.value: {
        "capabilities": [
            DeviceCapability.DISPLAY.value,
            DeviceCapability.AUDIO_OUTPUT.value,
            DeviceCapability.VOICE_INPUT.value,
            DeviceCapability.CAMERA.value
        ],
        "default_protocol": ConnectionProtocol.HTTP.value
    },
    DeviceType.ARDUINO.value: {
        "capabilities": [
            DeviceCapability.GPIO_CONTROL.value,
            DeviceCapability.SENSOR_READ.value,
            DeviceCapability.LED_CONTROL.value
        ],
        "default_protocol": ConnectionProtocol.SERIAL.value
    },
    DeviceType.RASPBERRY_PI.value: {
        "capabilities": [
            DeviceCapability.GPIO_CONTROL.value,
            DeviceCapability.SENSOR_READ.value,
            DeviceCapability.CAMERA.value,
            DeviceCapability.AUDIO_INPUT.value,
            DeviceCapability.AUDIO_OUTPUT.value
        ],
        "default_protocol": ConnectionProtocol.TCP.value
    },
    DeviceType.TEMPERATURE_SENSOR.value: {
        "capabilities": [DeviceCapability.SENSOR_READ.value],
        "default_protocol": ConnectionProtocol.SERIAL.value
    }
}


def create_device_context(
    device_id: str,
    device_type: str,
    **kwargs
) -> DeviceContext:
    """创建设备上下文的便捷函数"""
    template = DEVICE_TEMPLATES.get(device_type, {})
    
    return DeviceContext(
        device_id=device_id,
        device_type=device_type,
        capabilities=kwargs.get('capabilities', template.get('capabilities', [])),
        connection_protocol=kwargs.get('connection_protocol', template.get('default_protocol')),
        **{k: v for k, v in kwargs.items() if k not in ['capabilities', 'connection_protocol']}
    )