# Background Jobs System - NATS + Redis

完整的后台任务处理系统，使用 NATS JetStream 作为任务队列（替代 Celery），Redis 作为状态管理和进度发布。

## 🎯 架构概览

```
┌─────────────────┐
│   API Request   │
└────────┬────────┘
         │
         ├──────► enqueue_task()
         │                │
         │                ▼
         │        ┌───────────────┐
         │        │  NATS Queue   │◄────┐
         │        │  (JetStream)  │     │
         │        └───────┬───────┘     │
         │                │              │
         ▼                ▼              │
┌────────────────┐  ┌──────────────┐   │
│  Redis State   │  │Task Worker(s)│───┘
│   Management   │◄─┤  Pull Tasks  │
│  + Pub/Sub     │  │  Execute     │
└────────────────┘  └──────────────┘
         │                │
         │                │
         ▼                ▼
   Progress Events    Tool Execution
   (SSE Streaming)    (via MCP)
```

## ✅ 问题解决

### 1. Consul 服务发现配置

**问题**: NATS 和 Redis 服务在 Consul 中注册的名称与代码中查找的不同。

**解决方案**:
- NATS 服务名: `nats_grpc_service` (注意：使用下划线，不是连字符)
- Redis 服务名: `redis_grpc_service` (注意：使用下划线，不是连字符)

代码中已更新 `service_name_override` 参数：

```python
# nats_task_queue.py
self.nats_client = NATSClient(
    user_id=self.user_id,
    consul_registry=consul_registry,
    service_name_override="nats_grpc_service"  # ✅ 正确的服务名 (underscore)
)

# redis_state_manager.py
self.redis_client = RedisClient(
    user_id=self.user_id,
    organization_id=self.organization_id,
    consul_registry=consul_registry,
    service_name_override="redis_grpc_service",  # ✅ 正确的服务名 (underscore)
    lazy_connect=False
)
```

### 2. isa_model 导入修复

**问题**: `from isa_model.inference_client import AsyncISAModel` 模块不存在

**解决方案**: 使用正确的导入路径
```python
# ❌ 错误
from isa_model.inference_client import AsyncISAModel

# ✅ 正确
from isa_model import ISAModelClient as AsyncISAModel
```

已修复文件：
- `src/components/model_service.py`
- `src/clients/model_client.py`

## 📁 项目结构

```
src/services/background_jobs/
├── __init__.py              # 导出接口和高级 API
├── task_models.py           # 任务数据模型 (Pydantic)
├── nats_task_queue.py       # NATS JetStream 任务队列
├── redis_state_manager.py   # Redis 状态管理 + Pub/Sub
├── task_worker.py           # Worker 执行逻辑
├── simple_test.py           # 组件测试脚本
├── test_background_jobs.py  # 完整测试套件
└── README.md                # 本文档
```

## 🚀 快速开始

### 1. 入队任务

```python
from src.services.background_jobs import (
    enqueue_task,
    TaskDefinition,
    ToolCallInfo
)

# 创建任务
task = TaskDefinition(
    job_id="job_123",
    session_id="sess_456",
    user_id="user_789",
    tools=[
        ToolCallInfo(
            tool_name="web_crawl",
            tool_args={"url": "https://example.com"},
            tool_call_id="call_1"
        ),
        ToolCallInfo(
            tool_name="summarize",
            tool_args={"text": "..."},
            tool_call_id="call_2"
        )
    ],
    priority="high"  # "low", "normal", "high"
)

# 入队
sequence = enqueue_task(task)
print(f"Task queued with sequence: {sequence}")
```

### 2. 查询任务状态

```python
from src.services.background_jobs import get_task_status

progress = get_task_status("job_123")
print(f"Status: {progress.status}")
print(f"Progress: {progress.progress_percent}%")
print(f"Completed: {progress.completed_tools}/{progress.total_tools}")
```

### 3. 获取最终结果

```python
from src.services.background_jobs import get_task_result

result = get_task_result("job_123")
print(f"Status: {result.status}")
print(f"Success: {result.successful_tools}/{result.total_tools}")

for tool_result in result.results:
    print(f"  {tool_result.tool_name}: {tool_result.status}")
```

### 4. 启动 Worker

```bash
# 启动 worker 处理所有优先级任务
python -m app.services.background_jobs.task_worker --name worker-1

# 只处理高优先级任务
python -m app.services.background_jobs.task_worker --name worker-high --priority high

# 只处理普通优先级任务
python -m app.services.background_jobs.task_worker --name worker-normal --priority normal
```

## 📊 数据模型

### TaskDefinition
```python
class TaskDefinition(BaseModel):
    job_id: str                      # 任务 ID
    session_id: str                  # 会话 ID
    user_id: Optional[str]           # 用户 ID
    tools: List[ToolCallInfo]        # 工具调用列表
    config: Dict[str, Any]           # 运行时配置
    created_at: datetime             # 创建时间
    priority: Literal["low", "normal", "high"]  # 优先级
```

### TaskProgress
```python
class TaskProgress(BaseModel):
    job_id: str
    status: TaskStatus               # QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
    total_tools: int
    completed_tools: int
    failed_tools: int
    progress_percent: float
    current_tool: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
```

### TaskResult
```python
class TaskResult(BaseModel):
    job_id: str
    session_id: str
    status: TaskStatus
    total_tools: int
    successful_tools: int
    failed_tools: int
    results: List[ToolExecutionResult]
    started_at: datetime
    completed_at: datetime
    execution_time_seconds: float
```

## 🔧 配置

### 环境变量

无需额外配置！系统自动通过 Consul 发现服务：

- **NATS**: 自动发现 `nats-grpc-service`
- **Redis**: 自动发现 `redis-grpc-service`
- **Consul**: 默认 `localhost:8500`

### 队列配置

在 `nats_task_queue.py` 中：

```python
STREAM_NAME = "ISA_AGENT_TASKS"
SUBJECT_PREFIX = "isa.agent.tasks"

# Priority subjects
SUBJECT_HIGH = "isa.agent.tasks.high"
SUBJECT_NORMAL = "isa.agent.tasks.normal"
SUBJECT_LOW = "isa.agent.tasks.low"
```

## 🧪 测试

### 运行组件测试

```bash
python src/services/background_jobs/simple_test.py
```

测试内容：
- ✅ Consul 服务发现
- ✅ Redis 连接和操作
- ✅ NATS 连接和 JetStream
- ✅ 任务模型序列化

### 运行完整测试（需要在 Docker 环境中）

```bash
python src/services/background_jobs/test_background_jobs.py
```

## 📈 监控和统计

```python
from src.services.background_jobs import get_queue_statistics

stats = get_queue_statistics()
print(f"Queued: {stats['tasks_queued']}")
print(f"Completed: {stats['tasks_completed']}")
print(f"Failed: {stats['tasks_failed']}")
print(f"Active: {stats['active_tasks']}")
```

## 🔄 集成到现有代码

### tool_node.py 集成示例

```python
async def _queue_background_job(self, tool_info_list, state, config):
    """Queue tools as background job"""
    from ..services.background_jobs import (
        enqueue_task,
        TaskDefinition,
        ToolCallInfo
    )

    job_id = f"job_{uuid.uuid4().hex[:12]}"

    tools = [
        ToolCallInfo(
            tool_name=t[0],
            tool_args=t[1],
            tool_call_id=t[2]
        )
        for t in tool_info_list
    ]

    task = TaskDefinition(
        job_id=job_id,
        session_id=state.get("session_id"),
        user_id=state.get("user_id"),
        tools=tools,
        config=dict(config),
        priority="normal"
    )

    sequence = enqueue_task(task)

    return {
        "status": "queued",
        "job_id": job_id,
        "nats_sequence": sequence,
        "poll_url": f"/api/v1/jobs/{job_id}",
        "sse_url": f"/api/v1/jobs/{job_id}/stream"
    }
```

## 🎯 核心特性

- ✅ **服务发现**: 通过 Consul 自动发现 NATS 和 Redis
- ✅ **优先级队列**: 支持 high/normal/low 三个优先级
- ✅ **进度追踪**: 实时任务进度更新
- ✅ **Pub/Sub**: Redis pub/sub 用于进度事件流
- ✅ **容错性**: 任务失败自动重试
- ✅ **可扩展**: 支持多 Worker 横向扩展
- ✅ **持久化**: NATS JetStream 持久化消息
- ✅ **监控**: 完整的统计和健康检查

## 🚨 注意事项

1. **Docker 网络**: 在本地测试时无法连接 Docker 内部网络服务，需要在 staging 环境测试
2. **Worker 部署**: 需要单独启动 Worker 进程来处理任务
3. **Redis TTL**: 任务状态默认保留 1 小时，结果保留 2 小时
4. **NATS Stream**: 最多保留 100k 消息，500MB 存储

## 📝 后续工作

- [ ] 更新 `tool_node.py` 集成新的任务队列
- [ ] 更新 `jobs.py` API 端点
- [ ] 创建 Worker Dockerfile 和部署配置
- [ ] 添加 Prometheus 监控指标
- [ ] 实现任务取消和超时机制

## 📚 相关文档

- [NATS Client 使用指南](/Users/xenodennis/Documents/Fun/isA_Cloud/isA_common/docs/how_to_nats_client.md)
- [Redis Client 使用指南](/Users/xenodennis/Documents/Fun/isA_Cloud/isA_common/docs/how_to_redis_client.md)
- [Consul 服务发现](https://www.consul.io/docs/discovery)

## 🎉 总结

通过本次重构：
1. ✅ 移除了 Celery 依赖
2. ✅ 使用 NATS JetStream 作为高性能任务队列
3. ✅ 使用 isA_common 统一客户端库
4. ✅ 通过 Consul 实现服务发现
5. ✅ 修复了所有导入问题
6. ✅ 完善了测试覆盖

**系统已准备好在 staging 环境中部署！** 🚀
