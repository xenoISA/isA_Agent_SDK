"""
Session Service Checkpointer - 使用 session_service 的 messages API 作为持久化后端

核心思路：
- 利用 session_service 现有的 messages API
- 将 checkpoint 序列化后作为特殊的 message 存储
- message_type = "checkpoint" 来标识 checkpoint 消息
- metadata 中存储 checkpoint_id 和其他元信息

优势：
- 无需修改 session_service（使用现有 API）
- 消息和 checkpoint 存储在同一个微服务
- 支持跨服务访问和查询
- 统一的数据管理和备份
"""
from __future__ import annotations

from typing import Iterator, Optional, Any, Dict
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
    CheckpointMetadata,
    ChannelVersions,
)
from langchain_core.runnables import RunnableConfig
from isa_agent_sdk.utils.logger import api_logger
import json
import httpx
from datetime import datetime


class SessionServiceCheckpointer(BaseCheckpointSaver):
    """
    使用 session_service messages API 作为后端的 Checkpointer

    实现策略：
    - checkpoint 作为特殊的 message 存储（message_type="checkpoint"）
    - role="system"
    - content 存储序列化的 checkpoint 数据
    - metadata 存储 checkpoint_id 和其他元信息
    """

    def __init__(
        self,
        session_service_url: str,
        auth_token: Optional[str] = None,
        user_id: str = "system",
        timeout: float = 30.0,
        **kwargs
    ):
        """
        初始化 Session Service Checkpointer

        Args:
            session_service_url: session_service 的基础 URL (例如: http://localhost:8203)
            auth_token: 可选的认证 token
            user_id: 用于 API 调用的 user_id（默认 "system"）
            timeout: HTTP 请求超时时间（秒）
        """
        super().__init__(**kwargs)
        self.base_url = session_service_url.rstrip('/')
        self.auth_token = auth_token
        self.user_id = user_id
        self.client = httpx.AsyncClient(timeout=timeout)
        api_logger.info(f"✅ SessionServiceCheckpointer initialized | url={self.base_url}")

    def _get_headers(self) -> dict:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def _ensure_session_exists(self, session_id: str, user_id: str) -> bool:
        """确保 session 存在，如果不存在则创建

        Args:
            session_id: 会话 ID
            user_id: 用户 ID

        Returns:
            bool: 是否成功（session 已存在或成功创建）
        """
        try:
            # Check if session exists
            url = f"{self.base_url}/api/v1/sessions/{session_id}"
            response = await self.client.get(url, headers=self._get_headers())

            if response.status_code == 200:
                return True  # Session exists
            elif response.status_code == 404:
                # Session doesn't exist, create it
                api_logger.info(f"[CHECKPOINT:SESSION] Creating session | session_id={session_id}")
                create_url = f"{self.base_url}/api/v1/sessions"
                create_payload = {
                    "user_id": user_id,
                    "session_id": session_id,  # Top-level field (not in metadata!)
                    "metadata": {
                        "source": "checkpointer",
                        "auto_created": True
                    }
                }
                create_response = await self.client.post(
                    create_url,
                    json=create_payload,
                    headers=self._get_headers()
                )

                if create_response.status_code in [200, 201]:
                    api_logger.info(f"[CHECKPOINT:SESSION] Session created | session_id={session_id}")
                    return True
                else:
                    api_logger.error(f"[CHECKPOINT:SESSION] Failed to create session | status={create_response.status_code} | response={create_response.text}")
                    return False
            else:
                api_logger.error(f"[CHECKPOINT:SESSION] Unexpected status checking session | status={response.status_code}")
                return False

        except Exception as e:
            api_logger.error(f"[CHECKPOINT:SESSION] Error ensuring session exists: {e}")
            return False

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """序列化 checkpoint

        dumps_typed() returns a tuple of (type_str, bytes_data)
        We need to store both parts to reconstruct later
        """
        try:
            type_str, bytes_data = self.serde.dumps_typed(checkpoint)
            # Store as JSON with both type and base64-encoded data
            import base64
            return json.dumps({
                "type": type_str,
                "data": base64.b64encode(bytes_data).decode('utf-8')
            })
        except Exception as e:
            api_logger.error(f"Failed to serialize checkpoint: {e}")
            raise

    def _deserialize_checkpoint(self, checkpoint_str: str) -> Checkpoint:
        """反序列化 checkpoint

        Reconstruct the tuple (type_str, bytes_data) expected by loads_typed()
        """
        try:
            import base64
            checkpoint_dict = json.loads(checkpoint_str)
            type_str = checkpoint_dict["type"]
            bytes_data = base64.b64decode(checkpoint_dict["data"])
            # loads_typed expects a tuple of (type_str, bytes_data)
            return self.serde.loads_typed((type_str, bytes_data))
        except Exception as e:
            api_logger.error(f"Failed to deserialize checkpoint: {e}")
            raise

    # ==================== 核心方法实现 ====================

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        异步获取 checkpoint（从 session messages 中查找）

        Args:
            config: 包含 thread_id (session_id) 的配置

        Returns:
            CheckpointTuple 或 None
        """
        try:
            api_logger.info(f"[CHECKPOINT:LOAD] aget_tuple called | config_keys={list(config.keys())}")
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config["configurable"].get("checkpoint_id")
            api_logger.info(f"[CHECKPOINT:LOAD] Starting load | thread_id={thread_id} | checkpoint_id={checkpoint_id}")

            # 获取所有 messages，过滤出 checkpoint 类型
            url = f"{self.base_url}/api/v1/sessions/{thread_id}/messages"
            params = {
                "limit": 100,  # 获取最近的 messages
                "user_id": self.user_id
            }

            response = await self.client.get(url, params=params, headers=self._get_headers())

            if response.status_code == 404:
                api_logger.info(f"[CHECKPOINT:LOAD] No session found | thread_id={thread_id}")
                return None

            response.raise_for_status()
            data = response.json()

            # session_service 返回格式：{"messages": [...], "total": N, "page": 1, "page_size": 100}
            messages = data.get("messages", data.get("data", []))

            # 过滤出 checkpoint 消息
            checkpoint_messages = [
                msg for msg in messages
                if msg.get("message_type") == "checkpoint"
            ]

            api_logger.info(
                f"[CHECKPOINT:LOAD] Session messages retrieved | "
                f"thread_id={thread_id} | "
                f"total_messages={len(messages)} | "
                f"checkpoint_messages={len(checkpoint_messages)}"
            )

            if not checkpoint_messages:
                api_logger.info(f"[CHECKPOINT:LOAD] No checkpoints found | thread_id={thread_id}")
                return None

            # 如果指定了 checkpoint_id，查找特定的
            if checkpoint_id:
                checkpoint_msg = next(
                    (msg for msg in checkpoint_messages
                     if msg.get("metadata", {}).get("checkpoint_id") == checkpoint_id),
                    None
                )
            else:
                # 否则返回最新的（按 created_at 排序）
                checkpoint_messages.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                checkpoint_msg = checkpoint_messages[0] if checkpoint_messages else None

            if not checkpoint_msg:
                api_logger.info(f"[CHECKPOINT:LOAD] Checkpoint not found | checkpoint_id={checkpoint_id}")
                return None

            # 反序列化 checkpoint
            checkpoint_data = checkpoint_msg.get("content", "")
            checkpoint = self._deserialize_checkpoint(checkpoint_data)

            # Log checkpoint details
            channel_values = checkpoint.get("channel_values", {})
            messages_in_checkpoint = channel_values.get("messages", [])
            api_logger.info(
                f"[CHECKPOINT:LOAD] Checkpoint loaded | "
                f"thread_id={thread_id} | "
                f"checkpoint_id={checkpoint_msg.get('metadata', {}).get('checkpoint_id')} | "
                f"messages_count={len(messages_in_checkpoint)} | "
                f"channels={list(channel_values.keys())}"
            )

            # 从 metadata 中提取信息（metadata是扁平结构）
            msg_metadata = checkpoint_msg.get("metadata", {})
            # 从JSON字符串反序列化 checkpoint_metadata
            metadata_json = msg_metadata.get("checkpoint_metadata_json", "{}")
            metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
            parent_checkpoint_id = msg_metadata.get("parent_checkpoint_id")

            # 构建 parent_config
            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": parent_checkpoint_id
                    }
                }

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            api_logger.error(f"HTTP error getting checkpoint: {e}")
            return None
        except Exception as e:
            api_logger.error(f"Failed to get checkpoint: {e}")
            import traceback
            api_logger.error(traceback.format_exc())
            return None

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """同步版本（兼容性）"""
        import asyncio
        try:
            return asyncio.run(self.aget_tuple(config))
        except RuntimeError:
            # 如果已经在事件循环中
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.aget_tuple(config))

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        异步保存 checkpoint（作为 session message）

        Args:
            config: 配置信息
            checkpoint: 要保存的 checkpoint
            metadata: 元数据
            new_versions: 新版本信息

        Returns:
            更新后的配置
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = checkpoint.get("id", str(checkpoint.get("ts", "")))

            # 确保 session 存在
            session_exists = await self._ensure_session_exists(thread_id, self.user_id)
            if not session_exists:
                api_logger.error(f"[CHECKPOINT:SAVE] Cannot save checkpoint - session creation failed | thread_id={thread_id}")
                # 返回配置以避免完全失败
                return {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id
                    }
                }

            # 序列化 checkpoint
            serialized_checkpoint = self._serialize_checkpoint(checkpoint)

            # 构建 message payload（使用 session_service messages API）
            # 注意：metadata 必须是扁平结构，嵌套对象会被转换为浮点数
            payload = {
                "role": "system",
                "content": serialized_checkpoint,
                "message_type": "checkpoint",  # 标识为 checkpoint 消息
                "metadata": {
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": metadata.get("parent_checkpoint_id", ""),
                    "timestamp": datetime.now().isoformat(),
                    # 将 checkpoint metadata 序列化为 JSON 字符串
                    "checkpoint_metadata_json": json.dumps(metadata),
                    "channel_versions_json": json.dumps(new_versions)
                },
                "user_id": self.user_id
            }

            # 保存到 session_service（使用 messages API）
            url = f"{self.base_url}/api/v1/sessions/{thread_id}/messages"
            response = await self.client.post(
                url,
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()

            result = response.json()
            # Session service returns the message object directly (has message_id, session_id, etc.)
            # Success is indicated by status 200 and presence of message_id
            if not result.get("message_id"):
                raise Exception(f"Failed to save checkpoint: Invalid response format")

            # Log detailed checkpoint save info
            channel_values = checkpoint.get("channel_values", {})
            messages_in_checkpoint = channel_values.get("messages", [])
            api_logger.info(
                f"[CHECKPOINT:SAVE] Checkpoint saved | "
                f"thread_id={thread_id} | "
                f"checkpoint_id={checkpoint_id} | "
                f"messages_count={len(messages_in_checkpoint)} | "
                f"channels={list(channel_values.keys())} | "
                f"metadata_keys={list(metadata.keys())}"
            )

            # Log last few messages for debugging
            if messages_in_checkpoint:
                last_messages = messages_in_checkpoint[-3:]  # Last 3 messages
                for idx, msg in enumerate(last_messages):
                    msg_content = str(msg.get("content", ""))[:100] if hasattr(msg, "get") else str(msg)[:100]
                    msg_role = msg.get("role", "unknown") if hasattr(msg, "get") else getattr(msg, "type", "unknown")
                    api_logger.info(
                        f"[CHECKPOINT:SAVE] Recent message {idx+1} | "
                        f"role={msg_role} | "
                        f"content_preview={msg_content}"
                    )

            # 返回更新后的 config
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id
                }
            }

        except Exception as e:
            api_logger.error(f"Failed to save checkpoint: {e}")
            import traceback
            api_logger.error(traceback.format_exc())
            raise

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """同步版本（兼容性）"""
        import asyncio
        try:
            return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.aput(config, checkpoint, metadata, new_versions))

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> list[CheckpointTuple]:
        """
        异步列出 checkpoints（从 session messages 中过滤）

        Args:
            config: 基础配置（包含 thread_id）
            filter: 过滤条件
            before: 列出此配置之前的 checkpoints
            limit: 最大返回数量

        Returns:
            CheckpointTuple 列表
        """
        try:
            if not config:
                return []

            thread_id = config["configurable"]["thread_id"]

            # 获取所有 messages
            url = f"{self.base_url}/api/v1/sessions/{thread_id}/messages"
            params = {
                "limit": limit or 100,
                "user_id": self.user_id
            }

            response = await self.client.get(
                url,
                params=params,
                headers=self._get_headers()
            )

            if response.status_code == 404:
                return []

            response.raise_for_status()
            result = response.json()

            # session_service 返回格式：{"messages": [...], "total": N, "page": 1, "page_size": 100}
            messages = result.get("messages", result.get("data", []))

            # 过滤出 checkpoint 消息
            checkpoint_messages = [
                msg for msg in messages
                if msg.get("message_type") == "checkpoint"
            ]

            # 反序列化所有 checkpoints
            checkpoint_tuples = []
            for msg in checkpoint_messages:
                try:
                    checkpoint_data = msg.get("content", "")
                    checkpoint = self._deserialize_checkpoint(checkpoint_data)

                    msg_metadata = msg.get("metadata", {})
                    # 从JSON字符串反序列化
                    metadata_json = msg_metadata.get("checkpoint_metadata_json", "{}")
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                    checkpoint_id = msg_metadata.get("checkpoint_id")
                    parent_checkpoint_id = msg_metadata.get("parent_checkpoint_id")

                    parent_config = None
                    if parent_checkpoint_id:
                        parent_config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_id": parent_checkpoint_id
                            }
                        }

                    checkpoint_tuples.append(
                        CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_id": checkpoint_id
                                }
                            },
                            checkpoint=checkpoint,
                            metadata=metadata,
                            parent_config=parent_config
                        )
                    )
                except Exception as e:
                    api_logger.warning(f"Failed to deserialize checkpoint: {e}")
                    continue

            # 按时间排序（最新的在前）
            checkpoint_tuples.sort(
                key=lambda x: x.metadata.get("timestamp", ""),
                reverse=True
            )

            return checkpoint_tuples

        except Exception as e:
            api_logger.error(f"Failed to list checkpoints: {e}")
            import traceback
            api_logger.error(traceback.format_exc())
            return []

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """同步版本（兼容性）"""
        import asyncio
        try:
            checkpoints = asyncio.run(self.alist(config, filter=filter, before=before, limit=limit))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            checkpoints = loop.run_until_complete(self.alist(config, filter=filter, before=before, limit=limit))
        return iter(checkpoints)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        Save writes to pending writes table (for streaming mode)

        Args:
            config: Configuration with thread_id
            writes: List of (channel, value) tuples to write
            task_id: Task ID for these writes
            task_path: Task path for nested tasks (optional)

        Note: This is called during streaming to save intermediate writes.
        For SessionServiceCheckpointer, we don't need to persist these separately
        as they will be included in the final checkpoint via aput().
        """
        # For our implementation, we don't need to persist writes separately
        # They will be included in the checkpoint when aput() is called
        pass

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Sync version of aput_writes"""
        # No-op for our implementation
        pass

    async def aclose(self):
        """清理资源"""
        await self.client.aclose()
        api_logger.info("✅ SessionServiceCheckpointer closed")

    def __del__(self):
        """析构时清理"""
        try:
            import asyncio
            asyncio.create_task(self.aclose())
        except:
            pass
