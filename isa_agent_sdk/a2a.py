#!/usr/bin/env python3
"""
A2A (Agent-to-Agent) support utilities.

This module provides:
- A lightweight JSON-RPC A2A client
- A server adapter that maps A2A methods to isa_agent_sdk query/ask
- Agent Card helpers
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from ._query import ask
from .options import ISAAgentOptions

try:
    from fastapi import Request as FastAPIRequest
except Exception:  # pragma: no cover - FastAPI may be optional
    FastAPIRequest = Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _extract_text_from_message(message: Dict[str, Any]) -> str:
    parts = message.get("parts", []) or []
    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(t for t in texts if t)


def _build_message(role: str, text: str, message_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "kind": "message",
        "messageId": message_id or _new_id("msg"),
        "role": role,
        "parts": [{"kind": "text", "text": text}],
    }


def _task_payload(task_id: str, state: str, message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kind": "task",
        "id": task_id,
        "status": {
            "state": state,
            "timestamp": _now_iso(),
        },
    }
    if message is not None:
        payload["artifacts"] = [message]
    return payload


@dataclass
class A2AAgentCard:
    """Minimal Agent Card structure for A2A discovery."""

    name: str
    url: str
    version: str = "0.1.0"
    description: str = "isA agent"
    provider_org: str = "isA"
    documentation_url: Optional[str] = None
    skills: List[Dict[str, Any]] = field(default_factory=list)
    default_input_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text/plain"])
    supports_streaming: bool = True
    supports_push_notifications: bool = True
    token_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        security_schemes: Dict[str, Any] = {}
        security: List[Dict[str, Any]] = []
        if self.token_url:
            security_schemes["oauth2_cc"] = {
                "oauth2SecurityScheme": {
                    "flows": {
                        "clientCredentials": {
                            "tokenUrl": self.token_url,
                            "scopes": {
                                "a2a.invoke": "Send/stream messages",
                                "a2a.tasks.read": "Read task state",
                                "a2a.tasks.cancel": "Cancel task",
                            },
                        }
                    }
                }
            }
            security.append({"oauth2_cc": ["a2a.invoke"]})

        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "provider": {"organization": self.provider_org},
            "documentationUrl": self.documentation_url,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "capabilities": {
                "streaming": self.supports_streaming,
                "pushNotifications": self.supports_push_notifications,
            },
            "skills": self.skills,
            "securitySchemes": security_schemes,
            "security": security,
        }


class A2AClient:
    """Simple A2A JSON-RPC client."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        auth_token: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.auth_token = auth_token
        self.default_headers = default_headers or {}

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = dict(self.default_headers)
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if extra:
            headers.update(extra)
        return headers

    async def get_agent_card(self) -> Dict[str, Any]:
        import httpx

        card_url = f"{self.base_url}/.well-known/agent-card.json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(card_url, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def rpc(
        self,
        method: str,
        params: Dict[str, Any],
        rpc_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        import httpx

        payload = {
            "jsonrpc": "2.0",
            "id": _new_id("rpc"),
            "method": method,
            "params": params,
        }
        url = rpc_url or self.base_url
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers(headers))
            resp.raise_for_status()
            return resp.json()

    async def send_message(
        self,
        rpc_url: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return await self.rpc(
            "message/send",
            {
                "message": _build_message("user", text),
                "metadata": metadata or {},
            },
            rpc_url=rpc_url,
            headers=headers,
        )

    async def stream_message(
        self,
        rpc_url: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        import httpx

        payload = {
            "jsonrpc": "2.0",
            "id": _new_id("rpc"),
            "method": "message/stream",
            "params": {
                "message": _build_message("user", text),
                "metadata": metadata or {},
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", rpc_url, json=payload, headers=self._headers(headers)) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw:
                        continue
                    yield json.loads(raw)


class A2AServerAdapter:
    """
    A2A server adapter for JSON-RPC method handling.

    Use `handle_rpc()` from your HTTP endpoint.
    """

    def __init__(
        self,
        *,
        options: Optional[ISAAgentOptions] = None,
        runner: Optional[Callable[[str, Optional[ISAAgentOptions]], Awaitable[str]]] = None,
    ):
        self.options = options or ISAAgentOptions()
        self._runner = runner or self._run_with_sdk
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._push_config: Dict[str, Dict[str, Any]] = {}
        self._max_tasks = 1000
        self._task_ttl_seconds = 3600  # 1 hour

    async def _run_with_sdk(self, prompt: str, options: Optional[ISAAgentOptions]) -> str:
        return await ask(prompt, options=options or self.options)

    def _evict_stale_tasks(self) -> None:
        """Remove completed/failed tasks older than TTL, enforce max size."""
        if len(self._tasks) <= self._max_tasks:
            return
        now = datetime.now(timezone.utc)
        to_delete = []
        for tid, task in self._tasks.items():
            status = task.get("status", {}).get("state", "")
            if status in ("completed", "failed"):
                updated = task.get("status", {}).get("timestamp")
                if updated:
                    try:
                        age = (now - datetime.fromisoformat(updated)).total_seconds()
                        if age > self._task_ttl_seconds:
                            to_delete.append(tid)
                    except (ValueError, TypeError):
                        to_delete.append(tid)
        for tid in to_delete:
            del self._tasks[tid]

    async def _execute_task(self, task_id: str, prompt: str) -> None:
        self._evict_stale_tasks()
        self._tasks[task_id] = _task_payload(task_id, "working")
        try:
            text = await self._runner(prompt, self.options)
            msg = _build_message("agent", text)
            self._tasks[task_id] = _task_payload(task_id, "completed", msg)
        except Exception as e:
            self._tasks[task_id] = {
                **_task_payload(task_id, "failed"),
                "error": {"message": str(e)},
            }

    async def handle_rpc(self, request: Dict[str, Any], *, extended_card: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {}) or {}

        try:
            if method in {"message/send", "SendMessage"}:
                message = params.get("message", {})
                prompt = _extract_text_from_message(message)
                if params.get("async") is True:
                    task_id = _new_id("task")
                    self._tasks[task_id] = _task_payload(task_id, "submitted")
                    asyncio.create_task(self._execute_task(task_id, prompt))
                    return {"jsonrpc": "2.0", "id": req_id, "result": self._tasks[task_id]}

                text = await self._runner(prompt, self.options)
                result = _build_message("agent", text)
                return {"jsonrpc": "2.0", "id": req_id, "result": result}

            if method in {"message/stream", "SendStreamingMessage"}:
                # HTTP layer should call `stream_rpc_events()` for SSE output.
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"accepted": True, "streaming": True},
                }

            if method in {"tasks/get", "GetTask"}:
                task_id = params.get("taskId")
                task = self._tasks.get(task_id)
                if not task:
                    raise KeyError(f"Unknown task: {task_id}")
                return {"jsonrpc": "2.0", "id": req_id, "result": task}

            if method in {"tasks/cancel", "CancelTask"}:
                task_id = params.get("taskId")
                task = self._tasks.get(task_id)
                if not task:
                    raise KeyError(f"Unknown task: {task_id}")
                task["status"]["state"] = "canceled"
                task["status"]["timestamp"] = _now_iso()
                return {"jsonrpc": "2.0", "id": req_id, "result": task}

            if method in {"tasks/resubscribe", "TaskResubscription"}:
                task_id = params.get("taskId")
                task = self._tasks.get(task_id)
                if not task:
                    raise KeyError(f"Unknown task: {task_id}")
                return {"jsonrpc": "2.0", "id": req_id, "result": task}

            if method == "tasks/pushNotificationConfig/set":
                task_id = params.get("taskId")
                config = params.get("config", {})
                self._push_config[task_id] = config
                return {"jsonrpc": "2.0", "id": req_id, "result": {"taskId": task_id, "config": config}}

            if method == "tasks/pushNotificationConfig/get":
                task_id = params.get("taskId")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"taskId": task_id, "config": self._push_config.get(task_id)},
                }

            if method == "tasks/pushNotificationConfig/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": self._push_config}

            if method == "tasks/pushNotificationConfig/delete":
                task_id = params.get("taskId")
                self._push_config.pop(task_id, None)
                return {"jsonrpc": "2.0", "id": req_id, "result": {"taskId": task_id, "deleted": True}}

            if method in {"agent/getAuthenticatedExtendedCard", "GetAuthenticatedExtendedCard"}:
                return {"jsonrpc": "2.0", "id": req_id, "result": extended_card or {}}

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    async def stream_rpc_events(self, request: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield JSON-RPC payloads for SSE transport for `message/stream`.
        """
        req_id = request.get("id")
        params = request.get("params", {}) or {}
        message = params.get("message", {})
        prompt = _extract_text_from_message(message)

        task_id = _new_id("task")
        start = _task_payload(task_id, "submitted")
        yield {"jsonrpc": "2.0", "id": req_id, "result": start}

        self._tasks[task_id] = _task_payload(task_id, "working")
        yield {"jsonrpc": "2.0", "id": req_id, "result": self._tasks[task_id]}

        try:
            text = await self._runner(prompt, self.options)
            msg = _build_message("agent", text)
            done = _task_payload(task_id, "completed", msg)
            self._tasks[task_id] = done
            yield {"jsonrpc": "2.0", "id": req_id, "result": done}
        except Exception as e:
            failed = {**_task_payload(task_id, "failed"), "error": {"message": str(e)}}
            self._tasks[task_id] = failed
            yield {"jsonrpc": "2.0", "id": req_id, "result": failed}


def register_a2a_fastapi_routes(
    app: Any,
    *,
    adapter: A2AServerAdapter,
    agent_card: Dict[str, Any],
    rpc_path: str = "/a2a",
    card_path: str = "/.well-known/agent-card.json",
    extended_card: Optional[Dict[str, Any]] = None,
    auth_validator: Optional[Callable[[Any], Awaitable[None]]] = None,
) -> None:
    """
    Register A2A routes on a FastAPI app.

    Args:
        app: FastAPI application instance
        adapter: A2A server adapter
        agent_card: Base agent card payload
        rpc_path: JSON-RPC endpoint path
        card_path: Agent Card discovery path
        extended_card: Optional authenticated extended card payload
        auth_validator: Optional async function(request) to enforce auth
    """
    from fastapi.responses import JSONResponse, StreamingResponse

    @app.get(card_path)
    async def a2a_agent_card() -> Dict[str, Any]:
        return agent_card

    @app.post(rpc_path)
    async def a2a_rpc(request: FastAPIRequest) -> Any:
        if auth_validator is not None:
            await auth_validator(request)

        body = await request.json()
        method = body.get("method")
        if method in {"message/stream", "SendStreamingMessage", "tasks/resubscribe", "TaskResubscription"}:
            async def event_gen() -> AsyncIterator[str]:
                async for payload in adapter.stream_rpc_events(body):
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        result = await adapter.handle_rpc(body, extended_card=extended_card)
        return JSONResponse(result)


def build_auth_service_token_validator(
    auth_service_base_url: str,
    *,
    required_scopes: Optional[List[str]] = None,
    provider: str = "isa_user",
) -> Callable[[Any], Awaitable[None]]:
    """
    Build a FastAPI-compatible auth validator using auth_service /verify-token.
    """
    from fastapi import HTTPException, status

    async def _validate(request: Any) -> None:
        import httpx

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

        token = header.split(" ", 1)[1].strip()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{auth_service_base_url.rstrip('/')}/api/v1/auth/verify-token",
                json={"token": token, "provider": provider},
            )
            if resp.status_code >= 400:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed")
            data = resp.json()

        if not data.get("valid"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=data.get("error", "Invalid token"))

        if required_scopes:
            permissions = set(data.get("permissions") or [])
            if not set(required_scopes).issubset(permissions):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient scope")

    return _validate
