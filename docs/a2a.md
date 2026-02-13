# A2A Guide

This guide explains how to use `isa_agent_sdk` for Agent-to-Agent (A2A) communication, based on a verified local test between two `isA_Agent` instances.

## What Was Verified

- `auth_service` running on `http://localhost:8201`
- Agent instance A on `http://localhost:8101`
- Agent instance B on `http://localhost:8102`
- A2A server endpoint on each instance:
  - `GET /.well-known/agent-card.json`
  - `POST /a2a`
- End-to-end client call from 8101 to 8102 succeeded via:
  - `POST /api/v1/a2a/test-client`

## SDK Components

Use these exports from `isa_agent_sdk`:

- `A2AAgentCard`
- `A2AClient`
- `A2AServerAdapter`
- `register_a2a_fastapi_routes`
- `build_auth_service_token_validator`

## Server Integration (FastAPI)

```python
from fastapi import FastAPI
from isa_agent_sdk import (
    A2AAgentCard,
    A2AServerAdapter,
    register_a2a_fastapi_routes,
    build_auth_service_token_validator,
)

app = FastAPI()
adapter = A2AServerAdapter()

card = A2AAgentCard(
    name="isA Agent",
    url="http://localhost:8102/a2a",
    token_url="http://localhost:8201/oauth/token",
).to_dict()

register_a2a_fastapi_routes(
    app,
    adapter=adapter,
    agent_card=card,
    rpc_path="/a2a",
    card_path="/.well-known/agent-card.json",
    auth_validator=build_auth_service_token_validator(
        "http://localhost:8201",
        required_scopes=["a2a.invoke"],
    ),
)
```

## Client Call Example

```python
from isa_agent_sdk import A2AClient

client = A2AClient("http://localhost:8102", auth_token="<bearer_token>")
card = await client.get_agent_card()
resp = await client.send_message("http://localhost:8102/a2a", "Reply exactly: A2A_OK")
```

## Local Test Command

```bash
curl -X POST http://localhost:8101/api/v1/a2a/test-client \
  -H "Content-Type: application/json" \
  -d '{
    "target_base_url":"http://localhost:8102",
    "target_rpc_url":"http://localhost:8102/a2a",
    "message":"Reply exactly: A2A_OK",
    "timeout":180
  }'
```

Expected shape:

```json
{
  "status": "success",
  "agent_card_name": "isA Agent",
  "rpc_response": { "jsonrpc": "2.0", "result": { "kind": "message" } }
}
```

## Notes

- For quick local functional testing, `A2A_AUTH_REQUIRED=false` can be used.
- For real interoperability testing, enable auth and pass OAuth token from `auth_service /oauth/token`.
- Current `ask()` aggregation may duplicate text in some responses; transport itself is working.
