# Agent Creation Feature Status

Last updated: 2026-02-12
Owner: isA Console + OS Services + isA Agent SDK

---

## Goal

Provide four creation paths in isA Console:
1. Basic agent (no-code, runs in cloud_os).
2. Template-based agent (no-code, prefilled use cases).
3. Generated agent (from requirement prompt via isA_Vibe).
4. Local development agent (scaffold download + local build + deploy to cloud_os).

## Product Epic

**Epic Name**: Self-Serve Agent Creation & Deployment

**Problem**
Users want to create and run agents without deep infrastructure knowledge. They also want a path to advanced custom agents when no-code options are insufficient.

**Outcome**
Users can create and run agents end-to-end from the Console using four paths (basic, template, generate, local dev), with cloud_os as the default runtime and code_deployment for custom agent deployment.

**Success Metrics**
- Time to first runnable agent < 10 minutes for Basic/Template/Generate.
- >= 60% of new agents created via no-code flows.
- >= 90% successful deploys for local dev flow.
- Reduced support requests for "how to deploy agent" by 50%.

---

## Architecture Overview

```
Console UI (Next.js, port 3001)
    |
    v
isA_Agent (FastAPI, port 8080) --- persistent service ---
    |-- Config CRUD (PostgreSQL via isA_Agent_SDK AgentConfigStore)
    |-- Template listing (hardcoded: basic-assistant, researcher)
    |-- Scaffold download (zip of isA_Agent/ directory)
    |-- Auth (require_auth_or_internal_service)
    |-- Chat routing:
    |       cloud_shared --> SDK query() directly (in-process)
    |       cloud_pool   --> pool_manager --> cloud_os --> Docker VM
    |       deployment   --> pool_manager --> custom_agent pool
    |       multi_agent  --> MultiAgentRunner (DAG-first)
    |
    v
pool_manager (port 8090)
    |-- Warm pool: auto-creates 10 agent VMs on startup
    |-- Acquire/execute/stream/release lifecycle
    |-- Executor calls container's /api/v1/agents/chat endpoint
    |-- Redis (password: staging_redis_2024) for pool state
    |
    v
cloud_os (port 8086)
    |-- Docker backend (macOS dev)
    |-- Firecracker/Ignite backend (Linux prod)
    |
    v
Docker Container (isa-agent-sdk:latest, port 8888 inside container)
    |-- isA Agent SDK (query, tools, skills)
    |-- Same main.py as isA_Agent service
    |-- Reaches host via host.docker.internal
    |-- isA_Model service (port 8082) for LLM inference
```

**Key architectural insight**: `isA_Agent/main.py` serves dual roles:
1. As the **Agent Service** (persistent, port 8080): handles Config CRUD, templates, scaffold, chat routing
2. As the **agent runtime** inside Docker containers (port 8888): handles `/api/v1/agents/chat` for direct query execution

The chat request body uses `message` (string), NOT `messages` (array). The `stream` field (boolean, default `true`) controls SSE vs JSON response.

---

## Current Status by Scenario

### Scenario 0: Security -- COMPLETE

All sensitive endpoints require `require_auth_or_internal_service`. 6 security fixes applied and tested.

**Fixes applied:**
| Fix | File | Description |
|-----|------|-------------|
| 0A: SQL injection whitelist | `agent_config_store.py` | `_ALLOWED_UPDATE_COLUMNS` frozenset; `update_config()` rejects unknown column names with `ValueError` |
| 0B: Dedup in memory fallback | `agent_config_store.py` | `list_configs()` deduplicates by config ID after merging owner + org configs |
| 0C: Auth on `/api/v1/agents/chat` | `isA_Agent/main.py` | Added `Depends(require_auth_or_internal_service)`; removed anonymous fallback |
| 0D: Auth on `/query` | `isA_Agent/main.py` | Was completely unauthenticated |
| 0E: Auth on swarm endpoints | `isA_Agent/main.py` | `/swarm`, `/swarm/stream`, `/swarm/dag` were completely unauthenticated |
| 0F: DB failure detection | `agent_config_store.py` | `create_config()` and `create_multi_spec()` raise `RuntimeError` if INSERT returns no row |

**Auth methods supported (checked in order):**
1. `X-Internal-Service: true` + `X-Internal-Service-Secret: <secret>` (service-to-service, returns user_id `"internal-service"`)
2. `Authorization: Bearer <jwt>` (verified via POST to auth_service `/api/v1/auth/verify-token`)
3. `X-User-Id: <user_id>` or `user-id: <user_id>` header (direct, simplest for testing)

**Public endpoints (no auth):**
- `GET /api/v1/agents/templates` (read-only)
- `GET /api/v1/agents/templates/{id}` (read-only)
- `GET /health`

### Scenario 1: Basic Agent -- E2E VERIFIED (Streaming + Non-Streaming)

**Status**: Fully working end-to-end. Config CRUD persisted to PostgreSQL. Chat works via both `cloud_shared` and `cloud_pool` paths. Streaming and non-streaming both verified.

**E2E test results (2026-02-12):**
```
1. Create agent config:
   POST /api/v1/agents/configs -> 200, returns config with id, owner_id, source=basic

2. GET /api/v1/agents/configs/{id} -> 200, all fields match

3. GET /api/v1/agents/configs -> 200, created agent appears in list

4. PATCH /api/v1/agents/configs/{id} -> 200, updated fields change, others preserved

5. Non-streaming chat (cloud_pool):
   POST /api/v1/agents/chat {agent_config_id, message, stream: false}
   -> {"response": {"success": true, "result": {"response": "7"}}, "status": "success"}
   Duration: ~7-17 seconds (includes VM acquire + model inference)

6. Streaming chat (cloud_pool):
   POST /api/v1/agents/chat {agent_config_id, message, stream: true}
   -> SSE events: session_start -> system -> text -> result -> session_end -> [DONE]

7. DELETE /api/v1/agents/configs/{id} -> 200
   GET after delete -> 404
```

**What works:**
- Config CRUD: create, get, list, update, delete -- all persisted to PostgreSQL
- Auth enforced on all endpoints
- Chat via `cloud_shared` (direct SDK): streaming + non-streaming
- Chat via `cloud_pool`: streaming + non-streaming **fully verified E2E**
- Warm pool: pool_manager auto-creates 10 warm agent VMs on startup
- Agent VMs reachable via auto-allocated host ports on macOS Docker Desktop

**What's NOT yet tested:**
- Console UI wiring (API-only verification so far)

### Scenario 2: Template Agent -- E2E VERIFIED

**Status**: Fully working. Template data inheritance verified. Bug found and fixed.

**E2E test results (2026-02-12):**
```
1. GET /api/v1/agents/templates -> 200, returns basic-assistant + researcher

2. Create from template (no overrides):
   POST /api/v1/agents/configs {name, model_id, template_id: "researcher", source: "template"}
   -> 200, inherits template's system_prompt, tools, skills, mode

3. Verified template data:
   system_prompt: "You are a research-focused assistant..."
   tools: ["web_search", "web_crawl", "read_file", "write_file"]
   skills: ["web"]
   mode: "COLLABORATIVE"

4. Chat with template agent (cloud_pool):
   -> {"response": "Python was created in 1991.", "status": "success"}
```

**Bug found and fixed (template mode override):**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Template `mode` overridden by Pydantic default | `AgentConfigCreate.mode` defaults to `"REACTIVE"`. When creating from template, `request.mode or template.mode` evaluated to `"REACTIVE"` (truthy default), so template's `"COLLABORATIVE"` was never used. | Used `request.model_fields_set` to detect which fields the user explicitly provided. Only override template values for fields in `model_fields_set`. |

**Fix location**: `isA_Agent/main.py`, template creation handler. The `model_fields_set` approach correctly distinguishes "user sent mode=REACTIVE" from "Pydantic defaulted to REACTIVE".

**Static templates available:**
| ID | Name | Tools | Skills | Mode |
|----|------|-------|--------|------|
| `basic-assistant` | Basic Assistant | read_file, write_file, edit_file, bash_execute, glob_files | filesystem, code | COLLABORATIVE |
| `researcher` | Web Researcher | web_search, web_crawl, read_file, write_file | web | COLLABORATIVE |

### Scenario 3: Generated Agent (Vibe) -- UNIT TESTED, NOT E2E

**Status**: Creation service tested with zero mocks. Vibe HTTP endpoint not E2E testable without running Vibe service.

**What works (unit tested):**
- `AgentCreationService.create_generated()` creates agent config + optional multi-agent spec
- Multi-agent spec CRUD: list, get, update, delete via `/api/v1/agents/multi-specs/*`
- Entry agent resolution: explicit `entry_agent_id` > first agent in list > `ValueError` if empty
- Vibe-unreachable behavior: returns 500 with meaningful error, doesn't crash

**What's NOT testable without Vibe:**
- `POST /api/v1/agents/generate` — calls external Vibe service to generate agent spec

### Scenario 4: Local Dev Agent -- E2E VERIFIED (Config + Scaffold)

**Status**: Scaffold download and config creation fully working. Deployment pipeline not wired.

**E2E test results (2026-02-12):**
```
1. Scaffold download:
   GET /api/v1/agents/scaffold -> 200, valid zip (36 files, ~41KB)
   Contains: main.py, pyproject.toml, README.md, .isa/config.json, etc.
   Excludes: __pycache__, .pyc, .log files

2. Create local-dev config:
   POST /api/v1/agents/local-dev {name, description, deployment_id}
   -> 200, returns {agent: {source: "local", metadata: {deployment_id: "..."}}, scaffold_url}
```

**What's NOT yet wired:**
- Console UI upload/build/deploy flow
- Custom agent pool E2E (pool_manager has the pool, executor not tested)

### Scenario 5: Swarm Runtime -- FIXES APPLIED, NOT E2E TESTED

**Fixes applied:**
| Fix | File | Description |
|-----|------|-------------|
| 5A: Unknown handoff reason | `handoff.py` | Includes `reason` field in `HandoffResult` for unknown targets |
| 5B: Context truncation marker | `swarm.py` | Adds `[...context truncated...]` marker instead of silently dropping beginning |

---

## Bugs Found and Fixed (2026-02-12)

### Bug 1: Template mode/env override (CRITICAL)
- **Symptom**: Creating agent from template without specifying `mode` resulted in `REACTIVE` instead of template's `COLLABORATIVE`
- **Root cause**: Pydantic `AgentConfigCreate` model has `mode: str = "REACTIVE"` as default. The expression `request.mode or template.mode` always evaluated to `"REACTIVE"` because the Pydantic default is truthy.
- **Fix**: Use `request.model_fields_set` to check which fields the user actually sent. Only override template values for explicitly provided fields.
- **File**: `isA_Agent/main.py`, template creation handler
- **Found by**: Zero-mock unit tests that verified real template data inheritance

### Bug 2: Pool manager executor calling wrong endpoint (CRITICAL)
- **Symptom**: Streaming chat returned 404; non-streaming chat called `/query` with mismatched body format
- **Root cause**: `cloudos_agent_executor.py` called `POST /stream` (didn't exist) and `POST /query` (wrong body format) on the Docker container. The container's main.py only exposes `/api/v1/agents/chat`.
- **Fix**: Changed both `_execute_query()` and `stream()` to call `POST /api/v1/agents/chat` with the correct body format (`message` not `prompt`, `stream: true/false`).
- **File**: `isA_OS/os_services/pool_manager/src/executors/agent/cloudos_agent_executor.py`

### Bug 3: SSE `[DONE]` sentinel not handled (MINOR)
- **Symptom**: JSON parse error at end of streaming response
- **Root cause**: Executor tried to `json.loads("[DONE]")` which fails
- **Fix**: Check for `[DONE]` sentinel before parsing JSON, break out of loop
- **File**: same as Bug 2

---

## Unit Tests -- Zero Mocks (64 Agent Service + SDK tests)

All Agent Service tests use **zero mocks** — no `unittest.mock`, no `MagicMock`, no `patch`. Tests use the real `AgentCreationService`, real `AgentConfigStore` (in-memory fallback mode — a real supported code path), real hardcoded templates, and real scaffold zip files. Only auth dependency is overridden (returns test user_id).

### Agent Service Tests (`isA_Agent/tests/`)
| File | Tests | What's Verified |
|------|-------|-----------------|
| `test_scenario_basic.py` | 15 | Full CRUD lifecycle: every field round-trips correctly (name, description, model_id, system_prompt, tools, skills, mode, env, visibility, metadata, timestamps). Tests defaults, multiple modes/envs, update preservation, delete from both GET and LIST. |
| `test_scenario_template.py` | 12 | Template discovery (real content verification), data inheritance (system_prompt, tools, skills, mode from template), overrides take priority, partial overrides merge correctly, full lifecycle. |
| `test_scenario_generated.py` | 15 | Vibe-unreachable returns 500, single agent (no multi-spec), multi-agent creates spec with correct entry_agent, entry agent inferred from first agent, explicit entry_agent_id, empty agents raises ValueError, full multi-spec CRUD lifecycle. |
| `test_scenario_local_dev.py` | 9 | Real scaffold zip download, file content verification, file filtering (__pycache__, .pyc, .log excluded), local-dev config creation, full flow: scaffold -> create -> list -> get -> update -> delete. |
| `test_auth_gates.py` | 13 | 401/403 on all endpoints without auth |

### Agent SDK Tests (`isA_Agent_SDK/tests/`)
| File | Tests | Coverage |
|------|-------|----------|
| `test_agent_config_store.py` | SQL whitelist, dedup, DB failure | Config store security + integrity |
| `test_agent_creation.py` | Basic, template, local-dev creation | All creation paths |
| `test_agent_creation_generated.py` | Generated + multi-agent spec | Vibe generation flow |
| `test_multi_agent_runner.py` | DAG/swarm modes, entry override | Multi-agent runtime |
| `test_swarm.py` | Handoff, truncation, circular | Swarm orchestration |
| `test_dag.py` | DAG execution | DAG routing |

### Running Tests
```bash
# Agent Service tests (zero-mock, real data)
cd isA_Agent && pytest tests/ -v

# Agent SDK tests
cd isA_Agent_SDK && pytest tests/ -v
```

---

## Infrastructure Changes (2026-02-10 to 2026-02-12)

### Pool Manager (`isA_OS/os_services/pool_manager`)
| Change | File | Details |
|--------|------|---------|
| API route prefix | `main.py` | All routers now use `/api/v1` prefix (was missing, caused 404s) |
| CloudOS Agent Executor networking | `cloudos_agent_executor.py` | Removed broken `network_mode: "none"` (was ignored by cloud_os). Now uses `container_port` for auto-allocated host port mapping |
| Host port preference | same | Prefers `127.0.0.1:host_port` over container bridge IP (required on macOS Docker Desktop) |
| Service URL rewriting | same | Replaces `localhost` with `host.docker.internal` for ISA_API_URL, ISA_MODEL_URL, AUTH_SERVICE_URL |
| Auth headers | same | All requests to agent VMs include `X-User-Id` header |
| Resource limits fix | same | Fixed field names: `cpu`->`cpu_count`, `memory`->`memory_mb`, `disk`->`disk_size_gb` |
| Proxy socket | same | Made proxy socket registration best-effort (non-fatal if fails) |
| SDK client param fix | `pool_manager_client.py` | Changed `"query"` to `"prompt"` in `execute_query()` and `stream_query()` |
| **Executor endpoint fix** | `cloudos_agent_executor.py` | `_execute_query()` and `stream()` now call `/api/v1/agents/chat` instead of `/query` and `/stream`. Body format: `{"message": ..., "stream": bool}` |
| **SSE sentinel handling** | `cloudos_agent_executor.py` | `stream()` checks for `[DONE]` sentinel before JSON parsing |

### Cloud OS (`isA_OS/os_services/cloud_os`)
| Change | File | Details |
|--------|------|---------|
| container_port model | `src/models/vm_models.py` | Added optional `container_port: int` to `VMCreateRequest` |
| container_port backend | `src/backends/docker_backend.py` | `create_vm()` uses `container_port` for auto-allocated port mapping (instead of SSH port 22) |
| container_port passthrough | `src/services/vm_service.py` | Passes `container_port` from request to backend |
| start_vm fix | `src/backends/docker_backend.py` | Skips `docker start` if container already running (Docker Desktop for Mac bug: `docker start` on running container breaks port bindings) |
| Ignite backend compat | `src/backends/ignite_backend.py` | Added `container_port` parameter (unused, for interface compatibility) |

### Agent Service (`isA_Agent`)
| Change | File | Details |
|--------|------|---------|
| Auth on all endpoints | `main.py` | Added `Depends(require_auth_or_internal_service)` to `/query`, `/api/v1/agents/chat`, all swarm endpoints |
| **Template override fix** | `main.py` | Uses `model_fields_set` to prevent Pydantic defaults from overriding template values |
| Docker image | `deployment/docker/Dockerfile` | NEW: Builds `isa-agent-sdk:latest` from local source packages |
| Entrypoint | `deployment/docker/entrypoint.sh` | NEW: Handles cloud_os `tail -f /dev/null` pattern by starting agent server in background |
| Dev dependencies | `deployment/requirements/base_dev.txt` | Added editable install of `isa-common` from local source (JSONB codec fix) |

### Agent SDK (`isA_Agent_SDK`)
| Change | File | Details |
|--------|------|---------|
| DB failure detection | `services/agent_config_store.py` | `create_config()` and `create_multi_spec()` raise `RuntimeError` on DB write failure |
| SQL injection fix | `services/agent_config_store.py` | `update_config()` validates column names against `_ALLOWED_UPDATE_COLUMNS` whitelist |
| Deduplication | `services/agent_config_store.py` | `list_configs()` memory fallback deduplicates by config ID |
| Entry agent fix | `services/agent_creation.py` | Removed literal `"entry"` fallback; derives from agent list or raises ValueError |
| Runner entry override | `services/multi_agent_runner.py` | `entry_agent_override` parameter takes priority over `routing_spec` |
| Handoff reason | `agents/handoff.py` | Unknown handoff target includes `reason` field in `HandoffResult` |
| Context truncation | `agents/swarm.py` | Adds `[...context truncated...]` marker when truncating context |

---

## Docker Image

**Image**: `isa-agent-sdk:latest`
**Base**: python:3.12-slim
**Build command**:
```bash
cd /path/to/isA  # root of monorepo
docker build -t isa-agent-sdk:latest -f isA_Agent/deployment/docker/Dockerfile .
```

**What's included:**
- isA_common (JSONB codec for asyncpg)
- isA_Model (model inference client)
- isA_Agent_SDK (agent runtime, tools, skills)
- isA_Agent/main.py (agent service entry point)
- fastapi, uvicorn, sse-starlette, python-dotenv

**Environment variables (set by executor at runtime):**
- `SERVICE_PORT` - port agent listens on (default 8080, overridden to 8888 by executor)
- `ISA_API_URL` / `ISA_MODEL_URL` - model service URL (set to `host.docker.internal:8082`)
- `AUTH_SERVICE_URL` - auth service URL (set to `host.docker.internal:8201`)
- `ISA_SESSION_ID`, `ISA_USER_ID`, `ISA_MODEL`, `ISA_EXECUTION_MODE`, etc.

**IMPORTANT**: Rebuild image after code changes:
```bash
docker build -t isa-agent-sdk:latest -f isA_Agent/deployment/docker/Dockerfile .
```

After rebuilding, flush pool state and restart pool_manager to get new VMs:
```bash
redis-cli -a staging_redis_2024 KEYS "pool:instances:agent_*" | xargs redis-cli -a staging_redis_2024 DEL
redis-cli -a staging_redis_2024 DEL "pool:pool:agent:warm" "pool:pool:agent:active"
# Stop old containers
docker ps --filter "ancestor=isa-agent-sdk:latest" -q | xargs docker stop
docker ps -a --filter "ancestor=isa-agent-sdk:latest" -q | xargs docker rm
# Restart pool_manager — it will create new warm VMs with updated image
```

---

## Services + Endpoints

**Agent Service (isA_Agent, port 8080)**
- `GET /health` (public)
- `GET /api/v1/agents/templates` (public)
- `GET /api/v1/agents/templates/{template_id}` (public)
- `GET /api/v1/agents/configs` (auth required)
- `POST /api/v1/agents/configs` (auth required)
- `GET /api/v1/agents/configs/{agent_id}` (auth required)
- `PATCH /api/v1/agents/configs/{agent_id}` (auth required)
- `DELETE /api/v1/agents/configs/{agent_id}` (auth required)
- `GET /api/v1/agents/multi-specs` (auth required)
- `GET /api/v1/agents/multi-specs/{spec_id}` (auth required)
- `PATCH /api/v1/agents/multi-specs/{spec_id}` (auth required)
- `DELETE /api/v1/agents/multi-specs/{spec_id}` (auth required)
- `POST /api/v1/agents/chat` (auth required, routes by env, supports stream=true/false)
- `POST /api/v1/agents/generate` (auth required, needs Vibe service)
- `GET /api/v1/agents/scaffold` (auth required)
- `POST /api/v1/agents/local-dev` (auth required)
- `POST /query` (auth required, simple non-streaming)
- `POST /api/v1/agents/swarm` (auth required)
- `POST /api/v1/agents/swarm/stream` (auth required)
- `POST /api/v1/agents/swarm/dag` (auth required)

**Pool Manager (isA_OS, port 8090)**
- `GET /health` (public)
- `GET /api/v1/pools/{type}/stats` (public)
- `POST /api/v1/pools/{agent|custom_agent}/acquire`
- `POST /api/v1/pools/{agent|custom_agent}/{instance_id}/execute`
- `POST /api/v1/pools/{agent|custom_agent}/{instance_id}/stream`
- `POST /api/v1/pools/{agent|custom_agent}/{instance_id}/release`

**Cloud OS (isA_OS, port 8086)**
- `GET /health` (public)
- `POST /api/v1/vms` (create VM with optional `container_port`)
- `POST /api/v1/vms/{vm_id}/start`
- `POST /api/v1/vms/{vm_id}/stop`
- `DELETE /api/v1/vms/{vm_id}`
- `GET /api/v1/vms/{vm_id}/status`

---

## Production Readiness -- NOT READY

### Critical (Must Fix Before Production)
1. **No credential injection**: Proxy socket system (`APIProxyService`) is skipped in dev. Agent VMs call model service directly without API key management. Production requires `network_mode: "none"` + proxy socket for credential isolation.
2. **`host.docker.internal` is macOS-only**: Linux Docker uses bridge IPs directly or requires `--add-host=host.docker.internal:host-gateway`. Production uses Firecracker/Ignite, not Docker Desktop.
3. **No Docker image CI/CD**: Image built manually from local source. No versioning, no registry push, no .dockerignore (2GB+ build context).
4. **No rate limiting / billing**: Agent VM queries have no usage tracking or billing integration.
5. **Single-node warm pool**: Pool manager runs single-node with in-memory executor state. Redis tracks pool metadata but executor's `_instances` dict is lost on restart.
6. **No TLS**: All inter-service communication is plaintext HTTP.

### Medium Priority
7. **Stale Redis entries on restart**: If pool_manager restarts (including `--reload`), the executor's `_instances` dict is cleared but Redis still has entries for old VMs. Requests to these "ghost" instances fail with "Instance not found". **Workaround**: flush Redis agent keys and let pool_manager recreate VMs (see Docker Image section). **Proper fix**: reconcile Redis state with executor state on startup.
8. **`--reload` flag incompatible with pool state**: Using `uvicorn --reload` for pool_manager causes repeated state loss. Use without `--reload` for E2E testing.
9. **No graceful VM cleanup**: Agent VMs are not cleaned up on pool_manager shutdown in all cases.
10. **Template management**: Static templates only, no admin CRUD.
11. **Vibe integration**: Heuristic generator, not full Vibe pipeline.

### Low Priority
12. **Console wiring**: Cloud_pool chat path not wired in Console UI (only API-tested).
13. **Logging/monitoring**: No structured logging, no metrics, no alerting.

---

## How to Run / Test

### Prerequisites
- Docker Desktop running
- PostgreSQL running on localhost:5432 (default user: `postgres`, password: `postgres`, db: `postgres`)
- Redis running on localhost:6379 (password: `staging_redis_2024`)
- Build Docker image: `docker build -t isa-agent-sdk:latest -f isA_Agent/deployment/docker/Dockerfile /path/to/isA`
- isA_Model service running (port 8082) — provides LLM inference

**IMPORTANT**: PostgreSQL is REQUIRED for the Agent Service. Without it, the service falls back to in-memory storage which is wiped on every `--reload`. The service will silently use memory fallback if PG is unreachable — check logs for `"AgentConfigStore setup failed, using memory fallback"`.

### Start Services
```bash
# Terminal 1: Cloud OS
cd isA_OS/os_services/cloud_os && python -m uvicorn main:app --port 8086

# Terminal 2: Pool Manager (NO --reload for stability)
cd isA_OS/os_services/pool_manager && REDIS_PASSWORD=staging_redis_2024 python -m uvicorn main:app --host 0.0.0.0 --port 8090

# Terminal 3: Model Service (isA_Model)
cd isA_Model && python -m uvicorn main:app --port 8082

# Terminal 4: Agent Service
cd isA_Agent && python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### E2E Test Commands (verified working 2026-02-12)
```bash
# 1. Health checks
curl http://localhost:8080/health   # Agent Service
curl http://localhost:8090/health   # Pool Manager
curl http://localhost:8086/health   # Cloud OS
curl http://localhost:8082/health   # Model Service

# 2. Check warm pool (should show warm_instances > 0)
curl http://localhost:8090/api/v1/pools/agent/stats

# 3. Create basic agent
curl -X POST http://localhost:8080/api/v1/agents/configs \
  -H "user-id: test-user" \
  -H "Content-Type: application/json" \
  -d '{"name":"E2E Test","model_id":"claude-sonnet-4-5-20250929","system_prompt":"You are helpful. Keep answers short.","env":"cloud_pool"}'

# 4. Non-streaming chat (use agent_id from step 3)
curl -X POST http://localhost:8080/api/v1/agents/chat \
  -H "user-id: test-user" \
  -H "Content-Type: application/json" \
  -d '{"agent_config_id":"<agent_id>","message":"What is 2+2?","stream":false}'

# 5. Streaming chat
curl -N -X POST http://localhost:8080/api/v1/agents/chat \
  -H "user-id: test-user" \
  -H "Content-Type: application/json" \
  -d '{"agent_config_id":"<agent_id>","message":"What is 2+2?","stream":true}'

# 6. Create from template
curl -X POST http://localhost:8080/api/v1/agents/configs \
  -H "user-id: test-user" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Researcher","model_id":"claude-sonnet-4-5-20250929","template_id":"researcher","source":"template"}'

# 7. Download scaffold
curl -o scaffold.zip http://localhost:8080/api/v1/agents/scaffold \
  -H "user-id: test-user"

# 8. Create local-dev config
curl -X POST http://localhost:8080/api/v1/agents/local-dev \
  -H "user-id: test-user" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Local Agent","description":"Local dev","deployment_id":"deploy-001"}'

# 9. Delete agent
curl -X DELETE http://localhost:8080/api/v1/agents/configs/<agent_id> \
  -H "user-id: test-user"
```

### Run Unit Tests
```bash
# Agent Service tests (zero-mock, real data, 64 tests)
cd isA_Agent && pytest tests/ -v

# Agent SDK tests
cd isA_Agent_SDK && pytest tests/ -v
```

### Debugging Tips
- Check agent VM logs: `docker logs <container_name>`
- Check port mappings: `docker ps --format "table {{.Names}}\t{{.Ports}}"`
- Verify container endpoint works: `docker exec <container_name> curl -s http://localhost:8888/health`
- Check if PG fallback active: look for `"using memory fallback"` in agent service logs
- Flush stale Redis entries if pool gets stuck:
  ```bash
  # Flush agent pool state
  redis-cli -a staging_redis_2024 KEYS "pool:instances:agent_*" | xargs redis-cli -a staging_redis_2024 DEL
  redis-cli -a staging_redis_2024 DEL "pool:pool:agent:warm" "pool:pool:agent:active"
  # Stop stale containers
  docker ps --filter "ancestor=isa-agent-sdk:latest" -q | xargs docker stop
  docker ps -a --filter "ancestor=isa-agent-sdk:latest" -q | xargs docker rm
  # Pool manager will recreate warm VMs automatically
  ```
- If pool_manager shows instances but execute fails with "Instance not found": pool_manager was restarted and lost in-memory executor state. Flush Redis and restart pool_manager (without `--reload`).
- If ports disappear from `docker ps`: container was likely hit by `docker start` bug. Stop and restart pool_manager to recreate VMs.

---

## Remaining TODO
1. **Console UI wiring**: Connect Console frontend to Agent Service API for config CRUD + chat
2. **Local Dev deployment pipeline**: Console upload/build/deploy not wired (only scaffold + config creation works)
3. **Template management**: Only static templates (no admin CRUD)
4. **Vibe generation E2E**: Needs running Vibe service
5. **Custom agent pool E2E**: pool_manager has custom_agent pool but not E2E tested
6. **Production hardening**: Credential injection, CI/CD, billing, Linux networking, TLS
7. **Pool manager resilience**: Reconcile Redis state with executor state on startup to handle restarts gracefully

---

## All Files Modified (Cumulative)

- `isA_Agent_SDK/isa_agent_sdk/services/agent_config_store.py`
- `isA_Agent_SDK/isa_agent_sdk/services/agent_creation.py`
- `isA_Agent_SDK/isa_agent_sdk/services/multi_agent_runner.py`
- `isA_Agent_SDK/isa_agent_sdk/agents/swarm.py`
- `isA_Agent_SDK/isa_agent_sdk/agents/handoff.py`
- `isA_Agent_SDK/isa_agent_sdk/clients/pool_manager_client.py`
- `isA_Agent/main.py`
- `isA_Agent/deployment/docker/Dockerfile` (NEW)
- `isA_Agent/deployment/docker/entrypoint.sh` (NEW)
- `isA_Agent/deployment/requirements/base_dev.txt`
- `isA_Agent/tests/__init__.py` (NEW)
- `isA_Agent/tests/conftest.py` (NEW)
- `isA_Agent/tests/test_auth_gates.py` (NEW)
- `isA_Agent/tests/test_scenario_basic.py` (NEW)
- `isA_Agent/tests/test_scenario_template.py` (NEW)
- `isA_Agent/tests/test_scenario_generated.py` (NEW)
- `isA_Agent/tests/test_scenario_local_dev.py` (NEW)
- `isA_Agent_SDK/tests/test_agent_config_store.py` (NEW)
- `isA_Agent_SDK/tests/test_agent_creation.py` (NEW)
- `isA_Agent_SDK/tests/test_agent_creation_generated.py` (NEW)
- `isA_Agent_SDK/tests/test_multi_agent_runner.py` (NEW)
- `isA_OS/os_services/cloud_os/src/models/vm_models.py`
- `isA_OS/os_services/cloud_os/src/backends/base.py`
- `isA_OS/os_services/cloud_os/src/backends/docker_backend.py`
- `isA_OS/os_services/cloud_os/src/backends/ignite_backend.py`
- `isA_OS/os_services/cloud_os/src/services/vm_service.py`
- `isA_OS/os_services/pool_manager/main.py`
- `isA_OS/os_services/pool_manager/src/executors/agent/cloudos_agent_executor.py`
- `isA_user/microservices/account_service/clients/organization_client.py`
