# Deployment

### HTTP Client (for deployed apps)

```python
from isa_agent_sdk import ISAAgent

client = ISAAgent(base_url="http://localhost:8000")

### 5. **Deployment Scripts** ✅

**Files Created**:

1. ✅ `deployment/staging/scripts/start_worker.py`
   - Worker startup script with environment validation
   - Debug mode support
   - Clear error messages

2. ✅ `deployment/staging/scripts/manage_workers.sh`
   - Multi-worker management (start/stop/status/restart)
   - Configurable worker counts by priority
   - systemd installation support

3. ✅ `deployment/staging/agent-worker.service`
   - systemd service template
   - Auto-restart configuration
   - Resource limits
   - Security hardening

4. ✅ `deployment/staging/WORKER_DEPLOYMENT.md`
   - Complete deployment guide
   - Architecture diagrams
   - Troubleshooting guide
   - Best practices

---

### Deployment (4 files)

7. ✅ `deployment/staging/scripts/start_worker.py` - NEW
8. ✅ `deployment/staging/scripts/manage_workers.sh` - NEW
9. ✅ `deployment/staging/agent-worker.service` - NEW
10. ✅ `deployment/staging/WORKER_DEPLOYMENT.md` - NEW

**Total**: 10 files (6 modified, 4 new)

---

## References

- [README.md](./README.md)
- [deployment-guide.md](./deployment-guide.md)
- [human-in-the-loop.md](./human-in-the-loop.md)
- [isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md](./isa_agent_sdk/services/background_jobs/docs/INTEGRATION_COMPLETE.md)
