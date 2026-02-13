"""
Tests for the Task DAG system (models + scheduler).
"""
import pytest
from isa_agent_sdk.dag.models import DAGTask, DAGTaskStatus, DAGState
from isa_agent_sdk.dag.scheduler import DAGScheduler


# ---------------------------------------------------------------------------
# Model serialization tests
# ---------------------------------------------------------------------------

class TestDAGTaskSerialization:
    def test_round_trip(self):
        task = DAGTask(
            task_id="t1",
            title="Research",
            description="Do research",
            tools=["web_search"],
            priority="high",
            depends_on=["t0"],
            metadata={"estimated_duration_minutes": 5},
        )
        d = task.to_dict()
        restored = DAGTask.from_dict(d)
        assert restored.task_id == "t1"
        assert restored.depends_on == ["t0"]
        assert restored.metadata["estimated_duration_minutes"] == 5
        assert restored.status == DAGTaskStatus.PENDING

    def test_to_task_dict(self):
        task = DAGTask(
            task_id="t1",
            title="Build",
            description="Build it",
            tools=["compiler"],
            metadata={"lang": "python"},
        )
        flat = task.to_task_dict()
        assert flat["id"] == "t1"
        assert flat["title"] == "Build"
        assert flat["lang"] == "python"


class TestDAGStateSerialization:
    def test_round_trip(self):
        state = DAGState()
        state.tasks["a"] = DAGTask(task_id="a", title="A", description="A")
        state.tasks["b"] = DAGTask(
            task_id="b", title="B", description="B", depends_on=["a"]
        )
        state.execution_order = [["a"], ["b"]]
        state.completed_count = 1

        d = state.to_dict()
        restored = DAGState.from_dict(d)
        assert len(restored.tasks) == 2
        assert restored.tasks["b"].depends_on == ["a"]
        assert restored.execution_order == [["a"], ["b"]]
        assert restored.completed_count == 1

    def test_empty_state(self):
        state = DAGState()
        d = state.to_dict()
        restored = DAGState.from_dict(d)
        assert len(restored.tasks) == 0
        assert restored.execution_order == []


# ---------------------------------------------------------------------------
# DAGScheduler.build_dag
# ---------------------------------------------------------------------------

class TestBuildDAG:
    def test_basic(self):
        tasks = [
            {"id": "a", "title": "A", "description": "do A"},
            {"id": "b", "title": "B", "description": "do B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        assert "a" in dag.tasks
        assert "b" in dag.tasks
        assert dag.tasks["b"].depends_on == ["a"]

    def test_auto_id(self):
        tasks = [{"title": "X"}, {"title": "Y"}]
        dag = DAGScheduler.build_dag(tasks)
        assert "task_1" in dag.tasks
        assert "task_2" in dag.tasks

    def test_metadata_passthrough(self):
        tasks = [
            {"id": "a", "title": "A", "estimated_duration_minutes": 10}
        ]
        dag = DAGScheduler.build_dag(tasks)
        assert dag.tasks["a"].metadata["estimated_duration_minutes"] == 10

    def test_skips_non_dict(self):
        tasks = ["not_a_dict", {"id": "a", "title": "A"}]
        dag = DAGScheduler.build_dag(tasks)
        assert len(dag.tasks) == 1

    def test_task_id_field(self):
        tasks = [{"task_id": "custom_id", "title": "Custom"}]
        dag = DAGScheduler.build_dag(tasks)
        assert "custom_id" in dag.tasks


# ---------------------------------------------------------------------------
# DAGScheduler.validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_dag(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["a", "b"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert errors == []

    def test_self_dependency(self):
        tasks = [{"id": "a", "title": "A", "depends_on": ["a"]}]
        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert any("depends on itself" in e for e in errors)

    def test_missing_dependency(self):
        tasks = [{"id": "a", "title": "A", "depends_on": ["nonexistent"]}]
        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert any("non-existent" in e for e in errors)

    def test_cycle_detection(self):
        tasks = [
            {"id": "a", "title": "A", "depends_on": ["b"]},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert any("Cycle" in e for e in errors)

    def test_three_node_cycle(self):
        tasks = [
            {"id": "a", "title": "A", "depends_on": ["c"]},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["b"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert any("Cycle" in e for e in errors)

    def test_empty_dag_is_valid(self):
        dag = DAGState()
        errors = DAGScheduler.validate(dag)
        assert errors == []

    def test_no_deps_is_valid(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B"},
        ]
        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert errors == []


# ---------------------------------------------------------------------------
# DAGScheduler.compute_wavefronts
# ---------------------------------------------------------------------------

class TestComputeWavefronts:
    def test_linear_chain(self):
        """A -> B -> C produces three wavefronts."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["b"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        assert wavefronts == [["a"], ["b"], ["c"]]

    def test_diamond(self):
        """Diamond: A -> {B, C} -> D produces three wavefronts."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["a"]},
            {"id": "d", "title": "D", "depends_on": ["b", "c"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        assert wavefronts[0] == ["a"]
        assert sorted(wavefronts[1]) == ["b", "c"]
        assert wavefronts[2] == ["d"]

    def test_all_independent(self):
        """All independent tasks in one wavefront."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B"},
            {"id": "c", "title": "C"},
        ]
        dag = DAGScheduler.build_dag(tasks)
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        assert len(wavefronts) == 1
        assert sorted(wavefronts[0]) == ["a", "b", "c"]

    def test_single_task(self):
        tasks = [{"id": "a", "title": "A"}]
        dag = DAGScheduler.build_dag(tasks)
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        assert wavefronts == [["a"]]

    def test_empty_dag(self):
        dag = DAGState()
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        assert wavefronts == []

    def test_cycle_raises(self):
        tasks = [
            {"id": "a", "title": "A", "depends_on": ["b"]},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        with pytest.raises(ValueError, match="Cycle"):
            DAGScheduler.compute_wavefronts(dag)

    def test_wide_fan_out(self):
        """One root with many children."""
        tasks = [{"id": "root", "title": "Root"}]
        for i in range(5):
            tasks.append({"id": f"child_{i}", "title": f"Child {i}", "depends_on": ["root"]})
        dag = DAGScheduler.build_dag(tasks)
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        assert wavefronts[0] == ["root"]
        assert len(wavefronts[1]) == 5


# ---------------------------------------------------------------------------
# DAGScheduler.get_ready_tasks
# ---------------------------------------------------------------------------

class TestGetReadyTasks:
    def test_initial_ready(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == ["a"]

    def test_after_completion(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_completed(dag, "a", "done")
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == ["b"]

    def test_running_not_ready(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B"},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag.tasks["a"].status = DAGTaskStatus.RUNNING
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == ["b"]

    def test_all_complete_nothing_ready(self):
        tasks = [{"id": "a", "title": "A"}]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_completed(dag, "a", "done")
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == []


# ---------------------------------------------------------------------------
# DAGScheduler.mark_task_failed + cascading
# ---------------------------------------------------------------------------

class TestFailureCascade:
    def test_basic_cascade(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["b"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_failed(dag, "a", "boom")

        assert dag.tasks["a"].status == DAGTaskStatus.FAILED
        assert dag.tasks["b"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["c"].status == DAGTaskStatus.SKIPPED
        assert dag.failed_count == 1
        assert dag.skipped_count == 2

    def test_partial_cascade(self):
        """Only dependents of the failed task are skipped."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B"},
            {"id": "c", "title": "C", "depends_on": ["a"]},
            {"id": "d", "title": "D", "depends_on": ["b"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_failed(dag, "a", "error")

        assert dag.tasks["a"].status == DAGTaskStatus.FAILED
        assert dag.tasks["b"].status == DAGTaskStatus.PENDING  # not affected
        assert dag.tasks["c"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["d"].status == DAGTaskStatus.PENDING  # not affected

    def test_diamond_cascade(self):
        """Failure at root cascades through entire diamond."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["a"]},
            {"id": "d", "title": "D", "depends_on": ["b", "c"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_failed(dag, "a", "err")

        assert dag.tasks["b"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["c"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["d"].status == DAGTaskStatus.SKIPPED
        assert dag.skipped_count == 3

    def test_no_cascade_on_leaf_failure(self):
        """Failure on a leaf node doesn't cascade."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_completed(dag, "a", "done")
        dag = DAGScheduler.mark_task_failed(dag, "b", "err")

        assert dag.tasks["a"].status == DAGTaskStatus.COMPLETED
        assert dag.tasks["b"].status == DAGTaskStatus.FAILED
        assert dag.skipped_count == 0

    def test_completed_tasks_not_skipped(self):
        """Already completed tasks are not re-marked as skipped."""
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_completed(dag, "a", "done")
        dag.tasks["b"].status = DAGTaskStatus.COMPLETED
        dag = DAGScheduler.mark_task_failed(dag, "a", "late failure")
        # b was already completed - should stay completed
        # Note: a was already completed too but we forced a re-fail (edge case)
        # c was still pending so gets skipped
        assert dag.tasks["c"].status == DAGTaskStatus.SKIPPED


# ---------------------------------------------------------------------------
# DAGScheduler.is_dag_complete
# ---------------------------------------------------------------------------

class TestIsDagComplete:
    def test_all_completed(self):
        tasks = [{"id": "a", "title": "A"}]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_completed(dag, "a", "done")
        assert DAGScheduler.is_dag_complete(dag)

    def test_pending_not_complete(self):
        tasks = [{"id": "a", "title": "A"}]
        dag = DAGScheduler.build_dag(tasks)
        assert not DAGScheduler.is_dag_complete(dag)

    def test_failed_and_skipped_is_complete(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_failed(dag, "a", "err")
        assert DAGScheduler.is_dag_complete(dag)

    def test_empty_dag_is_complete(self):
        dag = DAGState()
        assert DAGScheduler.is_dag_complete(dag)

    def test_mixed_terminal_states(self):
        tasks = [
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B"},
            {"id": "c", "title": "C", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag = DAGScheduler.mark_task_completed(dag, "b", "ok")
        dag = DAGScheduler.mark_task_failed(dag, "a", "err")
        # c should be skipped by cascade
        assert DAGScheduler.is_dag_complete(dag)


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import ToolMessage


# ---------------------------------------------------------------------------
# Integration Test 1: ToolNode DAG detection
# ---------------------------------------------------------------------------

class TestToolNodeDAGDetection:
    """Integration tests for ToolNode._detect_autonomous_plan with DAG paths."""

    def _make_tool_node(self):
        """
        Create a ToolNode with its heavy dependencies mocked out so the
        constructor completes without hitting Consul / Redis / etc.
        """
        with patch(
            "isa_agent_sdk.nodes.tool_node.get_hil_service",
            return_value=MagicMock(),
        ), patch(
            "isa_agent_sdk.services.auto_detection.tool_profiler.get_tool_profiler",
            return_value=MagicMock(),
        ), patch(
            "isa_agent_sdk.nodes.tool_node.ToolHILDetector",
            return_value=MagicMock(),
        ), patch(
            "isa_agent_sdk.nodes.tool_node.ToolHILRouter",
            return_value=MagicMock(),
        ):
            from isa_agent_sdk.nodes.tool_node import ToolNode
            node = ToolNode()
        return node

    # -- DAG is built when tasks have depends_on --

    def test_detect_plan_builds_dag_with_depends_on(self):
        """When create_execution_plan returns tasks with depends_on,
        _detect_autonomous_plan should include a task_dag key."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "status": "success",
            "data": {
                "plan_id": "plan_001",
                "tasks": [
                    {"id": "t1", "title": "Setup", "description": "Initial setup"},
                    {"id": "t2", "title": "Build", "description": "Build step", "depends_on": ["t1"]},
                    {"id": "t3", "title": "Test", "description": "Test step", "depends_on": ["t2"]},
                ],
            },
        })

        tool_info_list = [("create_execution_plan", {"goal": "deploy"}, "call_0")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_0")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None, "Plan should be detected"
        assert "task_dag" in result, "DAG should be present when depends_on exists"
        dag_dict = result["task_dag"]
        assert "t1" in dag_dict["tasks"]
        assert "t2" in dag_dict["tasks"]
        assert "t3" in dag_dict["tasks"]

        # Verify wavefronts were computed (execution_order populated)
        assert len(dag_dict["execution_order"]) == 3  # linear chain -> 3 wavefronts

    def test_detect_plan_no_dag_without_depends_on(self):
        """When tasks do NOT have depends_on, result should have no task_dag key."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "status": "success",
            "data": {
                "plan_id": "plan_002",
                "tasks": [
                    {"id": "t1", "title": "Task A", "description": "A"},
                    {"id": "t2", "title": "Task B", "description": "B"},
                ],
            },
        })

        tool_info_list = [("create_execution_plan", {"goal": "flat"}, "call_1")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_1")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None, "Plan should still be detected"
        assert "task_dag" not in result, "No DAG when no depends_on"
        assert len(result["tasks"]) == 2

    def test_detect_plan_dag_validation_error_falls_back(self):
        """When DAG validation fails (e.g. cycle), the result should fall back
        to a flat task list with no task_dag."""
        node = self._make_tool_node()

        # Introduce a cycle: t1 -> t2 -> t1
        plan_payload = json.dumps({
            "status": "success",
            "data": {
                "plan_id": "plan_003",
                "tasks": [
                    {"id": "t1", "title": "First", "description": "A", "depends_on": ["t2"]},
                    {"id": "t2", "title": "Second", "description": "B", "depends_on": ["t1"]},
                ],
            },
        })

        tool_info_list = [("create_execution_plan", {}, "call_2")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_2")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None, "Plan should still be detected (flat fallback)"
        assert "task_dag" not in result, "DAG should NOT be present on validation error"
        assert len(result["tasks"]) == 2

    def test_detect_plan_missing_dep_falls_back(self):
        """A depends_on referencing a non-existent task should fail validation
        and fall back to a flat list."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "status": "success",
            "data": {
                "plan_id": "plan_004",
                "tasks": [
                    {"id": "t1", "title": "A", "description": "A", "depends_on": ["nonexistent"]},
                ],
            },
        })

        tool_info_list = [("create_execution_plan", {}, "call_3")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_3")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None
        assert "task_dag" not in result

    def test_detect_plan_diamond_dag(self):
        """Diamond dependency graph: A -> {B, C} -> D should produce 3 wavefronts."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "status": "success",
            "data": {
                "plan_id": "plan_005",
                "tasks": [
                    {"id": "a", "title": "A", "description": "Root"},
                    {"id": "b", "title": "B", "description": "Left", "depends_on": ["a"]},
                    {"id": "c", "title": "C", "description": "Right", "depends_on": ["a"]},
                    {"id": "d", "title": "D", "description": "Merge", "depends_on": ["b", "c"]},
                ],
            },
        })

        tool_info_list = [("create_execution_plan", {}, "call_4")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_4")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None
        assert "task_dag" in result
        dag_dict = result["task_dag"]
        assert len(dag_dict["execution_order"]) == 3
        assert dag_dict["execution_order"][0] == ["a"]
        assert sorted(dag_dict["execution_order"][1]) == ["b", "c"]
        assert dag_dict["execution_order"][2] == ["d"]

    def test_detect_plan_not_triggered_for_other_tools(self):
        """_detect_autonomous_plan returns None when the tool is not
        create_execution_plan."""
        node = self._make_tool_node()

        tool_info_list = [("web_search", {"query": "foo"}, "call_5")]
        tool_messages = [ToolMessage(content="search results", tool_call_id="call_5")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)
        assert result is None

    def test_detect_plan_error_response_returns_none(self):
        """An error response from create_execution_plan should be ignored."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "status": "error",
            "message": "Plan creation failed",
        })

        tool_info_list = [("create_execution_plan", {}, "call_6")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_6")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)
        assert result is None

    def test_detect_plan_auto_approved_format(self):
        """Auto-approved plan format should also trigger DAG building."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "auto_approved": True,
            "content": {
                "plan_id": "plan_auto",
                "tasks": [
                    {"id": "a", "title": "A", "description": "A"},
                    {"id": "b", "title": "B", "description": "B", "depends_on": ["a"]},
                ],
            },
        })

        tool_info_list = [("create_execution_plan", {}, "call_7")]
        tool_messages = [ToolMessage(content=plan_payload, tool_call_id="call_7")]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None
        assert "task_dag" in result
        assert len(result["task_dag"]["execution_order"]) == 2

    def test_detect_plan_with_multiple_tool_calls(self):
        """When multiple tools are called and one is create_execution_plan,
        only the plan tool's message is used for DAG detection."""
        node = self._make_tool_node()

        plan_payload = json.dumps({
            "status": "success",
            "data": {
                "plan_id": "plan_multi",
                "tasks": [
                    {"id": "a", "title": "A", "description": "A"},
                    {"id": "b", "title": "B", "description": "B", "depends_on": ["a"]},
                ],
            },
        })

        tool_info_list = [
            ("web_search", {"query": "test"}, "call_8a"),
            ("create_execution_plan", {"goal": "multi"}, "call_8b"),
        ]
        tool_messages = [
            ToolMessage(content="search results", tool_call_id="call_8a"),
            ToolMessage(content=plan_payload, tool_call_id="call_8b"),
        ]

        result = node._detect_autonomous_plan(tool_info_list, tool_messages)

        assert result is not None
        assert "task_dag" in result


# ---------------------------------------------------------------------------
# Integration Test 2: AgentExecutorNode DAG path selection
# ---------------------------------------------------------------------------

class TestAgentExecutorNodeDAGPathSelection:
    """Test that AgentExecutorNode selects the correct execution path
    based on presence/absence of task_dag in state."""

    def _make_executor_node(self):
        """Create an AgentExecutorNode with dependencies mocked."""
        with patch(
            "isa_agent_sdk.nodes.agent_executor_node.get_hil_service",
            return_value=MagicMock(),
        ):
            from isa_agent_sdk.nodes.agent_executor_node import AgentExecutorNode
            node = AgentExecutorNode()
        return node

    @pytest.mark.asyncio
    async def test_dag_path_taken_when_task_dag_present(self):
        """When state has task_dag, _execute_dag should be invoked."""
        node = self._make_executor_node()

        # Build a real DAG state dict
        tasks = [
            {"id": "a", "title": "A", "description": "Do A"},
            {"id": "b", "title": "B", "description": "Do B", "depends_on": ["a"]},
        ]
        dag = DAGScheduler.build_dag(tasks)
        dag.execution_order = DAGScheduler.compute_wavefronts(dag)
        dag_dict = dag.to_dict()

        state = {
            "messages": [],
            "remaining_steps": 50,
            "task_dag": dag_dict,
            "task_list": None,
            "current_task_index": 0,
            "execution_mode": "sequential",
            "completed_task_count": 0,
            "failed_task_count": 0,
        }

        mock_config = {"configurable": {}}

        # Mock _execute_dag and the mode/context helpers
        with patch.object(node, "_execute_dag", new_callable=AsyncMock) as mock_dag, \
             patch.object(node, "get_runtime_context", return_value={}), \
             patch.object(node, "stream_custom"):
            mock_dag.return_value = {"next_action": "call_model"}
            result = await node._execute_logic(state, mock_config)

            mock_dag.assert_called_once_with(state, mock_config, dag_dict)
            assert result["next_action"] == "call_model"

    @pytest.mark.asyncio
    async def test_flat_path_taken_when_task_dag_absent(self):
        """When state has no task_dag, flat-list execution path should be used."""
        node = self._make_executor_node()

        state = {
            "messages": [],
            "remaining_steps": 50,
            "task_dag": None,
            "task_list": [
                {"id": "t1", "title": "Flat task 1", "description": "Do it"},
            ],
            "current_task_index": 0,
            "execution_mode": "sequential",
            "completed_task_count": 0,
            "failed_task_count": 0,
            "execution_plan": {},
            "autonomous_tasks": None,
        }

        mock_config = {"configurable": {"mcp_service": MagicMock()}}

        with patch.object(node, "get_runtime_context", return_value={}), \
             patch.object(node, "stream_custom"), \
             patch.object(node, "stream_tool"), \
             patch.object(node, "_execute_single_task", new_callable=AsyncMock) as mock_exec, \
             patch.object(node, "_should_trigger_progress_checkpoint", return_value=False):
            mock_exec.return_value = "Task completed successfully"
            result = await node._execute_logic(state, mock_config)

            # _execute_dag should NOT be called (task_dag is None)
            # The sequential path should have been taken
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_tasks_path_when_state_empty(self):
        """When state has no task_dag and no tasks, _handle_no_tasks is called."""
        node = self._make_executor_node()

        state = {
            "messages": [],
            "remaining_steps": 50,
            "task_dag": None,
            "task_list": None,
            "current_task_index": 0,
            "execution_mode": "sequential",
            "completed_task_count": 0,
            "failed_task_count": 0,
            "execution_plan": {},
            "autonomous_tasks": None,
        }

        mock_config = {"configurable": {}}

        with patch.object(node, "get_runtime_context", return_value={}), \
             patch.object(node, "stream_custom"):
            result = await node._execute_logic(state, mock_config)
            assert result["next_action"] == "end"


# ---------------------------------------------------------------------------
# Integration Test 3: End-to-end DAG flow
# ---------------------------------------------------------------------------

class TestEndToEndDAGFlow:
    """End-to-end integration test covering the full DAG lifecycle:
    build -> validate -> compute_wavefronts -> get_ready_tasks ->
    mark_completed/failed -> is_dag_complete.
    """

    def test_full_lifecycle_success(self):
        """Simulate the complete lifecycle where all tasks succeed."""
        # Phase 1: Build (simulates what ToolNode._detect_autonomous_plan does)
        tasks = [
            {"id": "research", "title": "Research", "description": "Gather data"},
            {"id": "analyze", "title": "Analyze", "description": "Analyze data", "depends_on": ["research"]},
            {"id": "visualize", "title": "Visualize", "description": "Create charts", "depends_on": ["research"]},
            {"id": "report", "title": "Report", "description": "Write report", "depends_on": ["analyze", "visualize"]},
        ]

        dag = DAGScheduler.build_dag(tasks)

        # Phase 2: Validate
        errors = DAGScheduler.validate(dag)
        assert errors == [], f"Unexpected validation errors: {errors}"

        # Phase 3: Compute wavefronts
        wavefronts = DAGScheduler.compute_wavefronts(dag)
        dag.execution_order = wavefronts
        assert wavefronts == [["research"], sorted(["analyze", "visualize"]), ["report"]]

        # Phase 4: Serialize to dict (simulates being stored in AgentState)
        dag_dict = dag.to_dict()
        assert isinstance(dag_dict, dict)

        # Phase 5: Deserialize (simulates AgentExecutorNode reading from state)
        restored_dag = DAGState.from_dict(dag_dict)
        assert len(restored_dag.tasks) == 4

        # Phase 6: Wavefront 1 - get ready tasks
        ready = DAGScheduler.get_ready_tasks(restored_dag)
        assert ready == ["research"]

        # Phase 7: Execute wavefront 1 - mark completed
        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "research", "Data gathered")
        assert restored_dag.completed_count == 1
        assert not DAGScheduler.is_dag_complete(restored_dag)

        # Phase 8: Wavefront 2 - get ready tasks
        ready = DAGScheduler.get_ready_tasks(restored_dag)
        assert sorted(ready) == ["analyze", "visualize"]

        # Phase 9: Execute wavefront 2 - mark completed
        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "analyze", "Analysis done")
        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "visualize", "Charts created")
        assert restored_dag.completed_count == 3
        assert not DAGScheduler.is_dag_complete(restored_dag)

        # Phase 10: Wavefront 3 - get ready tasks
        ready = DAGScheduler.get_ready_tasks(restored_dag)
        assert ready == ["report"]

        # Phase 11: Execute wavefront 3 - mark completed
        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "report", "Report written")
        assert restored_dag.completed_count == 4
        assert DAGScheduler.is_dag_complete(restored_dag)

    def test_full_lifecycle_with_failure_cascade(self):
        """Simulate the lifecycle where an early task fails, cascading skips."""
        tasks = [
            {"id": "fetch", "title": "Fetch", "description": "Fetch data"},
            {"id": "clean", "title": "Clean", "description": "Clean data"},
            {"id": "transform", "title": "Transform", "description": "Transform data", "depends_on": ["fetch"]},
            {"id": "merge", "title": "Merge", "description": "Merge data", "depends_on": ["fetch", "clean"]},
            {"id": "load", "title": "Load", "description": "Load to DB", "depends_on": ["transform", "merge"]},
        ]

        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert errors == []

        wavefronts = DAGScheduler.compute_wavefronts(dag)
        dag.execution_order = wavefronts

        # Wavefront 1: fetch and clean are both ready (no deps)
        ready = DAGScheduler.get_ready_tasks(dag)
        assert sorted(ready) == ["clean", "fetch"]

        # clean succeeds
        dag = DAGScheduler.mark_task_completed(dag, "clean", "Cleaned")

        # fetch fails -> transform, merge, and load should all be skipped
        dag = DAGScheduler.mark_task_failed(dag, "fetch", "Connection timeout")

        assert dag.tasks["fetch"].status == DAGTaskStatus.FAILED
        assert dag.tasks["clean"].status == DAGTaskStatus.COMPLETED
        assert dag.tasks["transform"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["merge"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["load"].status == DAGTaskStatus.SKIPPED

        assert dag.failed_count == 1
        assert dag.skipped_count == 3
        assert dag.completed_count == 1

        # No more ready tasks
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == []

        # DAG is complete (all tasks terminal)
        assert DAGScheduler.is_dag_complete(dag)

    def test_full_lifecycle_partial_failure(self):
        """One branch fails but the other completes successfully."""
        tasks = [
            {"id": "root", "title": "Root", "description": "Root task"},
            {"id": "left", "title": "Left", "description": "Left branch", "depends_on": ["root"]},
            {"id": "right", "title": "Right", "description": "Right branch", "depends_on": ["root"]},
            {"id": "left_child", "title": "LeftChild", "description": "Left leaf", "depends_on": ["left"]},
            {"id": "right_child", "title": "RightChild", "description": "Right leaf", "depends_on": ["right"]},
        ]

        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert errors == []

        wavefronts = DAGScheduler.compute_wavefronts(dag)
        dag.execution_order = wavefronts

        # Wavefront 1: root
        dag = DAGScheduler.mark_task_completed(dag, "root", "Root done")

        # Wavefront 2: left and right become ready
        ready = DAGScheduler.get_ready_tasks(dag)
        assert sorted(ready) == ["left", "right"]

        dag = DAGScheduler.mark_task_completed(dag, "right", "Right done")
        dag = DAGScheduler.mark_task_failed(dag, "left", "Left error")

        # left_child should be skipped; right_child should be ready
        assert dag.tasks["left_child"].status == DAGTaskStatus.SKIPPED
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == ["right_child"]

        dag = DAGScheduler.mark_task_completed(dag, "right_child", "Right child done")
        assert DAGScheduler.is_dag_complete(dag)

        assert dag.completed_count == 3  # root, right, right_child
        assert dag.failed_count == 1     # left
        assert dag.skipped_count == 1    # left_child

    def test_serialization_roundtrip_preserves_progress(self):
        """Simulate serializing mid-execution DAG to AgentState and restoring it,
        verifying all progress data is preserved."""
        tasks = [
            {"id": "a", "title": "A", "description": "A"},
            {"id": "b", "title": "B", "description": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "description": "C", "depends_on": ["a"]},
            {"id": "d", "title": "D", "description": "D", "depends_on": ["b", "c"]},
        ]

        dag = DAGScheduler.build_dag(tasks)
        dag.execution_order = DAGScheduler.compute_wavefronts(dag)

        # Execute first wavefront
        dag = DAGScheduler.mark_task_completed(dag, "a", "done-a")

        # Serialize (simulates AgentState storage)
        dag_dict = dag.to_dict()

        # Restore (simulates AgentExecutorNode reading from state)
        restored = DAGState.from_dict(dag_dict)

        # Verify all state is preserved
        assert restored.completed_count == 1
        assert restored.tasks["a"].status == DAGTaskStatus.COMPLETED
        assert restored.tasks["a"].result == "done-a"
        assert restored.tasks["b"].status == DAGTaskStatus.PENDING
        assert restored.tasks["c"].status == DAGTaskStatus.PENDING
        assert restored.tasks["d"].status == DAGTaskStatus.PENDING

        # Continue execution on restored DAG
        ready = DAGScheduler.get_ready_tasks(restored)
        assert sorted(ready) == ["b", "c"]

        restored = DAGScheduler.mark_task_completed(restored, "b", "done-b")
        restored = DAGScheduler.mark_task_completed(restored, "c", "done-c")

        ready = DAGScheduler.get_ready_tasks(restored)
        assert ready == ["d"]

        restored = DAGScheduler.mark_task_completed(restored, "d", "done-d")
        assert DAGScheduler.is_dag_complete(restored)
        assert restored.completed_count == 4

    def test_toolnode_to_executor_dag_integration(self):
        """Simulate the complete ToolNode -> AgentExecutorNode handoff:
        1. ToolNode detects plan with depends_on and builds DAG
        2. DAG state dict is stored in AgentState
        3. AgentExecutorNode reads DAG and can schedule tasks."""
        # Step 1: Simulate ToolNode _detect_autonomous_plan building DAG
        raw_tasks = [
            {"id": "setup", "title": "Setup", "description": "Env setup", "tools": ["shell"]},
            {"id": "build", "title": "Build", "description": "Compile", "tools": ["compiler"], "depends_on": ["setup"]},
            {"id": "test", "title": "Test", "description": "Run tests", "tools": ["test_runner"], "depends_on": ["build"]},
            {"id": "deploy", "title": "Deploy", "description": "Ship it", "tools": ["deployer"], "depends_on": ["test"]},
        ]

        # This mirrors the code path in ToolNode._detect_autonomous_plan
        has_dependencies = any(
            task.get("depends_on") for task in raw_tasks if isinstance(task, dict)
        )
        assert has_dependencies

        dag_state = DAGScheduler.build_dag(raw_tasks)
        errors = DAGScheduler.validate(dag_state)
        assert errors == []

        dag_state.execution_order = DAGScheduler.compute_wavefronts(dag_state)
        dag_dict = dag_state.to_dict()

        # Step 2: Verify the dict could be placed in AgentState
        assert isinstance(dag_dict, dict)
        assert "tasks" in dag_dict
        assert "execution_order" in dag_dict

        # Step 3: Simulate AgentExecutorNode._execute_dag reading DAG
        restored_dag = DAGState.from_dict(dag_dict)

        # Wavefront scheduling mirrors _execute_dag logic
        ready_ids = DAGScheduler.get_ready_tasks(restored_dag)
        assert ready_ids == ["setup"]

        # Execute setup
        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "setup", "env ready")

        ready_ids = DAGScheduler.get_ready_tasks(restored_dag)
        assert ready_ids == ["build"]

        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "build", "compiled")

        ready_ids = DAGScheduler.get_ready_tasks(restored_dag)
        assert ready_ids == ["test"]

        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "test", "all pass")

        ready_ids = DAGScheduler.get_ready_tasks(restored_dag)
        assert ready_ids == ["deploy"]

        restored_dag = DAGScheduler.mark_task_completed(restored_dag, "deploy", "shipped")

        assert DAGScheduler.is_dag_complete(restored_dag)
        assert restored_dag.completed_count == 4
        assert restored_dag.failed_count == 0
        assert restored_dag.skipped_count == 0

    def test_dag_deadlock_scenario(self):
        """Simulate a scenario where the DAG reaches a state with no
        ready tasks but is not complete (deadlock due to partial failure
        leaving blocked tasks whose sole dependency was not the failed one,
        but rather a still-pending task that itself was not ready).

        In practice this happens when task B depends on A and C, A fails,
        but C is still running/pending. B gets skipped by cascade from A.
        So the DAG completes with no deadlock. Let's verify that."""
        tasks = [
            {"id": "a", "title": "A", "description": "A"},
            {"id": "c", "title": "C", "description": "C"},
            {"id": "b", "title": "B", "description": "B", "depends_on": ["a", "c"]},
        ]

        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        assert errors == []

        # a and c both ready initially
        ready = DAGScheduler.get_ready_tasks(dag)
        assert sorted(ready) == ["a", "c"]

        # a fails -> b gets skipped
        dag = DAGScheduler.mark_task_failed(dag, "a", "boom")
        assert dag.tasks["b"].status == DAGTaskStatus.SKIPPED

        # c is still pending and ready
        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == ["c"]

        dag = DAGScheduler.mark_task_completed(dag, "c", "done")
        assert DAGScheduler.is_dag_complete(dag)

    def test_wide_parallel_wavefront_lifecycle(self):
        """Test a wide fan-out pattern where many tasks execute in parallel."""
        root_task = {"id": "root", "title": "Root", "description": "Root"}
        child_tasks = [
            {"id": f"child_{i}", "title": f"Child {i}", "description": f"Parallel task {i}", "depends_on": ["root"]}
            for i in range(10)
        ]
        merge_task = {
            "id": "merge",
            "title": "Merge",
            "description": "Merge all",
            "depends_on": [f"child_{i}" for i in range(10)],
        }

        all_tasks = [root_task] + child_tasks + [merge_task]
        dag = DAGScheduler.build_dag(all_tasks)
        errors = DAGScheduler.validate(dag)
        assert errors == []

        wavefronts = DAGScheduler.compute_wavefronts(dag)
        dag.execution_order = wavefronts

        assert len(wavefronts) == 3
        assert wavefronts[0] == ["root"]
        assert len(wavefronts[1]) == 10
        assert wavefronts[2] == ["merge"]

        # Execute root
        dag = DAGScheduler.mark_task_completed(dag, "root", "Root done")
        ready = DAGScheduler.get_ready_tasks(dag)
        assert len(ready) == 10

        # Execute all children
        for i in range(10):
            dag = DAGScheduler.mark_task_completed(dag, f"child_{i}", f"child {i} done")

        ready = DAGScheduler.get_ready_tasks(dag)
        assert ready == ["merge"]

        dag = DAGScheduler.mark_task_completed(dag, "merge", "merged")
        assert DAGScheduler.is_dag_complete(dag)
        assert dag.completed_count == 12  # root + 10 children + merge

    def test_to_task_dict_used_by_executor(self):
        """Verify DAGTask.to_task_dict produces the format expected by
        AgentExecutorNode._execute_single_task."""
        task = DAGTask(
            task_id="deploy",
            title="Deploy App",
            description="Deploy the application",
            tools=["kubectl", "helm"],
            priority="high",
            metadata={"estimated_duration_minutes": 10, "region": "us-east-1"},
        )

        flat = task.to_task_dict()
        assert flat["id"] == "deploy"
        assert flat["title"] == "Deploy App"
        assert flat["description"] == "Deploy the application"
        assert flat["tools"] == ["kubectl", "helm"]
        assert flat["priority"] == "high"
        assert flat["estimated_duration_minutes"] == 10
        assert flat["region"] == "us-east-1"
