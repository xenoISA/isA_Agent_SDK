#!/usr/bin/env python3
"""
Background HIL Detector - detect long-running tasks that need background execution

Difference from ToolHILDetector:
- ToolHILDetector: detects HIL responses from MCP tools (source: MCP server)
- BackgroundHILDetector: detects long-running tasks at ToolNode level (source: ToolNode)

Responsibilities:
1. Analyze upcoming tool_calls list
2. Predict execution time based on tool types and historical data
3. Determine if execution_choice HIL should be triggered
4. Generate execution_choice HIL data

Trigger source: ToolNode (not MCP)
"""

from typing import List, Tuple, Optional, Dict, Any
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger


class BackgroundHILDetector:
    """Detect long-running tasks that need background execution"""

    def __init__(self, tool_profiler=None):
        """
        Initialize detector

        Args:
            tool_profiler: Tool performance profiler (optional, for historical data)
        """
        self.profiler = tool_profiler

        # Default tool execution time estimates (seconds)
        self.default_durations = {
            "web_crawl": 12.0,
            "web_crawl_compare": 45.0,  # Multiple URLs, parallel crawl + comparison
            "web_search": 3.0,
            "arxiv_search": 5.0,
            "file_download": 8.0,
            "image_generation": 15.0,
            "code_execution": 10.0,
        }

        # Threshold configuration (seconds)
        self.thresholds = {
            "quick_suggestion": 30,      # >30s suggest choice
            "background_recommend": 60,  # >60s recommend background
            "force_background": 300      # >5min force background
        }

    def should_offer_execution_choice(
        self,
        tool_info_list: List[Tuple[str, dict, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if execution choice should be offered

        Args:
            tool_info_list: [(tool_name, tool_args, tool_call_id), ...]

        Returns:
            execution_choice data structure if HIL needed, None otherwise
        """
        # 1. Predict total execution time
        estimated_time = self._estimate_total_duration(tool_info_list)

        # 2. Check if exceeds threshold
        if estimated_time < self.thresholds["quick_suggestion"]:
            logger.debug(
                f"[BackgroundHIL] Task duration {estimated_time:.1f}s < "
                f"{self.thresholds['quick_suggestion']}s, no choice needed"
            )
            return None

        # 3. Analyze task composition
        task_composition = self._analyze_task_composition(tool_info_list)

        # 4. Generate execution_choice HIL data
        return self._generate_execution_choice_data(
            estimated_time=estimated_time,
            tool_info_list=tool_info_list,
            task_composition=task_composition
        )

    def _estimate_total_duration(self, tool_info_list: List[Tuple[str, dict, str]]) -> float:
        """
        Predict total execution time (seconds)

        Priority:
        1. Use tool_profiler historical data (if available)
        2. Use default time estimates
        3. Unknown tools default to 5 seconds

        Args:
            tool_info_list: [(tool_name, tool_args, tool_call_id), ...]

        Returns:
            Estimated total time (seconds)
        """
        total_duration = 0.0

        for tool_name, tool_args, _ in tool_info_list:
            # Try profiler first
            if self.profiler:
                try:
                    predicted = self.profiler.predict_duration(tool_name, tool_args)
                    if predicted > 0:
                        total_duration += predicted
                        continue
                except Exception as e:
                    logger.debug(f"[BackgroundHIL] Profiler prediction failed: {e}")

            # Use default estimate
            duration = self.default_durations.get(tool_name, 5.0)
            total_duration += duration

        logger.info(
            f"[BackgroundHIL] Estimated duration: {total_duration:.1f}s "
            f"for {len(tool_info_list)} tools"
        )
        return total_duration

    def _analyze_task_composition(
        self,
        tool_info_list: List[Tuple[str, dict, str]]
    ) -> Dict[str, Any]:
        """
        Analyze task composition

        Returns:
            {
                "total_tools": int,
                "tool_breakdown": {"web_crawl": 5, "web_search": 3},
                "primary_task_type": "web_crawling" | "web_searching" | "mixed",
                "sources": [...],
                "queries": [...],
            }
        """
        tool_breakdown = {}
        sources = []
        queries = []

        for tool_name, tool_args, _ in tool_info_list:
            # Count tool types
            tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1

            # Extract web_crawl URLs
            if tool_name == "web_crawl":
                url = tool_args.get("url", "unknown")
                sources.append(url)

            # Extract web_search queries
            if tool_name == "web_search":
                query = tool_args.get("query", "unknown")
                queries.append(query)

        # Determine primary task type
        primary_type = self._determine_primary_task_type(tool_breakdown)

        return {
            "total_tools": len(tool_info_list),
            "tool_breakdown": tool_breakdown,
            "primary_task_type": primary_type,
            "sources": sources,
            "queries": queries
        }

    def _determine_primary_task_type(self, tool_breakdown: Dict[str, int]) -> str:
        """
        Determine primary task type

        Args:
            tool_breakdown: {"web_crawl": 5, "web_search": 2}

        Returns:
            "web_crawling" | "web_searching" | "code_execution" | "mixed"
        """
        if not tool_breakdown:
            return "unknown"

        # Find most common tool type
        max_tool = max(tool_breakdown.items(), key=lambda x: x[1])
        max_tool_name = max_tool[0]
        max_count = max_tool[1]

        # If one tool type is >70%, it's the primary type
        total = sum(tool_breakdown.values())
        if max_count / total > 0.7:
            if max_tool_name == "web_crawl":
                return "web_crawling"
            elif max_tool_name == "web_search":
                return "web_searching"
            elif max_tool_name == "code_execution":
                return "code_execution"

        return "mixed"

    def _generate_execution_choice_data(
        self,
        estimated_time: float,
        tool_info_list: List[Tuple[str, dict, str]],
        task_composition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate execution_choice HIL data

        Returns format follows HIL standard structure for ScenarioHandler.handle_execution_choice_scenario()

        Returns:
            {
                "estimated_time_seconds": 45.6,
                "tool_count": 5,
                "task_type": "web_crawling",
                "task_composition": {...},
                "recommendation": "background" | "comprehensive" | "quick",
                "options": [...],
                "prompt": "How do you want to execute this task?",
                "context": {...}
            }
        """
        tool_count = task_composition["total_tools"]
        task_type = task_composition["primary_task_type"]

        # Generate recommendation
        recommendation = self._generate_recommendation(estimated_time)

        # Generate option descriptions
        options = self._generate_execution_options(
            estimated_time=estimated_time,
            tool_count=tool_count
        )

        # Generate prompt
        prompt = self._generate_execution_prompt(
            estimated_time=estimated_time,
            tool_count=tool_count,
            task_type=task_type,
            task_composition=task_composition
        )

        return {
            "estimated_time_seconds": estimated_time,
            "tool_count": tool_count,
            "task_type": task_type,
            "task_composition": task_composition,
            "recommendation": recommendation,
            "options": options,
            "prompt": prompt,
            "context": {
                "tool_breakdown": task_composition["tool_breakdown"],
                "sources_count": len(task_composition.get("sources", [])),
                "queries_count": len(task_composition.get("queries", []))
            }
        }

    def _generate_recommendation(self, estimated_time: float) -> str:
        """
        Generate recommendation based on estimated time

        Returns:
            "quick" | "comprehensive" | "background"
        """
        if estimated_time > self.thresholds["background_recommend"]:
            return "background"
        elif estimated_time > self.thresholds["quick_suggestion"]:
            return "comprehensive"
        else:
            return "quick"

    def _generate_execution_options(
        self,
        estimated_time: float,
        tool_count: int
    ) -> List[Dict[str, Any]]:
        """
        Generate execution options

        Returns:
            [
                {"value": "quick", "label": "Quick", "description": "..."},
                ...
            ]
        """
        quick_time = min(30, estimated_time * 0.3)
        quick_tools = min(3, tool_count)

        return [
            {
                "value": "quick",
                "label": "Quick",
                "description": f"Fast response (~{quick_time:.0f}s, {quick_tools} sources)",
                "estimated_time": quick_time
            },
            {
                "value": "comprehensive",
                "label": "Comprehensive",
                "description": f"Wait for all {tool_count} sources (~{estimated_time:.0f}s)",
                "estimated_time": estimated_time
            },
            {
                "value": "background",
                "label": "Background",
                "description": f"Run in background, get job_id immediately (~{estimated_time:.0f}s total)",
                "estimated_time": estimated_time
            }
        ]

    def _generate_execution_prompt(
        self,
        estimated_time: float,
        tool_count: int,
        task_type: str,
        task_composition: Dict[str, Any]
    ) -> str:
        """
        Generate user-friendly prompt

        Returns:
            Execution choice prompt string
        """
        task_desc = {
            "web_crawling": f"crawling {tool_count} web pages",
            "web_searching": f"performing {tool_count} web searches",
            "code_execution": f"executing {tool_count} code operations",
            "mixed": f"executing {tool_count} mixed operations"
        }.get(task_type, f"executing {tool_count} operations")

        prompt = f"Long-running task detected: {task_desc} (~{estimated_time:.0f}s total)\n\nChoose execution mode:"

        # Add specific source information
        sources = task_composition.get("sources", [])
        queries = task_composition.get("queries", [])

        if sources and len(sources) <= 5:
            prompt += "\n\nSources to crawl:\n"
            for i, url in enumerate(sources[:5], 1):
                prompt += f"  {i}. {url}\n"
        elif sources:
            prompt += f"\n\nCrawling {len(sources)} web pages\n"

        if queries and len(queries) <= 3:
            prompt += "\nSearch queries:\n"
            for i, query in enumerate(queries[:3], 1):
                prompt += f"  {i}. {query}\n"

        return prompt.strip()
