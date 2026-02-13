"""
Config Extractor Mixin - Methods for extracting configuration from RunnableConfig
"""
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableConfig


class ConfigExtractorMixin:
    """Mixin providing configuration extraction methods for BaseNode"""

    def get_runtime_context(self, config: RunnableConfig) -> Dict[str, Any]:
        """
        Extract runtime context from config

        Args:
            config: LangGraph RunnableConfig with configurable context

        Returns:
            Runtime context dictionary
        """
        configurable = config.get("configurable", {})
        # Support both nested runtime_context and flat configurable structure
        # If runtime_context key exists, use it; otherwise return configurable itself
        return configurable.get("runtime_context", configurable)

    def get_user_id(self, config: RunnableConfig) -> str:
        """Get user ID from config context"""
        return config.get("configurable", {}).get("user_id", "")

    def get_thread_id(self, config: RunnableConfig) -> str:
        """Get thread ID from config context"""
        return config.get("configurable", {}).get("thread_id", "")

    def get_mcp_service(self, config: RunnableConfig) -> Optional[Any]:
        """Get MCP client from config context"""
        return config.get("configurable", {}).get("mcp_service")

    def get_session_manager(self, config: RunnableConfig) -> Any:
        """Get session manager from config context"""
        return config.get("configurable", {}).get("session_manager")

    def get_default_prompts(self, config: RunnableConfig) -> Dict[str, str]:
        """Get default prompts from config context"""
        return config.get("configurable", {}).get("default_prompts", {})

    def get_default_tools(self, config: RunnableConfig) -> List[Dict[str, Any]]:
        """Get default tools from config context"""
        return config.get("configurable", {}).get("default_tools", [])

    def get_default_resources(self, config: RunnableConfig) -> List[Dict[str, Any]]:
        """Get default resources from config context"""
        return config.get("configurable", {}).get("default_resources", [])

    def get_default_prompt(self, config: RunnableConfig, prompt_key: str, fallback: str = "") -> str:
        """
        Get default prompt from config context

        Args:
            config: LangGraph RunnableConfig
            prompt_key: Key for the prompt (e.g., 'entry_node_prompt', 'reason_node_prompt')
            fallback: Fallback text if prompt not found

        Returns:
            Default prompt text or fallback
        """
        default_prompts = self.get_default_prompts(config)
        return default_prompts.get(prompt_key, fallback)
