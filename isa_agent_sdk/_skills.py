#!/usr/bin/env python3
"""
isA Agent SDK - Skills System
=============================

Skills are specialized prompts that can be loaded and injected into agent context.
They leverage the existing MCP prompt/resource infrastructure.

Skills can be loaded from:
1. MCP prompts (via mcp_get_prompt)
2. MCP resources (via read_resource)
3. Local files
4. Inline definitions

Example:
    from isa_agent_sdk import ISAAgentOptions
    from isa_agent_sdk.skills import SkillManager, load_skill

    # Load a skill from MCP
    skill = await load_skill("code-review")

    # Use skills in query options
    options = ISAAgentOptions(
        skills=["code-review", "debug", "refactor"]
    )

    # Or manually manage skills
    manager = SkillManager()
    await manager.load("code-review")
    prompt_injection = manager.get_injection()
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .utils.logger import agent_logger

logger = agent_logger


@dataclass
class Skill:
    """
    A skill definition.

    Skills are specialized prompts that enhance agent capabilities
    for specific tasks (code review, debugging, etc.)
    """

    name: str
    """Unique skill identifier"""

    description: str
    """Short description of what this skill does"""

    prompt: str
    """The prompt content to inject"""

    triggers: List[str] = field(default_factory=list)
    """Keywords/patterns that trigger this skill"""

    category: str = "general"
    """Skill category (coding, research, writing, etc.)"""

    source: str = "inline"
    """Where this skill was loaded from (mcp, file, inline)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def matches_trigger(self, text: str) -> bool:
        """Check if text matches any trigger patterns"""
        if not self.triggers:
            return False
        text_lower = text.lower()
        return any(trigger.lower() in text_lower for trigger in self.triggers)


# Built-in skill definitions
BUILTIN_SKILLS: Dict[str, Skill] = {
    "code-review": Skill(
        name="code-review",
        description="Expert code reviewer focusing on quality, security, and best practices",
        prompt="""You are an expert code reviewer. When reviewing code:
- Check for bugs, edge cases, and potential errors
- Evaluate code quality, readability, and maintainability
- Look for security vulnerabilities (injection, XSS, auth issues)
- Suggest improvements following best practices
- Consider performance implications
- Provide specific, actionable feedback with examples""",
        triggers=["review", "code review", "check this code", "review my code"],
        category="coding"
    ),

    "debug": Skill(
        name="debug",
        description="Systematic debugger for finding and fixing issues",
        prompt="""You are an expert debugger. When debugging:
- Analyze error messages and stack traces carefully
- Form hypotheses about root causes
- Suggest diagnostic steps to narrow down the issue
- Consider common causes: null/undefined, type errors, async issues, edge cases
- Provide step-by-step debugging strategies
- Suggest fixes with explanations of why they work""",
        triggers=["debug", "error", "bug", "fix this", "not working", "broken"],
        category="coding"
    ),

    "refactor": Skill(
        name="refactor",
        description="Code refactoring expert for improving structure and maintainability",
        prompt="""You are an expert at code refactoring. When refactoring:
- Identify code smells and anti-patterns
- Apply SOLID principles appropriately
- Extract reusable functions/classes
- Improve naming for clarity
- Reduce complexity and nesting
- Maintain backwards compatibility where needed
- Explain the reasoning behind each change""",
        triggers=["refactor", "clean up", "improve", "restructure"],
        category="coding"
    ),

    "test-writer": Skill(
        name="test-writer",
        description="Test writing expert for comprehensive test coverage",
        prompt="""You are an expert test writer. When writing tests:
- Cover happy paths and edge cases
- Test error handling and boundary conditions
- Write clear test descriptions
- Use appropriate mocking/stubbing
- Follow testing best practices (AAA pattern, isolation)
- Aim for meaningful coverage, not just high numbers
- Consider unit, integration, and e2e tests as appropriate""",
        triggers=["test", "write tests", "add tests", "test coverage"],
        category="coding"
    ),

    "documentation": Skill(
        name="documentation",
        description="Technical documentation writer",
        prompt="""You are an expert technical writer. When writing documentation:
- Write clear, concise explanations
- Include practical examples
- Structure content logically
- Consider the audience's knowledge level
- Document both usage and implementation details
- Include troubleshooting sections where relevant
- Keep documentation maintainable and up-to-date""",
        triggers=["document", "docs", "readme", "explain this"],
        category="writing"
    ),

    "security-audit": Skill(
        name="security-audit",
        description="Security auditor for finding vulnerabilities",
        prompt="""You are a security expert. When auditing code:
- Check for OWASP Top 10 vulnerabilities
- Look for injection points (SQL, command, XSS)
- Verify authentication and authorization
- Check for sensitive data exposure
- Review cryptographic implementations
- Assess input validation and sanitization
- Consider attack vectors and threat models
- Provide severity ratings and remediation steps""",
        triggers=["security", "audit", "vulnerability", "secure"],
        category="security"
    ),
}


class SkillManager:
    """
    Manages skill loading and injection.

    Integrates with MCP for loading skills from prompts/resources.
    """

    def __init__(self):
        self._loaded_skills: Dict[str, Skill] = {}
        self._active_skills: List[str] = []

    @property
    def loaded_skills(self) -> Dict[str, Skill]:
        """Get all loaded skills"""
        return self._loaded_skills.copy()

    @property
    def active_skills(self) -> List[str]:
        """Get list of active skill names"""
        return self._active_skills.copy()

    def load_builtin(self, name: str) -> Optional[Skill]:
        """
        Load a built-in skill.

        Args:
            name: Built-in skill name

        Returns:
            Skill if found, None otherwise
        """
        if name in BUILTIN_SKILLS:
            skill = BUILTIN_SKILLS[name]
            self._loaded_skills[name] = skill
            return skill
        return None

    def load_inline(
        self,
        name: str,
        prompt: str,
        description: str = "",
        triggers: Optional[List[str]] = None,
        category: str = "custom"
    ) -> Skill:
        """
        Load a skill from inline definition.

        Args:
            name: Skill name
            prompt: Skill prompt content
            description: Skill description
            triggers: Trigger patterns
            category: Skill category

        Returns:
            Created Skill
        """
        skill = Skill(
            name=name,
            description=description or f"Custom skill: {name}",
            prompt=prompt,
            triggers=triggers or [],
            category=category,
            source="inline"
        )
        self._loaded_skills[name] = skill
        return skill

    async def load_from_mcp(
        self,
        name: str,
        mcp_client,
        prompt_name: Optional[str] = None
    ) -> Optional[Skill]:
        """
        Load a skill from MCP prompt.

        Args:
            name: Skill name
            mcp_client: MCP client instance
            prompt_name: MCP prompt name (defaults to skill_{name})

        Returns:
            Skill if loaded, None otherwise
        """
        try:
            mcp_prompt_name = prompt_name or f"skill_{name}"
            prompt_content = await mcp_client.get_prompt(mcp_prompt_name, {})

            if prompt_content:
                skill = Skill(
                    name=name,
                    description=f"MCP skill: {name}",
                    prompt=prompt_content,
                    source="mcp",
                    metadata={"mcp_prompt": mcp_prompt_name}
                )
                self._loaded_skills[name] = skill
                logger.info(f"Loaded skill '{name}' from MCP prompt '{mcp_prompt_name}'")
                return skill

            logger.warning(f"MCP prompt '{mcp_prompt_name}' not found for skill '{name}'")
            return None

        except Exception as e:
            logger.error(f"Failed to load skill '{name}' from MCP: {e}")
            return None

    async def load_from_resource(
        self,
        name: str,
        mcp_client,
        resource_uri: str
    ) -> Optional[Skill]:
        """
        Load a skill from MCP resource.

        Args:
            name: Skill name
            mcp_client: MCP client instance
            resource_uri: Resource URI

        Returns:
            Skill if loaded, None otherwise
        """
        try:
            resource = await mcp_client.read_resource(resource_uri)

            if resource and "contents" in resource:
                contents = resource["contents"]
                if contents and len(contents) > 0:
                    content = contents[0]
                    text = content.get("text", "")

                    if text:
                        skill = Skill(
                            name=name,
                            description=f"Resource skill: {name}",
                            prompt=text,
                            source="mcp_resource",
                            metadata={"resource_uri": resource_uri}
                        )
                        self._loaded_skills[name] = skill
                        logger.info(f"Loaded skill '{name}' from resource '{resource_uri}'")
                        return skill

            logger.warning(f"Resource '{resource_uri}' empty or not found for skill '{name}'")
            return None

        except Exception as e:
            logger.error(f"Failed to load skill '{name}' from resource: {e}")
            return None

    def load_from_file(self, name: str, file_path: Union[str, Path]) -> Optional[Skill]:
        """
        Load a skill from a local file.

        Args:
            name: Skill name
            file_path: Path to skill file (markdown or text)

        Returns:
            Skill if loaded, None otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Skill file not found: {file_path}")
                return None

            prompt_content = path.read_text(encoding="utf-8")

            skill = Skill(
                name=name,
                description=f"File skill: {name}",
                prompt=prompt_content,
                source="file",
                metadata={"file_path": str(path)}
            )
            self._loaded_skills[name] = skill
            logger.info(f"Loaded skill '{name}' from file '{file_path}'")
            return skill

        except Exception as e:
            logger.error(f"Failed to load skill '{name}' from file: {e}")
            return None

    async def load(
        self,
        name: str,
        mcp_client=None
    ) -> Optional[Skill]:
        """
        Load a skill by name, trying multiple sources.

        Order: builtin -> MCP prompt -> MCP resource

        Args:
            name: Skill name
            mcp_client: Optional MCP client for remote loading

        Returns:
            Skill if loaded, None otherwise
        """
        # Already loaded?
        if name in self._loaded_skills:
            return self._loaded_skills[name]

        # Try built-in
        skill = self.load_builtin(name)
        if skill:
            return skill

        # Try MCP if client available
        if mcp_client:
            # Try MCP prompt
            skill = await self.load_from_mcp(name, mcp_client)
            if skill:
                return skill

            # Try MCP resource
            skill = await self.load_from_resource(
                name, mcp_client, f"skill://{name}"
            )
            if skill:
                return skill

        logger.warning(f"Could not load skill '{name}' from any source")
        return None

    def activate(self, *skill_names: str) -> List[str]:
        """
        Activate skills for prompt injection.

        Args:
            *skill_names: Names of skills to activate

        Returns:
            List of successfully activated skill names
        """
        activated = []
        for name in skill_names:
            if name in self._loaded_skills:
                if name not in self._active_skills:
                    self._active_skills.append(name)
                activated.append(name)
            else:
                logger.warning(f"Cannot activate unloaded skill: {name}")
        return activated

    def deactivate(self, *skill_names: str) -> None:
        """Deactivate skills"""
        for name in skill_names:
            if name in self._active_skills:
                self._active_skills.remove(name)

    def deactivate_all(self) -> None:
        """Deactivate all skills"""
        self._active_skills.clear()

    def get_injection(self, separator: str = "\n\n---\n\n") -> str:
        """
        Get the combined prompt injection for all active skills.

        Args:
            separator: Separator between skill prompts

        Returns:
            Combined prompt string
        """
        if not self._active_skills:
            return ""

        parts = []
        for name in self._active_skills:
            if name in self._loaded_skills:
                skill = self._loaded_skills[name]
                parts.append(f"[SKILL: {skill.name}]\n{skill.prompt}")

        return separator.join(parts)

    def detect_skills(self, text: str) -> List[str]:
        """
        Detect which skills should be activated based on text triggers.

        Args:
            text: Input text to check

        Returns:
            List of skill names that match triggers
        """
        matching = []
        for name, skill in self._loaded_skills.items():
            if skill.matches_trigger(text):
                matching.append(name)
        return matching

    def auto_activate(self, text: str) -> List[str]:
        """
        Automatically activate skills based on text triggers.

        Args:
            text: Input text to check

        Returns:
            List of newly activated skill names
        """
        detected = self.detect_skills(text)
        return self.activate(*detected)


# Global skill manager instance
_global_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """Get the global skill manager instance"""
    global _global_skill_manager
    if _global_skill_manager is None:
        _global_skill_manager = SkillManager()
    return _global_skill_manager


def set_skill_manager(manager: SkillManager) -> None:
    """Set the global skill manager instance"""
    global _global_skill_manager
    _global_skill_manager = manager


# Convenience functions

async def load_skill(
    name: str,
    mcp_client=None
) -> Optional[Skill]:
    """
    Load a skill using the global manager.

    Args:
        name: Skill name
        mcp_client: Optional MCP client

    Returns:
        Skill if loaded
    """
    manager = get_skill_manager()
    return await manager.load(name, mcp_client)


def load_builtin_skill(name: str) -> Optional[Skill]:
    """
    Load a built-in skill.

    Args:
        name: Built-in skill name

    Returns:
        Skill if found
    """
    manager = get_skill_manager()
    return manager.load_builtin(name)


def activate_skills(*names: str) -> List[str]:
    """Activate skills by name"""
    manager = get_skill_manager()
    return manager.activate(*names)


def get_skill_injection() -> str:
    """Get prompt injection for active skills"""
    manager = get_skill_manager()
    return manager.get_injection()


def list_builtin_skills() -> List[str]:
    """List all available built-in skills"""
    return list(BUILTIN_SKILLS.keys())


__all__ = [
    # Core classes
    "Skill",
    "SkillManager",
    "BUILTIN_SKILLS",

    # Manager functions
    "get_skill_manager",
    "set_skill_manager",

    # Convenience functions
    "load_skill",
    "load_builtin_skill",
    "activate_skills",
    "get_skill_injection",
    "list_builtin_skills",
]
