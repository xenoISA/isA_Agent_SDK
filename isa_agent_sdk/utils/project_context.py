#!/usr/bin/env python3
"""
Project Context File Support (Claude SDK CLAUDE.md compatible)

Supports loading project memory from static markdown files:
- ISA.md (preferred)
- CLAUDE.md (Claude SDK compatible)
- .isa/CONTEXT.md (alternative location)

The content is injected into the system prompt as persistent project context.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Default file names to search for, in priority order
DEFAULT_CONTEXT_FILES = [
    "ISA.md",
    "CLAUDE.md",
    ".isa/CONTEXT.md",
    ".claude/CONTEXT.md",
]


def discover_project_context_file(
    start_dir: Optional[str] = None,
    max_depth: int = 3
) -> Optional[str]:
    """
    Discover project context file by walking up from start_dir.

    Searches for ISA.md, CLAUDE.md, or .isa/CONTEXT.md in:
    1. Current directory
    2. Parent directories (up to max_depth levels)

    Args:
        start_dir: Starting directory (defaults to cwd)
        max_depth: Maximum parent directories to check

    Returns:
        Path to context file if found, None otherwise
    """
    if start_dir is None:
        start_dir = os.getcwd()

    current = Path(start_dir).resolve()

    for _ in range(max_depth + 1):
        for filename in DEFAULT_CONTEXT_FILES:
            context_path = current / filename
            if context_path.exists() and context_path.is_file():
                logger.info(f"Discovered project context file: {context_path}")
                return str(context_path)

        # Move up to parent
        parent = current.parent
        if parent == current:
            # Reached root
            break
        current = parent

    logger.debug(f"No project context file found in {start_dir} or parents")
    return None


def load_project_context(
    source: Optional[str] = None,
    start_dir: Optional[str] = None
) -> str:
    """
    Load project context from file or string.

    Args:
        source: One of:
            - "auto": Auto-discover from project root
            - File path: Load from specific file
            - Content string: Use directly (if multi-line or no file extension)
            - None: Return empty string
        start_dir: Starting directory for auto-discovery

    Returns:
        Project context content as string (may be empty)

    Example:
        # Auto-discover
        context = load_project_context("auto")

        # Specific file
        context = load_project_context("./ISA.md")

        # Direct content
        context = load_project_context('''
            This project uses FastAPI.
            Always use type hints.
        ''')
    """
    if not source:
        return ""

    # Auto-discovery
    if source.lower() == "auto":
        discovered = discover_project_context_file(start_dir)
        if discovered:
            return _load_file(discovered)
        return ""

    # Check if it's a file path
    if _looks_like_path(source):
        resolved = _resolve_path(source, start_dir)
        if resolved and os.path.exists(resolved):
            return _load_file(resolved)
        logger.warning(f"Project context file not found: {source}")
        return ""

    # Treat as direct content
    return source.strip()


def _looks_like_path(s: str) -> bool:
    """Check if string looks like a file path"""
    # Has file extension
    if "." in os.path.basename(s) and not "\n" in s:
        return True
    # Starts with path indicators
    if s.startswith(("./", "../", "/", "~/")):
        return True
    # Is a known filename
    if s in DEFAULT_CONTEXT_FILES or os.path.basename(s) in DEFAULT_CONTEXT_FILES:
        return True
    return False


def _resolve_path(path: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Resolve a path relative to base_dir or cwd"""
    if os.path.isabs(path):
        return path

    base = base_dir or os.getcwd()
    resolved = os.path.join(base, path)
    return os.path.normpath(resolved)


def _load_file(path: str) -> str:
    """Load content from file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Loaded project context from {path} ({len(content)} chars)")
        return content.strip()
    except Exception as e:
        logger.error(f"Failed to load project context file {path}: {e}")
        return ""


def format_project_context_for_prompt(content: str) -> str:
    """
    Format project context for injection into system prompt.

    Args:
        content: Raw project context content

    Returns:
        Formatted content ready for prompt injection
    """
    if not content:
        return ""

    return f"""
## PROJECT CONTEXT

The following is persistent project context that should inform all responses:

{content}

---
"""


__all__ = [
    "discover_project_context_file",
    "load_project_context",
    "format_project_context_for_prompt",
    "DEFAULT_CONTEXT_FILES",
]
