#!/usr/bin/env python3
"""
Artifact Detector - 检测 AI 响应中的可渲染内容

支持检测:
- 代码块 (Markdown code blocks)
- HTML/SVG 标签
- 可视化图表代码
- React/JSX 组件

用于 Claude-style Artifact 功能
"""

import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ArtifactType:
    """Artifact 类型常量"""
    CODE = "code"           # 纯代码
    HTML = "html"           # HTML 内容
    SVG = "svg"             # SVG 图形
    REACT = "react"         # React 组件
    CHART = "chart"         # 图表可视化
    MARKDOWN = "markdown"   # Markdown 文档


class ArtifactDetector:
    """
    Artifact 检测器

    检测 AI 生成的内容中是否包含可渲染的 artifact，
    并提取相关信息用于前端展示
    """

    # 代码块正则表达式
    CODE_BLOCK_PATTERN = r'```(\w+)?\n([\s\S]+?)\n```'

    # 需要特殊处理的语言
    RENDERABLE_LANGUAGES = {
        'html': ArtifactType.HTML,
        'svg': ArtifactType.SVG,
        'jsx': ArtifactType.REACT,
        'tsx': ArtifactType.REACT,
        'react': ArtifactType.REACT,
    }

    # 代码类语言
    CODE_LANGUAGES = {
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
        'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'sql'
    }

    # 图表关键词
    CHART_KEYWORDS = [
        'chart', 'plot', 'graph', 'visualization',
        'Chart.js', 'echarts', 'd3', 'plotly', 'matplotlib'
    ]

    @classmethod
    def detect(cls, content: str) -> Optional[Dict[str, Any]]:
        """
        检测内容中的 artifact

        Args:
            content: AI 生成的文本内容

        Returns:
            artifact 信息字典，如果没有检测到则返回 None

        Example:
            >>> detector = ArtifactDetector()
            >>> content = "Here's a chart:\\n```html\\n<div>Chart</div>\\n```"
            >>> artifact = detector.detect(content)
            >>> artifact['type']
            'html'
        """
        if not content or not isinstance(content, str):
            return None

        # 检测代码块
        matches = re.finditer(cls.CODE_BLOCK_PATTERN, content)

        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()

            if not code:
                continue

            # 判断 artifact 类型
            artifact_type = cls._determine_type(language, code)

            if artifact_type:
                artifact_data = cls._create_artifact(
                    artifact_type=artifact_type,
                    language=language,
                    code=code,
                    original_content=content
                )

                logger.info(
                    f"[ARTIFACT_DETECTOR] detected | "
                    f"type={artifact_type} | "
                    f"language={language} | "
                    f"code_length={len(code)}"
                )

                return artifact_data

        return None

    @classmethod
    def _determine_type(cls, language: str, code: str) -> Optional[str]:
        """
        根据语言和代码内容判断 artifact 类型

        Args:
            language: 代码块标记的语言
            code: 代码内容

        Returns:
            artifact 类型，如果不是可渲染内容则返回 None
        """
        language_lower = language.lower()

        # 1. 直接语言匹配
        if language_lower in cls.RENDERABLE_LANGUAGES:
            return cls.RENDERABLE_LANGUAGES[language_lower]

        # 2. SVG 内容检测
        if code.strip().startswith('<svg') or '<svg' in code[:100]:
            return ArtifactType.SVG

        # 3. HTML 内容检测
        if (code.strip().startswith('<!DOCTYPE') or
            code.strip().startswith('<html') or
            '<html' in code[:100]):
            return ArtifactType.HTML

        # 4. 图表关键词检测
        if any(keyword.lower() in code.lower() for keyword in cls.CHART_KEYWORDS):
            # 如果是 HTML/JS 代码且包含图表库，归类为 HTML
            if language_lower in ['html', 'javascript', 'js']:
                return ArtifactType.HTML
            # 否则归类为图表
            return ArtifactType.CHART

        # 5. 代码语言检测 (可选择是否展示为 artifact)
        # 注释掉这部分可以只展示真正可渲染的内容
        # if language_lower in cls.CODE_LANGUAGES:
        #     return ArtifactType.CODE

        return None

    @classmethod
    def _create_artifact(
        cls,
        artifact_type: str,
        language: str,
        code: str,
        original_content: str
    ) -> Dict[str, Any]:
        """
        创建 artifact 数据结构

        Args:
            artifact_type: artifact 类型
            language: 编程语言
            code: 代码内容
            original_content: 原始完整内容

        Returns:
            artifact 数据字典
        """
        # 生成唯一 ID
        timestamp = int(datetime.now().timestamp() * 1000)
        artifact_id = f"artifact_{timestamp}"

        # 生成标题
        title = cls._generate_title(artifact_type, language)

        return {
            "id": artifact_id,
            "type": artifact_type,
            "title": title,
            "content": code,
            "language": language,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "original_length": len(original_content),
                "code_length": len(code),
                "has_multiple_blocks": original_content.count('```') > 2
            }
        }

    @classmethod
    def _generate_title(cls, artifact_type: str, language: str) -> str:
        """
        根据类型和语言生成友好的标题

        Args:
            artifact_type: artifact 类型
            language: 编程语言

        Returns:
            生成的标题
        """
        title_map = {
            ArtifactType.HTML: "网页预览",
            ArtifactType.SVG: "矢量图形",
            ArtifactType.REACT: "React 组件",
            ArtifactType.CHART: "数据图表",
            ArtifactType.CODE: f"{language.upper()} 代码",
            ArtifactType.MARKDOWN: "Markdown 文档"
        }

        return title_map.get(artifact_type, f"生成的 {language}")

    @classmethod
    def detect_multiple(cls, content: str) -> list[Dict[str, Any]]:
        """
        检测内容中的所有 artifacts (支持多个代码块)

        Args:
            content: AI 生成的文本内容

        Returns:
            artifact 列表
        """
        if not content or not isinstance(content, str):
            return []

        artifacts = []
        matches = re.finditer(cls.CODE_BLOCK_PATTERN, content)

        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()

            if not code:
                continue

            artifact_type = cls._determine_type(language, code)

            if artifact_type:
                artifact_data = cls._create_artifact(
                    artifact_type=artifact_type,
                    language=language,
                    code=code,
                    original_content=content
                )
                artifacts.append(artifact_data)

        if artifacts:
            logger.info(
                f"[ARTIFACT_DETECTOR] detected_multiple | "
                f"count={len(artifacts)} | "
                f"types={[a['type'] for a in artifacts]}"
            )

        return artifacts


# 便捷函数
def detect_artifact(content: str) -> Optional[Dict[str, Any]]:
    """
    便捷函数：检测内容中的第一个 artifact

    Args:
        content: AI 生成的文本内容

    Returns:
        artifact 信息字典，如果没有检测到则返回 None
    """
    return ArtifactDetector.detect(content)


def detect_all_artifacts(content: str) -> list[Dict[str, Any]]:
    """
    便捷函数：检测内容中的所有 artifacts

    Args:
        content: AI 生成的文本内容

    Returns:
        artifact 列表
    """
    return ArtifactDetector.detect_multiple(content)
