#!/usr/bin/env python3
"""Model configuration for isA_Model integration

Centralizes all LLM and embedding model settings:
- Default LLM model and provider
- Reasoning model for chain-of-thought
- Response model for summaries
- Embedding model and vector dimensions
"""
import os
from dataclasses import dataclass

def _int(val: str, default: int) -> int:
    try:
        return int(val) if val else default
    except ValueError:
        return default

def _float(val: str, default: float) -> float:
    try:
        return float(val) if val else default
    except ValueError:
        return default


@dataclass
class ModelConfig:
    """LLM and Embedding model configuration from isA_Model"""

    # ===========================================
    # ISA Model Service Connection
    # ===========================================
    service_url: str = "http://localhost:8082"
    api_key: str = ""

    # ===========================================
    # LLM Configuration
    # ===========================================
    # Default model for general tasks
    default_llm: str = "gpt-4.1-nano"
    default_llm_provider: str = "openai"

    # Reasoning model for complex tasks (chain-of-thought)
    reason_llm: str = "deepseek-r1"
    reason_llm_provider: str = "yyds"

    # Response model for summaries/formatting (faster, cheaper)
    response_llm: str = "gpt-5-nano"
    response_llm_provider: str = "openai"

    # Model parameters
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0

    # OpenAI API (legacy support)
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"

    # ===========================================
    # Embedding Configuration
    # ===========================================
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"

    # Vector dimensions
    # text-embedding-3-small: 1536
    # text-embedding-3-large: 3072
    vector_size: int = 1536

    # Distance metric: Cosine, Euclid, Dot, Manhattan
    distance_metric: str = "Cosine"

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Load model configuration from environment variables"""
        return cls(
            # ISA Model Service
            service_url=os.getenv("ISA_MODEL_URL") or os.getenv("ISA_API_URL", "http://localhost:8082"),
            api_key=os.getenv("ISA_MODEL_API_KEY") or os.getenv("ISA_API_KEY", ""),

            # LLM Configuration
            default_llm=os.getenv("AI_MODEL") or os.getenv("DEFAULT_LLM", "gpt-4.1-nano"),
            default_llm_provider=os.getenv("AI_PROVIDER") or os.getenv("DEFAULT_LLM_PROVIDER", "openai"),
            reason_llm=os.getenv("REASON_MODEL") or os.getenv("REASON_LLM", "deepseek-reasoner"),
            reason_llm_provider=os.getenv("REASON_MODEL_PROVIDER") or os.getenv("REASON_LLM_PROVIDER", "yyds"),
            response_llm=os.getenv("RESPONSE_MODEL") or os.getenv("RESPONSE_LLM", "gpt-5-nano"),
            response_llm_provider=os.getenv("RESPONSE_MODEL_PROVIDER") or os.getenv("RESPONSE_LLM_PROVIDER", "openai"),

            # Model parameters
            temperature=_float(os.getenv("AI_TEMPERATURE") or os.getenv("LLM_TEMPERATURE", "0.0"), 0.0),
            max_tokens=_int(os.getenv("LLM_MAX_TOKENS", "4096"), 4096),
            top_p=_float(os.getenv("LLM_TOP_P", "1.0"), 1.0),

            # OpenAI API
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),

            # Embedding Configuration
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            vector_size=_int(os.getenv("VECTOR_SIZE", "1536"), 1536),
            distance_metric=os.getenv("DISTANCE_METRIC", "Cosine"),
        )
