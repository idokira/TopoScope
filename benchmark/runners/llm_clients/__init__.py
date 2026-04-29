from __future__ import annotations

from typing import Dict, Type

from benchmark.runners.llm_clients.base import BaseLLMClient, ProviderConfig
from benchmark.runners.llm_clients.config_loader import list_providers, load_provider_config
from benchmark.runners.llm_clients.deepseek_client import DeepSeekClient
from benchmark.runners.llm_clients.gemini_client import GeminiClient
from benchmark.runners.llm_clients.llama_client import LlamaClient
from benchmark.runners.llm_clients.glm_client import GlmClient
from benchmark.runners.llm_clients.openai_client import OpenAIClient
from benchmark.runners.llm_clients.claude_client import ClaudeClient
from benchmark.runners.llm_clients.minimax_client import MiniMaxClient
from benchmark.runners.llm_clients.openai_compatible import OpenAICompatibleClient
from benchmark.runners.llm_clients.qwen_client import QwenClient


_PROVIDER_CLASS_MAP: Dict[str, Type[BaseLLMClient]] = {
    "qwen": QwenClient,
    "openai": OpenAIClient,
    "deepseek": DeepSeekClient,
    "gemini": GeminiClient,
    "llama": LlamaClient,
    "glm": GlmClient,
    "claude": ClaudeClient,
    "minimax": MiniMaxClient,
}


def build_client(provider: str | None = None, config_path: str = "benchmark/configs/llm_clients.json") -> BaseLLMClient:
    cfg = load_provider_config(provider=provider, config_path=config_path)
    cls = _PROVIDER_CLASS_MAP.get(cfg.provider, OpenAICompatibleClient)
    return cls(cfg)


__all__ = [
    "BaseLLMClient",
    "ProviderConfig",
    "OpenAICompatibleClient",
    "build_client",
    "load_provider_config",
    "list_providers",
]
