from __future__ import annotations

from typing import Any

from benchmark.runners.llm_clients.base import BaseLLMClient, ProviderConfig

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, config: ProviderConfig):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Please install it first.")

        super().__init__(config)
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def raw_client(self) -> Any:
        return self._client

    def responses_create(self, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        if "input" in payload:
            payload["messages"] = payload.pop("input")
        payload.setdefault("model", self.model)
        return self._client.chat.completions.create(**payload)
