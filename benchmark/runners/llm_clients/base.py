from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    provider: str
    base_url: str
    api_key: str
    model: str


class BaseLLMClient(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    @abstractmethod
    def raw_client(self) -> Any:
        raise NotImplementedError
