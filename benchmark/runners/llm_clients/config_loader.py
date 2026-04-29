from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from benchmark.runners.llm_clients.base import ProviderConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "benchmark/configs/llm_clients.json"


def _resolve_config_path(config_path: Path) -> Path:
    p = config_path.expanduser()
    if p.is_absolute():
        return p

    if p.exists():
        return p.resolve()

    return PROJECT_ROOT / p


def _read_config(config_path: Path) -> Dict[str, object]:
    resolved = _resolve_config_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"LLM config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid LLM config: root must be an object.")

    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("Invalid LLM config: 'providers' must be an object.")

    return data


def list_providers(config_path: str = str(DEFAULT_CONFIG_PATH)) -> Dict[str, Dict[str, object]]:
    data = _read_config(Path(config_path))
    providers = data.get("providers", {})
    return {str(k): dict(v) for k, v in providers.items() if isinstance(v, dict)}


def load_provider_config(
    provider: Optional[str] = None,
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> ProviderConfig:
    data = _read_config(Path(config_path))
    providers = data.get("providers", {})

    selected = provider or str(data.get("default_provider", "")).strip()
    if not selected:
        raise ValueError("No provider specified and default_provider is missing in config.")

    if selected not in providers:
        available = ", ".join(sorted(str(k) for k in providers.keys()))
        raise ValueError(f"Unknown provider '{selected}'. Available providers: {available}")

    p = providers[selected]
    if not isinstance(p, dict):
        raise ValueError(f"Provider config for '{selected}' must be an object.")

    base_url = str(p.get("base_url", "")).strip()
    model = str(p.get("model", "")).strip()

    api_key = str(p.get("api_key", "")).strip()
    if not api_key:
        api_key_env = str(p.get("api_key_env", "")).strip()
        if api_key_env:
            api_key = os.getenv(api_key_env, "").strip()

    if not base_url:
        raise ValueError(f"Provider '{selected}' is missing 'base_url'.")
    if not model:
        raise ValueError(f"Provider '{selected}' is missing 'model'.")
    if not api_key:
        raise ValueError(
            f"Provider '{selected}' has no API key. Set 'api_key' or export env var '{p.get('api_key_env', '')}'."
        )

    return ProviderConfig(
        provider=selected,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
