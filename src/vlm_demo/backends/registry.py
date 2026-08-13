"""Builds the backend a :class:`RunConfig` asks for."""

from __future__ import annotations

from vlm_demo.backends.base import VLMBackend
from vlm_demo.config import BackendKind, RunConfig


def create_backend(config: RunConfig) -> VLMBackend:
    """Instantiate (but do not yet prepare) the configured backend."""
    match config.backend:
        case BackendKind.MOCK:
            from vlm_demo.backends.mock import MockBackend

            return MockBackend(
                config.model,
                detect_at=config.mock_detect_at,
                latency=config.mock_latency,
            )
        case BackendKind.OPENAI_COMPAT:
            from vlm_demo.backends.openai_compat import OpenAICompatBackend

            assert config.base_url  # guaranteed by RunConfig validation
            return OpenAICompatBackend(
                config.model,
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.infer_timeout,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        case BackendKind.TRANSFORMERS:
            from vlm_demo.backends.transformers_local import TransformersBackend

            return TransformersBackend(
                config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
    raise AssertionError(f"unhandled backend {config.backend!r}")  # pragma: no cover
