"""Any OpenAI-compatible ``/chat/completions`` server: vLLM, Ollama, OpenRouter, ..."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx

from vlm_demo.backends.base import BackendError, InferenceResult, VLMBackend, to_data_url
from vlm_demo.video import Frame, Window

log = logging.getLogger(__name__)


def normalise_base_url(base_url: str) -> str:
    """``http://host:8000`` → ``http://host:8000/v1``; anything with a path is left alone."""
    url = base_url.rstrip("/")
    if urlsplit(url).path in ("", "/"):
        url += "/v1"
    return url


class OpenAICompatBackend(VLMBackend):
    name = "openai-compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_tokens: int = 128,
        temperature: float = 0.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = normalise_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client
        self._owns_client = client is None

    async def prepare(self) -> None:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=headers, timeout=self.timeout
            )

    def build_payload(self, prompt: str, frames: Sequence[Frame]) -> dict:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": to_data_url(frame)}} for frame in frames
        ]
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

    async def infer(
        self, prompt: str, frames: Sequence[Frame], window: Window
    ) -> InferenceResult:
        if self._client is None:
            await self.prepare()
        assert self._client is not None

        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/chat/completions", json=self.build_payload(prompt, frames)
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"request to {self.base_url} failed: {exc}") from exc

        if response.status_code >= 400:
            raise BackendError(
                f"{self.base_url} returned {response.status_code}: {response.text[:300]}"
            )
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response shape: {response.text[:300]}") from exc

        if isinstance(text, list):  # some servers return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return InferenceResult(
            text=(text or "").strip(),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"usage": body.get("usage", {}), "window": window.index},
        )

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None
