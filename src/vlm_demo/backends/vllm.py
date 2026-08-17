"""vLLM: the openai-compat backend, plus wait-for-ready and a warmup pass.

vLLM takes a while to load weights, and the first multimodal request pays extra
one-off costs (processor init, CUDA graph capture). ``prepare()`` absorbs both
before the first real window is scheduled.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import cv2
import httpx
import numpy as np

from vlm_demo.backends.base import BackendError
from vlm_demo.backends.openai_compat import OpenAICompatBackend

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"


def warmup_data_url(size: int) -> str:
    """A solid-grey JPEG at the same size as real frames, as a ``data:`` URL."""
    ok, buf = cv2.imencode(".jpg", np.full((size, size, 3), 127, np.uint8))
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class VLLMBackend(OpenAICompatBackend):
    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        ready_timeout: float = 600.0,
        warmup_frames: int = 8,
        warmup_frame_size: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(model, base_url=base_url or DEFAULT_BASE_URL, **kwargs)
        self.ready_timeout = ready_timeout
        self.warmup_frames = warmup_frames
        self.warmup_frame_size = warmup_frame_size

    async def prepare(self) -> None:
        await super().prepare()
        assert self._client is not None
        await self._wait_ready(self._client)
        await self._warmup(self._client)

    async def _wait_ready(self, client: httpx.AsyncClient) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while True:
            try:
                if (await client.get("/models", timeout=5.0)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise BackendError(
                    f"vLLM at {self.base_url} not ready after {self.ready_timeout:g}s"
                )
            log.info("waiting for vLLM at %s ...", self.base_url)
            await asyncio.sleep(2.0)

    async def _warmup(self, client: httpx.AsyncClient) -> None:
        image = warmup_data_url(self.warmup_frame_size)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe the images."}]
                    + [
                        {"type": "image_url", "image_url": {"url": image}}
                        for _ in range(self.warmup_frames)
                    ],
                }
            ],
            "max_tokens": 8,
        }
        started = time.perf_counter()
        try:
            # First multimodal pass can be far slower than steady state.
            response = await client.post(
                "/chat/completions", json=payload, timeout=max(self.timeout, 300.0)
            )
        except httpx.HTTPError as exc:
            raise BackendError(f"vLLM warmup request failed: {exc}") from exc
        if response.status_code >= 400:
            raise BackendError(
                f"vLLM warmup returned {response.status_code}: {response.text[:300]}"
            )
        log.info("vLLM warmup pass took %.1fs", time.perf_counter() - started)
