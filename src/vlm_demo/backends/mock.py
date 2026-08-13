"""A backend that needs no model, so the whole pipeline can be demoed and tested.

It answers in the shape the README's example prompt asks for: ``nothing happened`` until
``detect_at`` seconds of video time, then ``accident detected: ...``. Responses depend only on
the window, so runs are reproducible.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from vlm_demo.backends.base import InferenceResult, VLMBackend
from vlm_demo.video import Frame, Window

NOTHING = "nothing happened"


class MockBackend(VLMBackend):
    name = "mock"

    def __init__(
        self,
        model: str = "mock",
        *,
        detect_at: float | None = None,
        latency: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = model
        self.detect_at = detect_at
        self.latency = max(0.0, latency)

    async def infer(
        self, prompt: str, frames: Sequence[Frame], window: Window
    ) -> InferenceResult:
        started = time.perf_counter()
        if self.latency:
            await asyncio.sleep(self.latency)
        if self.detect_at is not None and window.t_end >= self.detect_at:
            text = (
                f"accident detected: two vehicles collide near the crossing "
                f"(first seen at {self.detect_at:.1f}s, {len(frames)} frames reviewed)"
            )
        else:
            text = NOTHING
        return InferenceResult(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"mock": True, "frames": len(frames)},
        )
