"""The contract every inference backend implements."""

from __future__ import annotations

import abc
import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vlm_demo.video import Frame, Window


class BackendError(RuntimeError):
    """Anything that stops a backend from answering: bad config, HTTP error, load failure."""


@dataclass(slots=True)
class InferenceResult:
    text: str
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class VLMBackend(abc.ABC):
    """A model that answers a prompt about a handful of frames.

    Implementations must be safe to call from the event loop: do network I/O with async
    clients and push blocking work (GPU generation, decoding) into a thread.
    """

    name: str = "backend"

    def __init__(self) -> None:
        self.notes: list[str] = []
        """Warnings raised while preparing, surfaced on the page next to the status."""

    async def prepare(self) -> None:
        """Load weights / open clients. Called once, before the first pass."""

    @abc.abstractmethod
    async def infer(
        self, prompt: str, frames: Sequence[Frame], window: Window
    ) -> InferenceResult:
        """Answer ``prompt`` for the frames sampled from ``window``."""

    async def aclose(self) -> None:
        """Release resources. Called on shutdown, even if :meth:`prepare` failed."""


def to_data_url(frame: Frame) -> str:
    """``data:`` URL for one frame, as OpenAI-compatible APIs expect."""
    return "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")
