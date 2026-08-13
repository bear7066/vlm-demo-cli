"""Per-run state and the fan-out of events to connected pages."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from pydantic import BaseModel

from vlm_demo.config import RunConfig
from vlm_demo.events import (
    RunState,
    SessionEvent,
    StatusEvent,
    VideoInfo,
    dump,
)
from vlm_demo.video import VideoMeta

log = logging.getLogger(__name__)

QUEUE_SIZE = 256
HISTORY_LIMIT = 2000


class Session:
    """One run: its config, its state, and every event it has produced so far.

    New connections (and page refreshes) replay :meth:`history`, so the feed survives a reload.
    """

    def __init__(self, config: RunConfig, video: VideoMeta, backend_name: str) -> None:
        self.config = config
        self.video = video
        self.backend_name = backend_name
        self.state = RunState.LOADING
        self._highlight = re.compile(config.highlight_regex)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ description

    def describe(self) -> SessionEvent:
        cfg = self.config
        return SessionEvent(
            prompt=cfg.prompt,
            model=cfg.model,
            backend=self.backend_name,
            window_sec=cfg.window_sec,
            num_frames=cfg.num_frames,
            pass_gap=cfg.pass_gap,
            pace=cfg.pace.value,
            highlight_regex=cfg.highlight_regex,
            total_passes=cfg.total_passes(self.video.duration),
            video=VideoInfo(
                filename=self.video.path.name,
                duration=self.video.duration,
                fps=self.video.fps,
                width=self.video.width,
                height=self.video.height,
                frame_count=self.video.frame_count,
            ),
        )

    def matches(self, text: str) -> bool:
        """Whether a response should be highlighted in the UI."""
        return bool(self._highlight.search(text))

    # ------------------------------------------------------------------ pub/sub

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    async def publish(self, event: BaseModel) -> None:
        payload = dump(event)
        self._history.append(payload)
        if len(self._history) > HISTORY_LIMIT:
            del self._history[: len(self._history) - HISTORY_LIMIT]
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # A stalled page must not slow the pipeline down.
                log.debug("dropping event for a slow subscriber")

    async def set_state(self, state: RunState, detail: str = "") -> None:
        self.state = state
        await self.publish(StatusEvent(state=state, detail=detail))
