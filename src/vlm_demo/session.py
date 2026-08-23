"""Per-run state and the fan-out of events to connected pages."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import quote

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
    The selected video and the model can change while the process lives (the user picks
    another video from the ``--input`` directory, or another model from the page), which starts
    the feed over — see :meth:`retarget`. ``config`` is replaced wholesale on a model switch, so
    read it through ``self`` rather than capturing it.
    """

    def __init__(
        self,
        config: RunConfig,
        backend_name: str,
        video: VideoMeta | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        self.config = config
        self.video = video
        self.backend_name = backend_name
        self.available_models = list(available_models or ())
        """Model ids the page offers as suggestions; fixed for the life of the process."""
        self.state = RunState.LOADING
        self._highlight = re.compile(config.highlight_regex)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ description

    def describe(self) -> SessionEvent:
        cfg = self.config
        video = self.video
        return SessionEvent(
            prompt=cfg.prompt,
            model=cfg.model,
            backend=self.backend_name,
            window_sec=cfg.window_sec,
            num_frames=cfg.num_frames,
            pass_gap=cfg.pass_gap,
            pace=cfg.pace.value,
            highlight_regex=cfg.highlight_regex,
            total_passes=cfg.total_passes(video.duration) if video else 0,
            video=None if video is None else video_info(video),
            available_models=self.available_models,
            model_locked=cfg.lock_model,
        )

    async def retarget(self, video: VideoMeta | None, state: RunState, detail: str = "") -> None:
        """Point the session at another video: drop the old feed and announce the new one.

        The history is cleared because every event in it describes the previous video's
        timeline; pages that reconnect must not replay it against the new one. The new
        description is not recorded either — every connection is sent a fresh one anyway.
        """
        self.video = video
        self._history.clear()
        await self.publish(self.describe(), record=False)
        await self.set_state(state, detail)

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

    async def publish(self, event: BaseModel, *, record: bool = True) -> None:
        """Fan an event out to every open page, and (by default) keep it for replay.

        ``record=False`` is for events that are re-sent in full on every connection anyway —
        the library listing — which would otherwise pile up in the history.
        """
        payload = dump(event)
        if record:
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


def video_info(meta: VideoMeta) -> VideoInfo:
    """Describe a probed video for the page, including where to stream it from.

    The ``v=`` stamp changes when the file does, so re-uploading over a name the browser has
    already cached still plays the new bytes.
    """
    return VideoInfo(
        filename=meta.path.name,
        url=f"/api/video/{quote(meta.path.name)}?v={int(meta.path.stat().st_mtime)}",
        duration=meta.duration,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        frame_count=meta.frame_count,
    )
