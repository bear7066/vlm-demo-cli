"""The websocket wire contract between the server and the page.

Every server → client message is one of the models below, serialised as JSON with a
discriminating ``type`` field. Client → server messages are handled in :mod:`vlm_demo.server`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class RunState(StrEnum):
    LOADING = "loading"
    """The backend is still being prepared (e.g. weights are loading)."""

    READY = "ready"
    """Waiting for the user to press Start."""

    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


class VideoInfo(BaseModel):
    """The video currently selected for analysis, once it has been probed."""

    filename: str
    url: str
    """Where the page should point its ``<video>``; carries a cache-busting stamp."""
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int


class LibraryVideo(BaseModel):
    """One entry of the ``--input`` directory, offered to the user as a choice."""

    name: str
    size_bytes: int
    modified: float


class LibraryEvent(BaseModel):
    """The contents of the ``--input`` directory; re-sent whenever they change."""

    type: Literal["library"] = "library"
    directory: str
    videos: list[LibraryVideo] = Field(default_factory=list)
    selected: str | None = None
    uploads_enabled: bool = True
    deletes_enabled: bool = True
    max_upload_mb: float = 0.0


class SessionEvent(BaseModel):
    """Sent first on every connection: what this run is about."""

    type: Literal["session"] = "session"
    prompt: str
    model: str
    backend: str
    window_sec: float
    num_frames: int
    pass_gap: float
    pace: str
    highlight_regex: str
    total_passes: int
    video: VideoInfo | None = None
    """``None`` until a video is selected — the directory may still be empty."""


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    state: RunState
    detail: str = ""


class PassStartedEvent(BaseModel):
    type: Literal["pass_started"] = "pass_started"
    index: int
    t_start: float
    t_end: float
    frame_ts: list[float] = Field(default_factory=list)


class PassResultEvent(BaseModel):
    type: Literal["pass_result"] = "pass_result"
    index: int
    t_start: float
    t_end: float
    text: str
    latency_ms: float
    frames_used: int
    matched: bool = False
    """True when the response matches ``--highlight-regex``; the UI highlights these."""


class PassSkippedEvent(BaseModel):
    type: Literal["pass_skipped"] = "pass_skipped"
    index: int
    t_start: float
    t_end: float
    reason: str
    count: int = 1
    """How many consecutive windows this marker stands for."""


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    index: int | None = None


Event = Annotated[
    SessionEvent
    | LibraryEvent
    | StatusEvent
    | PassStartedEvent
    | PassResultEvent
    | PassSkippedEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


def dump(event: BaseModel) -> dict[str, Any]:
    """JSON-safe dict for websocket / REST transport."""
    return event.model_dump(mode="json")
