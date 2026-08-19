"""Run configuration: the single source of truth for one CLI invocation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ConfigError(ValueError):
    """Raised when the flags the user passed cannot produce a valid run."""


class Pace(StrEnum):
    """What to do when inference is slower than ``pass_gap``."""

    REALTIME = "realtime"
    """Skip windows whose deadline has already passed; stay in sync with playback."""

    COMPLETE = "complete"
    """Never skip; run every window even if the feed falls behind the video."""


class BackendKind(StrEnum):
    MOCK = "mock"
    OPENAI_COMPAT = "openai-compat"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


DEFAULT_HIGHLIGHT_REGEX = r"(?i)detect"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one `vlm-demo` run needs, already validated."""

    input: Path
    """Directory of videos to choose from; the page picks which one to analyse."""
    prompt: str
    model: str

    backend: BackendKind
    base_url: str | None = None
    api_key: str | None = None

    window_sec: float = 2.0
    num_frames: int = 8
    pass_gap: float = 1.0
    pace: Pace = Pace.REALTIME
    max_inflight: int = 1

    host: str = "127.0.0.1"
    port: int = 3000
    open_browser: bool = True

    frame_max_size: int = 512
    jpeg_quality: int = 80

    infer_timeout: float = 30.0
    max_tokens: int = 128
    temperature: float = 0.0

    allow_upload: bool = True
    max_upload_mb: float = 1024.0

    highlight_regex: str = DEFAULT_HIGHLIGHT_REGEX
    dump_frames: Path | None = None
    log_level: str = "info"

    mock_detect_at: float | None = None
    mock_latency: float = 0.2

    def __post_init__(self) -> None:
        if not self.input.exists():
            raise ConfigError(f"input directory not found: {self.input}")
        if not self.input.is_dir():
            raise ConfigError(
                f"--input must be a directory of videos, not a file: {self.input}"
            )
        if not self.prompt.strip():
            raise ConfigError("--prompt must not be empty")
        if not self.model.strip():
            raise ConfigError("--model must not be empty")
        if self.window_sec <= 0:
            raise ConfigError("--window-sec must be > 0")
        if self.pass_gap <= 0:
            raise ConfigError("--pass-gap must be > 0")
        if self.num_frames < 1:
            raise ConfigError("--num-frames must be >= 1")
        if self.max_inflight < 1:
            raise ConfigError("--max-inflight must be >= 1")
        if self.frame_max_size < 32:
            raise ConfigError("--frame-max-size must be >= 32")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError("--jpeg-quality must be between 1 and 100")
        if self.infer_timeout <= 0:
            raise ConfigError("--infer-timeout must be > 0")
        if self.max_tokens < 1:
            raise ConfigError("--max-tokens must be >= 1")
        if self.max_upload_mb <= 0:
            raise ConfigError("--max-upload-mb must be > 0")
        if not 1 <= self.port <= 65535:
            raise ConfigError("--port must be between 1 and 65535")
        if self.backend is BackendKind.OPENAI_COMPAT and not self.base_url:
            raise ConfigError("the openai-compat backend needs --base-url (or $VLM_BASE_URL)")
        try:
            re.compile(self.highlight_regex)
        except re.error as exc:
            raise ConfigError(f"--highlight-regex is not a valid regex: {exc}") from exc

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1_000_000)

    def total_passes(self, duration: float) -> int:
        """How many inference passes a video of ``duration`` seconds yields.

        Pass ``k`` (1-based) fires at video time ``k * pass_gap``, so a 10s video with a
        1s gap gives exactly 10 passes.
        """
        return max(1, math.ceil(round(duration / self.pass_gap, 6)))

    def window_for(self, index: int, duration: float) -> tuple[float, float]:
        """The ``(t_start, t_end)`` video-time window covered by pass ``index``."""
        t_end = min(index * self.pass_gap, duration)
        return max(0.0, t_end - self.window_sec), t_end


def resolve_backend_kind(
    explicit: str | None, model: str, base_url: str | None
) -> BackendKind:
    """Pick a backend: explicit flag wins, then --base-url, then the model id shape."""
    if explicit:
        try:
            return BackendKind(explicit)
        except ValueError as exc:
            choices = ", ".join(k.value for k in BackendKind)
            raise ConfigError(f"unknown --backend {explicit!r}; choose one of: {choices}") from exc
    if model.lower().startswith("mock"):
        return BackendKind.MOCK
    if base_url:
        return BackendKind.OPENAI_COMPAT
    return BackendKind.TRANSFORMERS
