"""Video probing and sliding-window frame extraction.

Decoding uses OpenCV, which ships its own codecs — no system ``ffmpeg`` required. Everything
here is blocking and thread-safe (guarded by a lock); callers run it via ``asyncio.to_thread``
so the event loop stays free.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0
"""Used when a container does not report a usable frame rate."""

MAX_CACHED_FRAMES = 256
"""Encoded frames kept around so overlapping windows do not re-decode."""


@dataclass(frozen=True, slots=True)
class Window:
    """A slice of the video timeline that one inference pass looks at."""

    index: int
    t_start: float
    t_end: float


@dataclass(frozen=True, slots=True)
class Frame:
    """A single JPEG-encoded frame handed to a backend."""

    index: int
    """Source frame number."""
    t: float
    """Timestamp in the video, in seconds."""
    jpeg: bytes
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


class VideoError(RuntimeError):
    pass


def sample_timestamps(
    t_start: float, t_end: float, num_frames: int, duration: float
) -> list[float]:
    """Uniformly sample ``num_frames`` timestamps across ``[t_start, t_end]``.

    Both ends are included. Windows near t=0 are truncated rather than padded, and the end is
    clamped to the video duration.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    t_end = max(0.0, min(t_end, duration))
    t_start = max(0.0, min(t_start, t_end))
    if num_frames == 1 or t_end <= t_start:
        return [t_end] * num_frames
    step = (t_end - t_start) / (num_frames - 1)
    return [t_start + step * i for i in range(num_frames)]


class VideoSource:
    """Reads windows of frames out of a video file, encoded and ready to send.

    Windows overlap and advance monotonically during playback, so decoded frames are cached
    (as JPEG, which is small) and the reader walks forward instead of seeking per frame.
    Backward jumps — the user scrubbing the player — fall back to a seek.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_size: int = 512,
        jpeg_quality: int = 80,
        dump_dir: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_size = max_size
        self.jpeg_quality = jpeg_quality
        self.dump_dir = dump_dir
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._pos = 0
        """Index of the next frame ``cap.read()`` will return."""
        self._cache: OrderedDict[int, tuple[bytes, int, int]] = OrderedDict()
        self._meta: VideoMeta | None = None

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> VideoMeta:
        with self._lock:
            if self._cap is not None:
                assert self._meta is not None
                return self._meta
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise VideoError(f"could not open video: {self.path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not np.isfinite(fps) or fps <= 0:
                log.warning("video reports no usable fps, assuming %.1f", DEFAULT_FPS)
                fps = DEFAULT_FPS
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                frame_count = self._count_frames(cap)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            self._cap = cap
            self._pos = 0
            self._meta = VideoMeta(
                path=self.path,
                fps=fps,
                frame_count=frame_count,
                duration=frame_count / fps if frame_count else 0.0,
                width=width,
                height=height,
            )
            if self.dump_dir is not None:
                self.dump_dir.mkdir(parents=True, exist_ok=True)
            return self._meta

    @staticmethod
    def _count_frames(cap: cv2.VideoCapture) -> int:
        """Last resort for containers without a frame count: walk the file once."""
        count = 0
        while cap.grab():
            count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return count

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._cache.clear()

    @property
    def meta(self) -> VideoMeta:
        if self._meta is None:
            raise VideoError("VideoSource.open() has not been called")
        return self._meta

    # ------------------------------------------------------------------ extraction

    def extract_window(self, window: Window, num_frames: int) -> list[Frame]:
        """Return up to ``num_frames`` encoded frames sampled across ``window``.

        Duplicate source frames (short windows on low-fps video) are collapsed, so the result
        can be shorter than ``num_frames``.
        """
        meta = self.meta
        stamps = sample_timestamps(window.t_start, window.t_end, num_frames, meta.duration)
        last = max(meta.frame_count - 1, 0)

        wanted: list[tuple[int, float]] = []
        seen: set[int] = set()
        for t in stamps:
            idx = min(int(round(t * meta.fps)), last)
            if idx not in seen:
                seen.add(idx)
                wanted.append((idx, idx / meta.fps))

        frames: list[Frame] = []
        with self._lock:
            for idx, t in wanted:
                encoded = self._encoded_frame(idx)
                if encoded is None:
                    continue
                jpeg, w, h = encoded
                frames.append(Frame(index=idx, t=t, jpeg=jpeg, width=w, height=h))
        if not frames:
            raise VideoError(f"no frames decodable for window {window.t_start:.2f}-{window.t_end:.2f}s")
        if self.dump_dir is not None:
            self._dump(window, frames)
        return frames

    # ------------------------------------------------------------------ internals

    def _encoded_frame(self, idx: int) -> tuple[bytes, int, int] | None:
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        raw = self._read_at(idx)
        if raw is None:
            return None
        encoded = self._encode(raw)
        self._cache[idx] = encoded
        while len(self._cache) > MAX_CACHED_FRAMES:
            self._cache.popitem(last=False)
        return encoded

    def _read_at(self, idx: int) -> np.ndarray | None:
        """Decode source frame ``idx``, walking forward when it is just ahead of us."""
        cap = self._cap
        if cap is None:
            raise VideoError("VideoSource is not open")

        seek_threshold = max(int(self.meta.fps * 2), 8)
        if idx < self._pos or idx - self._pos > seek_threshold:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        while self._pos <= idx:
            ok, frame = cap.read()
            current = self._pos
            if not ok:
                log.debug("decode failed at frame %d", current)
                return None
            self._pos += 1
            if current == idx:
                return frame
            # Frames we skipped past are cheap to keep: the next window overlaps this one.
            if current not in self._cache:
                self._cache[current] = self._encode(frame)
        return None

    def _encode(self, bgr: np.ndarray) -> tuple[bytes, int, int]:
        h, w = bgr.shape[:2]
        longest = max(h, w)
        if longest > self.max_size:
            scale = self.max_size / longest
            bgr = cv2.resize(
                bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
            h, w = bgr.shape[:2]
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise VideoError("JPEG encoding failed")
        return bytes(buf), w, h

    def _dump(self, window: Window, frames: list[Frame]) -> None:
        assert self.dump_dir is not None
        for n, frame in enumerate(frames):
            name = f"pass{window.index:04d}_{n:02d}_t{frame.t:07.2f}.jpg"
            (self.dump_dir / name).write_bytes(frame.jpeg)
