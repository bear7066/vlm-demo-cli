"""The video directory behind ``--input``: listing, safe name resolution, uploads.

``--input`` points at a folder, not a file. The page lists what is in it, the user picks one
to analyse, and may add more by uploading. Everything a browser hands us — a file name, the
name to switch to — is untrusted, so it goes through :func:`sanitize_name` /
:meth:`VideoLibrary.resolve` before it ever touches the filesystem.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = frozenset(
    {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".ogv", ".wmv"}
)
"""Extensions treated as videos. Anything else in the directory is ignored."""

MAX_NAME_LENGTH = 120
UPLOAD_SUFFIX = ".part"
"""Uploads land under this suffix and are renamed once complete, so a half-written file
never shows up in the listing."""

_UNSAFE = re.compile(r"[^A-Za-z0-9 ._()\[\]-]+")
_RUNS = re.compile(r"_{2,}")


class LibraryError(ValueError):
    """A name the client asked for cannot be served or stored."""


class UploadTooLarge(LibraryError):
    """The upload ran past ``--max-upload-mb``; reported as HTTP 413."""


def sanitize_name(raw: str) -> str:
    """Reduce a client-supplied name to a plain, safe file name inside the library.

    Directory components are dropped (both separators — browsers on Windows send backslashes),
    unusual characters become ``_``, and the extension must be a known video one.
    """
    candidate = PureWindowsPath(PurePosixPath(raw.strip()).name).name
    stem, dot, suffix = candidate.rpartition(".")
    if not dot:
        raise LibraryError(f"{raw!r} has no file extension")
    suffix = "." + suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(VIDEO_SUFFIXES))
        raise LibraryError(f"{suffix!r} is not a video extension; expected one of: {allowed}")
    stem = _RUNS.sub("_", _UNSAFE.sub("_", stem)).strip(" .")
    if not stem:
        raise LibraryError(f"{raw!r} has no usable file name")
    return stem[: MAX_NAME_LENGTH - len(suffix)] + suffix


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


@dataclass(frozen=True, slots=True)
class VideoEntry:
    """One selectable video in the library."""

    name: str
    size_bytes: int
    modified: float

    @classmethod
    def of(cls, path: Path) -> VideoEntry:
        stat = path.stat()
        return cls(name=path.name, size_bytes=stat.st_size, modified=stat.st_mtime)


class VideoLibrary:
    """The directory ``--input`` points at, plus the rules for reading and writing in it."""

    def __init__(self, root: Path, *, max_upload_bytes: int, allow_upload: bool = True) -> None:
        self.root = Path(root).resolve()
        self.max_upload_bytes = max_upload_bytes
        self.allow_upload = allow_upload

    # ------------------------------------------------------------------ reading

    def entries(self) -> list[VideoEntry]:
        """Every video directly in the directory, sorted by name. Sub-folders are ignored."""
        found = []
        for path in sorted(self.root.iterdir(), key=lambda p: p.name.lower()):
            if is_video(path):
                found.append(VideoEntry.of(path))
        return found

    def names(self) -> list[str]:
        return [entry.name for entry in self.entries()]

    def default_name(self) -> str | None:
        """What to select on startup: the first video, or nothing if the folder is empty."""
        entries = self.entries()
        return entries[0].name if entries else None

    def resolve(self, name: str) -> Path:
        """Map a client-supplied name to a file in the library, or raise.

        Rejects anything that would escape the directory, including via a symlink.
        """
        path = (self.root / sanitize_name(name)).resolve()
        if path.parent != self.root:
            raise LibraryError(f"{name!r} is outside the video directory")
        if not is_video(path):
            raise LibraryError(f"no such video: {name}")
        return path

    # ------------------------------------------------------------------ writing

    def unique_name(self, name: str) -> str:
        """``clip.mp4`` → ``clip.mp4`` if free, else ``clip-1.mp4``, ``clip-2.mp4``, …"""
        stem, suffix = Path(name).stem, Path(name).suffix
        candidate = name
        n = 0
        while (self.root / candidate).exists():
            n += 1
            candidate = f"{stem}-{n}{suffix}"
        return candidate

    async def save(self, raw_name: str, chunks: AsyncIterator[bytes]) -> VideoEntry:
        """Stream an upload to disk under a safe, unused name and return its entry.

        The bytes are written to a temporary file first, so a failed or oversized upload
        leaves nothing selectable behind.
        """
        if not self.allow_upload:
            raise LibraryError("uploads are disabled (--no-upload)")
        name = self.unique_name(sanitize_name(raw_name))
        temp = self.root / f".{uuid.uuid4().hex}{UPLOAD_SUFFIX}"
        written = 0
        try:
            with temp.open("wb") as handle:
                async for chunk in chunks:
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        raise UploadTooLarge(
                            f"upload exceeds the {self.max_upload_bytes / 1e6:.0f} MB limit "
                            "(raise it with --max-upload-mb)"
                        )
                    handle.write(chunk)
            if written == 0:
                raise LibraryError("upload is empty")
            os.replace(temp, self.root / name)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        log.info("stored upload %s (%.1f MB)", name, written / 1e6)
        return VideoEntry.of(self.root / name)
