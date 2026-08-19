"""The ``--input`` directory: what it lists, what names it accepts, what it stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_demo.config import ConfigError, RunConfig
from vlm_demo.library import (
    LibraryError,
    UploadTooLarge,
    VideoLibrary,
    sanitize_name,
)


@pytest.fixture
def lib(tmp_path: Path) -> VideoLibrary:
    return VideoLibrary(tmp_path, max_upload_bytes=1_000)


async def feed(*chunks: bytes):
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------- names


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("clip.mp4", "clip.mp4"),
        ("My Clip (2).MP4", "My Clip (2).mp4"),
        ("../../etc/passwd.mp4", "passwd.mp4"),
        (r"C:\Users\me\Desktop\fall.mov", "fall.mov"),
        ("/abs/path/x.webm", "x.webm"),
        ("wei;rd*na|me.mkv", "wei_rd_na_me.mkv"),
        ("a.b.mp4", "a.b.mp4"),
    ],
)
def test_sanitize_name_reduces_to_a_safe_basename(raw, expected):
    assert sanitize_name(raw) == expected


@pytest.mark.parametrize("raw", ["notes.txt", "clip", "..", "....mp4", "payload.mp4.exe"])
def test_sanitize_name_rejects_what_is_not_a_video(raw):
    with pytest.raises(LibraryError):
        sanitize_name(raw)


# ---------------------------------------------------------------- listing


def test_entries_lists_only_videos_in_the_directory(lib: VideoLibrary):
    (lib.root / "b.mp4").write_bytes(b"x")
    (lib.root / "a.mov").write_bytes(b"xy")
    (lib.root / "notes.txt").write_text("ignored")
    (lib.root / "nested").mkdir()
    (lib.root / "nested" / "deep.mp4").write_bytes(b"x")

    assert [e.name for e in lib.entries()] == ["a.mov", "b.mp4"]
    assert lib.default_name() == "a.mov"
    assert lib.entries()[0].size_bytes == 2


def test_default_name_is_none_for_an_empty_directory(lib: VideoLibrary):
    assert lib.default_name() is None
    assert lib.entries() == []


def test_resolve_finds_a_video_and_refuses_anything_else(lib: VideoLibrary):
    (lib.root / "clip.mp4").write_bytes(b"x")
    assert lib.resolve("clip.mp4") == lib.root / "clip.mp4"
    with pytest.raises(LibraryError, match="no such video"):
        lib.resolve("missing.mp4")


def test_resolve_refuses_to_escape_the_directory(tmp_path: Path):
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    root = tmp_path / "vids"
    root.mkdir()
    lib = VideoLibrary(root, max_upload_bytes=1_000)
    with pytest.raises(LibraryError):
        lib.resolve("../secret.mp4")


def test_resolve_refuses_a_symlink_pointing_outside(tmp_path: Path):
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    root = tmp_path / "vids"
    root.mkdir()
    (root / "link.mp4").symlink_to(outside)
    with pytest.raises(LibraryError, match="outside"):
        VideoLibrary(root, max_upload_bytes=1_000).resolve("link.mp4")


# ---------------------------------------------------------------- uploads


async def test_save_stores_the_stream_under_a_safe_name(lib: VideoLibrary):
    entry = await lib.save("../My Clip.MP4", feed(b"abc", b"de"))
    assert entry.name == "My Clip.mp4"
    assert entry.size_bytes == 5
    assert (lib.root / "My Clip.mp4").read_bytes() == b"abcde"


async def test_save_never_overwrites_an_existing_video(lib: VideoLibrary):
    (lib.root / "clip.mp4").write_bytes(b"original")
    first = await lib.save("clip.mp4", feed(b"new"))
    second = await lib.save("clip.mp4", feed(b"newer"))
    assert (first.name, second.name) == ("clip-1.mp4", "clip-2.mp4")
    assert (lib.root / "clip.mp4").read_bytes() == b"original"


async def test_save_rejects_an_oversized_upload_and_leaves_nothing_behind(lib: VideoLibrary):
    with pytest.raises(UploadTooLarge):
        await lib.save("big.mp4", feed(b"x" * 600, b"x" * 600))
    assert lib.entries() == []
    assert list(lib.root.iterdir()) == [], "the partial file must be cleaned up"


async def test_save_rejects_an_empty_upload(lib: VideoLibrary):
    with pytest.raises(LibraryError, match="empty"):
        await lib.save("nothing.mp4", feed())
    assert list(lib.root.iterdir()) == []


async def test_save_honours_no_upload(tmp_path: Path):
    lib = VideoLibrary(tmp_path, max_upload_bytes=1_000, allow_upload=False)
    with pytest.raises(LibraryError, match="disabled"):
        await lib.save("clip.mp4", feed(b"x"))


# ---------------------------------------------------------------- config


def test_config_requires_a_directory(clip: Path, tmp_path: Path):
    with pytest.raises(ConfigError, match="must be a directory"):
        RunConfig(input=clip, prompt="p", model="mock", backend="mock")
    with pytest.raises(ConfigError, match="not found"):
        RunConfig(input=tmp_path / "nope", prompt="p", model="mock", backend="mock")
