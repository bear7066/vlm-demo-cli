from __future__ import annotations

from pathlib import Path

import pytest

from vlm_demo.video import VideoSource, Window, sample_timestamps

CLIP_FPS = 10.0
CLIP_SECONDS = 6.0


def test_sample_timestamps_spans_the_window_inclusively():
    stamps = sample_timestamps(2.0, 4.0, 5, duration=10.0)
    assert stamps[0] == pytest.approx(2.0)
    assert stamps[-1] == pytest.approx(4.0)
    assert stamps == sorted(stamps)
    assert len(stamps) == 5


def test_sample_timestamps_truncates_at_the_start_of_the_video():
    # A 2s window ending at 1s only has 1s of video behind it.
    stamps = sample_timestamps(-1.0, 1.0, 4, duration=10.0)
    assert stamps[0] == pytest.approx(0.0)
    assert stamps[-1] == pytest.approx(1.0)


def test_sample_timestamps_clamps_to_duration():
    stamps = sample_timestamps(9.0, 12.0, 3, duration=10.0)
    assert stamps[-1] == pytest.approx(10.0)


def test_probe_reads_metadata(clip: Path):
    source = VideoSource(clip)
    try:
        meta = source.open()
    finally:
        source.close()
    assert meta.fps == pytest.approx(CLIP_FPS, abs=0.5)
    assert meta.duration == pytest.approx(CLIP_SECONDS, abs=0.2)
    assert (meta.width, meta.height) == (160, 120)


def test_extract_window_returns_frames_inside_the_window(clip: Path):
    source = VideoSource(clip, max_size=64)
    source.open()
    try:
        frames = source.extract_window(Window(index=3, t_start=1.0, t_end=3.0), num_frames=8)
    finally:
        source.close()

    assert len(frames) == 8
    assert [f.t for f in frames] == sorted(f.t for f in frames)
    assert all(0.9 <= f.t <= 3.1 for f in frames)
    assert all(f.jpeg[:2] == b"\xff\xd8" for f in frames)  # JPEG SOI
    assert all(max(f.width, f.height) <= 64 for f in frames)


def test_overlapping_windows_reuse_decoded_frames(clip: Path):
    """Consecutive windows overlap; the second one must not re-decode what is cached."""
    source = VideoSource(clip)
    source.open()
    try:
        first = source.extract_window(Window(1, 0.0, 2.0), num_frames=8)
        cached_before = set(source._cache)
        second = source.extract_window(Window(2, 1.0, 3.0), num_frames=8)
    finally:
        source.close()

    decoded_up_to = max(f.index for f in first)
    overlap = {f.index for f in second if f.index <= decoded_up_to}
    assert overlap, "the two windows overlap in time"
    assert overlap <= cached_before, "the overlapping part should come from the cache"


def test_backward_seek_still_decodes(clip: Path):
    """The user scrubbing back must not break the sequential reader."""
    source = VideoSource(clip)
    source.open()
    try:
        source.extract_window(Window(1, 3.0, 5.0), num_frames=4)
        frames = source.extract_window(Window(2, 0.0, 1.0), num_frames=4)
    finally:
        source.close()
    assert frames and frames[0].t < 1.1
