"""Shared fixtures: a synthetic clip and a ready-made RunConfig."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from vlm_demo.config import BackendKind, RunConfig

CLIP_FPS = 10.0
CLIP_SECONDS = 6.0
CLIP_SIZE = (160, 120)  # width, height
EVENT_AT = 3.0
"""The synthetic clip turns red from this second on, so tests have something to point at."""


def write_clip(path: Path) -> Path:
    """Render a small deterministic clip: a box sliding right, going red at EVENT_AT."""
    width, height = CLIP_SIZE
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (width, height)
    )
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        pytest.skip("no usable mp4 encoder in this OpenCV build")
    total = int(CLIP_FPS * CLIP_SECONDS)
    for n in range(total):
        t = n / CLIP_FPS
        frame = np.full((height, width, 3), 24, dtype=np.uint8)
        x = int((width - 30) * n / max(total - 1, 1))
        colour = (0, 0, 220) if t >= EVENT_AT else (200, 200, 200)
        cv2.rectangle(frame, (x, 45), (x + 30, 75), colour, thickness=-1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_clip(tmp_path_factory.mktemp("video") / "sample.mp4")


@pytest.fixture
def library(clip: Path, tmp_path: Path) -> Path:
    """A per-test ``--input`` directory holding one copy of the clip.

    Tests that upload or select must not share a directory, hence the copy.
    """
    root = tmp_path / "vids"
    root.mkdir()
    shutil.copy(clip, root / clip.name)
    return root


@pytest.fixture
def make_config(library: Path):
    def _make(**overrides: Any) -> RunConfig:
        defaults: dict[str, Any] = {
            "input": library,
            "prompt": "detect if any accident happens",
            "model": "mock",
            "backend": BackendKind.MOCK,
            "window_sec": 2.0,
            "num_frames": 8,
            "pass_gap": 1.0,
            "mock_latency": 0.0,
            "mock_detect_at": EVENT_AT,
            "open_browser": False,
        }
        return RunConfig(**{**defaults, **overrides})

    return _make
