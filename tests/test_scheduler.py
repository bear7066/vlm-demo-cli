from __future__ import annotations

import asyncio
import time

import pytest

from vlm_demo.backends.mock import MockBackend
from vlm_demo.config import Pace
from vlm_demo.scheduler import MediaClock, Scheduler
from vlm_demo.session import Session
from vlm_demo.video import VideoSource


class EndedClock:
    """A player that has already reached the end of the video."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.stopped = False
        self.ended = True

    def now(self) -> float:
        return self.duration

    async def wait_until(self, target: float) -> bool:
        return False


class SpeedClock:
    """Video time running ``speed``× faster than wall time, to keep tests short."""

    def __init__(self, duration: float, speed: float = 10.0) -> None:
        self.duration = duration
        self.speed = speed
        self.stopped = False
        self._start = time.monotonic()

    @property
    def ended(self) -> bool:
        return self.now() >= self.duration - 1e-9

    def now(self) -> float:
        return min((time.monotonic() - self._start) * self.speed, self.duration)

    async def wait_until(self, target: float) -> bool:
        while True:
            if self.stopped:
                return False
            current = self.now()
            if current >= target - 1e-3:
                return True
            if self.ended:
                return False
            await asyncio.sleep(min((target - current) / self.speed, 0.02))


def build(config, clock, latency: float = 0.0):
    source = VideoSource(config.input, max_size=64)
    meta = source.open()
    backend = MockBackend(detect_at=config.mock_detect_at, latency=latency)
    session = Session(config, meta, backend.name)
    scheduler = Scheduler(config, session, source, backend, clock)
    return source, session, scheduler, meta


def of_type(session: Session, kind: str) -> list[dict]:
    return [event for event in session.history() if event["type"] == kind]


# ---------------------------------------------------------------- MediaClock


def test_media_clock_extrapolates_while_playing():
    clock = MediaClock(duration=10.0)
    clock.update(2.0, playing=True)
    time.sleep(0.05)
    assert clock.now() > 2.0
    clock.update(2.0, playing=False)
    time.sleep(0.05)
    assert clock.now() == pytest.approx(2.0)


def test_media_clock_never_runs_past_the_end():
    clock = MediaClock(duration=1.0)
    clock.update(0.99, playing=True)
    time.sleep(0.05)
    assert clock.now() <= 1.0


async def test_wait_until_returns_for_a_deadline_already_reached():
    clock = MediaClock(duration=5.0)
    clock.mark_ended()
    assert await clock.wait_until(4.0) is True


async def test_wait_until_gives_up_on_deadlines_past_the_end():
    clock = MediaClock(duration=5.0)
    clock.mark_ended()
    assert await clock.wait_until(5.5) is False


# ---------------------------------------------------------------- pacing


async def test_complete_pace_runs_every_window(make_config):
    config = make_config(pace=Pace.COMPLETE, pass_gap=1.0, num_frames=4)
    source, session, scheduler, meta = build(config, EndedClock(0.0))
    try:
        await scheduler.run()
    finally:
        source.close()

    total = config.total_passes(meta.duration)
    results = of_type(session, "pass_result")
    assert [r["index"] for r in results] == list(range(1, total + 1))
    assert of_type(session, "pass_skipped") == []
    assert of_type(session, "error") == []


async def test_complete_pace_reports_detection_after_the_event(make_config):
    config = make_config(pace=Pace.COMPLETE, pass_gap=1.0, num_frames=4)
    source, session, scheduler, _ = build(config, EndedClock(0.0))
    try:
        await scheduler.run()
    finally:
        source.close()

    results = of_type(session, "pass_result")
    assert [r["matched"] for r in results] == [
        r["t_end"] >= config.mock_detect_at for r in results
    ]
    assert results[0]["text"] == "nothing happened"
    assert results[-1]["matched"] is True


async def test_realtime_pace_skips_instead_of_falling_behind(make_config):
    # Each pass costs 0.05s of wall time = 0.5s of video time at speed 10, well over pass_gap.
    config = make_config(pace=Pace.REALTIME, pass_gap=0.2, window_sec=0.4, num_frames=2)
    clock = SpeedClock(6.0, speed=10.0)
    source, session, scheduler, meta = build(config, clock, latency=0.05)
    try:
        await scheduler.run()
    finally:
        source.close()

    total = config.total_passes(meta.duration)
    results = of_type(session, "pass_result")
    skipped = of_type(session, "pass_skipped")

    assert results, "some passes should still complete"
    assert len(results) < total, "a slow backend must not keep up with every window"
    assert skipped, "windows the player moved past must be reported as skipped"
    assert [r["index"] for r in results] == sorted(r["index"] for r in results)
    assert max(r["index"] for r in results) <= total

    # Regression: the loop used to rewind onto the window it had just handled and emit the
    # same skip over and over while a pass was in flight.
    reported = [event["index"] for event in skipped]
    assert len(reported) == len(set(reported)), "each window may be skipped at most once"
    assert not set(reported) & {r["index"] for r in results}


async def test_realtime_pace_keeps_up_when_the_backend_is_fast(make_config):
    """The headline case: a fast backend answers every window, with nothing skipped."""
    config = make_config(pace=Pace.REALTIME, pass_gap=0.5, window_sec=1.0, num_frames=2)
    source, session, scheduler, meta = build(config, SpeedClock(6.0, speed=20.0), latency=0.005)
    try:
        await asyncio.wait_for(scheduler.run(), timeout=10)
    finally:
        source.close()

    results = of_type(session, "pass_result")
    assert [r["index"] for r in results] == list(range(1, config.total_passes(meta.duration) + 1))
    assert of_type(session, "pass_skipped") == []


async def test_realtime_pace_finishes_when_playback_ends(make_config):
    config = make_config(pace=Pace.REALTIME, pass_gap=0.5, window_sec=1.0, num_frames=2)
    source, session, scheduler, _ = build(config, SpeedClock(6.0, speed=20.0))
    try:
        await asyncio.wait_for(scheduler.run(), timeout=10)
    finally:
        source.close()
    assert session.history()[-1] == {
        "type": "status",
        "state": "finished",
        "detail": "run complete",
    }


async def test_stop_ends_the_run_early(make_config):
    config = make_config(pace=Pace.COMPLETE, pass_gap=0.5, num_frames=2)
    source, session, scheduler, _ = build(config, EndedClock(0.0), latency=0.02)
    try:
        task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        scheduler.stop()
        await asyncio.wait_for(task, timeout=5)
    finally:
        source.close()

    total = config.total_passes(6.0)
    assert len(of_type(session, "pass_result")) < total
