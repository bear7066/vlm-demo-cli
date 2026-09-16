"""The media clock and the pass loop.

The browser owns the timeline: the ``<video>`` element reports ``currentTime`` over the
websocket and :class:`MediaClock` interpolates between those heartbeats. The scheduler wakes at
every pass deadline, cuts a window of frames and asks the backend about it.
"""

from __future__ import annotations

import asyncio
import logging
import time

from vlm_demo.backends.base import VLMBackend
from vlm_demo.config import Pace, RunConfig
from vlm_demo.events import (
    ErrorEvent,
    PassResultEvent,
    PassSkippedEvent,
    PassStartedEvent,
    RunState,
)
from vlm_demo.session import Session
from vlm_demo.video import Frame, VideoSource, Window

log = logging.getLogger(__name__)

EPS = 1e-3
IDLE_POLL = 0.1
"""How often to re-check the clock while the video is paused."""

MAX_SKIP_EVENTS = 6
"""Beyond this, consecutive skips collapse into a single marker."""


class MediaClock:
    """Video-time as reported by the player, extrapolated between heartbeats."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.playing = False
        self.ended = False
        self.stopped = False
        self._t = 0.0
        self._wall = time.monotonic()

    def update(self, t: float, playing: bool) -> None:
        self._t = max(0.0, min(t, self.duration))
        self._wall = time.monotonic()
        self.playing = playing
        if playing:
            self.ended = False

    def mark_ended(self) -> None:
        self._t = self.duration
        self._wall = time.monotonic()
        self.playing = False
        self.ended = True

    def stop(self) -> None:
        self.stopped = True
        self.playing = False

    def now(self) -> float:
        if not self.playing:
            return self._t
        return min(self._t + (time.monotonic() - self._wall), self.duration)

    async def wait_until(self, target: float) -> bool:
        """Block until video time reaches ``target``.

        Returns ``False`` if the video ended (or the run was stopped) first, which is how the
        caller learns there is no more playback to wait for.
        """
        while True:
            if self.stopped:
                return False
            current = self.now()
            if current >= target - EPS:
                return True
            if self.ended:
                return False
            remaining = target - current if self.playing else IDLE_POLL
            await asyncio.sleep(min(max(remaining, 0.01), IDLE_POLL))


class Scheduler:
    """Drives inference passes for one playback run."""

    def __init__(
        self,
        config: RunConfig,
        session: Session,
        video: VideoSource,
        backend: VLMBackend,
        clock: MediaClock,
    ) -> None:
        self.config = config
        self.session = session
        self.video = video
        self.backend = backend
        self.clock = clock
        self._stopped = False
        self._inflight = 0
        self._tasks: set[asyncio.Task[None]] = set()

    def stop(self) -> None:
        self._stopped = True

    async def run(self) -> None:
        cfg = self.config
        duration = self.video.meta.duration
        total = cfg.total_passes(duration)
        await self.session.set_state(RunState.RUNNING, f"{total} passes planned")
        log.info("scheduler started: %d passes, pace=%s", total, cfg.pace.value)

        index = 1
        try:
            while index <= total and not self._stopped:
                if cfg.pace is Pace.REALTIME:
                    index = await self._realign(index, total)
                    if index > total:
                        break

                deadline = min(index * cfg.pass_gap, duration)
                reached = await self.clock.wait_until(deadline)
                if self._stopped:
                    break
                if not reached and cfg.pace is Pace.REALTIME:
                    # Playback is over; in realtime mode there is nothing left to describe.
                    break

                if cfg.pace is Pace.REALTIME:
                    if self._inflight >= cfg.max_inflight:
                        await self._skip(index, index, "previous pass still running")
                    else:
                        self._spawn(index)
                else:
                    while self._inflight >= cfg.max_inflight and not self._stopped:
                        await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
                    if self._stopped:
                        break
                    # VideoSource shares one decoder; extract in order before launching
                    # concurrent network requests.
                    try:
                        prepared = await self._prepare_pass(index)
                    except Exception as exc:
                        log.exception("pass %d failed", index)
                        await self.session.publish(
                            ErrorEvent(index=index, message=f"pass {index} failed: {exc}")
                        )
                    else:
                        self._spawn(index, prepared)
                index += 1

            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        finally:
            pending = list(self._tasks)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        if not self._stopped:
            await self.session.set_state(RunState.FINISHED, "run complete")
            log.info("scheduler finished")

    # ------------------------------------------------------------------ internals

    async def _realign(self, index: int, total: int) -> int:
        """Follow the player: skip windows it has passed, rewind when the user seeks back.

        The ``EPS`` nudge matches :meth:`MediaClock.wait_until`, which returns a hair before
        the deadline; without it a window would be recomputed as the previous one.
        """
        wanted = int((self.clock.now() + EPS) // self.config.pass_gap) + 1
        if wanted > index:
            await self._skip(index, min(wanted - 1, total), "playback moved past this window")
            return wanted
        # Only a real backward seek rewinds; drifting inside one window must not replay it.
        if wanted < index - 1:
            log.debug("player seeked back to pass %d", wanted)
            return max(1, wanted)
        return index

    def _spawn(self, index: int, prepared: tuple[Window, list[Frame]] | None = None) -> None:
        self._inflight += 1
        task = asyncio.create_task(self._run_pass(index, prepared=prepared))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _skip(self, first: int, last: int, reason: str) -> None:
        if last < first:
            return
        duration = self.video.meta.duration
        if last - first + 1 > MAX_SKIP_EVENTS:
            t_start, _ = self.config.window_for(first, duration)
            _, t_end = self.config.window_for(last, duration)
            await self.session.publish(
                PassSkippedEvent(
                    index=first,
                    t_start=t_start,
                    t_end=t_end,
                    reason=reason,
                    count=last - first + 1,
                )
            )
            return
        for index in range(first, last + 1):
            t_start, t_end = self.config.window_for(index, duration)
            await self.session.publish(
                PassSkippedEvent(index=index, t_start=t_start, t_end=t_end, reason=reason)
            )

    async def _prepare_pass(self, index: int) -> tuple[Window, list[Frame]]:
        cfg = self.config
        t_start, t_end = cfg.window_for(index, self.video.meta.duration)
        window = Window(index=index, t_start=t_start, t_end=t_end)
        frames = await asyncio.to_thread(self.video.extract_window, window, cfg.num_frames)
        await self.session.publish(
            PassStartedEvent(
                index=index,
                t_start=t_start,
                t_end=t_end,
                frame_ts=[round(frame.t, 3) for frame in frames],
            )
        )
        return window, frames

    async def _run_pass(
        self, index: int, prepared: tuple[Window, list[Frame]] | None = None
    ) -> None:
        cfg = self.config
        try:
            window, frames = prepared or await self._prepare_pass(index)
            result = await asyncio.wait_for(
                self.backend.infer(cfg.prompt, frames, window), timeout=cfg.infer_timeout
            )
            await self.session.publish(
                PassResultEvent(
                    index=index,
                    t_start=window.t_start,
                    t_end=window.t_end,
                    text=result.text,
                    latency_ms=round(result.latency_ms, 1),
                    frames_used=len(frames),
                    matched=self.session.matches(result.text),
                )
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            await self.session.publish(
                ErrorEvent(index=index, message=f"pass {index} timed out after {cfg.infer_timeout:g}s")
            )
        except Exception as exc:
            log.exception("pass %d failed", index)
            await self.session.publish(
                ErrorEvent(index=index, message=f"pass {index} failed: {exc}")
            )
        finally:
            self._inflight -= 1
