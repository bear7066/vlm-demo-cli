"""FastAPI app: serves the page, streams the video, and carries the event websocket."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vlm_demo.backends.registry import create_backend
from vlm_demo.config import RunConfig
from vlm_demo.events import ErrorEvent, RunState, dump
from vlm_demo.scheduler import MediaClock, Scheduler
from vlm_demo.session import Session
from vlm_demo.video import VideoSource

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
CHUNK_SIZE = 512 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class AppState:
    """Everything the routes need, hung off ``app.state.run``."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.video = VideoSource(
            config.input,
            max_size=config.frame_max_size,
            jpeg_quality=config.jpeg_quality,
            dump_dir=config.dump_frames,
        )
        self.backend = create_backend(config)
        self.session: Session | None = None
        self.clock: MediaClock | None = None
        self.scheduler: Scheduler | None = None
        self.scheduler_task: asyncio.Task[None] | None = None

    def require_session(self) -> Session:
        if self.session is None:  # pragma: no cover - lifespan always sets it
            raise HTTPException(status_code=503, detail="session not ready")
        return self.session


def create_app(config: RunConfig) -> FastAPI:
    state = AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        meta = await asyncio.to_thread(state.video.open)
        state.session = Session(config, meta, state.backend.name)
        state.clock = MediaClock(meta.duration)
        log.info(
            "%s: %.2fs, %.2f fps, %dx%d → %d passes",
            meta.path.name,
            meta.duration,
            meta.fps,
            meta.width,
            meta.height,
            config.total_passes(meta.duration),
        )
        prepare = asyncio.create_task(_prepare_backend(state))
        try:
            yield
        finally:
            prepare.cancel()
            if state.scheduler is not None:
                state.scheduler.stop()
            if state.scheduler_task is not None:
                state.scheduler_task.cancel()
            await state.backend.aclose()
            await asyncio.to_thread(state.video.close)

    app = FastAPI(title="vlm-demo", lifespan=lifespan)
    app.state.run = state
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/config")
    async def api_config() -> dict[str, Any]:
        return dump(state.require_session().describe())

    @app.get("/api/events")
    async def api_events() -> dict[str, Any]:
        session = state.require_session()
        return {"state": session.state.value, "events": session.history()}

    @app.get("/api/video")
    async def api_video(request: Request) -> Any:
        return video_response(config.input, request.headers.get("range"))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await _websocket_endpoint(websocket, state)

    return app


# ---------------------------------------------------------------------- backend warm-up


async def _prepare_backend(state: AppState) -> None:
    session = state.require_session()
    await session.set_state(RunState.LOADING, f"preparing {state.backend.name} backend")
    try:
        await state.backend.prepare()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("backend preparation failed")
        await session.publish(ErrorEvent(message=f"backend failed to load: {exc}"))
        await session.set_state(RunState.ERROR, str(exc))
        return
    for note in state.backend.notes:
        await session.publish(ErrorEvent(message=note))
    await session.set_state(RunState.READY, "press Start to begin")


# ---------------------------------------------------------------------- video streaming


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``Range`` header into inclusive ``(start, end)`` offsets."""
    if not header:
        return None
    match = RANGE_RE.fullmatch(header.strip())
    if not match:
        return None
    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    elif raw_end:  # suffix range: last N bytes
        start = max(0, size - int(raw_end))
        end = size - 1
    else:
        return None
    end = min(end, size - 1)
    if start > end or start >= size:
        raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    return start, end


def video_response(path: Path, range_header: str | None) -> Any:
    """Serve the input video, honouring ``Range`` so the player can seek."""
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    span = parse_range(range_header, size)
    if span is None:
        return FileResponse(
            path, media_type=media_type, headers={"Accept-Ranges": "bytes"}
        )

    start, end = span
    length = end - start + 1

    def body() -> Iterator[bytes]:
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        body(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
            "Accept-Ranges": "bytes",
        },
    )


# ---------------------------------------------------------------------- websocket


async def _websocket_endpoint(websocket: WebSocket, state: AppState) -> None:
    await websocket.accept()
    session = state.require_session()
    queue = session.subscribe()
    try:
        await websocket.send_json(dump(session.describe()))
        for event in session.history():
            await websocket.send_json(event)

        sender = asyncio.create_task(_send_loop(websocket, queue))
        receiver = asyncio.create_task(_receive_loop(websocket, state))
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                log.debug("websocket task ended with %r", exc)
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)


async def _send_loop(websocket: WebSocket, queue: asyncio.Queue[dict[str, Any]]) -> None:
    while True:
        event = await queue.get()
        await websocket.send_json(event)


async def _receive_loop(websocket: WebSocket, state: AppState) -> None:
    while True:
        message = await websocket.receive_json()
        await handle_client_message(message, state)


async def handle_client_message(message: dict[str, Any], state: AppState) -> None:
    """Apply one client → server control message. Unknown types are ignored."""
    session = state.require_session()
    clock = state.clock
    if clock is None:  # pragma: no cover - lifespan always sets it
        return
    kind = message.get("type")
    t = float(message.get("t", clock.now()) or 0.0)

    match kind:
        case "start":
            clock.stopped = False
            clock.update(t, playing=True)
            await _ensure_scheduler(state)
        case "clock":
            clock.update(t, playing=bool(message.get("playing", True)))
        case "pause":
            clock.update(t, playing=False)
            if session.state is RunState.RUNNING:
                await session.set_state(RunState.PAUSED, "playback paused")
        case "resume":
            clock.update(t, playing=True)
            if session.state is RunState.PAUSED:
                await session.set_state(RunState.RUNNING)
        case "seek":
            clock.update(t, playing=bool(message.get("playing", False)))
        case "ended":
            clock.mark_ended()
        case "stop":
            clock.stop()
            if state.scheduler is not None:
                state.scheduler.stop()
        case _:
            log.debug("ignoring client message %r", kind)


async def _ensure_scheduler(state: AppState) -> None:
    """Start a pass loop if one is not already running (Start can be pressed again)."""
    if state.scheduler_task is not None and not state.scheduler_task.done():
        return
    session = state.require_session()
    if session.state is RunState.ERROR:
        await session.publish(ErrorEvent(message="backend is unavailable; see the server log"))
        return
    if session.state is RunState.LOADING:
        await session.publish(
            ErrorEvent(message="the backend is still loading — press Start again in a moment")
        )
        return
    assert state.clock is not None
    scheduler = Scheduler(state.config, session, state.video, state.backend, state.clock)
    state.scheduler = scheduler
    state.scheduler_task = asyncio.create_task(scheduler.run())
