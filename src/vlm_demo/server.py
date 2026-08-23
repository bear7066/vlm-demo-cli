"""FastAPI app: serves the page, the video library, and the event websocket."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import mimetypes
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vlm_demo.backends.registry import create_backend
from vlm_demo.config import BackendKind, RunConfig
from vlm_demo.events import ErrorEvent, LibraryEvent, LibraryVideo, RunState, dump
from vlm_demo.library import LibraryError, UploadTooLarge, VideoLibrary
from vlm_demo.models import cached_model_ids
from vlm_demo.scheduler import MediaClock, Scheduler
from vlm_demo.session import Session
from vlm_demo.video import VideoSource

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
CHUNK_SIZE = 512 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class ModelSwitchDisabled(RuntimeError):
    """Raised when the page asks for another model under ``--lock-model``."""


class SelectRequest(BaseModel):
    """``POST /api/select`` — which video in the library to analyse next."""

    name: str


class ModelRequest(BaseModel):
    """``POST /api/model`` — which model to load next; any id the backend accepts."""

    name: str


class AppState:
    """Everything the routes need, hung off ``app.state.run``.

    One process serves one prompt, but neither the *video* nor the *model* is fixed. The video
    is whichever file of the ``--input`` directory the page has selected (:meth:`select`); the
    model is whichever id the page last asked for (:meth:`set_model`). Both tear the old run
    down and start a fresh one, and both hold :attr:`_rebuild_lock` while they do, so they
    cannot interleave. The *backend kind* is settled at startup and never changes.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.library = VideoLibrary(
            config.input,
            max_upload_bytes=config.max_upload_bytes,
            allow_upload=config.allow_upload,
            allow_delete=config.allow_delete,
        )
        self.backend = create_backend(config)
        # Only the local backend loads weights from the HuggingFace cache, so it is the only one
        # we can offer a list for; everywhere else the page takes a typed-in id and nothing more.
        self.models = cached_model_ids() if config.backend is BackendKind.TRANSFORMERS else []
        self.session = Session(config, self.backend.name, available_models=self.models)
        self.video: VideoSource | None = None
        self.selected: str | None = None
        self.clock: MediaClock | None = None
        self.scheduler: Scheduler | None = None
        self.scheduler_task: asyncio.Task[None] | None = None
        self.backend_state = RunState.LOADING
        self.backend_detail = f"preparing {self.backend.name} backend"
        self.prepare_task: asyncio.Task[None] | None = None
        self._rebuild_lock = asyncio.Lock()

    # ------------------------------------------------------------------ run state

    def run_state(self) -> tuple[RunState, str]:
        """The state a freshly (re)targeted session should sit in, and why.

        Backend readiness and video selection are independent; the page can only start when
        both are settled.
        """
        if self.backend_state in (RunState.ERROR, RunState.LOADING):
            return self.backend_state, self.backend_detail
        if self.video is None:
            return RunState.READY, "pick a video to analyse"
        return RunState.READY, "press Start to begin"

    def library_event(self) -> LibraryEvent:
        return LibraryEvent(
            directory=str(self.library.root),
            videos=[
                LibraryVideo(name=e.name, size_bytes=e.size_bytes, modified=e.modified)
                for e in self.library.entries()
            ],
            selected=self.selected,
            uploads_enabled=self.config.allow_upload,
            deletes_enabled=self.config.allow_delete,
            max_upload_mb=self.config.max_upload_mb,
        )

    async def publish_library(self) -> None:
        """Tell every open page what is in the directory now.

        Not recorded in the history: connections are sent the current listing anyway, and a
        replayed stale one would fight with it.
        """
        await self.session.publish(self.library_event(), record=False)

    # ------------------------------------------------------------------ the backend

    def start_prepare(self) -> None:
        """Warm the current backend up in the background; the page watches the status for it."""
        self.prepare_task = asyncio.create_task(_prepare_backend(self))

    async def stop_prepare(self) -> None:
        """Drop the warm-up task, if any, and wait for it to stop touching the backend."""
        task, self.prepare_task = self.prepare_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def set_model(self, name: str) -> None:
        """Load ``name`` instead of the current model, keeping the video and everything else.

        The backend *kind* is not re-inferred — a run started against vLLM stays on vLLM, this
        only changes which model id it is pointed at. Whether ``name`` exists is not checked
        here: :meth:`~vlm_demo.backends.base.VLMBackend.prepare` is what loads it, and a bad id
        surfaces on the page exactly as a bad ``--model`` does at startup.
        """
        if self.config.lock_model:
            raise ModelSwitchDisabled("model switching is disabled (--lock-model)")
        name = name.strip()
        if not name:
            raise ValueError("model must not be empty")
        if name == self.config.model:
            return
        async with self._rebuild_lock:
            # Nothing else may hold the outgoing backend while we close it: a Scheduler keeps
            # a reference to it, and a warm-up in flight would report its readiness as ours.
            await self.stop_run()
            await self.stop_prepare()
            previous, self.config = self.backend, replace(self.config, model=name)
            self.session.config = self.config
            self.backend = create_backend(self.config)
            await previous.aclose()
            self.backend_state = RunState.LOADING
            self.backend_detail = f"loading {name}"
            # Same video, fresh feed: every response in the history came from the old model.
            await self.session.retarget(self.session.video, *self.run_state())
        self.start_prepare()

    # ------------------------------------------------------------------ selection

    async def select(self, name: str | None) -> None:
        """Make ``name`` the video under analysis (``None`` clears the selection).

        Raises :class:`LibraryError` if the name is not a video in the directory, and
        :class:`~vlm_demo.video.VideoError` if it cannot be decoded — in both cases the
        current selection is left untouched.
        """
        async with self._rebuild_lock:
            source: VideoSource | None = None
            meta = None
            if name is not None:
                path = self.library.resolve(name)
                source = VideoSource(
                    path,
                    max_size=self.config.frame_max_size,
                    jpeg_quality=self.config.jpeg_quality,
                    dump_dir=self.config.dump_frames,
                )
                meta = await asyncio.to_thread(source.open)

            await self.stop_run()
            previous, self.video, self.selected = self.video, source, name
            self.clock = MediaClock(meta.duration) if meta is not None else None
            if previous is not None:
                await asyncio.to_thread(previous.close)

            if meta is not None:
                log.info(
                    "%s: %.2fs, %.2f fps, %dx%d → %d passes",
                    meta.path.name,
                    meta.duration,
                    meta.fps,
                    meta.width,
                    meta.height,
                    self.config.total_passes(meta.duration),
                )
            await self.session.retarget(meta, *self.run_state())
            await self.publish_library()

    async def delete(self, name: str) -> None:
        """Remove ``name`` from the directory, and move on if it was the one being analysed.

        Deleting the selection drops it first — that stops the run and closes the file — and
        then falls through to whatever is left, so the page never points at a video that is
        no longer there.
        """
        target = self.library.resolve(name)
        if self.selected is not None and self.library.resolve(self.selected) == target:
            await self.select(None)
            self.library.delete(name)
            with contextlib.suppress(Exception):
                await self.select(self.library.default_name())
        else:
            self.library.delete(name)
        await self.publish_library()

    async def stop_run(self) -> None:
        """Halt the pass loop, if any, and wait for it to let go of the video."""
        if self.scheduler is not None:
            self.scheduler.stop()
        task, self.scheduler_task, self.scheduler = self.scheduler_task, None, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def require_video(self) -> VideoSource:
        if self.video is None:
            raise HTTPException(status_code=409, detail="no video selected")
        return self.video


def create_app(config: RunConfig) -> FastAPI:
    state = AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Start on the first video in the directory, so the common case (a folder of clips)
        # behaves exactly like the old single-file flag. An empty folder is fine: the page
        # shows the uploader instead.
        default = state.library.default_name()
        if default is not None:
            try:
                await state.select(default)
            except Exception:
                log.exception("could not open %s; starting with no video selected", default)
        else:
            log.info("%s holds no videos yet — upload one from the page", state.library.root)
        state.start_prepare()
        try:
            yield
        finally:
            await state.stop_prepare()
            await state.stop_run()
            await state.backend.aclose()
            if state.video is not None:
                await asyncio.to_thread(state.video.close)

    app = FastAPI(title="vlm-demo", lifespan=lifespan)
    app.state.run = state
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/config")
    async def api_config() -> dict[str, Any]:
        return dump(state.session.describe())

    @app.get("/api/events")
    async def api_events() -> dict[str, Any]:
        return {"state": state.session.state.value, "events": state.session.history()}

    @app.get("/api/library")
    async def api_library() -> dict[str, Any]:
        return dump(state.library_event())

    @app.post("/api/select")
    async def api_select(body: SelectRequest) -> dict[str, Any]:
        try:
            await state.select(body.name)
        except LibraryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"cannot open {body.name}: {exc}") from exc
        return dump(state.session.describe())

    @app.post("/api/model")
    async def api_model(body: ModelRequest) -> dict[str, Any]:
        """Point the run at another model. The page gets the new session on the websocket."""
        try:
            await state.set_model(body.name)
        except ModelSwitchDisabled as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return dump(state.session.describe())

    @app.post("/api/videos", status_code=201)
    async def api_upload(
        request: Request,
        name: str = Query(..., description="File name the upload should be stored under."),
    ) -> dict[str, Any]:
        """Store one uploaded video in the ``--input`` directory.

        The body is the raw file, streamed straight to disk — one request per file, which
        keeps memory flat and gives the page a progress bar per upload.
        """
        if not state.config.allow_upload:
            raise HTTPException(status_code=403, detail="uploads are disabled (--no-upload)")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > state.config.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds the {state.config.max_upload_mb:g} MB limit",
            )
        try:
            entry = await state.library.save(name, request.stream())
        except UploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except LibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            log.exception("could not store upload %r", name)
            raise HTTPException(status_code=500, detail=f"could not store upload: {exc}") from exc

        # Nothing was playable before, so the upload becomes the selection; otherwise leave
        # whatever the user is watching alone.
        if state.selected is None:
            with contextlib.suppress(Exception):
                await state.select(entry.name)
        await state.publish_library()
        return {
            "video": dump(
                LibraryVideo(
                    name=entry.name, size_bytes=entry.size_bytes, modified=entry.modified
                )
            ),
            "selected": state.selected,
        }

    @app.delete("/api/videos/{name}")
    async def api_delete(name: str) -> dict[str, Any]:
        """Remove one video from the ``--input`` directory. This deletes the file."""
        if not state.config.allow_delete:
            raise HTTPException(status_code=403, detail="deleting is disabled (--no-delete)")
        try:
            await state.delete(name)
        except LibraryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            log.exception("could not delete %r", name)
            raise HTTPException(status_code=500, detail=f"could not delete: {exc}") from exc
        return {"deleted": name, "selected": state.selected}

    @app.get("/api/video")
    async def api_video(request: Request) -> Any:
        return video_response(state.require_video().path, request.headers.get("range"))

    @app.get("/api/video/{name}")
    async def api_video_named(name: str, request: Request) -> Any:
        try:
            path = state.library.resolve(name)
        except LibraryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return video_response(path, request.headers.get("range"))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await _websocket_endpoint(websocket, state)

    return app


# ---------------------------------------------------------------------- backend warm-up


async def _prepare_backend(state: AppState) -> None:
    session = state.session
    await session.set_state(*state.run_state())
    try:
        await state.backend.prepare()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("backend preparation failed")
        state.backend_state, state.backend_detail = RunState.ERROR, str(exc)
        await session.publish(ErrorEvent(message=f"backend failed to load: {exc}"))
        await session.set_state(*state.run_state())
        return
    state.backend_state, state.backend_detail = RunState.READY, ""
    for note in state.backend.notes:
        await session.publish(ErrorEvent(message=note))
    await session.set_state(*state.run_state())


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
    """Serve one video from the library, honouring ``Range`` so the player can seek."""
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
    session = state.session
    queue = session.subscribe()
    try:
        await websocket.send_json(dump(session.describe()))
        await websocket.send_json(dump(state.library_event()))
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
    session = state.session
    clock = state.clock
    if clock is None:  # no video selected: there is no timeline to drive
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
    session = state.session
    if session.state is RunState.ERROR:
        await session.publish(ErrorEvent(message="backend is unavailable; see the server log"))
        return
    if session.state is RunState.LOADING:
        await session.publish(
            ErrorEvent(message="the backend is still loading — press Start again in a moment")
        )
        return
    if state.video is None or state.clock is None:
        await session.publish(ErrorEvent(message="pick a video from the list first"))
        return
    scheduler = Scheduler(state.config, session, state.video, state.backend, state.clock)
    state.scheduler = scheduler
    state.scheduler_task = asyncio.create_task(scheduler.run())
