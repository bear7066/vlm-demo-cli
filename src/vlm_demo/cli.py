"""`vlm-demo` — the command line entry point."""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from vlm_demo.config import (
    DEFAULT_HIGHLIGHT_REGEX,
    BackendKind,
    ConfigError,
    Pace,
    RunConfig,
    resolve_backend_kind,
)
from vlm_demo.library import VideoLibrary
from vlm_demo.server import create_app

app = typer.Typer(
    add_completion=False,
    help="Stream a video to a vision-language model and watch its responses live.",
)

LOOPBACK = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


@app.command()
def run(
    input: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory of videos to choose from on the page."),
    ],
    prompt: Annotated[
        str, typer.Option("--prompt", "-p", help="Prompt sent with every window of frames.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model id, e.g. a HuggingFace repo id.")
    ],
    backend: Annotated[
        BackendKind | None,
        typer.Option("--backend", help="Force a backend instead of inferring one."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", envvar="VLM_BASE_URL", help="OpenAI-compatible endpoint."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="VLM_API_KEY", help="Bearer token for --base-url."),
    ] = None,
    window_sec: Annotated[
        float, typer.Option("--window-sec", help="Seconds of video each window covers.")
    ] = 2.0,
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Frames sampled per window.")
    ] = 8,
    pass_gap: Annotated[
        float, typer.Option("--pass-gap", help="Video seconds between inference passes.")
    ] = 1.0,
    pace: Annotated[
        Pace,
        typer.Option("--pace", help="realtime: skip late windows. complete: run every window."),
    ] = Pace.REALTIME,
    max_inflight: Annotated[
        int, typer.Option("--max-inflight", help="Concurrent passes (realtime pace only).")
    ] = 1,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port for the web UI.")] = 3000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the browser on startup.")
    ] = True,
    frame_max_size: Annotated[
        int, typer.Option("--frame-max-size", help="Longest side of each frame, in pixels.")
    ] = 512,
    jpeg_quality: Annotated[
        int, typer.Option("--jpeg-quality", help="JPEG quality of encoded frames (1-100).")
    ] = 80,
    infer_timeout: Annotated[
        float, typer.Option("--infer-timeout", help="Per-pass timeout in seconds.")
    ] = 30.0,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Generation length limit.")
    ] = 128,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature; 0 is greedy.")
    ] = 0.0,
    allow_upload: Annotated[
        bool,
        typer.Option("--upload/--no-upload", help="Let the page add videos to --input."),
    ] = True,
    max_upload_mb: Annotated[
        float, typer.Option("--max-upload-mb", help="Size limit for one uploaded video.")
    ] = 1024.0,
    highlight_regex: Annotated[
        str, typer.Option("--highlight-regex", help="Responses matching this are highlighted.")
    ] = DEFAULT_HIGHLIGHT_REGEX,
    dump_frames: Annotated[
        Path | None,
        typer.Option("--dump-frames", help="Write every frame sent to the model here."),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", help="Server log level.")] = "info",
    mock_detect_at: Annotated[
        float | None,
        typer.Option("--mock-detect-at", help="mock backend: 'detect' from this second on."),
    ] = None,
    mock_latency: Annotated[
        float, typer.Option("--mock-latency", help="mock backend: fake seconds per pass.")
    ] = 0.2,
) -> None:
    """Serve the demo page for a directory of videos + prompt + model."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = RunConfig(
            input=input,
            prompt=prompt,
            model=model,
            backend=resolve_backend_kind(backend, model, base_url),
            base_url=base_url,
            api_key=api_key,
            window_sec=window_sec,
            num_frames=num_frames,
            pass_gap=pass_gap,
            pace=pace,
            max_inflight=max_inflight,
            host=host,
            port=port,
            open_browser=open_browser,
            frame_max_size=frame_max_size,
            jpeg_quality=jpeg_quality,
            infer_timeout=infer_timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_upload=allow_upload,
            max_upload_mb=max_upload_mb,
            highlight_regex=highlight_regex,
            dump_frames=dump_frames,
            log_level=log_level,
            mock_detect_at=mock_detect_at,
            mock_latency=mock_latency,
        )
    except ConfigError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    found = len(VideoLibrary(config.input, max_upload_bytes=config.max_upload_bytes).entries())
    typer.secho(f"\n  videos  {config.input}", bold=True)
    typer.echo(
        f"          {found} found"
        + ("" if config.allow_upload else "; uploads disabled")
        + (" — upload one from the page" if not found and config.allow_upload else "")
    )
    typer.echo(f"  model   {config.model}  (backend: {config.backend.value})")
    typer.echo(
        f"  passes  every {config.pass_gap:g}s, {config.num_frames} frames "
        f"from the last {config.window_sec:g}s, pace={config.pace.value}"
    )
    typer.secho(f"  open    {url}\n", fg=typer.colors.GREEN, bold=True)

    if config.open_browser and host in LOOPBACK:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(config), host=host, port=port, log_level=log_level.lower())


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
