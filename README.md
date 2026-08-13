# vlm-demo-cli

Stream a video file to a vision-language model and watch its responses appear, round by round,
on a simple web page while the video plays.

```bash
uv run vlm-demo \
  --input ./video.mp4 \
  --prompt "detect if any accident happens, if yes, say 'accident detected: <desc>', otherwise just say 'nothing happened'" \
  --model google/gemma-4-e4b-it
```

This opens <http://127.0.0.1:3000>. The page shows the video, the prompt and the backend model.
Nothing runs until you press **Start**. From then on the video plays and, in lockstep with
playback, the backend extracts a sliding window of frames and sends them to the model. Each
inference pass appends one response to the feed on the right.

With the defaults (`--window-sec 2 --num-frames 8 --pass-gap 1`) a 10-second video produces
10 passes: every second, the 8 frames sampled from the previous 2 seconds are sent to the model.

## Install

```bash
uv sync                 # runtime + dev deps
uv sync --extra gpu     # additionally: torch / torchvision / transformers, for the local backend
```

No system `ffmpeg` is required — decoding uses `opencv-python-headless`, which bundles its own
codecs.

### CUDA build

PyPI's `torch` is built for CUDA 13, which **dropped Volta (sm_70)** and needs a 580+ driver. This
repo therefore pins `torch`/`torchvision` to the CUDA 12.6 index in `pyproject.toml`
(`[[tool.uv.index]]` + `[tool.uv.sources]`), which is what the V100s here need. On other hardware,
point that block at a different `https://download.pytorch.org/whl/<backend>` index, or delete it to
fall back to PyPI. To check what you ended up with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torchvision` is required, not optional: recent image processors (Gemma 4's among them) import
`torchvision.transforms` directly, and without it `AutoProcessor.from_pretrained` fails with
`Could not import module 'Gemma4Processor'`.

## Backends

The backend is chosen automatically from your flags, or forced with `--backend`.

| Backend | When it is picked | Notes |
| --- | --- | --- |
| `mock` | `--model mock…` | No GPU, deterministic. Use it to demo/verify the whole pipeline. |
| `openai-compat` | `--base-url` is set (or `VLM_BASE_URL`) | Any `/v1/chat/completions` server: vLLM, Ollama, OpenRouter, … |
| `transformers` | anything else (a HuggingFace repo id) | Loads the model in-process with `transformers`. |

```bash
# 1. no model needed — proves the timeline, the websocket and the UI
uv run vlm-demo -i ./video.mp4 -p "detect if any accident happens…" \
  -m mock --mock-detect-at 6 --mock-latency 0.3

# 2. against a served model
uv run vlm-demo -i ./video.mp4 -p "…" \
  -m Qwen/Qwen2.5-VL-7B-Instruct --base-url http://localhost:8000/v1

# 3. locally, in-process
uv sync --extra gpu
uv run vlm-demo -i ./video.mp4 -p "…" -m google/gemma-4-e4b-it --backend transformers
```

## Pacing

Real models are not always faster than the video. `--pace` decides what happens when an
inference pass takes longer than `--pass-gap`:

- `realtime` (default) — windows whose deadline has already slipped are **skipped** (and shown as
  a muted marker in the feed). Responses always describe what is on screen right now.
- `complete` — nothing is skipped, so you always get `ceil(duration / pass_gap)` responses, but
  the feed can lag behind playback.

Measured on a V100 with `google/gemma-4-E2B-it`, fp16, 4 frames at 512px: ~3s for the first pass
then ~0.75s each. Bigger models, more frames or larger frames push past the default 1s gap, so on
this class of hardware raise `--pass-gap` (2–3s is comfortable) or switch to `--pace complete`.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input, -i` | — | Video file to play and analyse. |
| `--prompt, -p` | — | Prompt sent with every window. |
| `--model, -m` | — | Model id. |
| `--backend` | auto | `mock`, `openai-compat` or `transformers`. |
| `--base-url` / `--api-key` | `$VLM_BASE_URL` / `$VLM_API_KEY` | For `openai-compat`. |
| `--window-sec` | `2.0` | Length of the frame window, in video seconds. |
| `--num-frames` | `8` | Frames sampled per window. |
| `--pass-gap` | `1.0` | Video seconds between inference passes. |
| `--pace` | `realtime` | `realtime` or `complete`. |
| `--max-inflight` | `1` | Concurrent inference passes (`realtime` only). |
| `--host` / `--port` | `127.0.0.1` / `3000` | Where the web UI is served. |
| `--open / --no-open` | `--open` | Open the browser automatically. |
| `--frame-max-size` | `512` | Longest side of each frame, in pixels. |
| `--jpeg-quality` | `80` | JPEG quality for encoded frames. |
| `--infer-timeout` | `30.0` | Per-pass timeout, in seconds. |
| `--max-tokens` / `--temperature` | `128` / `0.0` | Generation settings. |
| `--highlight-regex` | `(?i)detect` | Responses matching this are highlighted. |
| `--dump-frames` | — | Directory to write every sent frame to, for debugging. |
| `--mock-detect-at` / `--mock-latency` | `none` / `0.2` | `mock` backend behaviour. |
| `--log-level` | `info` | Server log level. |

## How it works

```
CLI (typer) ──▶ RunConfig ──▶ FastAPI app ──▶ uvicorn
                                  │
 browser ◀── WS events ── Session ◀── Scheduler ──▶ VideoSource ──▶ frames (JPEG)
         ── clock/control ─▶                 └────▶ VLMBackend.infer(prompt, frames)
```

The **browser is the clock**. The `<video>` element reports `currentTime` over the websocket
several times a second; the scheduler extrapolates between heartbeats, so pausing the video pauses
inference and seeking moves the analysis with it. Pass `k` fires at video time `k * pass_gap` and
covers `[k * pass_gap - window_sec, k * pass_gap]`.

Layout:

```
src/vlm_demo/
  cli.py          typer entrypoint
  config.py       RunConfig + validation
  events.py       websocket wire contract (pydantic)
  video.py        decoding and window sampling
  scheduler.py    media clock + pacing policies
  session.py      run state, event fan-out, replay buffer
  server.py       FastAPI routes and /ws
  backends/       base protocol, registry, mock / openai_compat / transformers_local
  web/            index.html, app.js, styles.css (no build step)
```

## Tests

```bash
uv run pytest
```

The suite synthesises a small clip on the fly, so no fixture video is committed.
