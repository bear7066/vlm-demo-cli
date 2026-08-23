# vlm-demo-cli

Stream a video to a vision-language model and watch its responses appear, round by round, on a
simple web page while the video plays.

```bash
uv run vlm-demo \
  --input ./vids \
  --prompt "detect if any accident happens, if yes, say 'accident detected: <desc>', otherwise just say 'nothing happened'" \
  --model google/gemma-4-e4b-it
```

This opens <http://127.0.0.1:3000>. `--input` is a **folder of videos**: the page lists what is
in it, you pick one (and can upload more — see [Choosing a video](#choosing-a-video)), and it
shows that video alongside the prompt and the backend model.

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
| `vllm` | `--backend vllm` | `openai-compat` plus: waits for the server, then runs a warmup pass so the first window is fast. `--base-url` defaults to `http://localhost:8000`. |
| `transformers` | anything else (a HuggingFace repo id) | Loads the model in-process with `transformers`. |

```bash
# 1. no model needed — proves the timeline, the websocket and the UI
uv run vlm-demo -i ./vids -p "detect if any accident happens…" \
  -m mock --mock-detect-at 6 --mock-latency 0.3

# 2. against a served model
uv run vlm-demo -i ./vids -p "…" \
  -m Qwen/Qwen2.5-VL-7B-Instruct --base-url http://localhost:8000/v1

# 3. locally, in-process
uv sync --extra gpu
uv run vlm-demo -i ./vids -p "…" -m google/gemma-4-e4b-it --backend transformers
```

## vLLM in Docker

```bash
# just the model server (waits for readiness, sends a warmup request):
./scripts/run-vllm.sh [model]
uv run vlm-demo -i ./vids -p "…" --backend vllm \
  -m THChou1220/gemma-4-e2b-kinetics54K-enhanced-fall_FFT

# the whole stack: vLLM + this app on a private compose network, UI on port 3000
./scripts/run-stack.sh          # = docker compose up --build
```

The gemma-4-e2b finetune checkpoint can't be served as-is: transformers deduplicates the tied
KV tensors of the last `num_kv_shared_layers` layers on save, and vLLM then fails with
*"Following weights were not initialized from checkpoint"*. `scripts/merge-base.py` copies
those tensors back from `google/gemma-4-E2B-it` into `models/gemma-4-e2b-fall-merged/`;
`run-stack.sh` runs it automatically, and both scripts serve the merged copy under the
original model name so `-m` stays the same.

Both are tuned via env vars — `MODEL`, `VIDS` (the video folder, default `./vids`), `PROMPT`,
`VLLM_IMAGE` (default: `vllm/vllm-openai:gemma4-cu130`, the local DGX Spark build), `HF_TOKEN`.
Which clip runs is no longer a launch flag: the page picks it. `./vids` is mounted read-write so
uploads from the page land on the host — files the container creates are owned by root, so
`sudo chown` them if that gets in your way, or pass `--no-upload` / `--no-delete` in the
compose command. The `vllm`
backend polls `/v1/models` until the server is up (weights can take minutes to load) and then
sends one dummy request with `--num-frames` grey frames at `--frame-max-size`, so processor
init and CUDA graph capture are paid before your first real window.

## Choosing a video

`--input` is a directory, and the page's **videos** card lists every video directly inside it
(`.mp4 .m4v .mov .mkv .webm .avi .mpg .mpeg .ogv .wmv`; sub-folders are ignored). The first one
is selected at startup, and clicking another switches to it: the pass loop stops, the feed and
the timeline are cleared, and the new clip is loaded with the same prompt and model. Every open
page follows along, since the switch is broadcast over the websocket.

**Add your own** with *Add video…*, or by dropping files onto the card — several at once is
fine, and they upload one by one with a progress bar. Uploads are stored in the same `--input`
directory, so they survive a restart and are just another entry in the list. Names are reduced
to a plain file name (no directories, no surprises), and an upload never overwrites an existing
video — `clip.mp4` arriving twice becomes `clip.mp4` and `clip-1.mp4`. Only the extensions above
are accepted, and `--max-upload-mb` (default 1024) caps a single file.

**Delete one** with the `×` beside its name. This removes the file from `--input` for good, so
the page asks first. Deleting the clip you are watching stops the run and moves the selection to
the next video in the list (or to nothing, if that was the last one); deleting any other clip just
drops it from the list and leaves playback alone. Every open page follows along, as with a switch.

If the folder starts out empty, the page opens with the uploader and nothing selected; the first
video you add becomes the selection. Pass `--no-upload` to make the directory read-only from the
page — the list still works, only adding is refused — and `--no-delete` to hide the `×` and refuse
removals.

## Choosing a model

`--model` says which model to *start* on; the box in the side panel switches to another without
restarting the process. Type any id the backend accepts — a HuggingFace repo id such as
`THChou1220/gemma-4-e2b-kinetics54K-enhanced-fall_FFT` for the `transformers` backend — and press
Load (or Enter).

Under `transformers` the box also suggests what is already in your HuggingFace cache
(`~/.cache/huggingface/hub/models--*`, or wherever `$HF_HUB_CACHE` / `$HF_HOME` point), so
previously downloaded models are one click away. Anything *not* in that list still loads normally
— it is a list of suggestions, not a whitelist, and an uncached id is simply downloaded. The other
backends have no local cache to read, so they offer no suggestions and take a typed id only.

Switching stops the run, frees the old model (VRAM included) and loads the new one; the status
pill goes back to `loading` and the feed is cleared, because every response in it came from the
previous model. The video you had selected stays selected. A model that does not exist fails the
same way a bad `--model` does at startup: the pill turns red and the reason lands in the feed.

The *backend* is settled at launch and does not change — switching only re-points the run at
another model id.

Pass `--lock-model` for demos where the model must not move: the box becomes read-only and
`POST /api/model` answers 403.

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
| `--input, -i` | — | Directory of videos; the page picks which one to analyse. |
| `--prompt, -p` | — | Prompt sent with every window. |
| `--model, -m` | — | Model id to start on; the page can switch to another. |
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
| `--upload / --no-upload` | `--upload` | Let the page add videos to `--input`. |
| `--delete / --no-delete` | `--delete` | Let the page remove videos from `--input`. |
| `--max-upload-mb` | `1024` | Size limit for one uploaded video. |
| `--lock-model / --no-lock-model` | `--no-lock-model` | Pin `--model`; the page cannot change it. |
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

Prompt and pacing are fixed for the process; the *video* and the *model* are not. Picking a
video (`POST /api/select`) or uploading one (`POST /api/videos?name=…`, the raw file as the body)
re-points the session at a new `VideoSource` and announces it to every page. Deleting one
(`DELETE /api/videos/{name}`) unlinks the file, and re-points the session if it was the video
being analysed. `POST /api/model` swaps the backend for one built around another model id, closing
the old one first; both paths hold the same lock, so they cannot interleave.

Layout:

```
src/vlm_demo/
  cli.py          typer entrypoint
  config.py       RunConfig + validation
  library.py      the --input directory: listing, name safety, uploads, deletes
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
