#!/usr/bin/env bash
# Bring up vLLM + the demo app; the UI appears on http://<host>:3000.
# Knobs (env): MODEL, VIDS (default ./vids), PROMPT, VLLM_IMAGE, HF_TOKEN.
# Which clip is analysed is chosen on the page, not here.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "${VIDS:-./vids}"
# The finetune checkpoint alone is unservable (missing KV-shared layer tensors);
# build the merged one first if it isn't there yet.
if [ -z "${MODEL_PATH:-}" ] && [ ! -d models/gemma-4-e2b-fall-merged ]; then
    uv run python scripts/merge-base.py
fi
exec docker compose up --build "$@"
