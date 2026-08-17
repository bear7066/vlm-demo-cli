#!/usr/bin/env bash
# Serve a model with vLLM in Docker, wait until it answers, then warm it up.
#
#   ./scripts/run-vllm.sh [model]           # default: the fall-detection Gemma finetune
#   VLLM_IMAGE=... VLLM_PORT=... ./scripts/run-vllm.sh
set -euo pipefail

MODEL="${1:-THChou1220/gemma-4-e2b-kinetics54K-enhanced-fall_FFT}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:gemma4-cu130}"   # local arm64 build for the DGX Spark
PORT="${VLLM_PORT:-8000}"
NAME="${VLLM_NAME:-vlm-demo-vllm}"

# Serve the merged checkpoint when one exists (the raw gemma-4-e2b finetune is
# missing its KV-shared layers' tensors; scripts/merge-base.py rebuilds them).
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVE_ARGS=("$MODEL")
if [ -d "$REPO_DIR/models/gemma-4-e2b-fall-merged" ] && [[ "$MODEL" == *gemma-4-e2b* ]]; then
    SERVE_ARGS=(/models/gemma-4-e2b-fall-merged --served-model-name "$MODEL")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc host \
    -p "$PORT:8000" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$REPO_DIR/models:/models:ro" \
    -e HF_TOKEN \
    "$IMAGE" "${SERVE_ARGS[@]}" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 16, "video": 0}'

echo "waiting for vLLM on port $PORT (first run downloads + loads weights) ..."
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    if [ -z "$(docker ps -q -f "name=^$NAME$")" ]; then
        echo "vLLM container died:" >&2
        docker logs "$NAME" 2>&1 | tail -30 >&2
        exit 1
    fi
    sleep 2
done

echo "warming up ..."
# ponytail: 1x1 JPEG exercises the multimodal path; the app re-warms at real frame size.
JPEG="/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AP7+KKKK/9k="
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\", \"max_tokens\": 8, \"messages\": [{\"role\": \"user\", \"content\": [
          {\"type\": \"text\", \"text\": \"Describe the image.\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$JPEG\"}}]}]}" \
    >/dev/null

echo "vLLM ready: http://127.0.0.1:$PORT/v1  (model: $MODEL)"
echo "run:  uv run vlm-demo --backend vllm -m \"$MODEL\" ..."
