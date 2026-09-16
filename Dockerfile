FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# opencv-python-headless still links against glib
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN mkdir -p /app/vids
COPY pyproject.toml uv.lock README.md ./
COPY src src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 3000
ENTRYPOINT ["vlm-demo"]
CMD ["-i", "/app/vids", "-p", "Describe the main action briefly in 2~6 words.", "-m", "gemma-4-e4b-kinetics54K-enhanced-fall_FFT", "--backend", "openai-compat", "--host", "0.0.0.0", "--port", "3000", "--no-open", "--lock-model", "--no-delete"]
