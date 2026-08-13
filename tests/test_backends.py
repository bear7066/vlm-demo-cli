from __future__ import annotations

import base64
import json

import httpx
import pytest

from vlm_demo.backends.base import BackendError
from vlm_demo.backends.mock import NOTHING, MockBackend
from vlm_demo.backends.openai_compat import OpenAICompatBackend, normalise_base_url
from vlm_demo.backends.registry import create_backend
from vlm_demo.config import BackendKind, ConfigError, RunConfig, resolve_backend_kind
from vlm_demo.video import Frame, Window


def make_frames(n: int = 3) -> list[Frame]:
    return [Frame(index=i, t=float(i), jpeg=b"\xff\xd8jpeg", width=64, height=48) for i in range(n)]


# ---------------------------------------------------------------- selection


@pytest.mark.parametrize(
    ("explicit", "model", "base_url", "expected"),
    [
        (None, "mock", None, BackendKind.MOCK),
        (None, "google/gemma-4-e4b-it", None, BackendKind.TRANSFORMERS),
        (None, "google/gemma-4-e4b-it", "http://x/v1", BackendKind.OPENAI_COMPAT),
        ("mock", "google/gemma-4-e4b-it", "http://x/v1", BackendKind.MOCK),
    ],
)
def test_backend_selection(explicit, model, base_url, expected):
    assert resolve_backend_kind(explicit, model, base_url) is expected


def test_unknown_backend_is_rejected():
    with pytest.raises(ConfigError, match="unknown --backend"):
        resolve_backend_kind("magic", "m", None)


def test_openai_compat_requires_a_base_url(clip):
    with pytest.raises(ConfigError, match="needs --base-url"):
        RunConfig(input=clip, prompt="p", model="m", backend=BackendKind.OPENAI_COMPAT)


def test_registry_builds_the_configured_backend(make_config):
    assert create_backend(make_config()).name == "mock"


def test_each_backend_gets_its_own_notes_list(make_config):
    first, second = create_backend(make_config()), create_backend(make_config())
    first.notes.append("careful")
    assert second.notes == [], "notes must not be shared between backend instances"


# ---------------------------------------------------------------- mock


async def test_mock_backend_is_deterministic():
    backend = MockBackend(detect_at=3.0, latency=0.0)
    before = await backend.infer("p", make_frames(), Window(1, 0.0, 2.0))
    after = await backend.infer("p", make_frames(), Window(2, 2.0, 4.0))
    again = await backend.infer("p", make_frames(), Window(3, 2.0, 4.0))

    assert before.text == NOTHING
    assert after.text.startswith("accident detected:")
    assert after.text == again.text


async def test_mock_backend_without_detection_never_fires():
    backend = MockBackend(detect_at=None, latency=0.0)
    result = await backend.infer("p", make_frames(), Window(9, 8.0, 10.0))
    assert result.text == NOTHING


# ---------------------------------------------------------------- openai-compatible


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1"),
        ("http://localhost:8000/", "http://localhost:8000/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1"),
    ],
)
def test_base_url_normalisation(given, expected):
    assert normalise_base_url(given) == expected


async def test_openai_compat_sends_prompt_and_frames_and_parses_the_reply():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": " accident detected: a crash "}}],
                "usage": {"total_tokens": 42},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://localhost:8000/v1",
        headers={"Authorization": "Bearer secret"},
    )
    backend = OpenAICompatBackend(
        "some/model", base_url="http://localhost:8000", max_tokens=64, client=client
    )
    frames = make_frames(4)
    result = await backend.infer("find accidents", frames, Window(1, 0.0, 2.0))

    assert result.text == "accident detected: a crash"
    assert result.raw["usage"]["total_tokens"] == 42
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"

    content = seen["body"]["messages"][0]["content"]
    assert seen["body"]["model"] == "some/model"
    assert seen["body"]["max_tokens"] == 64
    assert content[0] == {"type": "text", "text": "find accidents"}
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == len(frames)
    encoded = images[0]["image_url"]["url"]
    assert encoded.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(encoded.split(",", 1)[1]) == frames[0].jpeg

    await client.aclose()


async def test_openai_compat_reports_server_errors():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="overloaded")),
        base_url="http://localhost:8000/v1",
    )
    backend = OpenAICompatBackend("m", base_url="http://localhost:8000", client=client)
    with pytest.raises(BackendError, match="503"):
        await backend.infer("p", make_frames(), Window(1, 0.0, 1.0))
    await client.aclose()


async def test_openai_compat_reports_unexpected_payloads():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"oops": 1})),
        base_url="http://localhost:8000/v1",
    )
    backend = OpenAICompatBackend("m", base_url="http://localhost:8000", client=client)
    with pytest.raises(BackendError, match="unexpected response shape"):
        await backend.infer("p", make_frames(), Window(1, 0.0, 1.0))
    await client.aclose()
