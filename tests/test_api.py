from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vlm_demo.config import Pace
from vlm_demo.server import create_app, parse_range


@pytest.fixture
def client(make_config):
    config = make_config(pace=Pace.COMPLETE, pass_gap=1.0, num_frames=2, mock_latency=0.0)
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_index_and_static_assets_are_served(client):
    assert "<video" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_config_endpoint_describes_the_run(client):
    body = client.get("/api/config").json()
    assert body["backend"] == "mock"
    assert body["prompt"].startswith("detect")
    assert body["num_frames"] == 2
    assert body["total_passes"] == 6  # 6s clip, 1s gap
    assert body["video"]["width"] == 160


def test_video_is_served_whole(client):
    response = client.get("/api/video")
    assert response.status_code == 200
    assert response.headers["accept-ranges"] == "bytes"
    assert len(response.content) > 0


def test_video_honours_range_requests(client):
    full = client.get("/api/video").content
    response = client.get("/api/video", headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 10-19/{len(full)}"
    assert response.content == full[10:20]


def test_unsatisfiable_range_is_rejected(client):
    assert client.get("/api/video", headers={"Range": "bytes=99999999-"}).status_code == 416


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        ("bytes=0-99", 1000, (0, 99)),
        ("bytes=500-", 1000, (500, 999)),
        ("bytes=-100", 1000, (900, 999)),
        ("bytes=0-99999", 1000, (0, 999)),
        (None, 1000, None),
        ("bytes=abc", 1000, None),
    ],
)
def test_parse_range(header, size, expected):
    assert parse_range(header, size) == expected


def test_websocket_streams_results_after_start(client):
    with client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        assert first["type"] == "session"
        assert first["model"] == "mock"

        # Wait for the backend to be ready, then start playback from the beginning.
        while True:
            event = socket.receive_json()
            if event["type"] == "status" and event["state"] == "ready":
                break
        socket.send_json({"type": "start", "t": 0.0, "playing": True})

        results = []
        while len(results) < 3:
            event = socket.receive_json()
            if event["type"] == "pass_result":
                results.append(event)
            assert event["type"] != "error", event

        assert [r["index"] for r in results] == [1, 2, 3]
        assert results[0]["text"] == "nothing happened"
        assert results[0]["frames_used"] == 2
        assert results[0]["t_end"] == pytest.approx(1.0)


def test_history_is_replayed_to_late_joiners(client):
    with client.websocket_connect("/ws") as socket:
        while True:
            event = socket.receive_json()
            if event["type"] == "status" and event["state"] == "ready":
                break
        socket.send_json({"type": "start", "t": 0.0, "playing": True})
        while socket.receive_json()["type"] != "pass_result":
            pass

    replay = client.get("/api/events").json()
    assert any(event["type"] == "pass_result" for event in replay["events"])
