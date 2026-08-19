from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from vlm_demo.config import Pace
from vlm_demo.server import create_app, parse_range


@pytest.fixture
def client(make_config):
    config = make_config(pace=Pace.COMPLETE, pass_gap=1.0, num_frames=2, mock_latency=0.0)
    with TestClient(create_app(config)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(make_config, tmp_path):
    """A run pointed at a directory with nothing in it yet."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with TestClient(create_app(make_config(input=empty))) as test_client:
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


# ---------------------------------------------------------------- the library


def test_library_lists_the_input_directory_and_preselects_a_video(client):
    body = client.get("/api/library").json()
    assert [v["name"] for v in body["videos"]] == ["sample.mp4"]
    assert body["selected"] == "sample.mp4"
    assert body["uploads_enabled"] is True


def test_an_empty_directory_starts_with_no_selection(empty_client):
    library = empty_client.get("/api/library").json()
    assert library["videos"] == [] and library["selected"] is None
    assert empty_client.get("/api/config").json()["video"] is None
    assert empty_client.get("/api/video").status_code == 409


def test_selecting_another_video_retargets_the_run(client, library):
    shutil.copy(library / "sample.mp4", library / "second.mp4")
    body = client.post("/api/select", json={"name": "second.mp4"}).json()
    assert body["video"]["filename"] == "second.mp4"
    assert body["video"]["url"].startswith("/api/video/second.mp4")
    assert client.get("/api/library").json()["selected"] == "second.mp4"
    assert client.get("/api/config").json()["video"]["filename"] == "second.mp4"


def test_selecting_a_missing_video_is_a_404_and_changes_nothing(client):
    assert client.post("/api/select", json={"name": "nope.mp4"}).status_code == 404
    assert client.post("/api/select", json={"name": "../../etc/passwd.mp4"}).status_code == 404
    assert client.get("/api/library").json()["selected"] == "sample.mp4"


def test_switching_videos_clears_the_previous_feed(client, library):
    shutil.copy(library / "sample.mp4", library / "second.mp4")
    with client.websocket_connect("/ws") as socket:
        _drain_until_ready(socket)
        socket.send_json({"type": "start", "t": 0.0, "playing": True})
        while socket.receive_json()["type"] != "pass_result":
            pass
    assert any(e["type"] == "pass_result" for e in client.get("/api/events").json()["events"])

    client.post("/api/select", json={"name": "second.mp4"})
    replay = client.get("/api/events").json()["events"]
    assert not [e for e in replay if e["type"] == "pass_result"]


# ---------------------------------------------------------------- uploads


def test_uploading_adds_a_video_to_the_directory(client, library):
    response = client.post(
        "/api/videos", params={"name": "my clip.mp4"}, content=b"not really a video"
    )
    assert response.status_code == 201
    assert response.json()["video"]["name"] == "my clip.mp4"
    assert (library / "my clip.mp4").read_bytes() == b"not really a video"
    assert "my clip.mp4" in [v["name"] for v in client.get("/api/library").json()["videos"]]
    # Something was already selected, so the upload must not steal playback.
    assert client.get("/api/library").json()["selected"] == "sample.mp4"


def test_uploading_into_an_empty_directory_selects_the_upload(empty_client, clip):
    response = empty_client.post(
        "/api/videos", params={"name": "first.mp4"}, content=clip.read_bytes()
    )
    assert response.status_code == 201
    assert response.json()["selected"] == "first.mp4"
    assert empty_client.get("/api/config").json()["video"]["filename"] == "first.mp4"


def test_upload_names_are_sanitised_and_never_clobber(client, library):
    client.post("/api/videos", params={"name": "../../escape.mp4"}, content=b"a")
    client.post("/api/videos", params={"name": "sample.mp4"}, content=b"b")
    names = [v["name"] for v in client.get("/api/library").json()["videos"]]
    assert "escape.mp4" in names and "sample-1.mp4" in names
    assert not (library.parent / "escape.mp4").exists()
    assert (library / "sample.mp4").stat().st_size > 1, "the original must be untouched"


def test_upload_rejects_a_non_video_extension(client):
    response = client.post("/api/videos", params={"name": "payload.sh"}, content=b"rm -rf /")
    assert response.status_code == 400
    assert "video extension" in response.json()["detail"]


def test_upload_rejects_a_file_over_the_limit(make_config, tmp_path):
    config = make_config(max_upload_mb=0.001)  # 1 kB
    with TestClient(create_app(config)) as client:
        response = client.post("/api/videos", params={"name": "big.mp4"}, content=b"x" * 5_000)
    assert response.status_code == 413


def test_uploads_can_be_switched_off(make_config):
    with TestClient(create_app(make_config(allow_upload=False))) as client:
        assert client.get("/api/library").json()["uploads_enabled"] is False
        response = client.post("/api/videos", params={"name": "x.mp4"}, content=b"x")
        assert response.status_code == 403


# ---------------------------------------------------------------- video streaming


def test_video_is_served_whole(client):
    response = client.get("/api/video")
    assert response.status_code == 200
    assert response.headers["accept-ranges"] == "bytes"
    assert len(response.content) > 0


def test_a_named_video_is_served_and_traversal_is_refused(client, library):
    shutil.copy(library / "sample.mp4", library / "second.mp4")
    assert client.get("/api/video/second.mp4").status_code == 200
    assert client.get("/api/video/..%2F..%2Fetc%2Fpasswd").status_code == 404


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


def _drain_until_ready(socket):
    """Read events until the backend reports it is ready to run passes."""
    while True:
        event = socket.receive_json()
        if event["type"] == "status" and event["state"] == "ready":
            return


def test_websocket_opens_with_the_session_and_the_library(client):
    with client.websocket_connect("/ws") as socket:
        first, second = socket.receive_json(), socket.receive_json()
    assert first["type"] == "session" and first["model"] == "mock"
    assert second["type"] == "library"
    assert [v["name"] for v in second["videos"]] == ["sample.mp4"]


def test_websocket_announces_a_video_switch_to_every_page(client, library):
    shutil.copy(library / "sample.mp4", library / "second.mp4")
    with client.websocket_connect("/ws") as socket:
        _drain_until_ready(socket)
        client.post("/api/select", json={"name": "second.mp4"})
        seen = [socket.receive_json() for _ in range(3)]
    session = next(e for e in seen if e["type"] == "session")
    assert session["video"]["filename"] == "second.mp4"
    assert next(e for e in seen if e["type"] == "library")["selected"] == "second.mp4"


def test_websocket_streams_results_after_start(client):
    with client.websocket_connect("/ws") as socket:
        first = socket.receive_json()
        assert first["type"] == "session"
        assert first["model"] == "mock"

        # Wait for the backend to be ready, then start playback from the beginning.
        _drain_until_ready(socket)
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
        _drain_until_ready(socket)
        socket.send_json({"type": "start", "t": 0.0, "playing": True})
        while socket.receive_json()["type"] != "pass_result":
            pass

    replay = client.get("/api/events").json()
    assert any(event["type"] == "pass_result" for event in replay["events"])
