"""A restart must restore uploaded clips from the persistent bucket."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vlm_demo.s3_library import S3VideoLibrary
from vlm_demo.server import create_app


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def list_objects_v2(self, **kwargs):
        return {
            "Contents": [
                {"Key": key, "Size": len(data)}
                for key, data in self.objects.items()
                if key.startswith(kwargs["Prefix"])
            ]
        }

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])

    def delete_object(self, *, Bucket, Key):
        self.objects.pop(Key, None)


async def feed(data: bytes):
    yield data


async def test_upload_survives_a_fresh_local_directory(tmp_path: Path):
    s3 = FakeS3()
    first = tmp_path / "first"
    first.mkdir()
    library = S3VideoLibrary(first, bucket="vlm-demo-cli", client=s3, max_upload_bytes=100)
    await library.save("clip.mp4", feed(b"video-data"))
    assert s3.objects == {"videos/clip.mp4": b"video-data"}

    second = tmp_path / "second"
    second.mkdir()
    restored = S3VideoLibrary(second, bucket="vlm-demo-cli", client=s3, max_upload_bytes=100)
    restored.restore()
    assert restored.default_name() == "clip.mp4"
    assert restored.resolve("clip.mp4").read_bytes() == b"video-data"


async def test_failed_persistent_upload_is_not_listed(tmp_path: Path):
    class BrokenS3(FakeS3):
        def upload_file(self, path, bucket, key):
            raise RuntimeError("offline")

    library = S3VideoLibrary(tmp_path, bucket="vlm-demo-cli", client=BrokenS3(), max_upload_bytes=100)
    with pytest.raises(OSError, match="persistent video upload failed"):
        await library.save("clip.mp4", feed(b"video-data"))
    assert library.entries() == []


def test_app_restores_video_for_playback(monkeypatch, make_config, clip):
    s3 = FakeS3()
    s3.objects[f"videos/{clip.name}"] = clip.read_bytes()
    monkeypatch.setenv("VLM_VIDEO_BUCKET", "vlm-demo-cli")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://storage.example.test")
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setattr("vlm_demo.s3_library.boto3.client", lambda *args, **kwargs: s3)

    with TestClient(create_app(make_config())) as client:
        assert client.get("/api/config").json()["video"]["filename"] == clip.name
        response = client.get(f"/api/video/{clip.name}", headers={"Range": "bytes=0-9"})
        assert response.status_code == 206
        assert response.content == clip.read_bytes()[:10]


def test_deleting_video_removes_bucket_object(monkeypatch, make_config, clip, tmp_path):
    s3 = FakeS3()
    s3.objects[f"videos/{clip.name}"] = clip.read_bytes()
    monkeypatch.setenv("VLM_VIDEO_BUCKET", "vlm-demo-cli")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://storage.example.test")
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setattr("vlm_demo.s3_library.boto3.client", lambda *args, **kwargs: s3)

    root = tmp_path / "stored"
    root.mkdir()
    with TestClient(create_app(make_config(input=root, allow_delete=True))) as client:
        assert client.get("/api/library").json()["deletes_enabled"] is True
        assert client.delete(f"/api/videos/{clip.name}").status_code == 200
        assert client.get("/api/library").json()["videos"] == []
    assert s3.objects == {}
