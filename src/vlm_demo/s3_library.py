"""Persist the video directory in an S3-compatible bucket across app restarts."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from vlm_demo.library import LibraryError, VideoEntry, VideoLibrary, sanitize_name


class S3VideoLibrary(VideoLibrary):
    def __init__(self, root: Path, *, bucket: str, client: Any = None, **kwargs: Any) -> None:
        super().__init__(root, **kwargs)
        self.bucket = bucket
        if client is None:
            required = ("AWS_ENDPOINT_URL_S3", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
            missing = [name for name in required if not os.getenv(name)]
            if missing:
                raise ValueError(f"missing Neon Object Storage settings: {', '.join(missing)}")
            client = boto3.client(
                "s3",
                endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
                region_name=os.environ["AWS_REGION"],
                config=Config(s3={"addressing_style": "path"}),
            )
        self.client = client

    @staticmethod
    def key(name: str) -> str:
        return f"videos/{name}"

    def restore(self) -> None:
        """Rebuild the local playback cache from the durable bucket at startup."""
        token: str | None = None
        while True:
            query: dict[str, Any] = {"Bucket": self.bucket, "Prefix": "videos/"}
            if token:
                query["ContinuationToken"] = token
            page = self.client.list_objects_v2(**query)
            for item in page.get("Contents", []):
                key = item["Key"]
                name = key.removeprefix("videos/")
                try:
                    if key != self.key(name) or sanitize_name(name) != name:
                        continue
                except LibraryError:
                    continue
                path = self.root / name
                if path.is_file() and path.stat().st_size == item["Size"]:
                    continue
                temp = self.root / f".{uuid.uuid4().hex}.part"
                try:
                    self.client.download_file(self.bucket, key, str(temp))
                    os.replace(temp, path)
                finally:
                    temp.unlink(missing_ok=True)
            if not page.get("IsTruncated"):
                break
            token = page["NextContinuationToken"]

    async def save(self, raw_name: str, chunks: AsyncIterator[bytes]) -> VideoEntry:
        entry = await super().save(raw_name, chunks)
        path = self.root / entry.name
        try:
            await asyncio.to_thread(
                self.client.upload_file, str(path), self.bucket, self.key(entry.name)
            )
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise OSError("persistent video upload failed") from exc
        return entry

    def delete(self, name: str) -> None:
        path = self.resolve(name)
        if not self.allow_delete:
            raise LibraryError("deleting is disabled (--no-delete)")
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self.key(path.name))
        except Exception as exc:
            raise OSError("persistent video delete failed") from exc
        super().delete(name)
