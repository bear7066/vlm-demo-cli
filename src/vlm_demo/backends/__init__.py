"""Pluggable inference backends."""

from vlm_demo.backends.base import BackendError, InferenceResult, VLMBackend
from vlm_demo.backends.registry import create_backend

__all__ = ["BackendError", "InferenceResult", "VLMBackend", "create_backend"]
