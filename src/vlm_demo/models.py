"""The models the page can offer as suggestions: whatever the HuggingFace cache already holds.

Only the ``transformers`` backend loads weights from that cache, so it is the only backend the
server offers a list for — see :class:`~vlm_demo.server.AppState`. Reading the directory
directly keeps ``huggingface_hub`` out of the base dependencies; it only ships with the gpu
extra.
"""

from __future__ import annotations

import os
from pathlib import Path

HUB_PREFIX = "models--"
"""Cache directories are ``models--owner--repo`` (and ``datasets--…``, which we skip)."""


def hub_cache_dir() -> Path:
    """Where HuggingFace keeps downloaded repos, honouring ``HF_HUB_CACHE`` / ``HF_HOME``."""
    if env := os.environ.get("HF_HUB_CACHE"):
        return Path(env)
    home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    return Path(home) / "hub"


def cached_model_ids() -> list[str]:
    """Repo ids already in the local cache, sorted. Empty if there is no cache directory."""
    root = hub_cache_dir()
    if not root.is_dir():
        return []
    return sorted(
        {
            path.name[len(HUB_PREFIX) :].replace("--", "/")
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(HUB_PREFIX)
        }
    )
