"""In-process inference with HuggingFace ``transformers``.

Imports are deferred so the rest of the CLI works without ``torch`` installed
(``uv sync --extra gpu`` adds it). Generation is blocking, so it runs in a worker thread,
serialised by a lock — one GPU pass at a time.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Sequence
from typing import Any

from vlm_demo.backends.base import BackendError, InferenceResult, VLMBackend
from vlm_demo.video import Frame, Window

log = logging.getLogger(__name__)


def _dtype_kwarg_name() -> str:
    """`torch_dtype` was renamed to `dtype` in transformers 4.56."""
    import transformers

    try:
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class TransformersBackend(VLMBackend):
    name = "transformers"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        device_map: str = "auto",
    ) -> None:
        super().__init__()
        self.model_id = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device_map = device_map
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._dtype: Any = None
        self._lock = asyncio.Lock()

    async def prepare(self) -> None:
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load)

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BackendError(
                "the transformers backend needs the gpu extra: `uv sync --extra gpu`"
            ) from exc

        self._torch = torch
        if torch.cuda.is_available():
            # Native bf16 starts at Ampere (sm_80). Do not trust is_bf16_supported(): on Volta
            # it reports True for *emulated* bf16, which is slower than plain fp16.
            major, _ = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32
            self._warn_about_cpu(torch)

        log.info("loading %s (dtype=%s, device_map=%s)", self.model_id, dtype, self.device_map)
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                attn_implementation="sdpa",
                **{_dtype_kwarg_name(): dtype},
            )
        except Exception as exc:
            raise BackendError(f"could not load {self.model_id!r}: {exc}") from exc
        self._model.eval()
        self._dtype = dtype
        log.info("model ready on %s", getattr(self._model, "device", "?"))

    def _warn_about_cpu(self, torch: Any) -> None:
        """A 4B VLM on CPU looks broken rather than misconfigured — say so loudly."""
        note = (
            f"no usable CUDA device: torch {torch.__version__} is built for CUDA "
            f"{torch.version.cuda}. Running on CPU, which is far too slow for realtime "
            f"playback — raise --pass-gap a lot, or install a torch build matching your "
            f"driver (`UV_TORCH_BACKEND=auto uv sync --extra gpu`)."
        )
        log.warning(note)
        self.notes.append(note)

    async def infer(
        self, prompt: str, frames: Sequence[Frame], window: Window
    ) -> InferenceResult:
        if self._model is None:
            await self.prepare()
        started = time.perf_counter()
        async with self._lock:
            text = await asyncio.to_thread(self._generate, prompt, list(frames))
        return InferenceResult(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"window": window.index, "frames": len(frames)},
        )

    def _generate(self, prompt: str, frames: list[Frame]) -> str:
        from PIL import Image

        torch = self._torch
        images = [Image.open(io.BytesIO(f.jpeg)).convert("RGB") for f in frames]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self._encode(messages, images, prompt)
        inputs = inputs.to(self._model.device, dtype=self._dtype)
        input_len = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_tokens}
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        return self._processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

    def _encode(self, messages: list[dict], images: list[Any], prompt: str) -> Any:
        """Prefer the chat-template path; fall back for processors that reject it."""
        try:
            return self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError, KeyError) as exc:
            log.debug("chat template path failed (%s), falling back to processor()", exc)

        placeholder = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        try:
            text = self._processor.apply_chat_template(
                [{"role": "user", "content": placeholder}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:  # processor without a chat template at all
            text = prompt
        return self._processor(text=[text], images=images, return_tensors="pt")

    async def aclose(self) -> None:
        model, self._model, self._processor = self._model, None, None
        if model is not None and self._torch is not None and self._torch.cuda.is_available():
            del model
            self._torch.cuda.empty_cache()
