#!/usr/bin/env python3
"""Fill tensors missing from a finetune checkpoint with the base model's.

Gemma-4 e2b ties the KV projections of its last ``num_kv_shared_layers`` layers,
so transformers' deduplicating save drops their k_proj/v_proj/k_norm from the
finetune checkpoint. vLLM refuses to load without them ("Following weights were
not initialized from checkpoint"). Those tensors are unused at runtime, so
copying the base model's copies back in restores a servable checkpoint.

    uv run python scripts/merge-base.py            # writes models/gemma-4-e2b-fall-merged
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

FINETUNE = "THChou1220/gemma-4-e2b-kinetics54K-enhanced-fall_FFT"
BASE = "google/gemma-4-E2B-it"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune", default=FINETUNE)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", type=Path, default=Path("models/gemma-4-e2b-fall-merged"))
    args = ap.parse_args()

    ft_map = get_safetensors_metadata(args.finetune).weight_map  # tensor -> shard file
    base_map = get_safetensors_metadata(args.base).weight_map
    missing = sorted(set(base_map) - set(ft_map))
    if not missing:
        print("nothing missing; finetune checkpoint is already complete")
        return
    print(f"{len(missing)} tensors missing from the finetune, e.g. {missing[0]}")

    ft_dir = Path(snapshot_download(args.finetune))
    tensors = {}
    for shard in sorted(set(ft_map.values())):
        with safe_open(ft_dir / shard, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    for shard in sorted({base_map[k] for k in missing}):
        path = hf_hub_download(args.base, shard)
        with safe_open(path, framework="pt") as f:
            for key in missing:
                if base_map[key] == shard:
                    tensors[key] = f.get_tensor(key)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out / "model.safetensors", metadata={"format": "pt"})
    # save_file honours a restrictive umask; the vLLM container reads as another user
    (args.out / "model.safetensors").chmod(0o644)
    for aux in ft_dir.iterdir():  # config, tokenizer, processor, chat template, ...
        if aux.is_file() and not aux.name.endswith(".safetensors"):
            shutil.copy(aux, args.out / aux.name)
    print(f"merged {len(tensors)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
