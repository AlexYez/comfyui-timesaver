"""Extract LaMa weights from TorchScript big-lama.pt into safetensors.

big-lama.pt is a TorchScript artefact (graph + weights). safetensors
stores only tensors, so the resulting file contains the state_dict
alone — the original .pt is not modified.

Usage:
    python tools/convert_lama_to_safetensors.py [--src PATH] [--dst PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


DEFAULT_SRC = Path(r"D:\AiApps\ComfyUI\comfyui\ComfyUI\models\lama\big-lama.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert big-lama.pt to safetensors.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="TorchScript .pt path.")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output .safetensors path (default: <src_dir>/big-lama.safetensors).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src: Path = args.src
    if not src.is_file():
        print(f"[error] source not found: {src}", file=sys.stderr)
        return 1

    dst: Path = args.dst or src.with_suffix(".safetensors")
    print(f"[info] loading TorchScript model: {src}")
    model = torch.jit.load(str(src), map_location="cpu")
    model.eval()

    raw_state = model.state_dict()
    if not raw_state:
        print("[error] state_dict is empty — nothing to export.", file=sys.stderr)
        return 2

    cleaned: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    total_bytes = 0
    for key, tensor in raw_state.items():
        if not isinstance(tensor, torch.Tensor):
            skipped.append(f"{key} ({type(tensor).__name__})")
            continue
        flat = tensor.detach().to("cpu").contiguous().clone()
        cleaned[key] = flat
        total_bytes += flat.numel() * flat.element_size()

    print(f"[info] tensors: {len(cleaned)}; total weight bytes: {total_bytes / 1024 / 1024:.2f} MiB")
    if skipped:
        print("[warn] skipped non-tensor entries:")
        for entry in skipped:
            print(f"  - {entry}")

    metadata = {"format": "pt", "source": src.name}
    print(f"[info] writing safetensors: {dst}")
    save_file(cleaned, str(dst), metadata=metadata)

    src_mib = src.stat().st_size / 1024 / 1024
    dst_mib = dst.stat().st_size / 1024 / 1024
    print(f"[done] source: {src_mib:.2f} MiB -> safetensors: {dst_mib:.2f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
