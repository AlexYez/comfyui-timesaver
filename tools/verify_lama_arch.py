"""Numerical equivalence check: pure-PyTorch LaMa vs TorchScript big-lama.pt.

Loads both implementations on CPU, runs a fixed random image+mask through
each, and reports max abs diff. Expectation: < 1e-4 (FFT path can introduce
tiny float drift but composite output should be near-bitwise identical).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from nodes.image.lama_cleanup._lama_arch import build_lama_inpainter


JIT_PATH = Path(r"D:\AiApps\ComfyUI\comfyui\ComfyUI\models\lama\big-lama.pt")
ST_PATH = Path(r"D:\AiApps\ComfyUI\comfyui\ComfyUI\models\lama\big-lama.safetensors")


def main() -> int:
    print(f"[info] loading TorchScript: {JIT_PATH}")
    jit_model = torch.jit.load(str(JIT_PATH), map_location="cpu").eval()

    print(f"[info] loading safetensors: {ST_PATH}")
    state = load_file(str(ST_PATH), device="cpu")
    print(f"[info] safetensors keys: {len(state)}")

    print("[info] building pure-PyTorch model and loading state_dict...")
    pt_model = build_lama_inpainter(state)

    torch.manual_seed(0)
    img = torch.rand(1, 3, 64, 64)
    mask = (torch.rand(1, 1, 64, 64) > 0.7).float()

    with torch.no_grad():
        out_jit = jit_model(img, mask)
        out_pt = pt_model(img, mask)

    diff = (out_jit - out_pt).abs()
    print(f"[info] output shape: jit={tuple(out_jit.shape)} pt={tuple(out_pt.shape)}")
    print(f"[info] max abs diff: {diff.max().item():.3e}")
    print(f"[info] mean abs diff: {diff.mean().item():.3e}")

    threshold = 1e-4
    if diff.max().item() > threshold:
        print(f"[fail] max diff {diff.max().item():.3e} exceeds threshold {threshold}")
        return 1
    print(f"[pass] max diff under {threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
