"""Static invariants for the production codebase (regex / file scans).

These tests do NOT import nodes — they read source files and check
project-level rules:

- production code must not import `comfy_api.latest` (Step 1 of the
  modernization plan pinned everything to `comfy_api.v0_0_2`);
- `requirements.txt` must not pin `torch*` (broken installs on user side);
- no new hardcoded `.cuda()` / `torch.device("cuda")` may appear in
  `nodes/` beyond the frozen list of known violations;
- production code must not call `print(...progress...)` for progress
  reporting — use `comfy.utils.ProgressBar` instead;
- node files in `nodes/` follow the `ts_` prefix loader convention.

These are CPU-only, dependency-free, and fast. Safe to run on any Python.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NODES_DIR = PACKAGE_ROOT / "nodes"

# Bundled third-party model code that lives under `nodes/` but is NOT part
# of our production node code (downloaded reference implementations of
# RIFE / Video-Depth-Anything). Loader skips them via the `ts_` prefix
# rule; static invariants must skip them too.
BUNDLED_HELPER_DIRS: frozenset[str] = frozenset(
    {
        "frame_interpolation_models",
        "video_depth_anything",
    }
)


def _iter_node_py_files() -> list[Path]:
    out: list[Path] = []
    for path in NODES_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if any(part in BUNDLED_HELPER_DIRS for part in path.relative_to(NODES_DIR).parts):
            continue
        out.append(path)
    return out


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _format_hits(path: Path, line_numbers: list[int]) -> list[str]:
    rel = path.relative_to(PACKAGE_ROOT).as_posix()
    return [f"{rel}:{lineno}" for lineno in line_numbers]


def test_no_comfy_api_latest_in_production():
    """Step 1 invariant: every production file imports `comfy_api.v0_0_2`.

    `comfy_api.latest` is an unstable alias per ComfyUI V3 migration docs;
    pinning to `v0_0_2` defends against silent surface changes between
    ComfyUI releases. New violations will fail this test.
    """
    pattern = re.compile(r"\bfrom\s+comfy_api\.latest\b")
    hits: list[str] = []
    for path in _iter_node_py_files():
        for lineno, line in enumerate(_read_lines(path), start=1):
            if pattern.search(line):
                hits.extend(_format_hits(path, [lineno]))

    assert not hits, (
        "Production code must import from `comfy_api.v0_0_2`, not `comfy_api.latest`.\n"
        "Offending lines:\n  " + "\n  ".join(hits)
    )


def test_no_torch_in_requirements():
    """`torch*` in `requirements.txt` breaks installs on Apple Silicon, ROCm
    and CPU-only setups (forces a CUDA wheel)."""
    requirements = PACKAGE_ROOT / "requirements.txt"
    assert requirements.is_file(), "requirements.txt missing"

    pattern = re.compile(r"^(torch|torchvision|torchaudio)([<>=!~]|$)")
    offenders: list[str] = []
    for lineno, line in enumerate(_read_lines(requirements), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if pattern.match(stripped):
            offenders.append(f"requirements.txt:{lineno}: {stripped}")

    assert not offenders, (
        "torch* must not appear in requirements.txt.\n"
        "Offending lines:\n  " + "\n  ".join(offenders)
    )


# Frozen list of known `.cuda()` / `torch.device("cuda")` hardcodes.
# This is a regression-only test: new violations fail; fixes must update
# this set (delete entries) and the test will pass.
KNOWN_CUDA_HARDCODES: frozenset[str] = frozenset(
    {
        "nodes/image/lama_cleanup/_lama_helpers.py:424",
        "nodes/audio/ts_whisper.py:578",
        "nodes/audio/ts_whisper.py:582",
    }
)


def test_no_new_cuda_hardcodes():
    """Block new `.cuda()` / `torch.device("cuda")` calls.

    Use `comfy.model_management.get_torch_device()` instead — it picks the
    right device on CUDA / ROCm / MPS / CPU and respects user overrides.

    The set above documents pre-existing violations grandfathered in. Each
    entry should eventually be fixed; this test only blocks _new_ ones.
    """
    # `.cuda()` method calls OR `torch.device("cuda"...)` constructor
    # (covers both bare `"cuda"` and conditional `"cuda" if ... else ...`
    # forms). Tightened from the previous version which missed the
    # ternary form in `_lama_helpers.py:424`.
    pattern = re.compile(
        r'\.cuda\(\)|torch\.device\(\s*[\'"]cuda[\'"]'
    )
    actual: set[str] = set()
    for path in _iter_node_py_files():
        for lineno, line in enumerate(_read_lines(path), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                rel = path.relative_to(PACKAGE_ROOT).as_posix()
                actual.add(f"{rel}:{lineno}")

    new_violations = sorted(actual - KNOWN_CUDA_HARDCODES)
    assert not new_violations, (
        "New hardcoded `.cuda()` / `torch.device(\"cuda\")` call(s):\n  "
        + "\n  ".join(new_violations)
        + "\nUse `comfy.model_management.get_torch_device()` instead."
    )

    fixed = sorted(KNOWN_CUDA_HARDCODES - actual)
    if fixed:
        # Soft signal: not a failure, just a nudge to clean up the frozen set.
        pytest.skip(
            "Some grandfathered violations are gone — please remove these "
            "entries from KNOWN_CUDA_HARDCODES:\n  " + "\n  ".join(fixed)
        )


def test_no_print_progress_reporting():
    """Reserve `print(...)` for one-off scripts; in node bodies use
    `comfy.utils.ProgressBar` (V1) or `api.execution.set_progress` (V3)
    so progress shows up in the ComfyUI UI instead of the server console."""
    pattern = re.compile(r"\bprint\s*\([^)]*progress", re.IGNORECASE)
    offenders: list[str] = []
    for path in _iter_node_py_files():
        for lineno, line in enumerate(_read_lines(path), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                rel = path.relative_to(PACKAGE_ROOT).as_posix()
                offenders.append(f"{rel}:{lineno}: {stripped}")

    assert not offenders, (
        "`print(...progress...)` is not allowed in node code.\n"
        "Offending lines:\n  " + "\n  ".join(offenders)
    )


def test_node_files_follow_ts_prefix_convention():
    """Loader convention (CLAUDE.md §6, §7): every public node file is
    named `ts_*.py`. Files without that prefix in `nodes/<category>/` are
    private helpers and must start with `_`."""
    offenders: list[str] = []
    for path in _iter_node_py_files():
        name = path.name
        if name == "__init__.py":
            continue
        if name.startswith("_") or name.startswith("ts_"):
            continue
        if any(part.startswith("_") for part in path.relative_to(NODES_DIR).parts):
            # Inside an underscore-prefixed subpackage — already private.
            continue
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        offenders.append(rel)

    assert not offenders, (
        "Node files must start with `ts_` (public) or `_` (private helper).\n"
        "Offending files:\n  " + "\n  ".join(offenders)
    )
