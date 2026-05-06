"""Local pre-flight checks for `comfyui-timesaver`.

Run this BEFORE `git push` to catch regressions early without spinning
up GitHub CI. Designed to be invoked under the ComfyUI portable Python
(which has numpy / torch / PIL / comfy_api), but each section degrades
gracefully if a dependency is missing.

This script DOES NOT start, stop, or reload ComfyUI. The `--full` extra
section connects to an already-running ComfyUI on 127.0.0.1:8188 (the
user manages the server lifecycle). If the server is down the extra
tests skip themselves cleanly.

Important: after editing production code in `nodes/`, a running ComfyUI
still holds the old modules in memory. Restart the server (UI → Restart
or Ctrl+C + relaunch) to actually exercise the change before running
`--full`.

Usage:
    python tools/preflight.py             # run everything
    python tools/preflight.py --quick     # skip pytest (compileall + contracts only)
    python tools/preflight.py --full      # also run extra tests against
                                          # a running ComfyUI (connects only)
    python tools/preflight.py -h

Sections (in order):
    1. compileall            — every .py parses
    2. contracts drift       — node schema snapshot is in sync
    3. pytest (offline)      — unit + contract + static-invariants tests
    4. pytest (extra)        — browser / live-API tests, only with --full;
                              auto-skip if ComfyUI not reachable.

Exits non-zero on the first failure unless `--continue-on-error` is set,
in which case all sections run and the script exits with the worst code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# Windows consoles default to cp1251/cp866 and choke on Unicode.
# Switch to UTF-8 if the stream supports reconfigure (Python 3.7+).
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


# ──────────────────────────────────────────────────────────────────────
# Output helpers (ASCII-safe — no Unicode box drawing or check marks)
# ──────────────────────────────────────────────────────────────────────

def _emit(symbol: str, label: str, detail: str = "") -> None:
    line = f"[preflight] {symbol} {label}"
    if detail:
        line += f" -- {detail}"
    print(line, flush=True)


def _section(title: str) -> None:
    bar = "-" * 72
    print(f"\n{bar}\n[preflight] >> {title}\n{bar}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────

def section_compileall() -> int:
    _section("compileall — syntax check on every .py")
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "."],
        cwd=PACKAGE_ROOT,
    )
    return proc.returncode


def section_contracts() -> int:
    _section("contracts — node schema snapshot drift")
    proc = subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "tools" / "build_node_contracts.py"), "--check"],
        cwd=PACKAGE_ROOT,
    )
    return proc.returncode


def section_pytest_offline() -> int:
    _section("pytest — offline tests (no browser, no live ComfyUI)")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--ignore=tests/test_browser_smoke.py",
        "--ignore=tests/test_comfyui_live_api.py",
    ]
    proc = subprocess.run(cmd, cwd=PACKAGE_ROOT)
    return proc.returncode


def section_pytest_extra() -> int:
    _section(
        "pytest -- extras (connects to ComfyUI on 127.0.0.1:8188; "
        "auto-skip if server is down)"
    )
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_browser_smoke.py",
        "tests/test_comfyui_live_api.py",
        "-q",
    ]
    proc = subprocess.run(cmd, cwd=PACKAGE_ROOT)
    return proc.returncode


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _run_section(
    name: str,
    runner: Callable[[], int],
    failures: list[str],
    continue_on_error: bool,
) -> bool:
    started = time.monotonic()
    try:
        rc = runner()
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        rc = 1
        _emit("[FAIL]", name, f"crashed: {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - started
    if rc == 0:
        _emit("[OK]", name, f"OK in {elapsed:.1f}s")
        return True

    failures.append(name)
    _emit("[FAIL]", name, f"failed (exit {rc}) after {elapsed:.1f}s")
    return continue_on_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight checks for comfyui-timesaver. "
        "Run before `git push`.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip pytest (compileall + contracts only). ~2 seconds.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run browser smoke + live API tests against a running "
        "ComfyUI on 127.0.0.1:8188 (connects only; never starts the server). "
        "Restart ComfyUI manually after editing production code.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run every section even after a failure; report all at the end.",
    )
    args = parser.parse_args(argv)

    if args.quick and args.full:
        parser.error("--quick and --full are mutually exclusive")

    print(f"[preflight] python: {sys.executable}")
    print(f"[preflight] root:   {PACKAGE_ROOT}")

    sections: list[tuple[str, Callable[[], int]]] = [
        ("compileall", section_compileall),
        ("contracts", section_contracts),
    ]
    if not args.quick:
        sections.append(("pytest-offline", section_pytest_offline))
    if args.full:
        sections.append(("pytest-extra", section_pytest_extra))

    failures: list[str] = []
    overall_started = time.monotonic()
    for name, runner in sections:
        keep_going = _run_section(name, runner, failures, args.continue_on_error)
        if not keep_going:
            break

    elapsed = time.monotonic() - overall_started
    print()
    if failures:
        _emit(
            "[FAIL]",
            "preflight",
            f"FAILED in {elapsed:.1f}s -- sections: {', '.join(failures)}",
        )
        return 1
    _emit("[OK]", "preflight", f"all green in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
