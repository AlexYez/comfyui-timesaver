"""Regression guard against cp1251-over-utf8 mojibake (TECH_DEBT_AUDIT F-05, F-25).

The encoding bug corrupted Russian descriptions, tooltips, and comments by
re-encoding UTF-8 bytes through cp1251 with latin-1 fallback for undefined
positions. The bigram we scan for: Cyrillic capital ``\\u0420`` or ``\\u0421``
immediately followed by either a Latin-1 supplement char (``\\u00A0..\\u00BF``)
or a Cyrillic supplement extension (``\\u0400..\\u040F``).

Real Russian text never produces those bigrams, so a hit is treated as
fail-loud regression. If you intentionally need such a sequence (extremely
rare), add the file path to ``_ALLOWED_PATHS`` with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Mojibake bigram: <Р|С> + <NBSP..¿|Ѐ..Џ>.
_MOJIBAKE_RE = re.compile(r"[РС][ -¿Ѐ-Џ]")

# Directories never scanned: VCS, build artifacts, vendored libs, caches.
_SKIP_PARTS = frozenset({".git", "__pycache__", ".cache", ".claude", ".pytest_cache"})

# Explicit per-file allow-list. Empty by default. Extend only with a reason.
_ALLOWED_PATHS: frozenset[str] = frozenset()


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        yield path


def test_no_mojibake_in_python_sources():
    hits: list[str] = []
    for path in _iter_python_files(PACKAGE_ROOT):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in _ALLOWED_PATHS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            match = _MOJIBAKE_RE.search(line)
            if match:
                snippet = line[max(0, match.start() - 20) : match.start() + 25].strip()
                hits.append(f"{rel}:{lineno}: {snippet!r}")

    assert not hits, (
        "Detected cp1251-over-utf8 mojibake. Recover with the byte-mapping "
        "decoder from the audit (see F-05/F-25). Hits:\n  " + "\n  ".join(hits)
    )
