"""Shared pytest fixtures for the comfyui-timesaver test suite.

The default pytest tmp_path on this Windows machine resolves under
%LOCALAPPDATA%\\Temp which is denied by an OS policy, so we provide a
project-local replacement that lives under tests/.cache and is cleaned
up after the test.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


_BASE_TMP = Path(__file__).resolve().parent / ".cache" / "tmp"


@pytest.fixture
def ts_tmp_path(request):
    _BASE_TMP.mkdir(parents=True, exist_ok=True)
    path = _BASE_TMP / f"{request.node.name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)

    def _cleanup():
        shutil.rmtree(path, ignore_errors=True)

    request.addfinalizer(_cleanup)
    return path
