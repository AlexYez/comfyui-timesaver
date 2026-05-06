"""Behaviour tests for TS_DownloadFilesNode helpers (audit F-42).

Targets pure-logic methods that do not require network or full ComfyUI runtime:
- ``_parse_file_list`` line parser (skips comments, validates URLs, resolves paths).
- ``_replace_hf_domain`` mirror substitution.
- ``_select_best_mirror`` deterministic single-mirror choice.
- ``_check_connectivity_to_targets`` with monkeypatched session.

The module is imported with ``folder_paths`` and ``ProgressBar`` stubbed so the
test stays CPU-/network-free and works on contributor machines without ComfyUI.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_stubs(monkeypatch, ts_tmp_path):
    """Stub heavy third-party deps so the module imports on a bare CI."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = str(ts_tmp_path / "comfy_root")
    folder_paths.models_dir = str(ts_tmp_path / "comfy_root" / "models")
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    comfy_utils = types.ModuleType("comfy.utils")

    class _ProgressBar:
        def __init__(self, total): self.total = total
        def update(self, n): pass
        def update_absolute(self, value, total=None): pass

    comfy_utils.ProgressBar = _ProgressBar
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.utils = comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)

    if "requests" not in sys.modules:
        requests_mod = types.ModuleType("requests")

        class _RequestException(Exception):
            pass

        class _Session:
            def __init__(self):
                self.headers = {}
                self.proxies = {}

            def mount(self, prefix, adapter): pass
            def head(self, url, timeout=None, allow_redirects=False):
                raise _RequestException("stub Session does not perform real HTTP")

        requests_mod.RequestException = _RequestException
        requests_mod.Session = _Session
        requests_utils_mod = types.ModuleType("requests.utils")
        requests_utils_mod.unquote = lambda s: s
        requests_mod.utils = requests_utils_mod
        requests_adapters_mod = types.ModuleType("requests.adapters")
        requests_adapters_mod.HTTPAdapter = type("HTTPAdapter", (), {"__init__": lambda self, **kw: None})
        requests_mod.adapters = requests_adapters_mod
        monkeypatch.setitem(sys.modules, "requests", requests_mod)
        monkeypatch.setitem(sys.modules, "requests.utils", requests_utils_mod)
        monkeypatch.setitem(sys.modules, "requests.adapters", requests_adapters_mod)

    if "urllib3" not in sys.modules:
        urllib3_mod = types.ModuleType("urllib3")
        urllib3_util_mod = types.ModuleType("urllib3.util")
        urllib3_retry_mod = types.ModuleType("urllib3.util.retry")
        urllib3_retry_mod.Retry = type("Retry", (), {"__init__": lambda self, **kw: None})
        urllib3_mod.util = urllib3_util_mod
        urllib3_util_mod.retry = urllib3_retry_mod
        monkeypatch.setitem(sys.modules, "urllib3", urllib3_mod)
        monkeypatch.setitem(sys.modules, "urllib3.util", urllib3_util_mod)
        monkeypatch.setitem(sys.modules, "urllib3.util.retry", urllib3_retry_mod)

    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = lambda iterable=None, **kw: iterable if iterable is not None else iter([])
        monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    # Stub comfy_api.v0_0_2.IO so the V3 schema declaration in
    # ts_downloader imports without dragging in the full ComfyUI runtime
    # (real comfy_api needs comfy.cli_args, which the test does not stub).
    comfy_api_mod = types.ModuleType("comfy_api")
    latest_mod = types.ModuleType("comfy_api.v0_0_2")

    class _StubInput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _StubOutput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _StubComfyType:
        Input = _StubInput
        Output = _StubOutput

    class _StubSchema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _StubNodeOutput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _StubIO:
        class ComfyNode:
            pass
        Schema = _StubSchema
        NodeOutput = _StubNodeOutput
        String = _StubComfyType
        Boolean = _StubComfyType
        Combo = _StubComfyType
        Int = _StubComfyType

    latest_mod.IO = _StubIO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api_mod)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest_mod)


@pytest.fixture
def downloader_module(monkeypatch, ts_tmp_path):
    """Import nodes.files.ts_downloader with safe stubs.

    Uses the project-local ``ts_tmp_path`` fixture (see tests/conftest.py)
    instead of pytest's default tmp_path because the OS Temp folder is
    permission-locked on the maintainer's Windows machine.
    """
    root = Path(__file__).resolve().parents[1]
    _install_stubs(monkeypatch, ts_tmp_path)
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.files.ts_downloader", None)
    return importlib.import_module("nodes.files.ts_downloader")


def test_parse_file_list_skips_blanks_and_comments(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    text = "\n".join([
        "# comment line",
        "",
        "https://example.com/a.bin /tmp/a",
        "   ",
        "https://example.com/b.bin /tmp/b",
    ])
    parsed = node._parse_file_list(text)
    urls = [item["url"] for item in parsed]
    assert urls == ["https://example.com/a.bin", "https://example.com/b.bin"]


def test_parse_file_list_rejects_invalid_url(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    text = "ftp://example.com/file /tmp/x"
    parsed = node._parse_file_list(text)
    assert parsed == []


def test_parse_file_list_resolves_models_alias(downloader_module, ts_tmp_path, monkeypatch):
    fake_fp = types.SimpleNamespace(base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models"))
    monkeypatch.setattr(downloader_module, "folder_paths", fake_fp)
    node = downloader_module.TS_DownloadFilesNode()
    parsed = node._parse_file_list("https://example.com/x.bin models/checkpoints")
    assert len(parsed) == 1
    target = Path(parsed[0]["target_dir"])
    assert target == (Path(ts_tmp_path) / "models" / "checkpoints").resolve()


def test_replace_hf_domain_substitutes_only_huggingface_host(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    src = "https://huggingface.co/repo/file"
    assert node._replace_hf_domain(src, "hf-mirror.com") == "https://hf-mirror.com/repo/file"
    # Non-HF URL is left alone.
    assert node._replace_hf_domain("https://example.com/x", "hf-mirror.com") == "https://example.com/x"
    # Empty/canonical target -> no-op.
    assert node._replace_hf_domain(src, "") == src
    assert node._replace_hf_domain(src, "huggingface.co") == src


def test_replace_hf_domain_strips_protocol_prefix(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    src = "https://huggingface.co/x"
    assert node._replace_hf_domain(src, "https://hf-mirror.com") == "https://hf-mirror.com/x"
    assert node._replace_hf_domain(src, "http://hf-mirror.com/") == "https://hf-mirror.com/x"


def test_select_best_mirror_picks_first_when_single_or_empty(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    fake_session = types.SimpleNamespace(head=lambda *a, **kw: None)
    assert node._select_best_mirror(fake_session, "") == "huggingface.co"
    assert node._select_best_mirror(fake_session, "hf-mirror.com") == "hf-mirror.com"


def test_check_connectivity_to_targets_returns_true_on_first_reachable(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    calls = []

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            calls.append(url)
            if "good.example" in url:
                return types.SimpleNamespace(status_code=200)
            raise downloader_module.requests.RequestException("nope")

    parsed = [
        {"url": "https://bad.example/a.bin", "target_dir": "/tmp"},
        {"url": "https://good.example/b.bin", "target_dir": "/tmp"},
    ]
    assert node._check_connectivity_to_targets(parsed, _Sess(), "huggingface.co") is True
    assert any("good.example" in u for u in calls)


def test_check_connectivity_to_targets_returns_false_when_all_fail(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            raise downloader_module.requests.RequestException("offline")

    parsed = [{"url": "https://offline.example/a.bin", "target_dir": "/tmp"}]
    assert node._check_connectivity_to_targets(parsed, _Sess(), "huggingface.co") is False
