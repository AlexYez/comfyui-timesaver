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

        class _TqdmStub:
            def __init__(self, iterable=None, **kwargs):
                self._iterable = iterable

            def __iter__(self):
                if self._iterable is None:
                    return iter([])
                return iter(self._iterable)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, n=1):
                pass

        tqdm_mod.tqdm = _TqdmStub
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


def test_check_connectivity_falls_back_to_secondary_hf_mirror(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    calls = []

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            calls.append(url)
            if "hf-mirror.com" in url:
                return types.SimpleNamespace(status_code=200)
            raise downloader_module.requests.RequestException("primary down")

    parsed = [{"url": "https://huggingface.co/repo/file.bin", "target_dir": "/tmp"}]
    assert node._check_connectivity_to_targets(parsed, _Sess(), "huggingface.co, hf-mirror.com") is True
    assert any("hf-mirror.com" in u for u in calls), \
        "secondary HF mirror must be probed even when first one is down"


def test_check_connectivity_probes_real_url_not_root(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    seen_urls = []

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            seen_urls.append(url)
            return types.SimpleNamespace(status_code=200)

    parsed = [{"url": "https://huggingface.co/owner/repo/resolve/main/file.safetensors", "target_dir": "/tmp"}]
    node._check_connectivity_to_targets(parsed, _Sess(), "huggingface.co")
    assert seen_urls and "/owner/repo/resolve/main/file.safetensors" in seen_urls[0]


def test_sanitize_filename_blocks_parent_traversal(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._sanitize_filename("../evil.bin") == "evil.bin"
    assert node._sanitize_filename("..\\..\\..\\evil.bin") == "evil.bin"
    sanitized_dotdot = node._sanitize_filename("..")
    assert ".." not in sanitized_dotdot and sanitized_dotdot.startswith("downloaded_file_")
    sanitized_dot = node._sanitize_filename(".")
    assert sanitized_dot.startswith("downloaded_file_")


def test_sanitize_filename_keeps_normal_model_names(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._sanitize_filename("model.safetensors") == "model.safetensors"
    assert node._sanitize_filename("flux.dev_v2.safetensors") == "flux.dev_v2.safetensors"


def test_sanitize_filename_strips_control_chars_and_forbidden(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    out = node._sanitize_filename('weird<>:"|?*\x00name.bin')
    assert "<" not in out and ">" not in out and "|" not in out and "\x00" not in out
    assert out.endswith("name.bin")


def test_sanitize_filename_truncates_overlong(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    long = ("x" * 400) + ".safetensors"
    out = node._sanitize_filename(long)
    assert len(out) <= 200
    assert out.endswith(".safetensors")


def test_is_zip_member_safe_accepts_normal_paths(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    root = str((ts_tmp_path / "extract").resolve())
    import os as _os
    _os.makedirs(root, exist_ok=True)
    assert node._is_zip_member_safe("model.safetensors", root) is True
    assert node._is_zip_member_safe("subdir/file.bin", root) is True


def test_is_zip_member_safe_rejects_traversal(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    root = str((ts_tmp_path / "extract").resolve())
    import os as _os
    _os.makedirs(root, exist_ok=True)
    assert node._is_zip_member_safe("../evil.bin", root) is False
    assert node._is_zip_member_safe("sub/../../evil.bin", root) is False
    assert node._is_zip_member_safe("/etc/passwd", root) is False
    assert node._is_zip_member_safe("..", root) is False


def test_is_zip_member_safe_rejects_windows_absolute(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    root = str((ts_tmp_path / "extract").resolve())
    import os as _os
    _os.makedirs(root, exist_ok=True)
    assert node._is_zip_member_safe("C:/Windows/System32/evil.dll", root) is False


def test_extract_zip_refuses_unsafe_archive(downloader_module, ts_tmp_path):
    import zipfile as _zipfile
    node = downloader_module.TS_DownloadFilesNode()
    extract_root = ts_tmp_path / "extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    zip_path = ts_tmp_path / "malicious.zip"
    with _zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("ok.txt", "ok")
        zf.writestr("../escape.txt", "boom")
    result = node._extract_zip(str(zip_path), str(extract_root))
    assert result is False
    assert zip_path.exists(), "archive must NOT be deleted on rejection"
    assert not (extract_root.parent / "escape.txt").exists(), "traversal payload must not be written"
    assert not (extract_root / "ok.txt").exists(), "no member must be extracted when archive is rejected"


def test_extract_zip_accepts_clean_archive(downloader_module, ts_tmp_path):
    import zipfile as _zipfile
    node = downloader_module.TS_DownloadFilesNode()
    extract_root = ts_tmp_path / "extract_ok"
    extract_root.mkdir(parents=True, exist_ok=True)
    zip_path = ts_tmp_path / "clean.zip"
    with _zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "a")
        zf.writestr("sub/b.txt", "b")
    result = node._extract_zip(str(zip_path), str(extract_root))
    assert result is True
    assert (extract_root / "a.txt").read_text() == "a"
    assert (extract_root / "sub" / "b.txt").read_text() == "b"
    assert not zip_path.exists(), "archive should be deleted after successful extraction"


def test_compute_sha256_matches_hashlib(downloader_module, ts_tmp_path):
    import hashlib as _hashlib
    node = downloader_module.TS_DownloadFilesNode()
    f = ts_tmp_path / "data.bin"
    payload = b"comfyui-timesaver" * 5000
    f.write_bytes(payload)
    expected = _hashlib.sha256(payload).hexdigest()
    actual = node._compute_sha256(str(f), chunk_size=1024)
    assert actual == expected
