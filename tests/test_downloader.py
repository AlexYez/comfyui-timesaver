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


def test_compute_sha256_empty_file(downloader_module, ts_tmp_path):
    import hashlib as _hashlib
    node = downloader_module.TS_DownloadFilesNode()
    f = ts_tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert node._compute_sha256(str(f)) == _hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# Pure helper coverage
# ---------------------------------------------------------------------------


def test_safe_int_parses_valid_values(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._safe_int(42) == 42
    assert node._safe_int("17") == 17
    assert node._safe_int(" 8 ") == 8


def test_safe_int_returns_default_on_garbage(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._safe_int(None) == -1
    assert node._safe_int("abc") == -1
    assert node._safe_int("12.5") == -1
    assert node._safe_int(None, default=99) == 99


def test_normalize_etag_variants(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._normalize_etag(None) is None
    assert node._normalize_etag("") is None
    assert node._normalize_etag('"abc123"') == "abc123"
    assert node._normalize_etag('W/"abc123"') == "abc123"
    assert node._normalize_etag('w/"abc123"') == "abc123"
    assert node._normalize_etag("  bare  ") == "bare"
    # Returns None when nothing left after stripping quotes
    assert node._normalize_etag('""') is None


def test_extract_total_size_from_content_range(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._extract_total_size_from_content_range("bytes 0-99/1000") == 1000
    assert node._extract_total_size_from_content_range("bytes 500-999/1000") == 1000
    assert node._extract_total_size_from_content_range("bytes 0-99/*") == -1
    assert node._extract_total_size_from_content_range(None) == -1
    assert node._extract_total_size_from_content_range("") == -1
    assert node._extract_total_size_from_content_range("garbage") == -1


def test_extract_remote_size_priority_order(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    # content-range wins
    h = {"content-range": "bytes 0-9/12345", "x-linked-size": "999", "content-length": "1"}
    assert node._extract_remote_size_from_headers(h) == 12345
    # x-linked-size beats content-length
    h = {"x-linked-size": "5000", "content-length": "1"}
    assert node._extract_remote_size_from_headers(h) == 5000
    # falls back to content-length
    assert node._extract_remote_size_from_headers({"content-length": "777"}) == 777
    # unknown
    assert node._extract_remote_size_from_headers({}) == -1


def test_is_hf_url_recognises_known_hosts(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._is_hf_url("https://huggingface.co/x") is True
    assert node._is_hf_url("https://www.huggingface.co/x") is True
    assert node._is_hf_url("https://cdn-lfs.huggingface.co/x") is True
    assert node._is_hf_url("https://hf-mirror.com/x") is True
    assert node._is_hf_url("https://example.com/huggingface.co") is False


def test_is_hf_url_rejects_domain_spoofing(downloader_module):
    """huggingface.co.evil.com must NOT be treated as an HF host."""
    node = downloader_module.TS_DownloadFilesNode()
    assert node._is_hf_url("https://huggingface.co.evil.com/x") is False
    assert node._is_hf_url("https://hf-mirror.com.evil.com/y") is False
    assert node._is_hf_url("https://evilhuggingface.co/y") is False


def test_replace_hf_domain_rejects_lookalike_host(downloader_module):
    """Lookalike host like huggingface.co.evil.com must be left alone."""
    node = downloader_module.TS_DownloadFilesNode()
    src = "https://huggingface.co.evil.com/x"
    assert node._replace_hf_domain(src, "hf-mirror.com") == src


def test_replace_hf_domain_preserves_query_and_fragment(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    src = "https://huggingface.co/repo/file?rev=main#frag"
    assert node._replace_hf_domain(src, "hf-mirror.com") == "https://hf-mirror.com/repo/file?rev=main#frag"


def test_extract_hf_expected_sha256_valid(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    sha = "a" * 64
    assert node._extract_hf_expected_sha256(sha, "https://huggingface.co/x") == sha
    # Uppercase normalised
    assert node._extract_hf_expected_sha256("A" * 64, "https://huggingface.co/x") == "a" * 64


def test_extract_hf_expected_sha256_rejects_non_hex(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._extract_hf_expected_sha256("not-a-sha", "https://huggingface.co/x") is None
    # Wrong length
    assert node._extract_hf_expected_sha256("a" * 63, "https://huggingface.co/x") is None
    assert node._extract_hf_expected_sha256("a" * 65, "https://huggingface.co/x") is None


def test_extract_hf_expected_sha256_rejects_non_hf_url(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    sha = "a" * 64
    assert node._extract_hf_expected_sha256(sha, "https://example.com/x") is None


def test_extract_hf_expected_sha256_handles_none_etag(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._extract_hf_expected_sha256(None, "https://huggingface.co/x") is None
    assert node._extract_hf_expected_sha256("", "https://huggingface.co/x") is None


def test_is_partial_meta_compatible_empty_meta(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._is_partial_meta_compatible(None, "u", 10, "e") is True
    assert node._is_partial_meta_compatible({}, "u", 10, "e") is True


def test_is_partial_meta_compatible_url_mismatch(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    meta = {"source_url": "https://old.example/a", "remote_size": 10, "remote_etag": "e"}
    assert node._is_partial_meta_compatible(meta, "https://new.example/a", 10, "e") is False


def test_is_partial_meta_compatible_size_mismatch(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    meta = {"source_url": "u", "remote_size": 5, "remote_etag": "e"}
    assert node._is_partial_meta_compatible(meta, "u", 10, "e") is False


def test_is_partial_meta_compatible_etag_mismatch(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    meta = {"source_url": "u", "remote_size": 10, "remote_etag": "old-etag"}
    assert node._is_partial_meta_compatible(meta, "u", 10, "new-etag") is False


def test_is_partial_meta_compatible_all_match(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    meta = {"source_url": "u", "remote_size": 10, "remote_etag": "e"}
    assert node._is_partial_meta_compatible(meta, "u", 10, "e") is True


def test_is_partial_meta_compatible_ignores_missing_remote_size(downloader_module):
    """When remote size is unknown (-1) we must not invalidate the .part."""
    node = downloader_module.TS_DownloadFilesNode()
    meta = {"source_url": "u", "remote_size": 5, "remote_etag": "e"}
    assert node._is_partial_meta_compatible(meta, "u", -1, "e") is True


def test_get_headers_for_hf_url(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    h = node._get_headers_for_url("https://huggingface.co/x", "secret", "")
    assert h.get("Authorization") == "Bearer secret"


def test_get_headers_for_hf_url_without_token(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    h = node._get_headers_for_url("https://huggingface.co/x", "", "")
    assert "Authorization" not in h


def test_get_headers_for_hf_mirror(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    h = node._get_headers_for_url("https://hf-mirror.com/x", "secret", "")
    assert h.get("Authorization") == "Bearer secret"


def test_get_headers_for_modelscope(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    h = node._get_headers_for_url("https://modelscope.cn/x", "", "ms-secret")
    assert h.get("Authorization") == "Bearer ms-secret"
    assert h.get("Referer") == "https://www.modelscope.cn/"


def test_get_headers_for_modelscope_without_token(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    h = node._get_headers_for_url("https://modelscope.cn/x", "", "")
    assert "Authorization" not in h
    assert h.get("Referer") == "https://www.modelscope.cn/"


def test_get_headers_for_plain_url(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    h = node._get_headers_for_url("https://example.com/x", "ignored", "ignored")
    assert h == {}


def test_process_dropbox_url_flips_dl0_to_dl1(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._process_dropbox_url("https://www.dropbox.com/sh/foo?dl=0") == "https://www.dropbox.com/sh/foo?dl=1"


def test_process_dropbox_url_appends_dl_param(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    out = node._process_dropbox_url("https://www.dropbox.com/sh/foo")
    assert out == "https://www.dropbox.com/sh/foo?dl=1"
    out2 = node._process_dropbox_url("https://www.dropbox.com/sh/foo?token=abc")
    assert out2 == "https://www.dropbox.com/sh/foo?token=abc&dl=1"


def test_process_dropbox_url_leaves_non_dropbox(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._process_dropbox_url("https://example.com/x") == "https://example.com/x"


def test_parse_mirror_domains_handles_inputs(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._parse_mirror_domains("") == ["huggingface.co"]
    assert node._parse_mirror_domains(None) == ["huggingface.co"]
    assert node._parse_mirror_domains("huggingface.co") == ["huggingface.co"]
    assert node._parse_mirror_domains("huggingface.co, hf-mirror.com") == ["huggingface.co", "hf-mirror.com"]
    # Dedupes
    assert node._parse_mirror_domains("a.com, a.com, b.com") == ["a.com", "b.com"]
    # Strips protocol and trailing slash
    assert node._parse_mirror_domains("https://a.com/, http://b.com/") == ["a.com", "b.com"]
    # All-empty after cleanup -> default
    assert node._parse_mirror_domains(" , , ") == ["huggingface.co"]


def test_build_connectivity_probes_expands_hf_mirrors(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    parsed = [{"url": "https://huggingface.co/repo/file.bin", "target_dir": "/tmp"}]
    probes = node._build_connectivity_probes(parsed, "huggingface.co, hf-mirror.com")
    assert set(probes.keys()) == {"https://huggingface.co", "https://hf-mirror.com"}
    assert probes["https://hf-mirror.com"] == "https://hf-mirror.com/repo/file.bin"


def test_build_connectivity_probes_non_hf_uses_real_url(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    parsed = [{"url": "https://files.example.com/a.bin", "target_dir": "/tmp"}]
    probes = node._build_connectivity_probes(parsed, "huggingface.co")
    assert probes == {"https://files.example.com": "https://files.example.com/a.bin"}


def test_build_connectivity_probes_mixed(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    parsed = [
        {"url": "https://huggingface.co/r/f.bin", "target_dir": "/tmp"},
        {"url": "https://example.com/x.bin", "target_dir": "/tmp"},
    ]
    probes = node._build_connectivity_probes(parsed, "huggingface.co, hf-mirror.com")
    assert "https://huggingface.co" in probes
    assert "https://hf-mirror.com" in probes
    assert "https://example.com" in probes


def test_select_best_mirror_picks_first_with_status_below_500(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            if "primary.example" in url:
                raise downloader_module.requests.RequestException("down")
            return types.SimpleNamespace(status_code=200)

    chosen = node._select_best_mirror(_Sess(), "primary.example, secondary.example")
    assert chosen == "secondary.example"


def test_select_best_mirror_skips_server_errors(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            if "first.example" in url:
                return types.SimpleNamespace(status_code=502)
            return types.SimpleNamespace(status_code=200)

    chosen = node._select_best_mirror(_Sess(), "first.example, second.example")
    assert chosen == "second.example"


def test_select_best_mirror_returns_first_when_all_fail(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()

    class _Sess:
        def head(self, url, timeout=None, allow_redirects=False):
            raise downloader_module.requests.RequestException("nope")

    chosen = node._select_best_mirror(_Sess(), "a.example, b.example")
    assert chosen == "a.example"


def test_check_connectivity_empty_list_returns_true(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()

    class _Sess:
        def head(self, *a, **kw):
            raise downloader_module.requests.RequestException("should not be called")

    assert node._check_connectivity_to_targets([], _Sess(), "huggingface.co") is True


# ---------------------------------------------------------------------------
# JSON / filesystem helpers
# ---------------------------------------------------------------------------


def test_write_then_read_json_file_roundtrip(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    target = ts_tmp_path / "meta.json"
    payload = {"source_url": "u", "remote_size": 1234, "sha256": "deadbeef"}
    assert node._write_json_file(str(target), payload) is True
    assert node._read_json_file(str(target)) == payload


def test_read_json_file_missing_returns_none(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._read_json_file(str(ts_tmp_path / "absent.json")) is None


def test_read_json_file_bad_content_returns_none(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    bad = ts_tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert node._read_json_file(str(bad)) is None


def test_read_json_file_non_dict_returns_none(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    arr = ts_tmp_path / "arr.json"
    arr.write_text("[1,2,3]", encoding="utf-8")
    assert node._read_json_file(str(arr)) is None


def test_write_json_file_does_not_leave_tmp_on_success(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    target = ts_tmp_path / "ok.json"
    node._write_json_file(str(target), {"k": "v"})
    assert not (ts_tmp_path / "ok.json.tmp").exists()


def test_remove_file_silent_removes_existing(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    p = ts_tmp_path / "x.txt"
    p.write_text("hi")
    node._remove_file_silent(str(p))
    assert not p.exists()


def test_remove_file_silent_ignores_missing(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    # Must not raise
    node._remove_file_silent(str(ts_tmp_path / "never.txt"))


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_get_filename_from_header_plain(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    headers = {"content-disposition": 'attachment; filename="model.safetensors"'}
    assert node._get_filename_from_header_map(headers) == "model.safetensors"


def test_get_filename_from_header_rfc5987(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    headers = {"content-disposition": "attachment; filename*=UTF-8''hello.bin"}
    # Stub requests_unquote is identity, but the parser must still extract the bare value.
    assert node._get_filename_from_header_map(headers) == "hello.bin"


def test_get_filename_from_header_prefers_utf8_form(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    headers = {"content-disposition": "attachment; filename=old.bin; filename*=UTF-8''new.bin"}
    assert node._get_filename_from_header_map(headers) == "new.bin"


def test_get_filename_from_header_missing(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._get_filename_from_header_map({}) is None
    assert node._get_filename_from_header_map({"x": "y"}) is None


# ---------------------------------------------------------------------------
# Target path resolution
# ---------------------------------------------------------------------------


def test_resolve_target_directory_handles_none_and_empty(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    assert node._resolve_target_directory(None) is None
    assert node._resolve_target_directory("") is None
    assert node._resolve_target_directory("   ") is None


def test_resolve_target_directory_strips_quotes(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    quoted = f'"{ts_tmp_path / "out"}"'
    resolved = node._resolve_target_directory(quoted)
    assert resolved == str((ts_tmp_path / "out").resolve())


def test_resolve_target_directory_relative_uses_base_path(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    resolved = node._resolve_target_directory("foo/bar")
    assert resolved == str((ts_tmp_path / "foo" / "bar").resolve())


def test_resolve_target_directory_leading_dot_slash(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    resolved = node._resolve_target_directory("./out")
    assert resolved == str((ts_tmp_path / "out").resolve())


# ---------------------------------------------------------------------------
# parse_file_list edge cases
# ---------------------------------------------------------------------------


def test_parse_file_list_skips_lines_without_target(downloader_module, ts_tmp_path, monkeypatch):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    parsed = node._parse_file_list("https://example.com/lonely.bin")
    assert parsed == []


def test_parse_file_list_inline_comments_left_alone(downloader_module, ts_tmp_path, monkeypatch):
    """Lines starting with # are dropped; lines that merely contain '#' inside path are kept."""
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    text = "\n".join([
        "# top comment",
        "https://example.com/a.bin /tmp/a",
        "  # indented comment must be skipped",
        "https://example.com/b.bin /tmp/b",
    ])
    parsed = node._parse_file_list(text)
    assert [p["url"] for p in parsed] == ["https://example.com/a.bin", "https://example.com/b.bin"]


# ---------------------------------------------------------------------------
# Session / schema smoke
# ---------------------------------------------------------------------------


def test_create_session_with_retries_sets_user_agent(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    session = node._create_session_with_retries()
    assert "User-Agent" in session.headers
    assert "Mozilla" in session.headers["User-Agent"]


def test_create_session_with_retries_attaches_proxy(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    session = node._create_session_with_retries("http://proxy.example:3128")
    assert session.proxies.get("http") == "http://proxy.example:3128"
    assert session.proxies.get("https") == "http://proxy.example:3128"


def test_create_session_with_retries_ignores_whitespace_proxy(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    session = node._create_session_with_retries("   ")
    assert session.proxies == {}


def test_node_mappings_exposes_class(downloader_module):
    mod = downloader_module
    assert "TS Files Downloader" in mod.NODE_CLASS_MAPPINGS
    assert mod.NODE_CLASS_MAPPINGS["TS Files Downloader"] is mod.TS_DownloadFilesNode
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["TS Files Downloader"] == "TS Files Downloader (Ultimate)"


def test_define_schema_preserves_contract(downloader_module):
    schema = downloader_module.TS_DownloadFilesNode.define_schema()
    assert schema.node_id == "TS Files Downloader"
    assert schema.display_name == "TS Files Downloader (Ultimate)"
    assert schema.category == "TS/Files"
    assert schema.is_output_node is True
    assert schema.outputs == []


# ---------------------------------------------------------------------------
# Additional zip member safety
# ---------------------------------------------------------------------------


def test_is_zip_member_safe_rejects_empty(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    root = str((ts_tmp_path / "extract").resolve())
    import os as _os
    _os.makedirs(root, exist_ok=True)
    assert node._is_zip_member_safe("", root) is False
    assert node._is_zip_member_safe(".", root) is False


def test_is_zip_member_safe_accepts_deep_subdir(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    root = str((ts_tmp_path / "extract").resolve())
    import os as _os
    _os.makedirs(root, exist_ok=True)
    assert node._is_zip_member_safe("a/b/c/d.bin", root) is True


def test_is_zip_member_safe_handles_backslash_separators(downloader_module, ts_tmp_path):
    node = downloader_module.TS_DownloadFilesNode()
    root = str((ts_tmp_path / "extract").resolve())
    import os as _os
    _os.makedirs(root, exist_ok=True)
    # Zip files sometimes carry backslash paths from Windows producers.
    # After normalisation a plain backslash-relative path stays safe; traversal still rejected.
    assert node._is_zip_member_safe("sub\\file.bin", root) is True
    assert node._is_zip_member_safe("..\\evil.bin", root) is False


# ---------------------------------------------------------------------------
# Sanitizer edge cases
# ---------------------------------------------------------------------------


def test_sanitize_filename_none_or_empty(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    a = node._sanitize_filename(None)
    b = node._sanitize_filename("")
    assert a.startswith("downloaded_file_")
    assert b.startswith("downloaded_file_")


def test_sanitize_filename_only_whitespace_or_dots(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._sanitize_filename("   ").startswith("downloaded_file_")
    assert node._sanitize_filename("....").startswith("downloaded_file_")


def test_sanitize_filename_strips_path_components(downloader_module):
    node = downloader_module.TS_DownloadFilesNode()
    assert node._sanitize_filename("/abs/path/model.bin") == "model.bin"
    assert node._sanitize_filename("a/b/c/file.safetensors") == "file.safetensors"


# ---------------------------------------------------------------------------
# Integration smoke for _download_single_file via mocked session
# ---------------------------------------------------------------------------


class _MockResponse:
    """Minimal requests-like response sufficient for _download_single_file paths."""

    def __init__(self, *, status_code=200, headers=None, body=b"", url=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise downloader_for_mock_response.requests.RequestException(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


# Set after fixture creates module; see _bind_mock_response.
downloader_for_mock_response = None


def _bind_mock_response(mod):
    global downloader_for_mock_response
    downloader_for_mock_response = mod


def test_download_single_file_happy_path(downloader_module, ts_tmp_path):
    _bind_mock_response(downloader_module)
    node = downloader_module.TS_DownloadFilesNode()
    payload = b"hello timesaver " * 100
    headers = {
        "content-length": str(len(payload)),
        "accept-ranges": "bytes",
        "etag": '"deadbeef"',
    }

    class _Sess:
        def head(self, url, **kw):
            r = _MockResponse(status_code=200, headers=headers, url=url)
            return r

        def get(self, url, **kw):
            return _MockResponse(status_code=200, headers=headers, body=payload, url=url)

    target = ts_tmp_path / "out"
    target.mkdir()
    ok = node._download_single_file(
        _Sess(),
        "https://example.com/model.safetensors",
        str(target),
        skip_existing=True,
        verify_size=True,
        chunk_size_bytes=16,
        hf_domain_active="huggingface.co",
        hf_token="",
        ms_token="",
        unzip_after_download=False,
        integrity_mode="size_only",
    )
    assert ok is True
    saved = target / "model.safetensors"
    assert saved.exists()
    assert saved.read_bytes() == payload
    # Meta file written next to model
    meta = target / "model.safetensors.tsmeta.json"
    assert meta.exists()


def test_download_single_file_skips_existing_size_match(downloader_module, ts_tmp_path):
    _bind_mock_response(downloader_module)
    node = downloader_module.TS_DownloadFilesNode()
    payload = b"already-on-disk" * 50
    target = ts_tmp_path / "out2"
    target.mkdir()
    existing = target / "data.bin"
    existing.write_bytes(payload)

    headers = {"content-length": str(len(payload)), "accept-ranges": "bytes"}
    calls = {"get": 0}

    class _Sess:
        def head(self, url, **kw):
            return _MockResponse(status_code=200, headers=headers, url=url)

        def get(self, url, **kw):
            calls["get"] += 1
            return _MockResponse(status_code=200, headers=headers, body=payload, url=url)

    ok = node._download_single_file(
        _Sess(),
        "https://example.com/data.bin",
        str(target),
        skip_existing=True,
        verify_size=True,
        chunk_size_bytes=16,
        hf_domain_active="huggingface.co",
        hf_token="",
        ms_token="",
        unzip_after_download=False,
        integrity_mode="size_only",
    )
    assert ok is True
    assert calls["get"] == 0, "must not re-download when local size matches remote"
    assert existing.read_bytes() == payload


def test_download_single_file_resumes_from_partial(downloader_module, ts_tmp_path):
    _bind_mock_response(downloader_module)
    node = downloader_module.TS_DownloadFilesNode()
    full_payload = b"timesaver-resume-test" * 20
    half = len(full_payload) // 2

    target = ts_tmp_path / "out3"
    target.mkdir()
    part = target / "resume.bin.part"
    part.write_bytes(full_payload[:half])
    node._write_json_file(str(target / "resume.bin.part.tsmeta.json"), {
        "source_url": "https://example.com/resume.bin",
        "remote_size": len(full_payload),
        "remote_etag": "abc",
    })

    head_headers = {
        "content-length": str(len(full_payload)),
        "accept-ranges": "bytes",
        "etag": '"abc"',
    }
    range_headers = {
        "content-range": f"bytes {half}-{len(full_payload)-1}/{len(full_payload)}",
        "content-length": str(len(full_payload) - half),
    }
    seen_ranges = []

    class _Sess:
        def head(self, url, **kw):
            return _MockResponse(status_code=200, headers=head_headers, url=url)

        def get(self, url, **kw):
            req_headers = kw.get("headers") or {}
            if "Range" in req_headers:
                seen_ranges.append(req_headers["Range"])
                return _MockResponse(
                    status_code=206, headers=range_headers,
                    body=full_payload[half:], url=url,
                )
            return _MockResponse(status_code=200, headers=head_headers, body=full_payload, url=url)

    ok = node._download_single_file(
        _Sess(),
        "https://example.com/resume.bin",
        str(target),
        skip_existing=False,
        verify_size=True,
        chunk_size_bytes=8,
        hf_domain_active="huggingface.co",
        hf_token="",
        ms_token="",
        unzip_after_download=False,
        integrity_mode="size_only",
    )
    assert ok is True, "resume must succeed"
    assert seen_ranges and seen_ranges[0].startswith(f"bytes={half}-")
    assert (target / "resume.bin").read_bytes() == full_payload
    assert not part.exists(), ".part must be promoted on success"


def test_download_single_file_returns_false_when_probe_fails(downloader_module, ts_tmp_path):
    _bind_mock_response(downloader_module)
    node = downloader_module.TS_DownloadFilesNode()
    target = ts_tmp_path / "out4"
    target.mkdir()

    class _Sess:
        def head(self, url, **kw):
            raise downloader_module.requests.RequestException("head down")

        def get(self, url, **kw):
            raise downloader_module.requests.RequestException("get down")

    ok = node._download_single_file(
        _Sess(),
        "https://example.com/missing.bin",
        str(target),
        skip_existing=True,
        verify_size=True,
        chunk_size_bytes=16,
        hf_domain_active="huggingface.co",
        hf_token="",
        ms_token="",
        unzip_after_download=False,
        integrity_mode="size_only",
    )
    assert ok is False
    assert list(target.iterdir()) == []


def test_execute_no_files_returns_quickly(downloader_module, monkeypatch, ts_tmp_path):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    result = node.execute(file_list="# only a comment\n")
    assert result is not None  # NodeOutput stub


def test_execute_disabled_short_circuits(downloader_module, monkeypatch, ts_tmp_path):
    monkeypatch.setattr(downloader_module, "folder_paths", types.SimpleNamespace(
        base_path=str(ts_tmp_path), models_dir=str(ts_tmp_path / "models")
    ))
    node = downloader_module.TS_DownloadFilesNode()
    called = {"session": 0}

    def _fake_session(*a, **kw):
        called["session"] += 1
        return None

    monkeypatch.setattr(downloader_module.TS_DownloadFilesNode, "_create_session_with_retries", staticmethod(_fake_session))
    result = node.execute(
        file_list="https://example.com/a.bin /tmp/a",
        enable=False,
    )
    assert result is not None
    assert called["session"] == 0, "disabled execute must skip all I/O"
