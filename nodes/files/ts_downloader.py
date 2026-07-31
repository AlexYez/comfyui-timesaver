# Standard library imports
import hashlib
import json
import logging
import os
import re
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

try:
    from requests.utils import unquote as requests_unquote
except ImportError:
    from urllib.parse import unquote as requests_unquote

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_downloader")
LOG_PREFIX = "[TS Downloader]"

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _resolve_pack_version() -> str:
    """Best-effort read of the pack version from pyproject.toml.

    Custom nodes are git-cloned (not pip-installed) into custom_nodes/, so
    importlib.metadata usually can't find us. Read the version straight from
    the sibling pyproject.toml with a small regex (no tomllib dependency, so
    this still works on Python 3.10). Falls back to "dev" if unreadable.
    """
    try:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        match = re.search(
            r'^\s*version\s*=\s*["\']([^"\']+)["\']',
            pyproject.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    except OSError:
        pass
    return "dev"


# Honest client identifier sent on every download request. This was previously
# a spoofed desktop-Chrome User-Agent, which static-analysis security scanners
# (including the Comfy registry scanner) flag as evasion/masquerading. Sending
# the pack's real name + version removes that signal and is the polite thing to
# advertise to the hosts we pull from.
_USER_AGENT = f"comfyui-timesaver/{_resolve_pack_version()}"


def _redact_proxy_url(proxy_url: str) -> str:
    """Return a proxy URL safe to log — scheme, host and port only.

    Proxy URLs routinely carry ``user:password@`` userinfo, which must never
    reach the console or a log file (CLAUDE.md §15).
    """
    candidate = (proxy_url or "").strip()
    if not candidate:
        return ""
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return "<proxy>"
    if not parsed.hostname:
        # Unparseable (e.g. a bare "host:port") — do not echo the raw value back.
        return "<proxy>"
    netloc = parsed.hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"***@{netloc}"
    scheme = f"{parsed.scheme}://" if parsed.scheme else ""
    return f"{scheme}{netloc}"


# One lock per destination file, shared by every node instance in this process.
# Two loader nodes pointing at the same URL (or the same graph queued twice)
# used to write the SAME "<file>.part" concurrently: the bytes interleaved and,
# with integrity_mode="size_only", the mixed result could still pass the size
# check and be accepted as a model. The lock serialises them instead — the
# second writer then finds the finished file and skips it. Locking the path
# rather than renaming the partial keeps resume-after-restart working.
_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}
_DOWNLOAD_LOCKS_GUARD = threading.Lock()


def _download_lock_for(path: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _DOWNLOAD_LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DOWNLOAD_LOCKS[key] = lock
        return lock


class TS_DownloadFilesNode(IO.ComfyNode):
    """
    A ComfyUI node to download files.
    Features: Offline Mode (Target-based check), Enable/Disable Toggle,
    Auto-Unzip, UI Progress Bar, Resume / Mirrors / Proxies.
    """

    # Offline detection tuning: a short (connect, read) timeout used by the
    # no-retry probe session (see execute) so a machine with no internet flips
    # to OFFLINE MODE within a few seconds per target instead of hanging for a
    # minute on retry storms.
    _CONNECTIVITY_TIMEOUT = (4, 6)

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS Files Downloader",
            display_name="TS Files Downloader (Ultimate)",
            category="TS/Files",
            description="Download every model a workflow needs from a list of URL + target-folder lines, with resume, mirrors and integrity checks.",
            is_output_node=True,
            inputs=[
                IO.String.Input(
                    "file_list",
                    default="https://www.dropbox.com/sh/example_folder?dl=0 /path/to/models\nhttps://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors /path/to/checkpoints",
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip="One download per line, formatted as '<url> <target_dir>'. Target may be absolute, a 'models/...' path, or a registered model folder name (checkpoints, loras, vae). Lines starting with # are ignored.",
                ),
                IO.Boolean.Input("skip_existing", default=True, tooltip="Skip a file that already exists in the target folder instead of downloading it again."),
                IO.Boolean.Input("verify_size", default=True, tooltip="Verify each finished file against the size (or HF SHA256) reported by the server; re-download or resume on mismatch."),
                IO.Int.Input("chunk_size_kb", default=4096, min=1, max=65536, step=1, tooltip="Streaming chunk size in KB. Larger values can be faster on fast links; smaller values use less memory."),
                IO.String.Input(
                    "hf_token", default="", multiline=False, optional=True,
                    tooltip="HuggingFace token for gated repos. WARNING: stored in the workflow JSON as plain text — do not share a file that contains it.",
                ),
                IO.String.Input("hf_domain", default="huggingface.co, hf-mirror.com", multiline=False, optional=True, tooltip="Comma-separated HuggingFace mirror domains. The first reachable one replaces huggingface.co in each URL. Useful in regions where the main host is slow or blocked."),
                IO.String.Input(
                    "proxy_url", default="", multiline=False, optional=True,
                    tooltip="Proxy URL (may contain credentials). WARNING: stored in the workflow JSON as plain text — do not share a file that contains it.",
                ),
                IO.String.Input(
                    "modelscope_token", default="", multiline=False, optional=True,
                    tooltip="ModelScope token. WARNING: stored in the workflow JSON as plain text — do not share a file that contains it.",
                ),
                IO.Boolean.Input("unzip_after_download", default=False, optional=True, tooltip="After a successful download, extract any .zip archive into its target folder and delete the archive."),
                IO.Boolean.Input("enable", default=True, optional=True, tooltip="Master switch. When off, the node does nothing and returns immediately."),
                IO.Combo.Input("integrity_mode", options=["hf_sha256_auto", "size_only"], default="hf_sha256_auto", optional=True, tooltip="Integrity check strategy. 'hf_sha256_auto' verifies HuggingFace files by SHA256 when available; 'size_only' checks byte size alone."),
            ],
            outputs=[],
        )

    @staticmethod
    def _create_session_with_retries(proxy_url=None, total_retries=3):
        session = requests.Session()
        if total_retries and total_retries > 0:
            retries = Retry(
                total=total_retries,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(['HEAD', 'GET'])
            )
            adapter = HTTPAdapter(max_retries=retries)
        else:
            # total_retries=0 -> a connection error raises on the first attempt
            # instead of retrying 3x with exponential backoff. The connectivity
            # probe uses this so OFFLINE MODE is detected fast (see execute).
            adapter = HTTPAdapter(max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({
            'User-Agent': _USER_AGENT,
            # Big binaries must arrive byte-identical to Content-Length. When a
            # server transparently gzips the stream, requests inflates it on the
            # fly, the file on disk grows PAST the advertised size, and every
            # size check (skip_existing, post-download verify) flags it corrupt
            # -> the same model re-downloads on every single run.
            'Accept-Encoding': 'identity',
        })

        if proxy_url and proxy_url.strip():
            logger.info(
                "%s TS_DownloadNode: Proxy enabled: %s",
                LOG_PREFIX,
                _redact_proxy_url(proxy_url),
            )
            session.proxies = {
                'http': proxy_url.strip(),
                'https': proxy_url.strip(),
            }
        return session

    @staticmethod
    def _replace_hf_domain(url, target_domain):
        if not target_domain or target_domain.strip() == "huggingface.co":
            return url
        clean_domain = target_domain.replace("https://", "").replace("http://", "").strip("/")
        pattern = r"(https?://)(www\.)?huggingface\.co(?=[/:?#]|$)"
        if re.search(pattern, url):
            return re.sub(pattern, f"\\1{clean_domain}", url)
        return url

    @staticmethod
    def _parse_mirror_domains(hf_domain_str):
        if not hf_domain_str:
            return ["huggingface.co"]
        domains = []
        for raw in hf_domain_str.split(','):
            cleaned = raw.replace("https://", "").replace("http://", "").strip("/").strip()
            if cleaned and cleaned not in domains:
                domains.append(cleaned)
        return domains or ["huggingface.co"]

    @classmethod
    def _build_connectivity_probes(cls, file_list_parsed, hf_domain_str):
        mirror_domains = cls._parse_mirror_domains(hf_domain_str)
        probes = {}
        for item in file_list_parsed:
            url = item['url']
            if cls._is_hf_url(url):
                for mirror in mirror_domains:
                    candidate_url = cls._replace_hf_domain(url, mirror)
                    try:
                        parsed = urlparse(candidate_url)
                        base = f"{parsed.scheme}://{parsed.netloc}"
                    except Exception as exc:
                        logger.debug("%s Could not parse URL '%s': %s", LOG_PREFIX, candidate_url, exc)
                        continue
                    probes.setdefault(base, candidate_url)
            else:
                try:
                    parsed = urlparse(url)
                    base = f"{parsed.scheme}://{parsed.netloc}"
                except Exception as exc:
                    logger.debug("%s Could not parse URL '%s': %s", LOG_PREFIX, url, exc)
                    continue
                probes.setdefault(base, url)
        return probes

    @classmethod
    def _check_connectivity_to_targets(cls, file_list_parsed, session, hf_domain_str):
        probes = cls._build_connectivity_probes(file_list_parsed, hf_domain_str)
        if not probes:
            return True

        logger.info(f"{LOG_PREFIX} Checking connectivity to targets: {list(probes.keys())} ...")
        for base_url, probe_url in probes.items():
            try:
                session.head(probe_url, timeout=cls._CONNECTIVITY_TIMEOUT, allow_redirects=True)
                logger.info(f"{LOG_PREFIX} Target '{base_url}' is REACHABLE.")
                return True
            except requests.RequestException:
                # Some hosts/CDNs drop bare HEAD requests. That must not flip a
                # perfectly online machine into OFFLINE MODE ("the node just
                # doesn't download") — settle reachability with a one-byte GET.
                try:
                    with session.get(
                        probe_url,
                        headers={"Range": "bytes=0-0"},
                        stream=True,
                        timeout=cls._CONNECTIVITY_TIMEOUT,
                        allow_redirects=True,
                    ):
                        pass
                    logger.info(f"{LOG_PREFIX} Target '{base_url}' is REACHABLE (via GET probe; HEAD rejected).")
                    return True
                except requests.RequestException:
                    logger.warning(f"{LOG_PREFIX} Target '{base_url}' is UNREACHABLE.")
                    continue

        return False

    @staticmethod
    def _select_best_mirror(session, domain_list_str):
        if not domain_list_str:
            return "huggingface.co"
        domains = [d.strip() for d in domain_list_str.split(',') if d.strip()]
        if not domains:
            return "huggingface.co"
        if len(domains) == 1:
            return domains[0]

        for domain in domains:
            clean_domain = domain.replace("https://", "").replace("http://", "").strip("/")
            test_url = f"https://{clean_domain}"
            try:
                response = session.head(test_url, timeout=2, allow_redirects=True)
                if response.status_code < 500:
                    return clean_domain
            except requests.RequestException:
                continue
        return domains[0]

    @staticmethod
    def _url_host(url):
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return ""
        if "@" in host:
            host = host.rsplit("@", 1)[-1]
        return host.split(":", 1)[0]

    @classmethod
    def _is_modelscope_url(cls, url):
        host = cls._url_host(url)
        return host == "modelscope.cn" or host.endswith(".modelscope.cn")

    @classmethod
    def _get_headers_for_url(cls, url, hf_token, ms_token, hf_domain_active=None):
        # Host-based matching only: the old substring checks ("hf-" in url,
        # "modelscope.cn" in url) attached the Bearer token to any URL that
        # merely contained those strings in its path — a credential leak to
        # arbitrary hosts. `hf_domain_active` extends the HF allowlist to the
        # user-configured mirror the download actually targets.
        headers = {}
        hf_hosts_ok = cls._is_hf_url(url)
        if not hf_hosts_ok and hf_domain_active and hf_token and hf_token.strip():
            # `hf_domain` is a widget, so its value travels INSIDE the workflow:
            # a graph from someone else can name any host there. Extending the
            # token allowlist to that value handed the user's HuggingFace token
            # to whatever host the shared file asked for, so the allowlist is
            # now static (_is_hf_url). A custom mirror still serves the
            # download — it just does not receive the token.
            mirror = cls._parse_mirror_domains(hf_domain_active)[0].lower()
            mirror = mirror.split("/", 1)[0].split(":", 1)[0]
            host = cls._url_host(url)
            if mirror and (host == mirror or host.endswith("." + mirror)):
                logger.warning(
                    f"{LOG_PREFIX} Mirror '{mirror}' is not a HuggingFace host — "
                    f"downloading without the token."
                )
        if hf_hosts_ok:
            if hf_token and hf_token.strip():
                headers["Authorization"] = f"Bearer {hf_token.strip()}"
        elif cls._is_modelscope_url(url):
            headers["Referer"] = "https://www.modelscope.cn/"
            if ms_token and ms_token.strip():
                headers["Authorization"] = f"Bearer {ms_token.strip()}"
        return headers

    @staticmethod
    def _process_dropbox_url(url):
        if "dropbox.com" in url:
            if "dl=0" in url:
                return url.replace("dl=0", "dl=1")
            if "dl=" not in url:
                return url + ("&dl=1" if "?" in url else "?dl=1")
        return url

    @staticmethod
    def _known_model_folder_root(folder_name: str) -> str | None:
        """Absolute root registered in ComfyUI for a model-folder name
        ("checkpoints", "loras", "vae", "diffusion_models", ...) — honouring
        extra_model_paths.yaml. None when the name is not a registered folder.
        """
        if not folder_paths:
            return None
        try:
            registered = getattr(folder_paths, "folder_names_and_paths", {}) or {}
            wanted = folder_name.lower()
            for name in registered.keys():
                if str(name).lower() != wanted:
                    continue
                paths = folder_paths.get_folder_paths(name)
                if paths:
                    return str(paths[0])
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} Known-folder lookup failed for '{folder_name}': {exc}")
        return None

    @staticmethod
    def _is_within(root, candidate) -> bool:
        """True when `candidate` resolves inside `root`.

        commonpath raises ValueError for paths on different Windows drives and
        OSError for an unusable root — both answer the question with "outside".
        """
        if not root:
            return False
        try:
            root_abs = os.path.abspath(str(root))
            candidate_abs = os.path.abspath(str(candidate))
            return os.path.commonpath([root_abs, candidate_abs]) == root_abs
        except (ValueError, OSError):
            return False

    @classmethod
    def _resolve_target_directory(cls, target_path):
        if target_path is None:
            return None

        cleaned = target_path.strip().strip('"').strip("'")
        if not cleaned:
            return None

        expanded = os.path.expandvars(os.path.expanduser(cleaned))

        # Honour a target only when it is TRULY absolute: a drive letter or a
        # UNC share on Windows, or any absolute path on POSIX. A bare leading
        # slash on Windows ("\models\vae" — an easy typo) also passes
        # os.path.isabs(), but abspath() would glue it to the CURRENT DRIVE
        # root and the models silently land in C:\models, outside ComfyUI.
        # Treat that rooted-but-driveless form as a relative path instead.
        drive, _tail = os.path.splitdrive(expanded)
        if os.path.isabs(expanded) and (drive or os.sep == "/"):
            resolved = os.path.abspath(expanded)
            # An absolute target stays honoured — "D:/models/checkpoints" is a
            # documented, deliberate choice. It is also the one form that can
            # name any location on the machine, and since the workflow scanner
            # exists a line may be written by a shared graph rather than typed
            # here, so a destination outside ComfyUI is worth saying out loud.
            base_path = getattr(folder_paths, "base_path", None) if folder_paths else None
            if base_path and not cls._is_within(base_path, resolved):
                logger.warning(
                    f"{LOG_PREFIX} Target '{target_path}' resolves OUTSIDE ComfyUI -> '{resolved}'. "
                    f"Honouring it, but check this line if you did not write it yourself."
                )
            return resolved

        normalized = expanded.replace("\\", "/").lstrip("/")
        while normalized.startswith("./"):
            normalized = normalized[2:].lstrip("/")
        if not normalized:
            return None

        first, _, rest = normalized.partition("/")
        first_lower = first.lower()

        # models[/...] -> the real ComfyUI models directory, never base_path.
        if first_lower == "models":
            models_root = None
            if folder_paths and getattr(folder_paths, "models_dir", None):
                models_root = folder_paths.models_dir
            if not models_root and folder_paths and getattr(folder_paths, "base_path", None):
                models_root = os.path.join(folder_paths.base_path, "models")
            if models_root:
                resolved = os.path.abspath(os.path.join(models_root, rest)) if rest else os.path.abspath(models_root)
                # "models/../../elsewhere" normalises straight out of the models
                # directory. A relative target names a folder INSIDE the root it
                # picked — anything that climbs back out is refused, not clamped,
                # so the line is reported instead of silently retargeted.
                if not cls._is_within(models_root, resolved):
                    logger.warning(
                        f"{LOG_PREFIX} Target '{target_path}' escapes the models directory. Skipping this line."
                    )
                    return None
                logger.info(f"{LOG_PREFIX} Target '{target_path}' -> '{resolved}'")
                return resolved

        # A registered model-folder name used directly ("checkpoints",
        # "loras/flux", "vae", ...) resolves into that model folder —
        # honouring extra_model_paths.yaml — instead of leaking into
        # ComfyUI's root as a stray sibling directory.
        known_root = cls._known_model_folder_root(first_lower)
        if known_root:
            resolved = os.path.abspath(os.path.join(known_root, rest)) if rest else os.path.abspath(known_root)
            if not cls._is_within(known_root, resolved):
                logger.warning(
                    f"{LOG_PREFIX} Target '{target_path}' escapes the '{first_lower}' model folder. Skipping this line."
                )
                return None
            logger.info(f"{LOG_PREFIX} Target '{target_path}' -> '{resolved}' (registered model folder)")
            return resolved

        if folder_paths and getattr(folder_paths, "base_path", None):
            resolved = os.path.abspath(os.path.join(folder_paths.base_path, normalized))
            if not cls._is_within(folder_paths.base_path, resolved):
                logger.warning(
                    f"{LOG_PREFIX} Target '{target_path}' escapes the ComfyUI directory. Skipping this line."
                )
                return None
            logger.info(f"{LOG_PREFIX} Target '{target_path}' -> '{resolved}'")
            return resolved

        return os.path.abspath(normalized)

    @classmethod
    def _parse_file_list(cls, file_list_text):
        files = []
        lines = file_list_text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                # A bare URL without a target path was previously dropped
                # with no message at all — the user just saw nothing happen.
                logger.warning(
                    f"{LOG_PREFIX} Line {i+1}: Expected '<url> <target_dir>' "
                    f"(two fields), got: {line[:120]!r}. Skipping."
                )
                continue
            url, target_path = parts[0].strip(), parts[1].strip()
            if not url.startswith(('http://', 'https://')):
                logger.warning(f"{LOG_PREFIX} Line {i+1}: Invalid URL.")
                continue
            target_path = cls._resolve_target_directory(target_path)
            if not target_path:
                logger.warning(f"{LOG_PREFIX} Line {i+1}: Invalid target path.")
                continue
            files.append({'url': url, 'target_dir': target_path})
        return files

    @staticmethod
    def _sanitize_filename(name, fallback_prefix="downloaded_file"):
        if not name:
            return f"{fallback_prefix}_{int(time.time())}"
        name = name.replace("\\", "/").split("/")[-1]
        if name in (".", ".."):
            return f"{fallback_prefix}_{int(time.time())}"
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
        name = name.strip().strip(".").strip()
        if not name:
            return f"{fallback_prefix}_{int(time.time())}"
        if len(name) > 200:
            stem, ext = os.path.splitext(name)
            name = stem[: max(1, 200 - len(ext))] + ext
        return name

    @staticmethod
    def _get_filename_from_header_map(headers):
        # Match the header case-insensitively. `headers` arrives as a plain dict
        # built from requests' CaseInsensitiveDict, and iterating that yields the
        # ORIGINAL casing ("Content-Disposition"), so a bare lowercase .get()
        # never matched: the server-supplied filename was always ignored and the
        # URL-derived name won (a Civitai /api/download/models/12345 link saved
        # as "12345", and skip-existing could not match a previous download).
        content_disposition = None
        for key, value in (headers or {}).items():
            if str(key).lower() == "content-disposition":
                content_disposition = value
                break
        if not content_disposition:
            return None
        fn_match_utf8 = re.search(r"filename\*=\s*UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if fn_match_utf8:
            return requests_unquote(fn_match_utf8.group(1).strip('" '))
        fn_match_plain = re.search(r'filename="?([^"]+)"?', content_disposition, re.IGNORECASE)
        if fn_match_plain:
            return requests_unquote(fn_match_plain.group(1).strip('" '))
        return None

    @staticmethod
    def _safe_int(value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_etag(etag_value):
        if not etag_value:
            return None
        etag = str(etag_value).strip()
        if etag.lower().startswith("w/"):
            etag = etag[2:].strip()
        etag = etag.strip().strip('"').strip("'")
        return etag or None

    @classmethod
    def _extract_total_size_from_content_range(cls, content_range):
        if not content_range:
            return -1
        match = re.search(r"/(\d+)$", str(content_range).strip())
        if not match:
            return -1
        return cls._safe_int(match.group(1), -1)

    @classmethod
    def _extract_remote_size_from_headers(cls, headers):
        total_from_range = cls._extract_total_size_from_content_range(headers.get("content-range"))
        if total_from_range > 0:
            return total_from_range

        size = cls._safe_int(headers.get("x-linked-size"), -1)
        if size > 0:
            return size
        size = cls._safe_int(headers.get("content-length"), -1)
        if size > 0:
            return size
        return -1

    @staticmethod
    def _is_hf_url(url):
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        if "@" in host:
            host = host.rsplit("@", 1)[-1]
        host = host.split(":", 1)[0]
        return (
            host == "huggingface.co"
            or host.endswith(".huggingface.co")
            or host == "hf-mirror.com"
            or host.endswith(".hf-mirror.com")
        )

    @classmethod
    def _extract_hf_expected_sha256(cls, remote_etag, final_url):
        if not remote_etag or not cls._is_hf_url(final_url):
            return None
        if re.fullmatch(r"[0-9a-fA-F]{64}", remote_etag):
            return remote_etag.lower()
        return None

    @staticmethod
    def _read_json_file(path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    @staticmethod
    def _write_json_file(path, payload):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return True
        except Exception as exc:
            # A meta file that never lands means e.g. the HF SHA256 gets
            # re-hashed on every run — worth a trace when debugging "slow" runs.
            logger.debug(f"{LOG_PREFIX} Could not write meta file '{path}': {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False

    @staticmethod
    def _remove_file_silent(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    @classmethod
    def _is_partial_meta_compatible(cls, meta, source_url, remote_size, remote_etag):
        if not isinstance(meta, dict):
            return True

        meta_url = meta.get("source_url")
        if meta_url and meta_url != source_url:
            return False

        meta_size = cls._safe_int(meta.get("remote_size"), -1)
        if remote_size > 0 and meta_size > 0 and meta_size != remote_size:
            return False

        meta_etag = meta.get("remote_etag")
        if meta_etag and remote_etag and meta_etag != remote_etag:
            return False

        return True

    @staticmethod
    def _compute_sha256(file_path, chunk_size=8 * 1024 * 1024, desc=None):
        hasher = hashlib.sha256()
        try:
            total = os.path.getsize(file_path)
        except OSError:
            total = 0
        comfy_pbar = ProgressBar(total) if (ProgressBar and total > 0) else None
        progress_desc = desc or f"SHA256: {os.path.basename(file_path)}"
        accumulated = 0
        ui_update_threshold = 8 * 1024 * 1024
        with open(file_path, "rb") as f, tqdm(
            total=total or None,
            unit='B', unit_scale=True, desc=progress_desc,
            mininterval=1.0, ncols=100, unit_divisor=1024,
        ) as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                chunk_len = len(chunk)
                pbar.update(chunk_len)
                if comfy_pbar:
                    accumulated += chunk_len
                    if accumulated >= ui_update_threshold:
                        comfy_pbar.update(accumulated)
                        accumulated = 0
            if comfy_pbar and accumulated > 0:
                comfy_pbar.update(accumulated)
        return hasher.hexdigest()

    @classmethod
    def _probe_remote_file(cls, session, processed_url, domain_headers):
        response = None
        used_get_probe = False

        try:
            response = session.head(
                processed_url,
                headers=domain_headers,
                allow_redirects=True,
                timeout=(10, 30),
            )
            response.raise_for_status()
        except requests.RequestException as head_error:
            if response is not None:
                response.close()
                response = None
            logger.warning(f"{LOG_PREFIX} HEAD probe failed: {head_error}. Trying lightweight GET probe...")
            probe_headers = domain_headers.copy()
            probe_headers["Range"] = "bytes=0-0"
            try:
                response = session.get(
                    processed_url,
                    stream=True,
                    headers=probe_headers,
                    timeout=(10, 30),
                    allow_redirects=True,
                )
                response.raise_for_status()
                used_get_probe = True
            except requests.RequestException as get_error:
                if response is not None:
                    response.close()
                logger.error(f"{LOG_PREFIX} Remote probe failed: {get_error}")
                return None

        try:
            remote_size = cls._extract_remote_size_from_headers(response.headers)
            remote_etag = cls._normalize_etag(response.headers.get("x-linked-etag") or response.headers.get("etag"))
            supports_ranges = "bytes" in str(response.headers.get("accept-ranges", "")).lower()
            if response.status_code == 206:
                supports_ranges = True

            final_url = response.url or processed_url
            hf_expected_sha256 = cls._extract_hf_expected_sha256(remote_etag, final_url)

            return {
                "final_url": final_url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "remote_size": remote_size,
                "remote_etag": remote_etag,
                "supports_ranges": supports_ranges,
                "hf_expected_sha256": hf_expected_sha256,
                "used_get_probe": used_get_probe,
            }
        finally:
            response.close()

    @staticmethod
    def _is_zip_member_safe(member_name, extract_root):
        if not member_name or member_name in (".", ".."):
            return False
        normalized = member_name.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized or normalized.endswith("/.."):
            return False
        head = normalized.split("/", 1)[0]
        if len(head) >= 2 and head[1] == ":":
            return False
        target = os.path.realpath(os.path.join(extract_root, member_name))
        if target == extract_root:
            return True
        return target.startswith(extract_root + os.sep)

    @classmethod
    def _extract_zip(cls, zip_path, extract_to):
        logger.info(f"{LOG_PREFIX} Auto-Unzip: Extracting '{os.path.basename(zip_path)}' to '{extract_to}'...")
        try:
            os.makedirs(extract_to, exist_ok=True)
            extract_root = os.path.realpath(extract_to)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for info in zip_ref.infolist():
                    if not cls._is_zip_member_safe(info.filename, extract_root):
                        logger.error(
                            "%s Refusing to extract unsafe zip member '%s' (path traversal).",
                            LOG_PREFIX, info.filename,
                        )
                        return False
                zip_ref.extractall(extract_to)
            logger.info(f"{LOG_PREFIX} Extraction complete. Deleting archive.")
            try:
                os.remove(zip_path)
            except OSError:
                pass
            cls._remove_file_silent(zip_path + ".tsmeta.json")
            cls._remove_file_silent(zip_path + ".part")
            cls._remove_file_silent(zip_path + ".part.tsmeta.json")
            return True
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Extraction failed: {e}")
            return False

    @classmethod
    def _download_single_file(cls, session, url, target_dir, skip_existing, verify_size, chunk_size_bytes, hf_domain_active, hf_token, ms_token, unzip_after_download, integrity_mode):
        response_get = None
        download_lock = None
        try:
            processed_url = cls._replace_hf_domain(url, hf_domain_active)
            processed_url = cls._process_dropbox_url(processed_url)

            os.makedirs(target_dir, exist_ok=True)
            domain_headers = cls._get_headers_for_url(
                processed_url, hf_token, ms_token, hf_domain_active=hf_domain_active
            )

            logger.info(f"{LOG_PREFIX} Connecting to: {processed_url}")
            remote_info = cls._probe_remote_file(session, processed_url, domain_headers)
            if not remote_info:
                return False

            remote_file_size = remote_info["remote_size"]
            remote_etag = remote_info["remote_etag"]
            final_direct_url = remote_info["final_url"] or processed_url
            supports_ranges = remote_info["supports_ranges"]
            hf_expected_sha256 = remote_info["hf_expected_sha256"]
            use_hf_sha256 = verify_size and integrity_mode == "hf_sha256_auto" and bool(hf_expected_sha256)

            filename_from_url = cls._sanitize_filename(
                os.path.basename(requests_unquote(url.split('?')[0].split('#')[0]))
            )
            header_name = cls._get_filename_from_header_map(remote_info["headers"])
            filename_from_header = cls._sanitize_filename(header_name) if header_name else None
            # The Content-Disposition name can differ between probe methods (a
            # HEAD often lacks the header that a GET later supplies), which used
            # to make skip_existing miss an already-downloaded file and pull the
            # same model again under the other name. Prefer whichever candidate
            # ALREADY EXISTS in the target dir; otherwise keep the old priority
            # (header name over URL name).
            candidates = [n for n in (filename_from_header, filename_from_url) if n]
            final_filename = next(
                (n for n in candidates if os.path.exists(os.path.join(target_dir, n))),
                candidates[0],
            )

            local_file_path = os.path.join(target_dir, final_filename)
            # Everything below reads, resumes, writes and renames this exact
            # file, so only one download of it may be in flight per process.
            download_lock = _download_lock_for(local_file_path)
            download_lock.acquire()
            temp_file_path = local_file_path + ".part"
            temp_meta_path = temp_file_path + ".tsmeta.json"
            final_meta_path = local_file_path + ".tsmeta.json"

            size_label = remote_file_size if remote_file_size > 0 else "Unknown"
            etag_label = remote_etag if remote_etag else "n/a"
            logger.info(f"{LOG_PREFIX} File: '{final_filename}' | Size: {size_label} | ETag: {etag_label} | Range: {'yes' if supports_ranges else 'no'} | Integrity: {integrity_mode}")

            if skip_existing and os.path.exists(local_file_path):
                if not verify_size:
                    logger.info(f"{LOG_PREFIX} File exists. Skipping (size verification disabled).")
                    if unzip_after_download and local_file_path.lower().endswith('.zip'):
                        cls._extract_zip(local_file_path, target_dir)
                    return True

                local_file_size = cls._safe_int(os.path.getsize(local_file_path), -1)
                if remote_file_size <= 0:
                    # The server reports no usable size (no Content-Length /
                    # x-linked-size / Content-Range). There is nothing to verify
                    # against — and re-downloading wouldn't make the result any
                    # more verifiable. The old code fell through this branch and
                    # pulled the same file again on EVERY run.
                    logger.info(
                        f"{LOG_PREFIX} File exists and the server reports no size to verify against. "
                        f"Keeping the existing file (delete it manually to force a re-download)."
                    )
                    if unzip_after_download and local_file_path.lower().endswith('.zip'):
                        cls._extract_zip(local_file_path, target_dir)
                    return True
                if remote_file_size > 0 and local_file_size == remote_file_size:
                    if use_hf_sha256:
                        cached_meta = cls._read_json_file(final_meta_path) or {}
                        cached_sha = str(cached_meta.get("sha256", "")).lower()
                        if cached_sha == hf_expected_sha256:
                            logger.info(f"{LOG_PREFIX} Verified existing HF file by cached SHA256. Skipping.")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                cls._extract_zip(local_file_path, target_dir)
                            return True

                        logger.info(f"{LOG_PREFIX} Verifying existing HF file SHA256 (one-time check)...")
                        actual_sha256 = cls._compute_sha256(local_file_path).lower()
                        if actual_sha256 == hf_expected_sha256:
                            logger.info(f"{LOG_PREFIX} Existing HF file SHA256 verified. Skipping.")
                            cls._write_json_file(final_meta_path, {
                                "source_url": url,
                                "resolved_url": processed_url,
                                "final_url": final_direct_url,
                                "remote_size": remote_file_size,
                                "remote_etag": remote_etag,
                                "sha256": actual_sha256,
                                "verified_at": int(time.time()),
                            })
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                cls._extract_zip(local_file_path, target_dir)
                            return True

                        logger.warning(f"{LOG_PREFIX} Existing HF file failed SHA256 verification. Re-downloading.")
                        cls._remove_file_silent(local_file_path)
                    else:
                        logger.info(f"{LOG_PREFIX} Verified existing file by size. Skipping.")
                        cls._write_json_file(final_meta_path, {
                            "source_url": url,
                            "resolved_url": processed_url,
                            "final_url": final_direct_url,
                            "remote_size": remote_file_size,
                            "remote_etag": remote_etag,
                            "sha256": None,
                            "verified_at": int(time.time()),
                        })
                        if unzip_after_download and local_file_path.lower().endswith('.zip'):
                            cls._extract_zip(local_file_path, target_dir)
                        return True
                elif remote_file_size > 0 and local_file_size < remote_file_size and not os.path.exists(temp_file_path):
                    try:
                        os.replace(local_file_path, temp_file_path)
                        cls._write_json_file(temp_meta_path, {
                            "source_url": url,
                            "resolved_url": processed_url,
                            "remote_size": remote_file_size,
                            "remote_etag": remote_etag,
                            "updated_at": int(time.time()),
                        })
                        logger.info(f"{LOG_PREFIX} Found truncated final file. Moved to .part for resume.")
                    except OSError as move_error:
                        logger.warning(f"{LOG_PREFIX} Could not promote truncated file to .part: {move_error}")
                elif remote_file_size > 0 and local_file_size > remote_file_size:
                    logger.warning(f"{LOG_PREFIX} Existing file is larger than remote. Re-downloading.")
                    cls._remove_file_silent(local_file_path)

            resume_byte_pos = 0
            file_mode = "wb"
            if os.path.exists(temp_file_path):
                temp_meta = cls._read_json_file(temp_meta_path)
                if not cls._is_partial_meta_compatible(temp_meta, url, remote_file_size, remote_etag):
                    logger.warning(f"{LOG_PREFIX} Existing .part belongs to a different file. Removing stale partial.")
                    cls._remove_file_silent(temp_file_path)
                    cls._remove_file_silent(temp_meta_path)
                else:
                    temp_file_size = cls._safe_int(os.path.getsize(temp_file_path), -1)
                    if temp_file_size < 0:
                        cls._remove_file_silent(temp_file_path)
                        cls._remove_file_silent(temp_meta_path)
                    elif remote_file_size > 0 and temp_file_size > remote_file_size:
                        logger.warning(f"{LOG_PREFIX} .part is larger than remote file. Removing stale partial.")
                        cls._remove_file_silent(temp_file_path)
                        cls._remove_file_silent(temp_meta_path)
                    elif remote_file_size > 0 and temp_file_size == remote_file_size:
                        logger.info(f"{LOG_PREFIX} .part size matches remote. Finalizing without re-download.")
                        part_is_valid = True
                        if use_hf_sha256:
                            logger.info(f"{LOG_PREFIX} Verifying completed .part with HF SHA256...")
                            part_sha256 = cls._compute_sha256(temp_file_path).lower()
                            if part_sha256 != hf_expected_sha256:
                                logger.error(f"{LOG_PREFIX} .part SHA256 mismatch. Removing corrupt partial.")
                                cls._remove_file_silent(temp_file_path)
                                cls._remove_file_silent(temp_meta_path)
                                part_is_valid = False
                        if part_is_valid:
                            os.replace(temp_file_path, local_file_path)
                            cls._remove_file_silent(temp_meta_path)
                            cls._write_json_file(final_meta_path, {
                                "source_url": url,
                                "resolved_url": processed_url,
                                "final_url": final_direct_url,
                                "remote_size": remote_file_size,
                                "remote_etag": remote_etag,
                                "sha256": hf_expected_sha256 if use_hf_sha256 else None,
                                "verified_at": int(time.time()),
                            })
                            logger.info(f"{LOG_PREFIX} Saved: {local_file_path}")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                cls._extract_zip(local_file_path, target_dir)
                            return True
                    elif temp_file_size > 0:
                        resume_byte_pos = temp_file_size
                        file_mode = "ab"
                        logger.info(f"{LOG_PREFIX} Resuming from {resume_byte_pos} bytes.")

            request_headers = domain_headers.copy()
            if resume_byte_pos > 0:
                request_headers["Range"] = f"bytes={resume_byte_pos}-"

            response_get = session.get(
                final_direct_url,
                stream=True,
                headers=request_headers,
                timeout=(15, 300),
                allow_redirects=True,
            )

            if resume_byte_pos > 0 and response_get.status_code == 416:
                response_get.close()
                response_get = None
                logger.warning(f"{LOG_PREFIX} Server rejected resume range (416). Restarting full download.")
                cls._remove_file_silent(temp_file_path)
                cls._remove_file_silent(temp_meta_path)
                resume_byte_pos = 0
                file_mode = "wb"
                response_get = session.get(
                    processed_url,
                    stream=True,
                    headers=domain_headers,
                    timeout=(15, 300),
                    allow_redirects=True,
                )
            elif resume_byte_pos > 0 and response_get.status_code != 206:
                response_get.close()
                response_get = None
                logger.warning(f"{LOG_PREFIX} Server ignored resume request. Restarting full download.")
                cls._remove_file_silent(temp_file_path)
                cls._remove_file_silent(temp_meta_path)
                resume_byte_pos = 0
                file_mode = "wb"
                response_get = session.get(
                    processed_url,
                    stream=True,
                    headers=domain_headers,
                    timeout=(15, 300),
                    allow_redirects=True,
                )

            response_get.raise_for_status()

            # Despite Accept-Encoding: identity some servers compress anyway;
            # requests then inflates the stream and the on-disk size can NEVER
            # match Content-Length. Disable size accounting for this transfer
            # rather than dead-looping on a false "corrupt" verdict.
            content_encoding = str(response_get.headers.get("content-encoding", "")).strip().lower()
            size_accounting_ok = not content_encoding or content_encoding == "identity"
            if not size_accounting_ok:
                logger.warning(
                    f"{LOG_PREFIX} Server forced content-encoding '{content_encoding}'; "
                    f"size verification disabled for this file."
                )
                remote_file_size = -1

            final_direct_url = response_get.url or final_direct_url
            get_reported_size = cls._extract_remote_size_from_headers(response_get.headers)
            if size_accounting_ok and remote_file_size <= 0 and get_reported_size > 0:
                remote_file_size = get_reported_size
            if response_get.status_code == 206:
                supports_ranges = True

            cls._write_json_file(temp_meta_path, {
                "source_url": url,
                "resolved_url": processed_url,
                "final_url": final_direct_url,
                "remote_size": remote_file_size,
                "remote_etag": remote_etag,
                "updated_at": int(time.time()),
            })

            total_size = remote_file_size if remote_file_size > 0 else None

            comfy_pbar = None
            if ProgressBar and total_size:
                comfy_pbar = ProgressBar(total_size)

            downloaded_since_update = 0
            ui_update_threshold = 1 * 1024 * 1024

            with open(temp_file_path, file_mode) as f, tqdm(
                total=total_size,
                unit='B', unit_scale=True, desc=f"DL: {final_filename}",
                initial=resume_byte_pos if file_mode == 'ab' else 0,
                mininterval=1.0, ncols=100, unit_divisor=1024
            ) as pbar:
                if comfy_pbar and resume_byte_pos > 0:
                    comfy_pbar.update(resume_byte_pos)
                for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                    if not chunk:
                        continue
                    f.write(chunk)
                    chunk_len = len(chunk)
                    pbar.update(chunk_len)
                    if comfy_pbar:
                        downloaded_since_update += chunk_len
                        if downloaded_since_update >= ui_update_threshold:
                            comfy_pbar.update(downloaded_since_update)
                            downloaded_since_update = 0
                if comfy_pbar and downloaded_since_update > 0:
                    comfy_pbar.update(downloaded_since_update)

            temp_final_size = cls._safe_int(os.path.getsize(temp_file_path), -1)
            if verify_size and remote_file_size > 0 and temp_final_size != remote_file_size:
                if temp_final_size < remote_file_size:
                    logger.warning(f"{LOG_PREFIX} Download incomplete ({temp_final_size}/{remote_file_size}). Keeping .part for resume.")
                else:
                    logger.error(f"{LOG_PREFIX} Downloaded file is larger than expected. Removing .part.")
                    cls._remove_file_silent(temp_file_path)
                    cls._remove_file_silent(temp_meta_path)
                return False

            verified_sha256 = None
            if use_hf_sha256:
                logger.info(f"{LOG_PREFIX} Verifying HF SHA256...")
                verified_sha256 = cls._compute_sha256(temp_file_path).lower()
                if verified_sha256 != hf_expected_sha256:
                    logger.error(f"{LOG_PREFIX} HF SHA256 mismatch. Removing corrupted file.")
                    cls._remove_file_silent(temp_file_path)
                    cls._remove_file_silent(temp_meta_path)
                    return False

            os.replace(temp_file_path, local_file_path)
            cls._remove_file_silent(temp_meta_path)
            cls._write_json_file(final_meta_path, {
                "source_url": url,
                "resolved_url": processed_url,
                "final_url": final_direct_url,
                "remote_size": remote_file_size,
                "remote_etag": remote_etag,
                "sha256": verified_sha256,
                "verified_at": int(time.time()),
            })
            logger.info(f"{LOG_PREFIX} Saved: {local_file_path}")
            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                cls._extract_zip(local_file_path, target_dir)

            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Download failed: {e}")
            return False
        finally:
            if response_get is not None:
                try:
                    response_get.close()
                except Exception as exc:
                    logger.debug("%s Closing GET probe response failed: %s", LOG_PREFIX, exc)
            if download_lock is not None:
                download_lock.release()

    @classmethod
    def execute(
        cls,
        file_list: str,
        skip_existing: bool = True,
        verify_size: bool = True,
        chunk_size_kb: int = 4096,
        hf_token: str = "",
        hf_domain: str = "huggingface.co, hf-mirror.com",
        proxy_url: str = "",
        modelscope_token: str = "",
        unzip_after_download: bool = False,
        enable: bool = True,
        integrity_mode: str = "hf_sha256_auto",
    ) -> IO.NodeOutput:
        if not enable:
            logger.info("%s Skipped (disabled).", LOG_PREFIX)
            return IO.NodeOutput()

        logger.info("%s Started.", LOG_PREFIX)
        chunk_size_bytes = max(1024, chunk_size_kb * 1024)
        integrity_mode_value = str(integrity_mode).strip().lower()
        if integrity_mode_value not in {"hf_sha256_auto", "size_only"}:
            logger.warning(f"{LOG_PREFIX} Unknown integrity_mode '{integrity_mode}'. Fallback to 'hf_sha256_auto'.")
            integrity_mode_value = "hf_sha256_auto"

        files_to_download = cls._parse_file_list(file_list)
        if not files_to_download:
            return IO.NodeOutput()

        # Offline detection must fail FAST. The download session retries
        # connection errors 3x with exponential backoff (~20s per dead host),
        # so probing connectivity through it turned a no-internet run into a
        # minute-long hang. Probe with a throwaway no-retry session (and a short
        # connect timeout) first; only build the retry-enabled download session
        # once we know at least one target is reachable. Both sessions are
        # context-managed so their connection pools never leak.
        with cls._create_session_with_retries(proxy_url, total_retries=0) as probe_session:
            if not cls._check_connectivity_to_targets(files_to_download, probe_session, hf_domain):
                logger.warning(f"{LOG_PREFIX} All target servers are unreachable. Switching to OFFLINE MODE. Execution finished.")
                return IO.NodeOutput()
            active_mirror = cls._select_best_mirror(probe_session, hf_domain)

        logger.info(f"{LOG_PREFIX} Using HF Mirror: '{active_mirror}'")

        success = 0
        failed = 0
        with cls._create_session_with_retries(proxy_url) as session:
            for file_info in files_to_download:
                if cls._download_single_file(session, file_info['url'], file_info['target_dir'], skip_existing, verify_size, chunk_size_bytes, active_mirror, hf_token, modelscope_token, unzip_after_download, integrity_mode_value):
                    success += 1
                else:
                    failed += 1

        logger.info("%s Done. Success: %d, Failed: %d", LOG_PREFIX, success, failed)
        return IO.NodeOutput()


NODE_CLASS_MAPPINGS = {
    "TS Files Downloader": TS_DownloadFilesNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Files Downloader": "TS Files Downloader (Ultimate)",
}
