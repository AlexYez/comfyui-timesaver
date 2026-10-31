"""Anonymous, bounded download-link checks for the Files Downloader button."""
from __future__ import annotations

import asyncio
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit

import requests
from aiohttp import web

from .._shared import make_route_registrars, resolve_prompt_server, url_target_is_public

MAX_ENTRIES = 8
MAX_TEXT = 65536
MAX_URL = 8192
SAMPLE_BYTES = 2048
MAX_REDIRECTS = 5
CHECK_SECONDS = 20
_BUSY = threading.Lock()
_SERVER = resolve_prompt_server(lambda _message: None)
_get, _post = make_route_registrars(_SERVER, lambda _message: None)
_TOKEN_KEYS = {"token", "hf_token", "access_token", "api_key", "apikey", "auth", "authorization", "auth_token"}
_SIGNED_KEYS = {"x-amz-signature", "x-goog-signature", "signature", "policy", "expires"}


def _credential_kind(url):
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        return "auth_url"
    keys = {key.lower() for key, _ in parse_qsl(parts.query, keep_blank_values=True)}
    if keys & _TOKEN_KEYS:
        return "auth_url"
    return "temporary" if keys & _SIGNED_KEYS else ""


def _public(url):
    try:
        return url_target_is_public(url)
    except ValueError:
        return False, "invalid address"


def check_url(url):
    """Read at most 2 KiB of a file, without tokens, netrc or browser cookies.

    Redirects are followed manually: requests otherwise consumes their whole
    bodies even with stream=True. Every hop is checked before contacting it.
    Signed CDN redirects from a public source are normal; a signed SOURCE link
    is reported as temporary instead of promising reusable public access.
    """
    from urllib3.exceptions import ReadTimeoutError

    from .ts_downloader import TS_DownloadFilesNode as Node

    try:
        credential = _credential_kind(url)
        if credential == "auth_url":
            return {"status": credential}
        if len(url) > MAX_URL:
            return {"status": "invalid"}
        current = Node._process_dropbox_url(url)
        deadline = time.monotonic() + CHECK_SECONDS
        with requests.Session() as session:
            # Disallow implicit .netrc authentication and environment credentials.
            # The check deliberately tests the source URL, not a token/mirror fallback.
            session.trust_env = False
            # Session.send builds Response.next through resolve_redirects even
            # when redirects are disabled, consuming the entire 3xx body.
            # This isolated session delegates ALL redirect handling to us.
            session.resolve_redirects = lambda *args, **kwargs: iter(())
            session.headers.update({"User-Agent": "comfyui-timesaver/CheckLinks",
                                    "Accept-Encoding": "identity", "Range": f"bytes=0-{SAMPLE_BYTES-1}"})
            for hop in range(MAX_REDIRECTS + 1):
                if _credential_kind(current) == "auth_url":
                    return {"status": "auth_url"}
                allowed, reason = _public(current)
                if not allowed:
                    return {"status": "unreachable" if "cannot be resolved" in reason else "blocked"}
                left = deadline - time.monotonic()
                if left <= 0:
                    return {"status": "timeout"}
                session.cookies.clear()
                with session.get(current, stream=True, allow_redirects=False,
                                 timeout=(min(5, left), min(8, left))) as response:
                    code = response.status_code
                    headers = response.headers
                    if code in (301, 302, 303, 307, 308):
                        target = headers.get("Location")
                        if not target or hop == MAX_REDIRECTS:
                            return {"status": "redirect_error", "http_status": code}
                        current = urljoin(current, target)
                        if re.search(r"/(?:login|signin|sign-in|oauth/authorize)(?:[/?#]|$)", current, re.I):
                            return {"status": "auth_required", "http_status": code}
                        continue
                    error_code = headers.get("X-Error-Code", "")
                    status = ("gated" if error_code in Node._HF_GATED_CODES else
                              "missing" if error_code in Node._HF_MISSING_FILE_CODES or code in (404, 410) else
                              "auth_required" if code == 401 else
                              "forbidden" if code == 403 else
                              "rate_limited" if code == 429 else
                              "http_error" if code not in (200, 206) else "")
                    if status:
                        return {"status": status, "http_status": code}
                    mime = headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                    if mime in ("text/html", "application/xhtml+xml"):
                        return {"status": "webpage", "http_status": code}
                    if headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
                        return {"status": "unverified", "http_status": code}
                    filename = Node._get_filename_from_header_map(headers)
                    filename = Node._sanitize_filename(filename or unquote(urlsplit(url).path.rsplit("/", 1)[-1]))
                    if Node._refused_download_suffix(filename):
                        return {"status": "unsafe_file", "http_status": code}
                    sample = response.raw.read(SAMPLE_BYTES, decode_content=False)
                    if not sample:
                        return {"status": "empty", "http_status": code}
                    sniff = sample.lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
                    if sniff.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
                        return {"status": "webpage", "http_status": code}
                    if sniff.startswith(b"version https://git-lfs.github.com/spec/v1"):
                        return {"status": "pointer", "http_status": code}
                    if code == 206 and not re.fullmatch(r"bytes 0-\d+/(?:\d+|\*)", headers.get("Content-Range", ""), re.I):
                        return {"status": "unverified", "http_status": code}
                    size = (Node._extract_total_size_from_content_range(headers.get("Content-Range"))
                            if code == 206 else Node._safe_int(headers.get("Content-Length"), -1))
                    return {"status": credential or "available", "http_status": code,
                            "filename": filename, "size": size, "sample_bytes": len(sample)}
    except (requests.Timeout, TimeoutError, ReadTimeoutError):
        return {"status": "timeout"}
    except requests.exceptions.SSLError:
        return {"status": "tls_error"}
    except (requests.RequestException, OSError):
        return {"status": "unreachable"}
    except Exception:  # Never expose credential-bearing CDN URLs.
        return {"status": "unverified"}


def _check_line(index, line):
    from .ts_downloader import TS_DownloadFilesNode as Node

    text = line.strip()
    if not text or text.startswith("#"):
        return {"line": index, "status": "skip"}
    parts = None
    for marker in (" → ", " -> "):
        if marker in text:
            parts = text.split(marker, 1)
            break
    parts = parts if parts is not None else text.split(maxsplit=1)
    if len(parts) != 2:
        return {"line": index, "status": "invalid"}
    url, target = (p.strip() for p in parts)
    try:
        if _credential_kind(url) == "auth_url":
            return {"line": index, "status": "auth_url"}
        if not Node.resolve_route_target_directory(target):
            return {"line": index, "status": "invalid_target"}
    except (ValueError, TypeError):
        return {"line": index, "status": "invalid"}
    return {"line": index, **check_url(url)}


def _locked_batch(lines):
    try:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="ts-link-check") as pool:
            return list(pool.map(lambda item: _check_line(*item), enumerate(lines)))
    finally:
        # A disconnected browser does not free the slot while its worker is
        # still checking links. Remaining frontend batches are never started.
        _BUSY.release()


@_post("/ts_downloader/check_links")
async def check_links_route(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Expected a JSON body."}, status=400)
    text = body.get("file_list") if isinstance(body, dict) else None
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        return web.json_response({"error": "Expected a bounded file list."}, status=400)
    lines = text.splitlines()
    if not lines or len(lines) > MAX_ENTRIES:
        return web.json_response({"error": f"Check 1–{MAX_ENTRIES} lines per request."}, status=400)
    if not _BUSY.acquire(blocking=False):
        return web.json_response({"error": "Another link check is still running."}, status=409)
    entries = await asyncio.shield(asyncio.to_thread(_locked_batch, lines))
    return web.json_response({"schema": 1, "anonymous": True, "entries": entries})
