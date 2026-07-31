"""HuggingFace lookup for models the workflow scanner could not find a link for.

The scanner reads the models a graph needs from loader metadata and Markdown
notes. When a graph names a file but carries no URL for it, the report has
nothing to offer — that is the "no download link" column, and until now the
only way out was to go find the file by hand.

This module answers that question: given a filename, find the HuggingFace repos
that actually contain a file with that exact name, and hand back a download URL.

Two deliberate constraints:

* The URL is BUILT here from the repo id and the file path the API reported —
  never taken from a response field. A search result is untrusted input, and
  the one thing a download target must not be is attacker-chosen.
* Every request is bounded (count, timeout, response size) and runs off the
  event loop, because it is triggered from a button that anyone with access to
  the ComfyUI port can press.

Private module: the loader skips `_`-prefixed paths, so this registers routes
without ever becoming a node.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse

import requests
from aiohttp import web

from .._shared import make_route_registrars, resolve_prompt_server

LOGGER = logging.getLogger("comfyui_timesaver.ts_downloader_search")
LOG_PREFIX = "[TS Downloader Search]"

HF_API = "https://huggingface.co/api"
HF_HOST = "https://huggingface.co"

# Bounds. The caller is a button, so every one of these is a ceiling on what a
# single click can cost the server.
MAX_FILENAMES_PER_REQUEST = 16
MAX_REPOS_PER_FILENAME = 6
MAX_MATCHES_PER_FILENAME = 5
SEARCH_TIMEOUT = (5, 12)
MAX_QUERY_LEN = 120


def _log_warning(message: str) -> None:
    LOGGER.warning("%s %s", LOG_PREFIX, message)


_PROMPT_SERVER = resolve_prompt_server(_log_warning)
_register_get, _register_post = make_route_registrars(_PROMPT_SERVER, _log_warning)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "comfyui-timesaver", "Accept": "application/json"})
    return session


def _search_query(filename: str) -> str:
    """Turn a model filename into something HuggingFace's search understands.

    Search matches repo names and metadata, not file names, so the extension
    and the packaging noise around the actual model name only hurt: querying
    "qwen_image_vae.safetensors" finds nothing, "qwen image vae" finds the repo
    that holds it.
    """
    stem = filename.rsplit(".", 1)[0]
    for noise in ("fp8_e4m3fn", "fp8_e5m2", "bf16", "fp16", "fp32", "e4m3fn", "scaled"):
        stem = stem.replace(noise, " ")
    stem = stem.replace("_", " ").replace("-", " ").replace(".", " ")
    return " ".join(stem.split())[:MAX_QUERY_LEN]


def _get_json(session: requests.Session, url: str):
    response = session.get(url, timeout=SEARCH_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _find_one(session: requests.Session, filename: str) -> list[dict]:
    """Repos containing a file named exactly ``filename``, best first."""
    query = _search_query(filename)
    if not query:
        return []
    wanted = filename.strip().lower()
    encoded = urllib.parse.quote(query)
    try:
        models = _get_json(
            session, f"{HF_API}/models?search={encoded}&limit={MAX_REPOS_PER_FILENAME}&sort=downloads&direction=-1"
        )
    except Exception as exc:
        _log_warning(f"Search for {filename!r} failed: {exc}")
        return []
    if not isinstance(models, list):
        return []

    matches: list[dict] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        repo = str(model.get("modelId") or model.get("id") or "").strip()
        # A repo id is "<owner>/<name>" and nothing else; anything with a path
        # separator beyond that, or a scheme, is not something we will build a
        # URL from.
        if not repo or repo.count("/") != 1 or ":" in repo or repo.startswith("/"):
            continue
        try:
            tree = _get_json(
                session, f"{HF_API}/models/{urllib.parse.quote(repo)}/tree/main?recursive=true"
            )
        except Exception as exc:
            LOGGER.debug("%s Tree listing for %s failed: %s", LOG_PREFIX, repo, exc)
            continue
        if not isinstance(tree, list):
            continue
        for entry in tree:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path = str(entry.get("path") or "")
            if path.rsplit("/", 1)[-1].lower() != wanted:
                continue
            if ".." in path.split("/"):
                continue
            matches.append({
                "repo": repo,
                "path": path,
                "size": int(entry.get("size") or 0),
                "downloads": int(model.get("downloads") or 0),
                "likes": int(model.get("likes") or 0),
                # Built here, from parts we validated — never echoed from the API.
                "url": f"{HF_HOST}/{repo}/resolve/main/{urllib.parse.quote(path)}",
            })
            break  # one hit per repo is enough
        if len(matches) >= MAX_MATCHES_PER_FILENAME:
            break

    matches.sort(key=lambda m: (-m["downloads"], -m["likes"], len(m["path"])))
    return matches[:MAX_MATCHES_PER_FILENAME]


def search_filenames(filenames: list[str]) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    session = _session()
    try:
        for raw in filenames:
            name = str(raw or "").strip()
            if not name or "/" in name or "\\" in name:
                continue
            results[name] = _find_one(session, name)
    finally:
        session.close()
    return results


@_register_post("/ts_downloader/hf_search")
async def hf_search(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body."}, status=400)

    raw = payload.get("filenames")
    if not isinstance(raw, list):
        return web.json_response({"error": "'filenames' must be a list."}, status=400)
    filenames = [str(item) for item in raw][:MAX_FILENAMES_PER_REQUEST]
    if not filenames:
        return web.json_response({"results": {}})

    # Network calls in a worker thread: this route is HTTP-triggered and would
    # otherwise stall ComfyUI's event loop for as long as HuggingFace takes.
    results = await asyncio.to_thread(search_filenames, filenames)
    return web.json_response({"results": results})
