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
# Queries per filename, from the most specific form down to the family name.
MAX_QUERIES_PER_FILENAME = 4


def _log_warning(message: str) -> None:
    LOGGER.warning("%s %s", LOG_PREFIX, message)


_PROMPT_SERVER = resolve_prompt_server(_log_warning)
_register_get, _register_post = make_route_registrars(_PROMPT_SERVER, _log_warning)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "comfyui-timesaver", "Accept": "application/json"})
    return session


def _search_queries(filename: str) -> list[str]:
    """Search terms for a model filename, most specific first.

    HuggingFace search matches repo names and metadata, never file names, so a
    filename is usually too specific to match anything:
    "mage_flow_turbo_int8_convrot" returns nothing while "mage flow" returns the
    repo that actually holds that file. Quantisation and packaging suffixes are
    dropped outright, and what remains is tried from the longest form down to
    the first two words — the shortest form is what finds a repo whose name is
    just the model family.
    """
    stem = filename.rsplit(".", 1)[0]
    for noise in ("fp8_e4m3fn", "fp8_e5m2", "bf16", "fp16", "fp32", "e4m3fn",
                  "int8", "int4", "nf4", "scaled", "convrot", "quantized"):
        stem = stem.replace(noise, " ")
    tokens = [t for t in stem.replace("_", " ").replace("-", " ").replace(".", " ").split() if t]
    queries: list[str] = []
    for count in range(len(tokens), 0, -1):
        query = " ".join(tokens[:count])[:MAX_QUERY_LEN]
        if query and query not in queries:
            queries.append(query)
    return queries[:MAX_QUERIES_PER_FILENAME]


def _get_json(session: requests.Session, url: str):
    response = session.get(url, timeout=SEARCH_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _find_one(session: requests.Session, filename: str, trees: dict) -> list[dict]:
    """Repos containing a file named exactly ``filename``, best first."""
    wanted = filename.strip().lower()
    seen_repos: set[str] = set()
    for query in _search_queries(filename):
        matches = _search_one_query(session, query, wanted, seen_repos, trees)
        if matches:
            return matches
    return []


def _search_one_query(
    session: requests.Session, query: str, wanted: str, seen_repos: set[str], trees: dict
) -> list[dict]:
    encoded = urllib.parse.quote(query)
    try:
        models = _get_json(
            session, f"{HF_API}/models?search={encoded}&limit={MAX_REPOS_PER_FILENAME}&sort=downloads&direction=-1"
        )
    except Exception as exc:
        _log_warning(f"Search for {query!r} failed: {exc}")
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
        if repo in seen_repos:
            continue  # a broader query re-lists repos the narrower one already read
        seen_repos.add(repo)
        try:
            tree = _get_json(
                session, f"{HF_API}/models/{urllib.parse.quote(repo)}/tree/main?recursive=true"
            )
        except Exception as exc:
            LOGGER.debug("%s Tree listing for %s failed: %s", LOG_PREFIX, repo, exc)
            continue
        if not isinstance(tree, list):
            continue
        # Kept for the batch: another filename may live in this same repo.
        trees[repo] = (tree, model)
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


def _match_in_seen_trees(wanted: str, trees: dict[str, tuple[list, dict]]) -> list[dict]:
    """Look for a file in repos already listed for the other names in this batch.

    HuggingFace search matches repo NAMES, so a file whose repo is named after a
    different family is unfindable on its own: qwen3vl_4b_bf16.safetensors lives
    in Comfy-Org/Mage-Flow and no query for "qwen3vl" will ever reach it. The
    models of one workflow, though, almost always ship together — and those
    repos are already downloaded and cached from the other lookups, so this
    costs nothing.
    """
    matches: list[dict] = []
    for repo, (tree, model) in trees.items():
        for entry in tree:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path = str(entry.get("path") or "")
            if path.rsplit("/", 1)[-1].lower() != wanted or ".." in path.split("/"):
                continue
            matches.append({
                "repo": repo,
                "path": path,
                "size": int(entry.get("size") or 0),
                "downloads": int(model.get("downloads") or 0),
                "likes": int(model.get("likes") or 0),
                "url": f"{HF_HOST}/{repo}/resolve/main/{urllib.parse.quote(path)}",
            })
            break
    matches.sort(key=lambda m: (-m["downloads"], -m["likes"], len(m["path"])))
    return matches[:MAX_MATCHES_PER_FILENAME]


def send_progress(operation_id: str, text: str, percent: float) -> None:
    """Report search progress to the browser.

    Looking through several repositories takes tens of seconds, and a button
    that simply sits there looks broken. Events are addressed to an
    operation_id so a second node's search cannot move this one's bar.
    """
    if not operation_id or _PROMPT_SERVER is None:
        return
    try:
        _PROMPT_SERVER.send_sync(
            "ts_downloader.search_progress",
            {"operation_id": operation_id, "text": text, "percent": max(0.0, min(100.0, percent))},
        )
    except Exception as exc:  # noqa: BLE001 - progress must never break the search
        LOGGER.debug("%s Could not send progress: %s", LOG_PREFIX, exc)


def search_filenames(filenames: list[str], operation_id: str = "") -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    trees: dict[str, tuple[list, dict]] = {}
    session = _session()
    try:
        names = []
        for raw in filenames:
            name = str(raw or "").strip()
            if not name or "/" in name or "\\" in name:
                continue
            names.append(name)
        total = max(1, len(names))
        for index, name in enumerate(names):
            # 0-90% for the searches; the sweep below finishes the bar.
            send_progress(operation_id, f"Searching {name} ({index + 1}/{len(names)})",
                          index / total * 90.0)
            results[name] = _find_one(session, name, trees)
        remaining = [name for name in names if not results[name]]
        if remaining:
            send_progress(operation_id, f"Checking {len(trees)} repository(ies) for the rest", 92.0)
            for name in remaining:
                results[name] = _match_in_seen_trees(name.lower(), trees)
        found = sum(1 for hits in results.values() if hits)
        send_progress(operation_id, f"Found {found} of {len(names)}", 100.0)
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

    operation_id = str(payload.get("operation_id") or "")
    # Network calls in a worker thread: this route is HTTP-triggered and would
    # otherwise stall ComfyUI's event loop for as long as HuggingFace takes.
    results = await asyncio.to_thread(search_filenames, filenames, operation_id)
    return web.json_response({"results": results})
