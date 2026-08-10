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
import os
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MAX_REPOS_PER_QUERY = 8
MAX_MATCHES_PER_FILENAME = 5
# Long enough for a slow link to HuggingFace. Measured on the maintainer's
# machine: a plain search is 0.7-4 s and the old 5-second connect ceiling was
# turning ordinary latency into "the search finds nothing".
SEARCH_TIMEOUT = (10, 25)
MAX_QUERY_LEN = 120
# Queries per filename, from the most specific form down to the family name.
MAX_QUERIES_PER_FILENAME = 5
# Repos to open from the full-text results, which are the least precise source.
MAX_FULLTEXT_REPOS = 5
# Names looked up at the same time. Small on purpose: this is someone
# else's public API, and the point is to stop waiting in single file, not
# to open as many sockets as the list is long.
SEARCH_WORKERS = 4


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

    The whole filename comes FIRST, extension and all. That is not a formality:
    a great many mirror repositories are named after the file they hold, so
    `t5xxl_fp8_e4m3fn_scaled.safetensors` finds three repos on the spot — while
    the trimmed form this used to start from ("t5xxl") found none of them. Then
    the stem, then progressively shorter forms with quantisation and packaging
    noise removed, down to the family name, which is what reaches a repo named
    after the model rather than after the file.
    """
    queries: list[str] = []

    def add(text: str) -> None:
        text = " ".join(str(text).split())[:MAX_QUERY_LEN]
        if text and text not in queries:
            queries.append(text)

    add(filename)
    stem = filename.rsplit(".", 1)[0]
    add(stem)
    for noise in ("fp8_e4m3fn", "fp8_e5m2", "bf16", "fp16", "fp32", "e4m3fn",
                  "int8", "int4", "nf4", "scaled", "convrot", "quantized"):
        stem = stem.replace(noise, " ")
    tokens = [t for t in stem.replace("_", " ").replace("-", " ").replace(".", " ").split() if t]
    for count in range(len(tokens), 0, -1):
        add(" ".join(tokens[:count]))
    return queries[:MAX_QUERIES_PER_FILENAME]


def _get_json(session: requests.Session, url: str):
    response = session.get(url, timeout=SEARCH_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _repo_id(model: dict) -> str:
    """The repo id, or "" when it is not one we will build a URL from.

    A repo id is "<owner>/<name>" and nothing else; anything with a further path
    separator, a scheme, or a leading slash is refused. Search results are
    untrusted input and the one thing a download target must not be is
    attacker-chosen.
    """
    repo = str(model.get("modelId") or model.get("id") or "").strip()
    if not repo or repo.count("/") != 1 or ":" in repo or repo.startswith("/"):
        return ""
    return repo


def _files_of(model: dict) -> list[str]:
    """File paths a model listing carries, from its `siblings`."""
    out = []
    for sibling in model.get("siblings") or []:
        if isinstance(sibling, dict):
            path = str(sibling.get("rfilename") or "")
            if path and ".." not in path.split("/"):
                out.append(path)
    return out


def _hit(repo: str, path: str, model: dict) -> dict:
    return {
        "repo": repo,
        "path": path,
        # The listing endpoints do not carry file sizes; the report shows the
        # repository, not the byte count, and the download learns the real size
        # from the server anyway.
        "size": 0,
        "downloads": int(model.get("downloads") or 0),
        "likes": int(model.get("likes") or 0),
        # Built here, from parts we validated — never echoed from the API.
        "url": f"{HF_HOST}/{repo}/resolve/main/{urllib.parse.quote(path)}",
    }


def _remember(repo: str, files: list[str], model: dict, known: dict) -> None:
    known.setdefault(repo, (files, model))


def _match_in(files: list[str], wanted: str) -> str:
    """The path in `files` whose basename is exactly `wanted`, or ""."""
    for path in files:
        if path.rsplit("/", 1)[-1].lower() == wanted:
            return path
    return ""


def _search_repos(session: requests.Session, query: str, known: dict) -> list[tuple[str, dict]]:
    """Repos matching `query`, WITH their file lists.

    `full=true` is what makes this one request instead of many: the listing
    comes back with each repo's `siblings`, so a candidate can be confirmed or
    dismissed without opening it. Listing the tree of every candidate was what
    made a search of twelve names take nearly two minutes.
    """
    encoded = urllib.parse.quote(query)
    try:
        models = _get_json(
            session,
            f"{HF_API}/models?search={encoded}&limit={MAX_REPOS_PER_QUERY}"
            f"&full=true&sort=downloads&direction=-1",
        )
    except Exception as exc:
        _log_warning(f"Search for {query!r} failed: {exc}")
        return []
    if not isinstance(models, list):
        return []
    found = []
    for model in models:
        if not isinstance(model, dict):
            continue
        repo = _repo_id(model)
        if not repo:
            continue
        files = _files_of(model)
        _remember(repo, files, model, known)
        found.append((repo, model))
    return found


def _fulltext_repos(session: requests.Session, filename: str) -> list[str]:
    """Repo ids from HuggingFace's own full-text search.

    Reaches what a name query cannot: a repository whose name says nothing about
    this file, but whose README lists it. Least precise of the three sources, so
    it runs last and only a handful of its answers are opened.
    """
    encoded = urllib.parse.quote(filename)
    try:
        payload = _get_json(
            session, f"{HF_API}/search/full-text?q={encoded}&type=model&limit=10"
        )
    except Exception as exc:
        LOGGER.debug("%s Full-text search for %r failed: %s", LOG_PREFIX, filename, exc)
        return []
    hits = payload.get("hits") if isinstance(payload, dict) else payload
    if not isinstance(hits, list):
        return []
    repos = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        owner = str(hit.get("repoOwner") or "").strip()
        name = str(hit.get("repoName") or "").strip()
        if not owner or not name or "/" in owner or "/" in name:
            continue
        repo = f"{owner}/{name}"
        if repo not in repos:
            repos.append(repo)
    return repos[:MAX_FULLTEXT_REPOS]


def _repo_files(session: requests.Session, repo: str, known: dict) -> tuple[list[str], dict]:
    """The files in `repo`, from cache or from one listing call.

    Asks the model endpoint rather than `tree/main`: it needs no assumption
    about the branch being called main, and it answers with the whole file list
    in one go.
    """
    cached = known.get(repo)
    if cached is not None:
        return cached
    try:
        model = _get_json(session, f"{HF_API}/models/{urllib.parse.quote(repo)}")
    except Exception as exc:
        LOGGER.debug("%s Listing %s failed: %s", LOG_PREFIX, repo, exc)
        known[repo] = ([], {})
        return [], {}
    if not isinstance(model, dict):
        known[repo] = ([], {})
        return [], {}
    files = _files_of(model)
    known[repo] = (files, model)
    return files, model


def _find_one(session: requests.Session, filename: str, known: dict) -> list[dict]:
    """Repos containing a file named exactly `filename`, best first.

    Three sources, cheapest and most precise first, and the first one that
    produces a match wins. Every answer is confirmed against the repository's
    own file list, so a hit means the file is really there — a search result
    that merely looked plausible would send the download somewhere useless.
    """
    wanted = filename.strip().lower()

    # Repos already opened for the other names in this batch, first of all: the
    # models of one workflow usually ship together, so the answer is often
    # already in hand. This used to run only after every name had paid for its
    # own full search — which is most of what made a search of twelve names
    # take two minutes.
    matches = _match_in_seen_repos(wanted, known)
    if matches:
        return matches

    for query in _search_queries(filename):
        for repo, model in _search_repos(session, query, known):
            path = _match_in(known.get(repo, ([], {}))[0], wanted)
            if path:
                matches.append(_hit(repo, path, model))
                if len(matches) >= MAX_MATCHES_PER_FILENAME:
                    break
        if matches:
            return _best(matches)

    for repo in _fulltext_repos(session, filename):
        files, model = _repo_files(session, repo, known)
        path = _match_in(files, wanted)
        if path:
            matches.append(_hit(repo, path, model))
            if len(matches) >= MAX_MATCHES_PER_FILENAME:
                break
    return _best(matches)


def _best(matches: list[dict]) -> list[dict]:
    matches.sort(key=lambda m: (-m["downloads"], -m["likes"], len(m["path"])))
    return matches[:MAX_MATCHES_PER_FILENAME]


def _match_in_seen_repos(wanted: str, known: dict[str, tuple[list, dict]]) -> list[dict]:
    """Look for a file among the repos already listed for this batch.

    A file whose repository is named after a different family is unfindable on
    its own: qwen3vl_4b_bf16.safetensors lives in Comfy-Org/Mage-Flow and no
    query for "qwen3vl" will ever reach it. The models of one workflow, though,
    almost always ship together — and those repos are already listed and cached
    from the other lookups, so this costs nothing.
    """
    matches: list[dict] = []
    # A snapshot, because searches run side by side: iterating the live mapping
    # while another name inserts into it raises.
    for repo, (files, model) in list(known.items()):
        path = _match_in(files, wanted)
        if path:
            matches.append(_hit(repo, path, model))
    return _best(matches)


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
    """Find a download link for each filename.

    Names are searched side by side. One at a time this was a minute-long wait
    on a button, and almost all of it was spent waiting on HuggingFace rather
    than doing anything — the requests for different names have nothing to do
    with each other. The pool is deliberately small: this runs against someone
    else's public API.
    """
    results: dict[str, list[dict]] = {}
    # repo -> (file paths, listing). Shared across the whole batch: the models
    # of one workflow usually ship from the same few repositories. Written from
    # several threads, and that is safe here — every write is a single
    # setdefault/assignment, and the worst a race can cost is one repeated
    # listing. Readers snapshot before iterating.
    known: dict[str, tuple[list, dict]] = {}
    names = []
    for raw in filenames:
        name = str(raw or "").strip()
        if not name or "/" in name or "\\" in name:
            continue
        names.append(name)
    if not names:
        return results

    total = len(names)
    local = threading.local()

    def session_for_this_thread() -> requests.Session:
        session = getattr(local, "session", None)
        if session is None:
            session = _session()
            local.session = session
        return session

    def look_for(name: str) -> tuple[str, list[dict]]:
        return name, _find_one(session_for_this_thread(), name, known)

    done = 0
    send_progress(operation_id, f"Searching {total} name(s)", 2.0)
    with ThreadPoolExecutor(max_workers=min(SEARCH_WORKERS, total)) as pool:
        pending = [pool.submit(look_for, name) for name in names]
        for future in as_completed(pending):
            try:
                name, hits = future.result()
            except Exception as exc:  # one name failing must not lose the rest
                _log_warning(f"Search failed: {exc}")
                continue
            results[name] = hits
            done += 1
            # 0-92% for the searches; the sweep below finishes the bar.
            send_progress(operation_id, f"Searched {done}/{total}: {name}",
                          done / total * 92.0)

    for name in names:
        results.setdefault(name, [])
    # A name that finished early could not see repos the others opened later.
    # One more pass over everything now known costs no request at all.
    remaining = [name for name in names if not results[name]]
    if remaining:
        send_progress(operation_id, f"Checking {len(known)} repository(ies) for the rest", 94.0)
        for name in remaining:
            results[name] = _match_in_seen_repos(name.lower(), known)
    found = sum(1 for hits in results.values() if hits)
    send_progress(operation_id, f"Found {found} of {total}", 100.0)
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


# ── where this machine actually keeps its models ─────────────────────────── #
#
# ComfyUI has read two directories per category for years and keeps both names
# alive: a text encoder is looked for in `models/text_encoders` AND in
# `models/clip`, a UNET in `diffusion_models` AND in `unet`. Which name is the
# "right" one is not ours to decide — the right one is the name this machine
# lives by.
#
# Auto-detect used to fill in the canonical name from a table, so someone with
# thirty encoders in `models/clip` was offered a download into an empty
# `models/text_encoders`. The file would be found (the engine looks in both),
# but it would not be where everything else is, and next time it would have to
# be hunted for by eye.
#
# So the answer here is the simple, checkable one: which of a category's
# directories CONTAINS models. That is the one being used.

# What makes a directory count as live. Placeholders such as
# `put_text_encoder_files_here` are not models — and they are exactly what made
# an empty folder look occupied.
_MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
                   ".onnx", ".sft", ".safetensor"}

# Enough to tell a used folder from an empty one. This is driven by a button in
# the UI, and walking a hundred thousand files to choose a folder name is a bad
# price for a hint.
_COUNT_CEILING = 64


def _count_models(directory: str) -> int:
    """How many model files live under `directory`, counted up to the ceiling."""
    found = 0
    try:
        for _root, _dirs, files in os.walk(directory):
            for name in files:
                if os.path.splitext(name)[1].lower() in _MODEL_SUFFIXES:
                    found += 1
                    if found >= _COUNT_CEILING:
                        return found
    except OSError as exc:
        LOGGER.debug("%s Cannot walk '%s': %s", LOG_PREFIX, directory, exc)
    return found


@_register_get("/ts_downloader/model_folders")
async def model_folders(_request: web.Request) -> web.StreamResponse:
    """Each category's directories and how many models are in each.

    Answers ``{"folders": {"<category>": [{"name": "clip", "path": "…",
    "exists": true, "models": 16}, …]}}``. Choosing a spelling from that is the
    frontend's job — it is the side that has to show the result to a person.
    """
    try:
        import folder_paths
    except ImportError:
        return web.json_response({"folders": {}, "available": False})

    def _collect() -> dict:
        out: dict = {}
        registered = getattr(folder_paths, "folder_names_and_paths", {}) or {}
        for category in registered:
            key = str(category)
            try:
                paths = folder_paths.get_folder_paths(key) or []
            except Exception as exc:  # a category registered without paths
                LOGGER.debug("%s No paths for '%s': %s", LOG_PREFIX, key, exc)
                continue
            entries = []
            for path in paths:
                text = str(path)
                if not text:
                    continue
                exists = os.path.isdir(text)
                entries.append({
                    "name": os.path.basename(os.path.normpath(text)).lower(),
                    "path": text,
                    "exists": exists,
                    "models": _count_models(text) if exists else 0,
                })
            if entries:
                out[key.strip().lower()] = entries
        return out

    # Walking the tree is blocking work; the event loop must not wait on it.
    folders = await asyncio.to_thread(_collect)
    return web.json_response({"folders": folders, "available": True})
