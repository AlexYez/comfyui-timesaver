"""Model-download jobs for TS Image Studio (and any other frontend).

The plan's §7.5 contract: one model = one job, two progress levels, no use
of the ComfyUI prompt queue. This module wraps the battle-tested engine of
TS_DownloadFilesNode as a service:

    POST /ts_downloader/fetch   {url, target, operation_id?} -> {job_id}
    POST /ts_downloader/cancel  {job_id}
    GET  /ts_downloader/jobs    -> {jobs: [...]}  (rehydration after reload)

WebSocket events: ``ts_downloader.fetch_progress``
    {job_id, filename, done_bytes, total_bytes, speed_bps, eta_s, status,
     error?}   status: queued|running|verifying|done|error|cancelled

At most two downloads run concurrently; the rest wait as ``queued``.
Cancellation flips a per-job flag checked between chunks; the .part file
stays, so a restart resumes from where it stopped (the engine already does
Range resumes).
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

from .._shared import make_route_registrars, resolve_prompt_server

_PROMPT_SERVER = resolve_prompt_server(lambda msg: None)
_register_get, _register_post = make_route_registrars(_PROMPT_SERVER, lambda msg: None)

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_GUARD = threading.Lock()
_CONCURRENCY = asyncio.Semaphore(2)
_EVENT = "ts_downloader.fetch_progress"
_THROTTLE_S = 0.25


def _emit(job: dict[str, Any]) -> None:
    if _PROMPT_SERVER is None:
        return
    now = time.monotonic()
    if job["status"] == "running" and now - job.get("_last_emit", 0.0) < _THROTTLE_S:
        return
    job["_last_emit"] = now
    payload = {k: v for k, v in job.items() if not k.startswith("_")}
    try:
        _PROMPT_SERVER.send_sync(_EVENT, payload)
    except Exception:
        pass


def _job_snapshot() -> list[dict[str, Any]]:
    with _JOBS_GUARD:
        return [{k: v for k, v in job.items() if not k.startswith("_")}
                for job in _JOBS.values()]


def _run_job_blocking(job: dict[str, Any]) -> None:
    from .ts_downloader import TS_DownloadFilesNode as Node

    url = job["_url"]
    target = job["_target"]
    resolved = Node._resolve_target_directory(target)
    if not resolved:
        raise RuntimeError(f"Target '{target}' did not resolve to a directory.")

    started = time.monotonic()
    state = {"last_done": 0, "last_t": started}

    def progress_cb(done: int, total: int, phase: str) -> None:
        if job.get("_cancel"):
            raise InterruptedError("cancelled by the user")
        now = time.monotonic()
        dt = max(1e-3, now - state["last_t"])
        speed = (done - state["last_done"]) / dt if done >= state["last_done"] else 0.0
        state["last_done"], state["last_t"] = done, now
        job.update({
            "status": "verifying" if phase == "verify" else "running",
            "done_bytes": int(done),
            "total_bytes": int(total),
            "speed_bps": float(speed),
            "eta_s": float((total - done) / speed) if speed > 1 and total > done else None,
        })
        _emit(job)

    session = Node._create_session_with_retries()
    try:
        ok = Node._download_single_file(
            session, url, resolved, True, True, 1024 * 1024,
            "huggingface.co", "", "", False, "hf_sha256_auto",
            progress_cb=progress_cb,
            # No run guard: this download belongs to no prompt, so interrupting
            # a prompt has nothing to say about it. Cancelling goes through
            # this job's own flag, checked in progress_cb above.
        )
    finally:
        try:
            session.close()
        except Exception:
            pass
    if not ok:
        # The engine swallows the InterruptedError our progress_cb raises and
        # reports a plain failure; restore the honest verdict.
        if job.get("_cancel"):
            raise InterruptedError("cancelled by the user")
        raise RuntimeError("Download failed — see the server log for the reason.")


async def _job_worker(job: dict[str, Any]) -> None:
    async with _CONCURRENCY:
        if job.get("_cancel"):
            job["status"] = "cancelled"
            _emit(job)
            return
        job["status"] = "running"
        _emit(job)
        try:
            await asyncio.to_thread(_run_job_blocking, job)
            job["status"] = "done"
        except InterruptedError:
            job["status"] = "cancelled"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
        job["_last_emit"] = 0.0
        _emit(job)


@_register_post("/ts_downloader/fetch")
async def fetch_route(request):
    from aiohttp import web

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body."}, status=400)
    url = str(data.get("url") or "").strip()
    target = str(data.get("target") or "").strip()
    if not url.lower().startswith(("http://", "https://")) or not target:
        return web.json_response({"error": "url and target are required."}, status=400)

    job_id = f"dl_{uuid.uuid4().hex[:10]}"
    job = {
        "job_id": job_id,
        "filename": url.rsplit("/", 1)[-1].split("?")[0],
        "url_host": url.split("/", 3)[2] if url.count("/") >= 3 else "",
        "target": target,
        "status": "queued",
        "done_bytes": 0,
        "total_bytes": 0,
        "speed_bps": 0.0,
        "eta_s": None,
        "operation_id": str(data.get("operation_id") or ""),
        "_url": url,
        "_target": target,
        "_cancel": False,
    }
    with _JOBS_GUARD:
        _JOBS[job_id] = job
    _emit(job)
    asyncio.create_task(_job_worker(job))
    return web.json_response({"job_id": job_id})


@_register_post("/ts_downloader/cancel")
async def cancel_route(request):
    from aiohttp import web

    try:
        data = await request.json()
    except Exception:
        data = {}
    job_id = str(data.get("job_id") or "")
    with _JOBS_GUARD:
        job = _JOBS.get(job_id)
    if job is None:
        return web.json_response({"error": "unknown job_id"}, status=404)
    job["_cancel"] = True
    return web.json_response({"ok": True})


@_register_get("/ts_downloader/jobs")
async def jobs_route(request):
    from aiohttp import web

    return web.json_response({"jobs": _job_snapshot()})
