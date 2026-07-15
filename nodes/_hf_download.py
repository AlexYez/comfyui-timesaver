"""Shared, resilient wrapper around ``huggingface_hub.snapshot_download``.

Every TS node that pulls weights from the Hub used to call ``snapshot_download``
directly, each with a different level of hardening. This module centralises the
two things that actually matter in the field:

* **Endpoint mirrors** — ``huggingface.co`` is not reachable everywhere. The
  endpoint list comes from the caller, else ``HF_ENDPOINT``/``HF_MIRROR`` in the
  environment, else the default hub. Each is tried in order.
* **Kwarg compatibility** — ``resume_download`` and ``endpoint`` have moved
  in/out of the ``snapshot_download`` signature across ``huggingface_hub``
  releases. A ``TypeError`` naming one of them retries without it instead of
  failing the download.

Node-specific policy (``allow_patterns``, repo fallbacks, progress reporting)
stays with the caller — this helper only owns the transport.

Note: ``nodes/llm/_qwen_engine.py`` keeps its own equivalent implementation
because it also tracks endpoint support to size its download estimates; it is
not a candidate for this helper without dragging that state along.

The loader skips ``_``-prefixed modules, so this is never registered as a node.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://huggingface.co"


def resolve_endpoints(endpoints: Sequence[str] | str | None = None) -> list[str]:
    """Normalise an endpoint list: explicit → environment → default hub.

    Accepts a comma-separated string or a sequence. Bare hosts get an https://
    scheme. Always returns at least one endpoint.
    """
    raw: list[str] = []
    if isinstance(endpoints, str):
        raw = [part.strip() for part in endpoints.split(",")]
    elif endpoints:
        raw = [str(part).strip() for part in endpoints]

    if not any(raw):
        env_value = os.environ.get("HF_ENDPOINT") or os.environ.get("HF_MIRROR") or ""
        raw = [part.strip() for part in env_value.split(",")]

    resolved = []
    for entry in raw:
        if not entry:
            continue
        if not entry.startswith(("http://", "https://")):
            entry = "https://" + entry
        if entry not in resolved:
            resolved.append(entry)

    if not resolved:
        resolved = [_DEFAULT_ENDPOINT]
    return resolved


def _snapshot_download_once(kwargs: dict[str, Any], log: logging.Logger, log_prefix: str) -> str:
    """One download attempt, retrying without kwargs the installed hub rejects."""
    # Lazy: huggingface_hub drags in HTTP/auth/cache infra and is only needed on
    # a cold download.
    from huggingface_hub import snapshot_download

    attempt = dict(kwargs)
    while True:
        try:
            return snapshot_download(**attempt)
        except TypeError as exc:
            message = str(exc)
            # Older/newer hub releases moved these; drop and retry rather than
            # failing a download over a signature difference.
            for fragile in ("resume_download", "endpoint", "local_dir_use_symlinks"):
                if fragile in message and fragile in attempt:
                    attempt.pop(fragile, None)
                    log.warning(
                        "%s huggingface_hub rejected '%s'; retrying without it.",
                        log_prefix,
                        fragile,
                    )
                    break
            else:
                raise


def snapshot_download_resilient(
    repo_id: str,
    local_dir: str,
    *,
    revision: str = "main",
    allow_patterns: Sequence[str] | None = None,
    token: str | None = None,
    endpoints: Sequence[str] | str | None = None,
    log: logging.Logger | None = None,
    log_prefix: str = "[TS HF Download]",
) -> str:
    """Download ``repo_id`` into ``local_dir``, cycling endpoints on failure.

    Args:
        repo_id: Hub repository id.
        local_dir: Destination directory.
        revision: Git revision to fetch (pinned to a branch/tag, never a
            mutable default, so a scanner can audit what we pull).
        allow_patterns: Restrict the fetched files (keep node-specific — a wide
            pattern on a mirror repo can pull gigabytes of extras).
        token: Hub token for gated repos.
        endpoints: Explicit mirror list; falls back to HF_ENDPOINT/HF_MIRROR.
        log: Logger to report attempts on.
        log_prefix: TS log prefix for the messages.

    Returns:
        The local path reported by ``snapshot_download``.

    Raises:
        RuntimeError: when every endpoint failed; the last error is included.
    """
    log = log or logger
    resolved_endpoints = resolve_endpoints(endpoints)
    clean_token = token.strip() if token and token.strip() else None
    last_error: Exception | None = None

    for index, endpoint in enumerate(resolved_endpoints):
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "resume_download": True,
        }
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        if clean_token:
            kwargs["token"] = clean_token
        # Only pass a non-default endpoint: on older hubs the kwarg does not
        # exist, and there is no reason to probe for it when we want the default.
        if endpoint != _DEFAULT_ENDPOINT:
            kwargs["endpoint"] = endpoint

        if len(resolved_endpoints) > 1:
            log.info(
                "%s Download attempt %d/%d via %s",
                log_prefix,
                index + 1,
                len(resolved_endpoints),
                endpoint,
            )
        try:
            return _snapshot_download_once(kwargs, log, log_prefix)
        except Exception as exc:  # noqa: BLE001 — transient/mirror-specific → next endpoint
            last_error = exc
            if index + 1 < len(resolved_endpoints):
                log.warning(
                    "%s Download failed on %s: %s. Trying next mirror.",
                    log_prefix,
                    endpoint,
                    exc,
                )
            continue

    raise RuntimeError(f"{log_prefix} Download of '{repo_id}' failed. Last error: {last_error}")
