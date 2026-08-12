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

Note: ``nodes/llm/_qwen_engine.py`` still carries its own equivalent
implementation. The two differ only in failure policy — the Qwen one stops
cycling mirrors on ``TypeError`` (a hub-version mismatch that switching
endpoints cannot fix), while this helper retries every endpoint — so merging
them means carrying that fail-fast rule over first.

The loader skips ``_``-prefixed modules, so this is never registered as a node.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://huggingface.co"


def is_official_hf_origin(url: str) -> bool:
    """Сам Hugging Face — и никакое зеркало.

    ⚠️ ОДНО правило на весь пак: кому можно отдать ``Authorization``. Токен
    открывает приватные репозитории и нередко имеет право записи, а зеркало —
    чужая машина. «Ведёт себя как HF» (тот же формат ссылок, тот же ETag с
    SHA256) достаточно, чтобы скачать файл и сверить хеш, но НЕ достаточно,
    чтобы получить токен.

    Дыру закрыли в загрузчике файлов, но она оставалась ещё в двух местах —
    в общем ``snapshot_download_resilient`` и в движке Qwen: токен вычислялся
    один раз ДО цикла по зеркалам и уезжал на каждое из них. Теперь спрашивают
    здесь все трое.
    """
    try:
        host = urlparse(str(url or "")).netloc.lower()
    except Exception:                       # noqa: BLE001 - мусор вместо адреса
        return False
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    host = host.split(":", 1)[0]
    return host == "huggingface.co" or host.endswith(".huggingface.co")


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


# ── закреплённые коммиты ─────────────────────────────────────────────────── #
#
# ⚠️ `main` — подвижная ветка. Для BiRefNet это прямое исполнение чужого
# кода (пак делает `exec_module` над скачанным `birefnet.py`), для
# остальных — подмена весов. Проверка целостности у huggingface_hub
# подтверждает, что файл доехал без искажений, но НЕ подтверждает, что
# содержимое то же самое, что вчера.
#
# Коммиты сняты 2026-08-10. Обновлять осознанно: новая архитектура
# заслуживает прогона, а не молчаливого приезда.
PINNED_REVISIONS = {
    "1038lab/BiRefNet": "4d000788a9698c7f8d67c8c6ce2b40c768f5b909",
    "ZhengPeng7/BiRefNet": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    "ZhengPeng7/BiRefNet-matting": "57f9f68b43ba337c75762b14cf3075d659007268",
    "ZhengPeng7/BiRefNet-portrait": "ecdeb6240ef23557dbd48ff27c59c1a88cbcb755",
    "ZhengPeng7/BiRefNet_HR": "a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
    "ZhengPeng7/BiRefNet_HR-matting": "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
    "ZhengPeng7/BiRefNet_dynamic": "280306042f57b7a33854319da62fd86aaa89ec4c",
    "ZhengPeng7/BiRefNet_dynamic-matting": "074df545be87034e74a96bf71566ecbbc4c15f0a",
    "ZhengPeng7/BiRefNet_lite": "7838f1c3472f827cd8ce13ab5ccc2ce48077360f",
    "ZhengPeng7/BiRefNet_lite-2K": "67d658fa863b1e716c3854270645e68860007d0e",
    "ZhengPeng7/BiRefNet_lite-matting": "99c33412e3f58e1f33187abdc8c435c645243690",
    "hustvl/vitmatte-small-composition-1k": "6a58ad7646403c1df626fbd746900aec7361ea1d",
    "hustvl/vitmatte-base-composition-1k": "bf486d01a7d9e3dbcc8400f7942835caf0eaf76e",
    "hustvl/vitmatte-small-distinctions-646": "6a0e75d7214b01f4d1163ede0f15b23afbbd480b",
    "hustvl/vitmatte-base-distinctions-646": "b58373f8dbbfbeb58157456e2e4949f9f872aa18",
    "depth-anything/Video-Depth-Anything-Small": "256875362cff76724b920335dfb4b29dd611f66e",
    "depth-anything/Video-Depth-Anything-Large": "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
    "huihui-ai/Huihui-Qwen3.5-2B-abliterated": "b2e291a65f29a9b148981fa5299caea5d35bd4c8",
    "huihui-ai/Huihui-Qwen3.5-4B-abliterated": "5581467dfd52bf338c782006a6cdce05c42594be",
    "huihui-ai/Huihui-Qwen3.5-9B-abliterated": "05b9e7c9b978ba29bdb8f50a49c30e4b91183339",
}


def pinned_revision(repo_id: str, default: str = "main") -> str:
    """Коммит, на котором закреплён репозиторий модели.

    Незнакомый репозиторий (человек указал свой) остаётся на `default` —
    запретить его нечем, но в журнал это попадает: оттуда тоже приедет
    код или веса, за которые пак не отвечает.
    """
    revision = PINNED_REVISIONS.get(str(repo_id or ""))
    if revision:
        return revision
    logger.warning(
        "[TS HF] %s is not pinned: taking '%s', whatever it holds right now.",
        repo_id, default,
    )
    return default


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
        # ⚠️ Токен — ТОЛЬКО официальному origin. Список endpoint-ов может
        # прийти из HF_ENDPOINT/HF_MIRROR, то есть указывать на чужую машину;
        # раньше он получал `Authorization` наравне с самим Hugging Face.
        if clean_token:
            if is_official_hf_origin(endpoint):
                kwargs["token"] = clean_token
            else:
                log.warning(
                    "%s %s is not huggingface.co — downloading without the token.",
                    log_prefix,
                    endpoint,
                )
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
