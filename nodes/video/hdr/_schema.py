"""Тонкая прослойка между чистым HDR-ядром и типами ComfyUI.

Всё, что знает про ``comfy_api``, собрано здесь и в самих нодах. Ядро
(``_acescct``, ``_primaries``, ``_resize``, ``_tonemap``, ``_exr_io``,
``_hdr_types``) остаётся обычным Python с torch и тестируется без ComfyUI.
"""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._hdr_types import HDR_CONFIG_TYPE, HDR_IMAGE_TYPE

# ⚠️ Имена типов — публичный контракт сокетов, менять нельзя.
HdrConfigIO = IO.Custom(HDR_CONFIG_TYPE)
HdrImageIO = IO.Custom(HDR_IMAGE_TYPE)

CATEGORY = "TS/Video/HDR"

# Отдельный сентинел: ``None`` ComfyUI передаёт для входа, который подключён,
# но ещё не вычислен, а нам надо отличать это от «не подключён вовсе».
MISSING = object()
