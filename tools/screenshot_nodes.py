"""Take canvas+DOM composite screenshots of every TS node.

Connects to a running ComfyUI on 127.0.0.1:8188 with headless Chromium via
Playwright, programmatically places each node on a clean graph, waits for
the Vue/DOM-widget render pass, and saves a clipped PNG per node into
``doc/screenshots/<stem>.png`` where ``<stem>`` is the basename of the
node's Python file (``nodes/utils/ts_math_int.py`` → ``ts_math_int.png``).

Why not ``canvas.toDataURL()``?  ComfyUI ноды состоят из двух слоёв:
HTML5 canvas (заголовок, сокеты, базовые виджеты) и DOM widgets через
``addDOMWidget`` (waveform-плеер, lama toolbar, blocks list, styles grid,
video player и т.п.). ``canvas.toDataURL()`` берёт только pixel buffer
canvas — DOM widgets теряются. Playwright делает screenshot через Chrome
DevTools Protocol (``Page.captureScreenshot``), который захватывает
финальный composited frame на уровне renderer, точно как видит человек.

Зависимости:

    python -m pip install playwright
    python -m playwright install chromium

Запуск:

    python tools/screenshot_nodes.py             # перешутить все ноды
    python tools/screenshot_nodes.py TS_Keyer    # перешутить конкретные
    python tools/screenshot_nodes.py --url http://127.0.0.1:8188

Опции CLI:

    --url URL              ComfyUI URL (default 127.0.0.1:8188).
    --out DIR              Папка для PNG (default doc/screenshots).
    --scale FLOAT          Zoom для рендеринга ноды (default 1.5).
    --no-headless          Запустить с видимым окном (отладка).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Reuse the AST-based contract collector to discover node ids and python files.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from tools import build_node_contracts as builder  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "playwright is not installed.\n"
        "Run: python -m pip install playwright && python -m playwright install chromium"
    ) from exc


COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_OUT = PACKAGE_ROOT / "doc" / "screenshots"

NODE_PLACE_JS_TEMPLATE = r"""
(typeId) => {
  return new Promise(resolve => {
    app.graph.clear();
    const node = LiteGraph.createNode(typeId);
    if (!node) { resolve({error: 'createNode returned null'}); return; }
    node.pos = [60, 30];
    app.graph.add(node);
    const ds = app.canvas.ds;
    const scale = SCALE_PLACEHOLDER;
    ds.scale = scale;
    // Place node so its top-left lands at viewport (220, 120) — clears the
    // workflow-tabs dropdown in the top-left corner of ComfyUI's toolbar.
    ds.offset[0] = 220/scale - 60;
    ds.offset[1] = 120/scale - 30;
    app.canvas.setDirty(true, true);
    app.canvas.draw(true, true);
    setTimeout(() => {
      app.canvas.draw(true, true);
      const cv = app.canvas.canvas;
      const cv_rect = cv.getBoundingClientRect();
      const w = node.size[0] * scale;
      const h = (node.size[1] + LiteGraph.NODE_TITLE_HEIGHT) * scale;
      const pad = 8;
      const x = cv_rect.left + (node.pos[0] + ds.offset[0]) * scale - pad;
      const y = cv_rect.top + (node.pos[1] - LiteGraph.NODE_TITLE_HEIGHT + ds.offset[1]) * scale - pad;
      resolve({
        x: Math.max(0, Math.floor(x)),
        y: Math.max(0, Math.floor(y)),
        w: Math.ceil(w + pad*2),
        h: Math.ceil(h + pad*2),
      });
    }, 1500);
  });
}
"""


def _build_targets(filters: list[str]) -> list[tuple[str, str]]:
    """Return [(node_id, file_stem), ...] for every node, filtered by ids if any."""
    contracts = builder.collect_contracts()
    pairs = []
    for node_id, contract in sorted(contracts.items()):
        stem = Path(contract.python_file).stem
        pairs.append((node_id, stem))

    if not filters:
        return pairs

    wanted = {f.lower() for f in filters}
    return [(nid, stem) for nid, stem in pairs if nid.lower() in wanted or stem.lower() in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filters", nargs="*", help="Node ids or file stems to screenshot (default: all).")
    parser.add_argument("--url", default=COMFYUI_URL, help=f"ComfyUI URL (default: {COMFYUI_URL})")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output directory (default: {DEFAULT_OUT.relative_to(PACKAGE_ROOT)})")
    parser.add_argument("--scale", type=float, default=1.5, help="Zoom level inside ComfyUI (default: 1.5)")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window (debug).")
    parser.add_argument("--wait", type=float, default=3.0, help="Seconds to wait after page load before first node (default 3).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _build_targets(args.filters)
    if not targets:
        print("No nodes match the given filters.", file=sys.stderr)
        return 2

    js = NODE_PLACE_JS_TEMPLATE.replace("SCALE_PLACEHOLDER", str(float(args.scale)))

    failed: list[tuple[str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.no_headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            locale="en-US",  # CRITICAL: forces English UI text (otherwise Chromium picks
                             # OS locale and ComfyUI translates labels: 'IMAGE' → 'ИЗОБРАЖЕНИЕ').
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = context.new_page()
        print(f"Opening {args.url} ...")
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "typeof app !== 'undefined' && typeof LiteGraph !== 'undefined' && app.graph",
            timeout=30000,
        )
        time.sleep(args.wait)
        print(f"ComfyUI ready. Capturing {len(targets)} nodes.")

        total = len(targets)
        for i, (node_id, stem) in enumerate(targets, 1):
            try:
                clip = page.evaluate(js, node_id)
                if not isinstance(clip, dict) or "error" in clip:
                    failed.append((stem, repr(clip)))
                    print(f"[{i:02d}/{total}] FAIL {stem}: {clip}")
                    continue
                time.sleep(0.4)
                out_path = out_dir / f"{stem}.png"
                page.screenshot(
                    path=str(out_path),
                    clip={"x": clip["x"], "y": clip["y"], "width": clip["w"], "height": clip["h"]},
                    type="png",
                )
                size = out_path.stat().st_size
                print(f"[{i:02d}/{total}] OK   {stem}.png ({clip['w']}x{clip['h']}, {size//1024} KB)")
            except Exception as exc:
                failed.append((stem, str(exc)))
                print(f"[{i:02d}/{total}] EXC  {stem}: {exc}")

        browser.close()

    print(f"\nDone. {total - len(failed)} ok, {len(failed)} failed.")
    if failed:
        for stem, err in failed:
            print(f"  - {stem}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
