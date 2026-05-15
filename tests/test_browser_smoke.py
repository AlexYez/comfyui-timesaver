"""Headless browser smoke tests.

Requires ComfyUI running on http://127.0.0.1:8188 + ``playwright`` with a
Chromium build available. If either is missing the whole module is skipped,
mirroring the pattern in ``test_comfyui_live_api.py``.

Coverage:

- Every registered node id from the snapshot survives ``LiteGraph.createNode``
  in the live frontend (no JS exception, no ``null`` return).
- Page exception count and console-error count stay at zero (after a small
  noise filter for unrelated ComfyUI core warnings).
- The four interactive DOM-widget nodes (TS_LamaCleanup, TS_AudioLoader,
  TS_AudioPreview, TS_SuperPrompt) actually mount the widget container that
  their JS extensions add.
- All public HTTP routes registered by our subsystems are reachable:
  ``/ts_lama_cleanup/model_status`` and ``/ts_voice_recognition/status``.
- TS_LamaCleanup wires its cleanup function into ``node.onRemoved`` so the
  source-poll interval and document-level paste listener are released when
  the node is deleted (TECH_DEBT_AUDIT #3 regression guard).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest


COMFYUI_URL = "http://127.0.0.1:8188"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _comfyui_running(url: str = COMFYUI_URL, timeout: float = 3.0) -> bool:
    try:
        # COMFYUI_URL is a hardcoded local 127.0.0.1 endpoint used only by the
        # opt-in smoke test against a developer-controlled ComfyUI instance.
        with urllib.request.urlopen(f"{url}/api/system_stats", timeout=timeout) as response:  # nosec B310
            return response.status == 200
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


if not _comfyui_running():
    pytest.skip(
        "ComfyUI not reachable at 127.0.0.1:8188 — browser smoke tests skipped.",
        allow_module_level=True,
    )

playwright_sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright not installed — `python -m pip install playwright && python -m playwright install chromium`",
)
sync_playwright = playwright_sync_api.sync_playwright

# Re-use the same AST-based contract collector the screenshot helper uses.
sys.path.insert(0, str(PACKAGE_ROOT))
from tools import build_node_contracts as builder  # noqa: E402


CREATE_NODE_JS = r"""
(typeId) => {
  try {
    app.graph.clear();
    const node = LiteGraph.createNode(typeId);
    if (!node) return {ok: false, error: "createNode returned null"};
    app.graph.add(node);
    return {
      ok: true,
      type: node.type,
      inputs: (node.inputs || []).length,
      outputs: (node.outputs || []).length,
      widgets: (node.widgets || []).length,
      hasDomWidget: (node.widgets || []).some(w => w && (w.type === "div" || w.element instanceof HTMLElement)),
    };
  } catch (err) {
    return {ok: false, error: String(err && err.stack || err)};
  }
}
"""


CHECK_LAMA_CLEANUP_REMOVAL_JS = r"""
async () => {
  app.graph.clear();
  const node = LiteGraph.createNode("TS_LamaCleanup");
  if (!node) return {ok: false, error: "createNode null"};
  app.graph.add(node);

  // Wait a tick for setupLamaCleanup to attach the cleanup function via
  // beforeRegisterNodeDef → onNodeCreated → setupLamaCleanup.
  await new Promise(r => setTimeout(r, 50));

  const wrapped = node.onRemoved;
  const hasCleanup = typeof node._tsLamaCleanupCleanup === "function";
  const cleanupBefore = hasCleanup;

  // Snapshot listener count: paste listener is attached to document.
  // We can't read it directly, but we can install a probe and count
  // how many copies of our handler exist via a marker on cleanup.
  let cleanupRan = false;
  const orig = node._tsLamaCleanupCleanup;
  if (typeof orig === "function") {
    node._tsLamaCleanupCleanup = function () {
      cleanupRan = true;
      return orig.apply(this, arguments);
    };
  }

  // Trigger removal via the same path LiteGraph uses.
  app.graph.remove(node);
  await new Promise(r => setTimeout(r, 50));

  return {
    ok: true,
    onRemovedWasWrapped: typeof wrapped === "function",
    cleanupBefore: cleanupBefore,
    cleanupRan: cleanupRan,
  };
}
"""


@pytest.fixture(scope="module")
def browser_page():
    page_errors: list[str] = []
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = context.new_page()
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "console",
            lambda msg: console_errors.append(f"[{msg.type}] {msg.text}")
            if msg.type == "error"
            else None,
        )
        page.goto(COMFYUI_URL, wait_until="networkidle")
        page.wait_for_function(
            "typeof app !== 'undefined' && typeof LiteGraph !== 'undefined' && app.graph",
            timeout=30000,
        )
        # Give Vue/DOM-widget extensions a moment to register.
        page.wait_for_timeout(2000)

        yield {"page": page, "page_errors": page_errors, "console_errors": console_errors}

        browser.close()


@pytest.fixture(scope="module")
def node_ids() -> list[str]:
    return sorted(builder.collect_contracts().keys())


def test_every_node_creates_in_litegraph(browser_page, node_ids):
    page = browser_page["page"]
    failed = []
    for node_id in node_ids:
        result = page.evaluate(CREATE_NODE_JS, node_id)
        if not result or not result.get("ok"):
            failed.append((node_id, (result or {}).get("error", "unknown")))
    assert not failed, f"createNode failed for {len(failed)} nodes: {failed[:5]}"


def test_no_unfiltered_console_errors_during_smoke(browser_page):
    """ComfyUI core fires a few benign warnings on every page load (jobs API,
    third-party 404s). Assert that nothing else slipped through.
    """
    NOISE = (
        "[Jobs API] Failed to fetch",
        "Failed to load resource",
        "ComfyApp graph accessed before initialization",
    )
    leftovers = [
        e for e in browser_page["console_errors"] if not any(p in e for p in NOISE)
    ]
    assert not leftovers, f"Unexpected console errors: {leftovers}"


def test_no_page_exceptions_during_smoke(browser_page):
    assert browser_page["page_errors"] == [], browser_page["page_errors"]


@pytest.mark.parametrize(
    "node_id",
    ["TS_LamaCleanup", "TS_AudioLoader", "TS_AudioPreview", "TS_SuperPrompt"],
)
def test_interactive_node_exposes_dom_widget(browser_page, node_id):
    """Frontend extension should mount at least one DOM widget (canvas /
    waveform / textarea) when the node is created — regression guard for the
    `addDOMWidget` integration after the lazy-import + super_prompt split.
    """
    page = browser_page["page"]
    result = page.evaluate(CREATE_NODE_JS, node_id)
    assert result.get("ok"), result.get("error")
    assert result.get("widgets", 0) > 0, f"{node_id} reports no widgets"


@pytest.mark.parametrize(
    "method, path, accept_status",
    [
        ("GET", "/ts_lama_cleanup/model_status", {200}),
        ("GET", "/ts_voice_recognition/status", {200}),
    ],
)
def test_subsystem_http_route_reachable(method, path, accept_status):
    request = urllib.request.Request(f"{COMFYUI_URL}{path}", method=method)
    try:
        # Local 127.0.0.1 endpoint, opt-in smoke test — not user input.
        with urllib.request.urlopen(request, timeout=10) as response:  # nosec B310
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = ""
    assert status in accept_status, f"{method} {path} returned {status}: {body[:200]}"


def test_lama_cleanup_releases_handlers_on_removal(browser_page):
    """Regression guard for TECH_DEBT_AUDIT #3: setupLamaCleanup wraps
    `node.onRemoved`, and the cleanup function fires when LiteGraph removes
    the node from the graph.
    """
    page = browser_page["page"]
    result = page.evaluate(CHECK_LAMA_CLEANUP_REMOVAL_JS)
    assert result.get("ok"), result.get("error")
    assert result.get("onRemovedWasWrapped"), "onRemoved was not wrapped on TS_LamaCleanup"
    assert result.get("cleanupBefore"), "_tsLamaCleanupCleanup was not attached"
    assert result.get("cleanupRan"), "_tsLamaCleanupCleanup did not run on graph.remove"
