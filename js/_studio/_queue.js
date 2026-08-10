// TS Studio kit — the job queue panel (ui-kit layer).
//
// The studio does NOT keep a shadow queue: ComfyUI already has one, and jobs
// sent from the graph, from another tab or from the studio must all show up in
// the same list. This panel is a view over GET /queue plus the four actions
// the server supports:
//
//   • stop the running job          → POST /interrupt
//   • drop one pending job          → POST /queue {delete: [id]}
//   • drop every pending job        → POST /queue {clear: true}
//   • reorder pending jobs          → there is no reorder endpoint, so the
//     panel clears the pending block and resubmits the prompts in the new
//     order. The prompts come back from /queue itself, so jobs that the
//     studio never sent survive the trip intact.
//
// Rows are labelled by reading the studio markers out of each queued prompt,
// which is why a foreign job still renders — it just says so.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-queue-styles";

export function ensureQueueStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-queue{display:flex;flex-direction:column;min-height:0;flex:1}
.ts-queue__head{display:flex;align-items:center;gap:6px;padding:8px;
    border-bottom:1px solid var(--ts-border)}
.ts-queue__count{flex:1;font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-queue__list{display:flex;flex-direction:column;gap:3px;padding:8px;overflow-y:auto;flex:1;
    min-height:0}
.ts-queue__row{flex:0 0 auto;display:flex;align-items:center;gap:6px;padding:5px 6px;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:var(--ts-sunken)}
.ts-queue__row.is-running{border-color:var(--ts-accent-line);background:var(--ts-accent-soft)}
.ts-queue__row.is-drag-over{border-color:var(--ts-accent)}
.ts-queue__handle{cursor:grab;color:var(--ts-muted);border:none;background:none;padding:0 2px;
    font-size:11px;letter-spacing:1px}
.ts-queue__body{flex:1;min-width:0;display:flex;flex-direction:column;gap:1px}
.ts-queue__title{font-size:var(--ts-fs-sm);color:var(--ts-text);overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
.ts-queue__sub{font-size:var(--ts-fs-xs);color:var(--ts-muted);overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
.ts-queue__x{border:none;background:none;color:var(--ts-muted);cursor:pointer;padding:0 3px;
    font-size:13px;line-height:1}
.ts-queue__x:hover{color:var(--ts-danger)}
.ts-queue__empty{padding:16px 10px;color:var(--ts-muted);font-size:var(--ts-fs-sm);
    text-align:center}
`;
    document.head.appendChild(style);
}

/** Human label for one queued prompt, read from the studio's own markers. */
export function describeJob(prompt, t) {
    const info = { title: t.queue.foreign, sub: "" };
    if (!prompt || typeof prompt !== "object") return info;
    let manifest = null;
    let text = "";
    for (const node of Object.values(prompt)) {
        const inputs = node?.inputs || {};
        if (node.class_type === "TS_StudioManifest") {
            try { manifest = JSON.parse(inputs.manifest || "{}"); } catch { manifest = null; }
        } else if (inputs.param_name === "prompt" && typeof inputs.value === "string") {
            text = inputs.value;
        }
    }
    if (manifest) {
        const family = manifest.family_label || manifest.family || "";
        const mode = t.modes[manifest.mode] || manifest.mode || "";
        info.title = [family, mode].filter(Boolean).join(" · ");
        info.sub = text.trim();
    } else {
        info.sub = text.trim();
    }
    return info;
}

/**
 * @param {object} options
 * @param {object} options.api ComfyUI api client.
 * @param {object} options.t Locale strings (needs t.queue and t.modes).
 * @param {(id: string) => void} [options.onInterrupt]
 * @returns {{element, refresh, setVisible, teardown, pendingCount}}
 */
export function createQueuePanel(options) {
    ensureQueueStyles();
    const { api, t } = options;

    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-queue`;

    const head = document.createElement("div");
    head.className = "ts-queue__head";
    const count = document.createElement("div");
    count.className = "ts-queue__count";
    const stopButton = document.createElement("button");
    stopButton.type = "button";
    stopButton.className = "ts-ui-btn";
    stopButton.textContent = t.queue.stopRunning;
    stopButton.title = t.queue.stopRunningTip;
    const clearButton = document.createElement("button");
    clearButton.type = "button";
    clearButton.className = "ts-ui-btn";
    clearButton.textContent = t.queue.clearPending;
    clearButton.title = t.queue.clearPendingTip;
    head.append(count, stopButton, clearButton);

    const list = document.createElement("div");
    list.className = "ts-queue__list";
    const empty = document.createElement("div");
    empty.className = "ts-queue__empty";
    empty.textContent = t.queue.queueEmpty;
    element.append(head, list, empty);

    let running = [];    // [{id, prompt}]
    let pending = [];    // [{id, prompt, extra}]
    let visible = false;
    let timer = 0;
    let busy = false;

    async function fetchQueue() {
        const response = await api.fetchApi("/queue");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const read = (rows) => (rows || []).map((row) => ({
            id: String(row[1]),
            prompt: row[2],
            extra: row[3] || {},
        }));
        running = read(payload.queue_running);
        pending = read(payload.queue_pending);
    }

    async function refresh() {
        if (busy) return;
        busy = true;
        try {
            await fetchQueue();
            render();
        } catch (err) {
            console.warn("[TS Studio] queue read failed", err);
        } finally {
            busy = false;
        }
    }

    let dragIndex = -1;

    function render() {
        list.textContent = "";
        const total = running.length + pending.length;
        count.textContent = t.queue.count(running.length, pending.length);
        empty.style.display = total ? "none" : "";
        stopButton.disabled = !running.length;
        clearButton.disabled = !pending.length;

        for (const job of running) list.appendChild(row(job, true, -1));
        pending.forEach((job, index) => list.appendChild(row(job, false, index)));
    }

    function row(job, isRunning, index) {
        const item = document.createElement("div");
        item.className = `ts-queue__row${isRunning ? " is-running" : ""}`;
        const info = describeJob(job.prompt, t);

        if (!isRunning) {
            const handle = document.createElement("button");
            handle.type = "button";
            handle.className = "ts-queue__handle";
            handle.textContent = "⋮⋮";
            handle.title = t.queue.reorderTip;
            item.appendChild(handle);
            item.draggable = true;
            item.addEventListener("dragstart", (event) => {
                dragIndex = index;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", String(index));
            });
            item.addEventListener("dragover", (event) => {
                if (dragIndex < 0) return;
                event.preventDefault();
                item.classList.add("is-drag-over");
            });
            item.addEventListener("dragleave", () => item.classList.remove("is-drag-over"));
            item.addEventListener("drop", (event) => {
                event.preventDefault();
                item.classList.remove("is-drag-over");
                if (dragIndex < 0 || dragIndex === index) return;
                const [moved] = pending.splice(dragIndex, 1);
                pending.splice(index, 0, moved);
                dragIndex = -1;
                render();
                applyOrder().catch((err) => console.warn("[TS Studio] reorder failed", err));
            });
        }

        const body = document.createElement("div");
        body.className = "ts-queue__body";
        const title = document.createElement("div");
        title.className = "ts-queue__title";
        title.textContent = isRunning ? `▶ ${info.title}` : info.title;
        const sub = document.createElement("div");
        sub.className = "ts-queue__sub";
        sub.textContent = info.sub;
        sub.title = info.sub;
        body.append(title, sub);
        item.appendChild(body);

        const x = document.createElement("button");
        x.type = "button";
        x.className = "ts-queue__x";
        x.textContent = "×";
        x.title = isRunning ? t.queue.stopRunningTip : t.queue.dropTip;
        x.addEventListener("click", () => {
            (isRunning ? interrupt() : drop(job.id))
                .catch((err) => console.warn("[TS Studio] queue action failed", err));
        });
        item.appendChild(x);
        return item;
    }

    async function post(path, body) {
        const response = await api.fetchApi(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body || {}),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response;
    }

    async function interrupt() {
        await post("/interrupt");
        options.onInterrupt?.(running[0]?.id);
        await refresh();
    }

    async function drop(id) {
        await post("/queue", { delete: [id] });
        await refresh();
    }

    async function clearPending() {
        await post("/queue", { clear: true });
        await refresh();
    }

    // No reorder endpoint exists: clear the pending block, then resubmit the
    // prompts in the order the user just dropped them into. The running job is
    // untouched, and each prompt keeps the extra_data it arrived with.
    async function applyOrder() {
        const order = pending.slice();
        await post("/queue", { clear: true });
        for (const job of order) {
            await post("/prompt", {
                prompt: job.prompt,
                client_id: job.extra?.client_id || api.clientId,
                extra_data: job.extra,
            });
        }
        await refresh();
    }

    stopButton.addEventListener("click", () => {
        interrupt().catch((err) => console.warn("[TS Studio] interrupt failed", err));
    });
    clearButton.addEventListener("click", () => {
        clearPending().catch((err) => console.warn("[TS Studio] clear failed", err));
    });

    // The status event fires on every queue change; the slow poll is the
    // safety net for changes made in another tab.
    const onStatus = () => { if (visible) refresh(); };
    api.addEventListener("status", onStatus);

    function setVisible(next) {
        visible = Boolean(next);
        clearInterval(timer);
        if (visible) {
            refresh();
            timer = setInterval(refresh, 2500);
        }
    }

    return {
        element,
        refresh,
        setVisible,
        pendingCount: () => pending.length,
        runningCount: () => running.length,
        teardown: () => {
            clearInterval(timer);
            api.removeEventListener("status", onStatus);
        },
    };
}
