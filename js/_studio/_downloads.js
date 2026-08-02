// TS Studio kit — model download panel (ui-kit layer).
//
// Renders the unavailable-backend story the plan demands (§7.5): every
// missing model is a row with its own Download button, a per-file progress
// bar (bytes, speed, eta, verifying state) and a cancel; above them one
// TOTAL bar aggregates bytes across active jobs. URLs come from the
// downloader pack's HF search; downloads run as fetch-route jobs — never
// through the prompt queue. Reload-safe: GET /jobs rehydrates active bars.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";
import { deckSection } from "./_shell.js";

const STYLE_ID = "ts-studio-downloads-styles";

export function ensureDownloadStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-dl{display:flex;flex-direction:column;gap:6px}
.ts-dl__total{display:none;flex-direction:column;gap:3px}
.ts-dl__total.is-active{display:flex}
.ts-dl__row{display:flex;flex-direction:column;gap:3px;padding:6px 8px;
    border-radius:var(--ts-radius-sm);background:var(--ts-sunken)}
.ts-dl__head{display:flex;align-items:center;gap:6px}
.ts-dl__name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    font-size:var(--ts-fs-sm)}
.ts-dl__meta{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-dl__bar{height:3px;border-radius:2px;background:var(--ts-border-soft);overflow:hidden;
    display:none}
.ts-dl__bar.is-active{display:block}
.ts-dl__bar div{height:100%;width:0%;background:var(--ts-accent);transition:width .2s ease}
.ts-dl__bar.is-verify div{background:var(--ts-success)}
`;
    document.head.appendChild(style);
}

function fmtBytes(n) {
    if (!n) return "0 MB";
    return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} MB`;
}

/**
 * @param {object} options {api, t, problems: [{backend, family, model, match, folder}], onResolved}
 * @returns {{element, teardown}}
 */
export function createDownloadPanel(options) {
    ensureDownloadStyles();
    const { api, t } = options;
    const section = deckSection(t.dl.dlHeader);
    const wrap = document.createElement("div");
    wrap.className = `${TS_UI_CLASS} ts-dl`;
    section.appendChild(wrap);

    const total = document.createElement("div");
    total.className = "ts-dl__total";
    const totalMeta = document.createElement("div");
    totalMeta.className = "ts-dl__meta";
    const totalBar = document.createElement("div");
    totalBar.className = "ts-dl__bar is-active";
    const totalFill = document.createElement("div");
    totalBar.appendChild(totalFill);
    total.append(totalMeta, totalBar);
    wrap.appendChild(total);

    const jobRows = new Map(); // job_id -> {fill, meta, bar, cancel}
    const jobStats = new Map(); // job_id -> {done, total}

    function refreshTotal() {
        let done = 0;
        let sum = 0;
        for (const stat of jobStats.values()) {
            done += stat.done;
            sum += stat.total;
        }
        total.classList.toggle("is-active", jobStats.size > 0 && sum > 0);
        if (sum > 0) {
            totalFill.style.width = `${Math.min(100, (done / sum) * 100)}%`;
            totalMeta.textContent = t.dl.total(jobStats.size, fmtBytes(done), fmtBytes(sum));
        }
    }

    function attachJob(jobId, row) {
        jobRows.set(jobId, row);
        jobStats.set(jobId, { done: 0, total: 0 });
        row.bar.classList.add("is-active");
        row.cancel.style.display = "";
        row.cancel.onclick = async () => {
            await api.fetchApi("/ts_downloader/cancel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ job_id: jobId }),
            });
        };
    }

    const onProgress = ({ detail }) => {
        const row = jobRows.get(detail?.job_id);
        if (!row) return;
        const stat = jobStats.get(detail.job_id);
        stat.done = detail.done_bytes || 0;
        stat.total = detail.total_bytes || 0;
        const pct = stat.total ? Math.min(100, (stat.done / stat.total) * 100) : 0;
        row.fill.style.width = `${pct}%`;
        row.bar.classList.toggle("is-verify", detail.status === "verifying");
        if (detail.status === "running") {
            const speed = detail.speed_bps ? `${(detail.speed_bps / 1e6).toFixed(1)} MB/s` : "";
            const eta = detail.eta_s ? ` · ${Math.round(detail.eta_s)}s` : "";
            row.meta.textContent = `${fmtBytes(stat.done)} / ${fmtBytes(stat.total)} · ${speed}${eta}`;
        } else if (detail.status === "verifying") {
            row.meta.textContent = t.dl.verifying;
        } else if (detail.status === "queued") {
            row.meta.textContent = t.dl.waiting;
        } else {
            row.meta.textContent = t.dl.status(detail.status) + (detail.error ? `: ${detail.error}` : "");
            row.bar.classList.remove("is-active");
            row.cancel.style.display = "none";
            jobRows.delete(detail.job_id);
            jobStats.delete(detail.job_id);
            if (detail.status === "done") options.onResolved?.();
        }
        refreshTotal();
    };
    api.addEventListener("ts_downloader.fetch_progress", onProgress);

    // Reload-safe: adopt jobs that were already running.
    api.fetchApi("/ts_downloader/jobs").then(async (res) => {
        if (!res.ok) return;
        const { jobs } = await res.json();
        for (const job of jobs) {
            if (!["queued", "running", "verifying"].includes(job.status)) continue;
            const row = makeRow(job.filename, null);
            attachJob(job.job_id, row);
            onProgress({ detail: job });
        }
    }).catch(() => {});

    function makeRow(label, problem) {
        const row = document.createElement("div");
        row.className = "ts-dl__row";
        const head = document.createElement("div");
        head.className = "ts-dl__head";
        const name = document.createElement("span");
        name.className = "ts-dl__name";
        name.textContent = label;
        name.title = label;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-ui-btn";
        button.textContent = t.dl.get;
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "ts-ui-btn";
        cancel.textContent = t.dl.stop;
        cancel.style.display = "none";
        head.append(name, button, cancel);
        const meta = document.createElement("div");
        meta.className = "ts-dl__meta";
        const bar = document.createElement("div");
        bar.className = "ts-dl__bar";
        const fill = document.createElement("div");
        bar.appendChild(fill);
        row.append(head, meta, bar);
        wrap.appendChild(row);
        const handle = { row, button, cancel, meta, bar, fill };
        if (problem) wireSearchAndFetch(handle, problem);
        else button.style.display = "none";
        return handle;
    }

    async function wireSearchAndFetch(handle, problem) {
        handle.button.onclick = async () => {
            handle.button.disabled = true;
            handle.meta.textContent = t.dl.searching;
            try {
                const search = await api.fetchApi("/ts_downloader/hf_search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ filenames: [problem.filenameHint] }),
                });
                const payload = await search.json();
                const best = (payload.results?.[problem.filenameHint] || [])[0];
                if (!best?.url) {
                    handle.meta.textContent = t.dl.notFound;
                    handle.button.disabled = false;
                    return;
                }
                const fetchRes = await api.fetchApi("/ts_downloader/fetch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        url: best.url,
                        target: `models/${problem.folder}`,
                    }),
                });
                const job = await fetchRes.json();
                if (job.error) throw new Error(job.error);
                attachJob(job.job_id, handle);
                handle.meta.textContent = t.dl.waiting;
            } catch (err) {
                handle.meta.textContent = t.dl.status("error") + `: ${err.message}`;
                handle.button.disabled = false;
            }
        };
    }

    for (const problem of options.problems || []) {
        makeRow(`${problem.familyLabel}: ${problem.filenameHint}`, problem);
    }

    return {
        element: section,
        teardown: () => api.removeEventListener("ts_downloader.fetch_progress", onProgress),
    };
}
