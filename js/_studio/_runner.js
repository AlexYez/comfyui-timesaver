// TS Studio kit — prompt runner (core layer, no DOM).
//
// Submits patched backend graphs through the page's own API client and turns
// the WebSocket event stream into per-job callbacks. Uses api.clientId so
// progress/preview/executed events arrive on the socket the page already
// holds. Cancellation is scoped: pending jobs leave the queue by id, and
// /interrupt fires ONLY while OUR prompt is the executing one — the studio
// must never kill a canvas render (plan §14).

export function createRunner(api) {
    const jobs = new Map(); // prompt_id -> {callbacks, status}
    let executingPromptId = null;

    function track(event, handler) {
        api.addEventListener(event, handler);
        return () => api.removeEventListener(event, handler);
    }

    const teardowns = [
        track("executing", ({ detail }) => {
            // detail: {node, display_node, prompt_id} — node === null means done.
            if (detail?.prompt_id) {
                executingPromptId = detail.node === null ? null : detail.prompt_id;
            }
        }),
        track("progress", ({ detail }) => {
            const job = jobs.get(detail?.prompt_id);
            if (job) job.callbacks.onProgress?.(detail.value, detail.max);
        }),
        track("b_preview", ({ detail }) => {
            // Binary previews carry no prompt id; they belong to whatever is
            // executing. Route them only when that is one of ours.
            const job = jobs.get(executingPromptId);
            if (job) job.callbacks.onPreview?.(detail); // detail is a Blob
        }),
        track("executed", ({ detail }) => {
            const job = jobs.get(detail?.prompt_id);
            if (!job) return;
            const images = detail?.output?.images || [];
            if (images.length) job.results.push(...images);
        }),
        track("execution_success", ({ detail }) => {
            finish(detail?.prompt_id, null);
        }),
        track("execution_error", ({ detail }) => {
            finish(detail?.prompt_id, detail?.exception_message || "execution error");
        }),
        track("execution_interrupted", ({ detail }) => {
            finish(detail?.prompt_id, null, true);
        }),
    ];

    function finish(promptId, error, interrupted = false) {
        const job = jobs.get(promptId);
        if (!job) return;
        jobs.delete(promptId);
        if (error) job.callbacks.onError?.(String(error));
        else if (interrupted) job.callbacks.onCancelled?.();
        else job.callbacks.onDone?.(job.results);
    }

    /**
     * @param {object} graph Patched prompt JSON.
     * @param {object} callbacks {onQueued, onProgress, onPreview, onDone, onError, onCancelled}
     * @returns {Promise<string>} prompt_id
     */
    async function submit(graph, callbacks) {
        const response = await api.fetchApi("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: graph, client_id: api.clientId }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) {
            const nodeErrors = payload.node_errors && Object.keys(payload.node_errors).length
                ? ` (${Object.entries(payload.node_errors).map(([n, e]) =>
                    `${n}: ${e.errors?.[0]?.message || "invalid"}`).join("; ")})`
                : "";
            throw new Error((payload.error?.message || `HTTP ${response.status}`) + nodeErrors);
        }
        const promptId = payload.prompt_id;
        jobs.set(promptId, { callbacks, results: [] });
        callbacks.onQueued?.(promptId);
        return promptId;
    }

    /** Cancel one studio job: dequeue if pending, interrupt only if ours runs. */
    async function cancel(promptId) {
        try {
            await api.fetchApi("/queue", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ delete: [promptId] }),
            });
        } catch (err) {
            console.warn("[TS Studio] queue delete failed", err);
        }
        if (executingPromptId === promptId) {
            await api.fetchApi("/interrupt", { method: "POST" });
        }
    }

    function activeCount() {
        return jobs.size;
    }

    function destroy() {
        for (const teardown of teardowns) teardown();
        jobs.clear();
    }

    return { submit, cancel, activeCount, destroy };
}
