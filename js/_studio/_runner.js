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
        // execution_start reliably carries prompt_id on every engine version.
        track("execution_start", ({ detail }) => {
            if (detail?.prompt_id) executingPromptId = detail.prompt_id;
        }),
        track("executing", ({ detail }) => {
            // detail: {node, display_node, prompt_id} — node === null means done.
            // Some engine versions omit prompt_id here; execution_start above
            // is the authoritative setter, this only clears on completion.
            if (detail?.prompt_id) {
                executingPromptId = detail.node === null ? null : detail.prompt_id;
            }
            // Какой узел сейчас считается. Полоса шагов сэмплера появляется
            // поздно, а до неё идут загрузка весов, текстовый энкодер и VAE —
            // минуты тишины, за которые непонятно, жив ли вообще прогон.
            const job = jobs.get(detail?.prompt_id ?? executingPromptId)
                || (jobs.size === 1 ? jobs.values().next().value : null);
            if (job && detail?.node !== undefined) job.callbacks.onNode?.(detail.node);
        }),
        // Свежие сборки ComfyUI шлют ещё и покомпонентный прогресс: сколько
        // узлов графа пройдено. Это тот самый «общий» ход, которого не хватало.
        track("progress_state", ({ detail }) => {
            const job = jobs.get(detail?.prompt_id ?? executingPromptId);
            if (!job || !detail?.nodes) return;
            const nodes = Object.values(detail.nodes);
            const done = nodes.filter((n) => n?.state === "finished").length;
            job.callbacks.onNodeProgress?.(done, nodes.length);
        }),
        track("progress", ({ detail }) => {
            const job = jobs.get(detail?.prompt_id);
            // Номер узла берётся ИЗ САМОГО прогресса: событие `executing` на
            // части сборок ComfyUI приходит пустым, без node, и определить по
            // нему этап нельзя. А здесь узел есть всегда — заодно это
            // единственный способ отличить шаги сэмплера от тайлов VAE.
            if (job) job.callbacks.onProgress?.(detail.value, detail.max, detail.node);
        }),
        track("b_preview", ({ detail }) => {
            // Binary previews carry no prompt id; they belong to whatever is
            // executing. Route via the tracked id, or — when the engine gave
            // us no id at all — to the single active job (measured: previews
            // arrived but "executing" lacked prompt_id on this build).
            let job = jobs.get(executingPromptId);
            if (!job && jobs.size === 1) job = jobs.values().next().value;
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
     * @param {object} callbacks {onQueued, onProgress, onNode, onNodeProgress,
     *   onPreview, onDone, onError, onCancelled}
     * @param {object} [callbacks.pngInfo] Extra tEXt chunks for saved images,
     *   as {chunkName: value}. The studio has no LiteGraph workflow to send,
     *   so extra_pnginfo would otherwise be absent and a saver would have
     *   nothing to attach the run snapshot to.
     * @returns {Promise<string>} prompt_id
     */
    async function submit(graph, callbacks) {
        const body = { prompt: graph, client_id: api.clientId };
        if (callbacks.pngInfo && Object.keys(callbacks.pngInfo).length) {
            body.extra_data = { extra_pnginfo: { ...callbacks.pngInfo } };
        }
        const response = await api.fetchApi("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
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
            // Выполняющийся прогон прервёт сервер, и `finish` придёт из
            // websocket-события — здесь ждать нечего.
            await api.fetchApi("/interrupt", { method: "POST" });
            return;
        }
        // ⚠️ Отложенный прогон сервер удаляет ТИХО: события `executed` по нему
        // не будет никогда, потому что он и не начинался. Раньше задание так и
        // оставалось в `jobs` — навсегда. `activeCount()` не возвращался к
        // нулю, студия считала, что работа идёт, и следующий запуск упирался в
        // несуществующую очередь. Закрываем задание сами.
        finish(promptId, null, true);
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
