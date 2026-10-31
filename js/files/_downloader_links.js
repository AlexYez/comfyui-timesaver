// Anonymous link checks; independent of local download status and graph execution.
import { api } from "/scripts/api.js";
import { ensureThemeStyles, pickLocaleStrings } from "../_theme.js";
import { openFullscreenOverlay } from "../_fullscreen.js";
import { parseLine, fileNameOf } from "./_downloader_list.js";

const TEXT = {
    en: {
        button: "Check Links", checking: "Checking…", title: "Check Links — public access",
        hint: "Check every source link without authentication, including files already on disk. No full downloads.",
        note: "Checks run directly from this server, without node tokens, cookies, mirrors or proxy settings. Only the start of each file is read.",
        scope: "Results describe access now from this server; another network or a later full download can differ.",
        empty: "The file list is empty.", close: "Close (Esc)",
        summary: (done, total, available) => `${done}/${total} checked · ${available} available without sign-in`,
        row: (line) => `Line ${line}`, http: (code) => `HTTP ${code}`,
        failed: "The check could not finish. Remaining links were not checked.",
        restart: "The server does not have Check Links yet. Restart ComfyUI after updating the pack.",
        busy: "The previous link check is still finishing. Try again in a few seconds.",
        changed: "The file list changed. Check the updated list again.",
        status: {
            pending: "Waiting", checking: "Checking…", available: "Available without sign-in",
            gated: "Repository requires access approval", auth_required: "Unavailable without sign-in",
            auth_url: "URL contains credentials or a token", forbidden: "Access denied",
            missing: "File or revision not found", webpage: "A webpage, not a direct file",
            pointer: "Git LFS pointer, not the model file",
            temporary: "Temporary signed link", timeout: "Timed out", unreachable: "Network or DNS error",
            tls_error: "TLS certificate error", rate_limited: "Too many requests — try later",
            redirect_error: "Invalid or excessive redirects", blocked: "Non-public or invalid address",
            invalid: "Invalid list entry", invalid_target: "Download folder is not accepted",
            unsafe_file: "Downloader refuses this file type", empty: "Empty response",
            http_error: "HTTP error", unverified: "File access could not be confirmed", not_checked: "Not checked",
        },
    },
    ru: {
        button: "Проверить ссылки", checking: "Проверка…", title: "Check Links — открытый доступ",
        hint: "Проверить все ссылки без авторизации, включая уже скачанные файлы. Без загрузки файлов целиком.",
        note: "Проверка идёт напрямую с этого сервера, без токенов, cookies, зеркал и прокси из настроек ноды. Читается только начало каждого файла.",
        scope: "Результат относится к доступу сейчас с этого сервера; в другой сети или при последующем скачивании условия могут отличаться.",
        empty: "Список файлов пуст.", close: "Закрыть (Esc)",
        summary: (done, total, available) => `Проверено ${done}/${total} · без входа доступны ${available}`,
        row: (line) => `Строка ${line}`, http: (code) => `HTTP ${code}`,
        failed: "Проверку не удалось завершить. Оставшиеся ссылки не проверены.",
        restart: "На сервере ещё нет Check Links. После обновления пака перезапустите ComfyUI.",
        busy: "Предыдущая проверка ещё завершается. Повторите через несколько секунд.",
        changed: "Список файлов изменился. Проверьте обновлённый список ещё раз.",
        status: {
            pending: "Ожидание", checking: "Проверка…", available: "Доступно без входа",
            gated: "Нужно получить доступ к репозиторию", auth_required: "Недоступно без входа",
            auth_url: "В ссылке есть токен или данные входа", forbidden: "Доступ запрещён",
            missing: "Файл или версия не найдены", webpage: "Веб-страница вместо файла",
            pointer: "Git LFS-указатель вместо файла модели",
            temporary: "Временная подписанная ссылка", timeout: "Время ожидания истекло",
            unreachable: "Ошибка сети или DNS", tls_error: "Ошибка TLS-сертификата",
            rate_limited: "Лимит запросов — попробуйте позже", redirect_error: "Ошибка перенаправлений",
            blocked: "Непубличный или некорректный адрес", invalid: "Ошибка в строке списка",
            invalid_target: "Папка загрузки не поддерживается", unsafe_file: "Этот тип файлов загрузчик не принимает",
            empty: "Пустой ответ", http_error: "Ошибка HTTP", unverified: "Доступ к файлу не подтверждён",
            not_checked: "Не проверено",
        },
    },
};

function styles() {
    ensureThemeStyles();
    if (document.getElementById("ts-fdl-link-styles")) return;
    const style = document.createElement("style");
    style.id = "ts-fdl-link-styles";
    style.textContent = `
.ts-fdl-link-report{display:flex;flex-direction:column;gap:12px;width:min(850px,90vw);max-height:78vh;
 padding:18px;border-radius:var(--ts-radius-lg);background:var(--ts-elevated);color:var(--ts-text);
 border:1px solid var(--ts-border)}
.ts-fdl-link-report p{margin:0;font-size:var(--ts-fs-sm);color:var(--ts-muted);line-height:1.5}
.ts-fdl-link-summary{font-size:var(--ts-fs);font-weight:600;font-variant-numeric:tabular-nums}
.ts-fdl-link-rows{min-height:0;overflow:auto;display:flex;flex-direction:column;gap:6px}
.ts-fdl-link-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(160px,1fr);gap:12px;
 padding:10px;border-radius:var(--ts-radius-sm);background:var(--ts-sunken);font-size:var(--ts-fs-sm)}
.ts-fdl-link-name{font-weight:600;overflow-wrap:anywhere}
.ts-fdl-link-host{font-size:var(--ts-fs-xs);color:var(--ts-muted);overflow-wrap:anywhere;margin-top:4px}
.ts-fdl-link-result{color:var(--ts-warning);align-self:center;overflow-wrap:anywhere}
.ts-fdl-link-row[data-status="available"] .ts-fdl-link-result{color:var(--ts-success)}
.ts-fdl-link-row[data-status="missing"] .ts-fdl-link-result,
.ts-fdl-link-row[data-status="webpage"] .ts-fdl-link-result,
.ts-fdl-link-row[data-status="invalid"] .ts-fdl-link-result{color:var(--ts-danger)}
.ts-fdl-link-error{color:var(--ts-warning)!important}
`;
    document.head.appendChild(style);
}

/** Non-serializing button: no changes to schema, widgets or saved file lists. */
export function createLinkCheckButton(getValue) {
    const t = pickLocaleStrings(TEXT);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn ts-fdl-check-links";
    button.textContent = t.button;
    button.title = t.hint;
    let overlay = null;
    let controller = null;
    let disposed = false;
    let generation = 0;

    button.addEventListener("click", async () => {
        if (disposed || button.disabled) return;
        overlay?.close();
        styles();
        const snapshot = String(getValue() ?? "");
        const lines = snapshot.split(/\r?\n/).map((raw, line) => ({ raw, line, ...parseLine(raw) }))
            .filter((entry) => !entry.blank && !entry.comment);
        const panel = document.createElement("div");
        panel.className = "ts-fdl-link-report";
        const title = document.createElement("div");
        title.className = "ts-ui-title";
        title.textContent = t.title;
        const note = document.createElement("p");
        note.textContent = t.note;
        const summary = document.createElement("div");
        summary.className = "ts-fdl-link-summary";
        summary.setAttribute("role", "status");
        const errorNote = document.createElement("p");
        errorNote.className = "ts-fdl-link-error";
        errorNote.hidden = true;
        const rows = document.createElement("div");
        rows.className = "ts-fdl-link-rows";
        const scope = document.createElement("p");
        scope.textContent = t.scope;
        panel.append(title, note, summary, errorNote, rows, scope);
        let done = 0;
        let available = 0;
        const updateSummary = () => { summary.textContent = lines.length ? t.summary(done, lines.length, available) : t.empty; };
        updateSummary();
        for (const entry of lines) {
            const row = document.createElement("div");
            row.className = "ts-fdl-link-row";
            row.dataset.line = String(entry.line);
            row.dataset.status = "pending";
            const main = document.createElement("div");
            const name = document.createElement("div");
            name.className = "ts-fdl-link-name";
            try { name.textContent = fileNameOf(entry.url); } catch { name.textContent = t.row(entry.line + 1); }
            const host = document.createElement("div");
            host.className = "ts-fdl-link-host";
            try { host.textContent = new URL(entry.url).hostname; } catch { host.textContent = t.row(entry.line + 1); }
            const result = document.createElement("div");
            result.className = "ts-fdl-link-result";
            result.textContent = t.status.pending;
            main.append(name, host);
            row.append(main, result);
            rows.appendChild(row);
            entry.row = row;
            entry.result = result;
        }
        const mark = (entry, status, code) => {
            entry.row.dataset.status = status;
            entry.result.textContent = (t.status[status] || t.status.unverified) + (code ? ` · ${t.http(code)}` : "");
        };
        const current = ++generation;
        const abort = new AbortController();
        controller = abort;
        button.disabled = true;
        button.textContent = t.checking;
        overlay = openFullscreenOverlay(panel, {
            center: true, closeOnBackdrop: true, trigger: button, closeTitle: t.close,
            onClose: () => {
                abort.abort();
                if (generation === current) {
                    ++generation;
                    controller = null;
                    overlay = null;
                    button.disabled = false;
                    button.textContent = t.button;
                }
            },
        });
        try {
            // Small batches keep progress visible and bound work after closing.
            for (let offset = 0; offset < lines.length; offset += 4) {
                if (abort.signal.aborted || disposed || generation !== current) break;
                if (String(getValue() ?? "") !== snapshot) throw new Error(t.changed);
                const batch = lines.slice(offset, offset + 4);
                for (const entry of batch) mark(entry, "checking");
                const response = await api.fetchApi("/ts_downloader/check_links", {
                    method: "POST", headers: { "Content-Type": "application/json" }, signal: abort.signal,
                    body: JSON.stringify({ file_list: batch.map((entry) => entry.raw).join("\n") }),
                });
                if (!response.ok) {
                    const reason = [404, 405].includes(response.status) ? t.restart
                        : response.status === 409 ? t.busy : `${t.failed} ${t.http(response.status)}`;
                    throw new Error(reason);
                }
                const payload = await response.json();
                if (abort.signal.aborted || disposed || generation !== current) break;
                if (String(getValue() ?? "") !== snapshot) throw new Error(t.changed);
                if (payload.anonymous !== true || !Array.isArray(payload.entries)) throw new Error(t.failed);
                for (let index = 0; index < batch.length; index++) {
                    const result = payload.entries.find((item) => item.line === index);
                    const status = result?.status === "skip" ? "unverified" : result?.status || "unverified";
                    mark(batch[index], status, result?.http_status);
                    done++;
                    if (status === "available") available++;
                }
                updateSummary();
            }
        } catch (error) {
            if (!abort.signal.aborted && !disposed && generation === current) {
                errorNote.hidden = false;
                errorNote.textContent = error?.message || t.failed;
                for (const entry of lines) if (["pending", "checking"].includes(entry.row.dataset.status)) mark(entry, "not_checked");
            }
        } finally {
            if (!disposed && generation === current) {
                button.disabled = false;
                button.textContent = t.button;
            }
        }
    });
    return {
        button,
        dispose() {
            disposed = true;
            ++generation;
            controller?.abort();
            overlay?.close();
            overlay = null;
        },
    };
}
