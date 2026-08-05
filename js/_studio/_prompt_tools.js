// TS Studio kit — the prompt media toolbar (ui-kit layer).
//
// Mounts into the prompt control's toolbar slot and turns the textarea into
// the studio's main instrument (plan §9.1): voice dictation (Whisper),
// attach-image + combined image+text enhance (SuperPrompt's own routes),
// enhance presets, and the style library with removable chips. Everything
// rides EXISTING pack services — this module is a client, not a copy.
//
// Tool visibility degrades gracefully: a missing route greys its button out
// with an explanatory title instead of breaking the field.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";
import { uploadImage, makeDropZone } from "./_dnd.js";

const STYLE_ID = "ts-studio-prompt-tools-styles";

export function ensurePromptToolStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-ptools{display:flex;align-items:center;gap:4px;padding-top:5px;flex-wrap:wrap}
.ts-ptools__btn{width:26px;height:26px;display:flex;align-items:center;justify-content:center;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);background:none;
    color:var(--ts-muted);cursor:pointer;padding:0}
.ts-ptools__btn:hover{color:var(--ts-text);border-color:var(--ts-border-strong)}
.ts-ptools__btn.is-active{color:var(--ts-accent);border-color:var(--ts-accent-line)}
.ts-ptools__btn:disabled{opacity:.4;cursor:default}
.ts-ptools__btn:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-ptools__btn--accent{background:var(--ts-accent-soft);color:var(--ts-accent);border-color:transparent}
.ts-ptools__select{max-width:110px}
.ts-ptools__rec{color:var(--ts-danger);font-size:var(--ts-fs-xs);display:none;align-items:center;gap:4px}
.ts-ptools__rec.is-active{display:inline-flex}
.ts-ptools__recdot{width:7px;height:7px;border-radius:50%;background:var(--ts-danger);
    animation:ts-ptools-pulse 1.1s ease-in-out infinite}
@keyframes ts-ptools-pulse{50%{opacity:.3}}
.ts-ptools__status{flex-basis:100%;font-size:var(--ts-fs-xs);color:var(--ts-muted);min-height:14px}
.ts-ptools__btn.is-drag-over{border-color:var(--ts-accent);
    background:var(--ts-accent-soft);color:var(--ts-accent)}
.ts-ptools__attach{position:absolute;top:6px;right:6px;width:44px;height:44px;border-radius:var(--ts-radius-sm);
    overflow:hidden;border:1px solid var(--ts-border);display:none}
.ts-ptools__attach.is-active{display:block}
.ts-ptools__attach img{width:100%;height:100%;object-fit:cover;display:block}
.ts-ptools__attachx{position:absolute;top:1px;right:1px;width:14px;height:14px;border-radius:50%;
    border:none;background:var(--ts-elevated);color:var(--ts-text);font-size:10px;line-height:1;
    cursor:pointer;padding:0}
.ts-ptools__chips{display:flex;flex-wrap:wrap;gap:4px;padding-top:4px}
.ts-ptools__chip{display:inline-flex;align-items:center;gap:5px;padding:2px 7px 2px 2px;
    border:1px solid var(--ts-border);border-radius:999px;background:var(--ts-sunken);
    color:var(--ts-text);font-size:var(--ts-fs-sm);cursor:pointer}
.ts-ptools__chip:hover{border-color:var(--ts-border-strong)}
.ts-ptools__chip span{color:var(--ts-muted)}
/* Выбранный стиль показывается своей же картинкой — той, что стоит на карточке
   в списке. Название стиля («Cinematic 35mm») говорит меньше, чем кадр в этом
   стиле, а на выбранное смотрят чаще, чем в список. */
.ts-ptools__chipthumb{width:20px;height:20px;border-radius:999px;object-fit:cover;
    flex:0 0 auto;display:block;background:var(--ts-bg)}
.ts-ptools__chip .ts-ptools__stylefallback{width:20px;height:20px;border-radius:999px;
    font-size:var(--ts-fs-xs);flex:0 0 auto;aspect-ratio:auto}
.ts-ptools__popover{position:absolute;z-index:40;left:0;right:0;top:calc(100% + 4px);
    display:none;flex-direction:column;gap:6px;padding:8px;max-height:340px;
    background:var(--ts-elevated);border:1px solid var(--ts-border);
    border-radius:var(--ts-radius);box-shadow:var(--ts-shadow)}
.ts-ptools__popover.is-open{display:flex}
/* auto-rows + start alignment keep every card at its natural height: a plain
   auto row in a scrolling grid squeezed the cards into slivers (measured). */
.ts-ptools__stylegrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;
    grid-auto-rows:min-content;align-content:start;overflow-y:auto;min-height:0;flex:1}
.ts-ptools__style{display:flex;flex-direction:column;align-self:start;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:none;color:var(--ts-text);cursor:pointer;padding:0 0 3px;overflow:hidden;
    font-size:var(--ts-fs-xs)}
.ts-ptools__style img,.ts-ptools__stylefallback{width:100%;aspect-ratio:1;object-fit:cover;
    display:block;flex:0 0 auto}
.ts-ptools__stylefallback{display:flex;align-items:center;justify-content:center;
    background:var(--ts-sunken);color:var(--ts-muted);font-size:var(--ts-fs-lg)}
.ts-ptools__style.is-selected{border-color:var(--ts-accent)}
.ts-ptools__style:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-ptools__stylename{display:block;padding:2px 4px 0;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;flex:0 0 auto}
.ts-ptools__styleempty{grid-column:1/-1;padding:10px;text-align:center;color:var(--ts-muted)}
`;
    document.head.appendChild(style);
}

// Prompt toolbar, same 24 grid as the rest of the studio. The wand's sparkles
// are drawn from one symmetric four-point shape at two sizes, so neither leans.
const TOOL_ICON_ATTRS = 'viewBox="0 0 24 24" width="14" height="14" fill="none" '
    + 'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
    + 'stroke-linejoin="round"';

const SVG = {
    mic: `<svg ${TOOL_ICON_ATTRS}><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/></svg>`,
    image: `<svg ${TOOL_ICON_ATTRS}><rect x="3" y="5" width="18" height="14" rx="2.2"/><circle cx="8.6" cy="9.8" r="1.5"/><path d="M4 17l4.8-4.8 3.4 3.4 3-3L21 16.6"/></svg>`,
    palette: `<svg ${TOOL_ICON_ATTRS}><path d="M12 3a9 9 0 1 0 0 18h1a2 2 0 0 0 0-4h-1a2 2 0 0 1 0-4h5a4 4 0 0 0 4-4c0-3.5-4-6-9-6z"/><circle cx="7.6" cy="11.2" r="1"/><circle cx="9.6" cy="7.4" r="1"/><circle cx="13.8" cy="6.6" r="1"/></svg>`,
    wand: `<svg ${TOOL_ICON_ATTRS}><path d="M4.5 19.5L13.5 10.5"/><path d="M17 3l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z"/><path d="M20 13l.6 1.4 1.4.6-1.4.6-.6 1.4-.6-1.4-1.4-.6 1.4-.6z"/></svg>`,
};

function insertAtCursor(textarea, text) {
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? start;
    const before = textarea.value.slice(0, start);
    const after = textarea.value.slice(end);
    const glue = before && !/\s$/.test(before) ? " " : "";
    textarea.value = `${before}${glue}${text}${after}`;
    const cursor = (before + glue + text).length;
    textarea.setSelectionRange(cursor, cursor);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
}

/**
 * @param {object} options
 * @param {HTMLTextAreaElement} options.textarea The prompt field.
 * @param {HTMLElement} options.slot Toolbar slot inside the prompt control.
 * @param {object} options.api ComfyUI api client.
 * @param {object} options.objectInfo For the enhance preset list.
 * @param {object} options.t Locale strings (studio dictionary).
 * @param {string} options.locale "en" | "ru".
 * @param {object[]} [options.initialStyles] Styles to start selected.
 * @returns {{getStylePrompts: () => string[], teardown: () => void}}
 */
export function mountPromptTools(options) {
    ensurePromptToolStyles();
    const { textarea, slot, api, objectInfo, t, locale } = options;
    const wrap = textarea.closest(".ts-studio__prompt") || slot.parentElement;
    wrap.style.position = "relative";

    const bar = document.createElement("div");
    bar.className = `${TS_UI_CLASS} ts-ptools`;
    slot.appendChild(bar);

    const status = document.createElement("div");
    status.className = "ts-ptools__status";
    const setStatus = (text) => { status.textContent = text || ""; };

    const teardowns = [];
    // initialStyles carries the selection across a deck rebuild: switching
    // models must not silently drop the styles someone picked.
    const state = { attached: "", styles: [...(options.initialStyles || [])] };

    // ── voice dictation ─────────────────────────────────────────────────── //
    const micButton = toolButton(SVG.mic, t.pt.mic);
    const recBadge = document.createElement("span");
    recBadge.className = "ts-ptools__rec";
    const recDot = document.createElement("span");
    recDot.className = "ts-ptools__recdot";
    const recTime = document.createElement("span");
    recBadge.append(recDot, recTime);
    const hqChip = document.createElement("button");
    hqChip.type = "button";
    hqChip.className = "ts-ptools__btn";
    hqChip.style.width = "auto";
    hqChip.style.padding = "0 6px";
    hqChip.textContent = "HQ";
    hqChip.title = t.pt.hq;

    let recorder = null;
    let recChunks = [];
    let recTimer = 0;
    let recStarted = 0;

    async function toggleRecording() {
        if (recorder) {
            recorder.stop();
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recorder = new MediaRecorder(stream);
            recChunks = [];
            recorder.ondataavailable = (event) => event.data.size && recChunks.push(event.data);
            recorder.onstop = async () => {
                clearInterval(recTimer);
                recBadge.classList.remove("is-active");
                micButton.classList.remove("is-active");
                stream.getTracks().forEach((track) => track.stop());
                const blob = new Blob(recChunks, { type: recorder.mimeType || "audio/webm" });
                recorder = null;
                await transcribe(blob);
            };
            recorder.start();
            recStarted = Date.now();
            recTime.textContent = "0:00";
            recBadge.classList.add("is-active");
            micButton.classList.add("is-active");
            recTimer = setInterval(() => {
                const seconds = Math.floor((Date.now() - recStarted) / 1000);
                recTime.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
            }, 500);
        } catch (err) {
            setStatus(t.pt.micDenied);
            console.warn("[TS Studio] microphone", err);
        }
    }

    async function transcribe(blob) {
        setStatus(t.pt.transcribing);
        try {
            const form = new FormData();
            form.append("audio", blob, "dictation.webm");
            if (hqChip.classList.contains("is-active")) form.append("high_quality", "true");
            const response = await api.fetchApi("/ts_voice_recognition/transcribe",
                { method: "POST", body: form });
            const payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
            const text = String(payload.text ?? payload.transcription ?? "").trim();
            if (text) insertAtCursor(textarea, text);
            setStatus("");
        } catch (err) {
            setStatus(t.pt.opFailed(err.message));
        }
    }

    micButton.addEventListener("click", toggleRecording);
    hqChip.addEventListener("click", () => hqChip.classList.toggle("is-active"));
    const onVoiceStatus = ({ detail }) => {
        if (detail?.text) setStatus(`${detail.text}${detail.percent ? ` ${Math.round(detail.percent)}%` : ""}`);
        if (detail?.percent >= 100) setStatus("");
    };
    api.addEventListener("ts_voice_recognition.status", onVoiceStatus);
    teardowns.push(() => api.removeEventListener("ts_voice_recognition.status", onVoiceStatus));

    // ── attach image (drop / paste / pick) ──────────────────────────────── //
    const attachButton = toolButton(SVG.image, t.pt.attach);
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.className = "ts-ui-file";
    document.body.appendChild(fileInput);
    teardowns.push(() => fileInput.remove());

    const attachPreview = document.createElement("div");
    attachPreview.className = "ts-ptools__attach";
    const attachImg = document.createElement("img");
    attachImg.alt = "";
    const attachX = document.createElement("button");
    attachX.type = "button";
    attachX.className = "ts-ptools__attachx";
    attachX.textContent = "×";
    attachX.title = t.pt.detach;
    attachPreview.append(attachImg, attachX);
    wrap.appendChild(attachPreview);

    async function attachBlob(blob, name) {
        try {
            setStatus(t.pt.uploading);
            state.attached = await uploadImage(api, blob, name || "studio_ref.png");
            attachImg.src = URL.createObjectURL(blob);
            attachPreview.classList.add("is-active");
            attachButton.classList.add("is-active");
            setStatus("");
        } catch (err) {
            setStatus(t.pt.opFailed(err.message));
        }
    }

    attachButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) attachBlob(file, file.name);
        fileInput.value = "";
    });
    attachX.addEventListener("click", () => {
        state.attached = "";
        attachPreview.classList.remove("is-active");
        attachButton.classList.remove("is-active");
    });

    const onDrop = (event) => {
        const file = [...(event.dataTransfer?.files || [])].find((f) => f.type.startsWith("image/"));
        if (!file) return;
        event.preventDefault();
        event.stopPropagation();
        attachBlob(file, file.name);
    };
    const onDragOver = (event) => {
        if ([...(event.dataTransfer?.types || [])].includes("Files")) event.preventDefault();
    };
    const onPaste = (event) => {
        const item = [...(event.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
        if (!item) return;
        event.preventDefault();
        attachBlob(item.getAsFile(), "pasted.png");
    };
    textarea.addEventListener("drop", onDrop);
    textarea.addEventListener("dragover", onDragOver);
    textarea.addEventListener("paste", onPaste);
    // The button that attaches a picture is also the place to drop one — and
    // through the shared drop service, so a card dragged out of the asset
    // browser works exactly like a file from the desktop.
    teardowns.push(makeDropZone(attachButton, {
        max: 1,
        onDrop: async ([item]) => attachBlob(await item.getBlob(), item.name),
    }));
    teardowns.push(() => {
        textarea.removeEventListener("drop", onDrop);
        textarea.removeEventListener("dragover", onDragOver);
        textarea.removeEventListener("paste", onPaste);
    });

    // ── enhance preset + AI button ──────────────────────────────────────── //
    const presetSelect = document.createElement("select");
    presetSelect.className = "ts-ui-select ts-ptools__select";
    presetSelect.title = t.pt.preset;
    const presetSpec = objectInfo?.TS_SuperPrompt?.input?.required?.system_preset
        ?? objectInfo?.TS_SuperPrompt?.input?.optional?.system_preset;
    // V1 nodes serialise a combo as [options, meta]; V3 as ["COMBO",
    // {options}]. Read both shapes.
    const presets = Array.isArray(presetSpec?.[0]) ? presetSpec[0]
        : Array.isArray(presetSpec?.[1]?.options) ? presetSpec[1].options : [];
    const defaultPreset = presetSpec?.[1]?.default;
    for (const preset of presets) {
        const option = document.createElement("option");
        option.value = preset;
        option.textContent = preset;
        option.selected = preset === defaultPreset;
        presetSelect.appendChild(option);
    }

    const aiButton = toolButton(SVG.wand, t.pt.enhance);
    aiButton.classList.add("ts-ptools__btn--accent");
    let enhanceSeed = 0;
    aiButton.addEventListener("click", async () => {
        aiButton.disabled = true;
        setStatus(state.attached ? t.pt.enhancingImage : t.pt.enhancing);
        try {
            enhanceSeed += 1;
            const response = await api.fetchApi("/ts_super_prompt/enhance", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: textarea.value,
                    system_preset: presetSelect.value || undefined,
                    attached_image: state.attached || undefined,
                    seed: enhanceSeed,
                }),
            });
            const payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
            textarea.value = String(payload.text || "").trim();
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
            setStatus("");
        } catch (err) {
            setStatus(t.pt.opFailed(err.message));
        } finally {
            aiButton.disabled = false;
        }
    });

    if (!presets.length) {
        aiButton.disabled = true;
        aiButton.title = t.pt.noSuperPrompt;
        presetSelect.disabled = true;
    }

    // ── styles ──────────────────────────────────────────────────────────── //
    const styleButton = toolButton(SVG.palette, t.pt.styles);
    const popover = document.createElement("div");
    popover.className = "ts-ptools__popover";
    const styleSearch = document.createElement("input");
    styleSearch.type = "text";
    styleSearch.className = "ts-ui-input";
    styleSearch.placeholder = t.pt.styleSearch;
    const styleGrid = document.createElement("div");
    styleGrid.className = "ts-ptools__stylegrid";
    popover.append(styleSearch, styleGrid);
    wrap.appendChild(popover);

    const chips = document.createElement("div");
    chips.className = "ts-ptools__chips";

    let allStyles = [];
    let stylesLoaded = false;

    async function loadStyles() {
        if (stylesLoaded) return;
        try {
            const response = await api.fetchApi("/ts_styles");
            const payload = await response.json();
            allStyles = (payload.styles || []).filter((s) => s.prompt);
            stylesLoaded = true;
            renderStyleGrid("");
        } catch (err) {
            setStatus(t.pt.opFailed(err.message));
        }
    }

    function styleName(style) {
        return (locale === "ru" && style.name_ru) ? style.name_ru : style.name || style.id;
    }

    function renderStyleGrid(query) {
        const needle = query.trim().toLowerCase();
        styleGrid.textContent = "";
        let shown = 0;
        for (const style of allStyles) {
            const label = styleName(style);
            if (needle && !`${label} ${style.name} ${style.id}`.toLowerCase().includes(needle)) continue;
            shown += 1;
            const card = document.createElement("button");
            card.type = "button";
            card.className = "ts-ptools__style";
            card.classList.toggle("is-selected", state.styles.some((s) => s.id === style.id));
            if (style.preview) {
                const img = document.createElement("img");
                img.loading = "lazy";
                img.alt = label;
                img.src = `/ts_styles/preview?path=${encodeURIComponent(style.preview)}`;
                // A missing preview must not collapse the card: swap in the
                // same-sized initial tile so the grid stays even.
                img.addEventListener("error", () => img.replaceWith(fallbackTile(label)));
                card.appendChild(img);
            } else {
                card.appendChild(fallbackTile(label));
            }
            const name = document.createElement("span");
            name.className = "ts-ptools__stylename";
            name.textContent = label;
            name.title = label;
            card.appendChild(name);
            card.addEventListener("click", () => {
                const index = state.styles.findIndex((s) => s.id === style.id);
                if (index >= 0) state.styles.splice(index, 1);
                else state.styles.push(style);
                renderChips();
                renderStyleGrid(styleSearch.value);
            });
            styleGrid.appendChild(card);
        }
        if (!shown) {
            const note = document.createElement("div");
            note.className = "ts-ptools__styleempty";
            note.textContent = stylesLoaded ? t.pt.stylesEmpty : t.pt.stylesLoading;
            styleGrid.appendChild(note);
        }
    }

    function fallbackTile(label) {
        const tile = document.createElement("span");
        tile.className = "ts-ptools__stylefallback";
        tile.textContent = (label || "?").trim().charAt(0).toUpperCase();
        return tile;
    }

    function renderChips() {
        chips.textContent = "";
        for (const style of state.styles) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "ts-ptools__chip";
            chip.title = t.pt.removeStyle;
            const x = document.createElement("span");
            x.textContent = "×";
            if (style.preview) {
                const thumb = document.createElement("img");
                thumb.className = "ts-ptools__chipthumb";
                thumb.alt = "";
                thumb.loading = "lazy";
                thumb.src = `/ts_styles/preview?path=${encodeURIComponent(style.preview)}`;
                // Пропавшее превью не должно оставлять дыру в ярлычке.
                thumb.addEventListener("error", () => {
                    thumb.replaceWith(fallbackTile(styleName(style)));
                }, { once: true });
                chip.appendChild(thumb);
            } else {
                chip.appendChild(fallbackTile(styleName(style)));
            }
            chip.append(document.createTextNode(styleName(style)), x);
            chip.addEventListener("click", () => {
                state.styles = state.styles.filter((s) => s.id !== style.id);
                renderChips();
                if (popover.classList.contains("is-open")) renderStyleGrid(styleSearch.value);
            });
            chips.appendChild(chip);
        }
    }

    styleButton.addEventListener("click", () => {
        const open = !popover.classList.contains("is-open");
        popover.classList.toggle("is-open", open);
        styleButton.classList.toggle("is-active", open);
        if (open) {
            renderStyleGrid(styleSearch.value);   // shows the loading note first
            loadStyles();
            styleSearch.focus();
        }
    });
    styleSearch.addEventListener("input", () => renderStyleGrid(styleSearch.value));
    const onDocDown = (event) => {
        if (!popover.contains(event.target) && event.target !== styleButton
            && !styleButton.contains(event.target)) {
            popover.classList.remove("is-open");
            styleButton.classList.remove("is-active");
        }
    };
    document.addEventListener("pointerdown", onDocDown);
    teardowns.push(() => document.removeEventListener("pointerdown", onDocDown));

    // ── assemble ────────────────────────────────────────────────────────── //
    bar.append(micButton, recBadge, hqChip, attachButton, presetSelect, styleButton, aiButton);
    slot.append(status, chips);

    function toolButton(svg, title) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-ptools__btn";
        button.title = title;
        button.setAttribute("aria-label", title);
        button.innerHTML = svg;
        return button;
    }

    renderChips();

    return {
        getStylePrompts: () => state.styles.map((s) => String(s.prompt || "").trim()).filter(Boolean),
        getStyleNames: () => state.styles.map((s) => styleName(s)),
        // The full style objects, so a rebuilt toolbar can restore them.
        getSelectedStyles: () => state.styles.map((s) => ({ ...s })),
        teardown: () => teardowns.forEach((fn) => fn()),
    };
}
