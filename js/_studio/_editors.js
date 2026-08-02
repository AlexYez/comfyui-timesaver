// TS Studio kit — external editor registry (ui-kit layer).
//
// Some families already have a full authoring interface shipped with their
// node. The studio does NOT reimplement those: it opens the node's own editor
// and stores whatever state that editor produces. When the node improves, the
// studio inherits the improvement — nothing here to keep in sync.
//
// A provider is {id, open(options) -> Promise<state|null>, prepare?}. The
// manifest names one through a "designer" control, so adding a family with an
// editor is a manifest line plus one provider.

const PROVIDERS = new Map();

/** Extension point: register an editor provider by id. */
export function registerEditorProvider(provider) {
    PROVIDERS.set(provider.id, provider);
}

export function getEditorProvider(id) {
    return PROVIDERS.get(id) || null;
}

// ── Ideogram Designer (TS_IdeogramDesigner's own editor) ────────────────── //
//
// Its caption builder, its aspect handling, its Generate-Prompt button (which
// calls the pack's SuperPrompt engine) — all reused as-is. The studio only
// hands it the current design and takes the saved one back.
registerEditorProvider({
    id: "ideogram",
    label: { en: "Ideogram Designer", ru: "Дизайнер Ideogram" },

    async open({ design, prompt, aspect, megapixels }) {
        const [{ openIdeogramEditor }, shared, { markOverlayAbove }] = await Promise.all([
            import("../ideogram/_ideogram_editor.js"),
            import("../ideogram/_ideogram_shared.js"),
            import("../_fullscreen.js"),
        ]);
        const presets = await shared.loadPresets();
        const start = seedDesign(design, prompt, aspect, megapixels, shared);
        return new Promise((resolve) => {
            let saved = null;
            let release = () => {};
            const settle = () => { release(); resolve(saved); };
            openIdeogramEditor(null, {
                design: start,
                presets,
                graphRef: null,
                // The editor commits on close, so this fires exactly once.
                onSave: (next) => { saved = next; },
            });
            // It mounts its own fullscreen overlay at the same depth as the
            // studio's; lift it so it paints above and owns Escape.
            const overlay = document.querySelector(".ts-ideoe-overlay");
            if (overlay) release = markOverlayAbove(overlay);
            // The editor has no "closed" callback, so settle when its overlay
            // leaves the document.
            const timer = setInterval(() => {
                if (document.querySelector(".ts-ideoe-overlay")) return;
                clearInterval(timer);
                settle();
            }, 200);
        });
    },
});

/**
 * The studio's own format picker stays authoritative: whatever aspect and
 * megapixels the deck shows are written into the design before the editor
 * opens, so the two never disagree about the frame.
 */
function seedDesign(design, prompt, aspect, megapixels, shared) {
    const base = design && typeof design === "object" ? { ...design } : {};
    if (aspect) {
        const token = String(aspect).replace(":", "x");
        if (shared.ASPECT_RATIOS.includes(token)) base.aspect_ratio = token;
    }
    if (Number.isFinite(Number(megapixels))) base.megapixels = Number(megapixels);
    if (prompt && !String(base.high_level_description || "").trim()) {
        base.high_level_description = String(prompt);
    }
    base.blocks = Array.isArray(base.blocks) ? base.blocks : [];
    return base;
}

/** Apply the studio's frame choice to a design the editor returned. */
export function applyFrameToDesign(design, aspect, megapixels) {
    const next = design && typeof design === "object" ? { ...design } : {};
    if (aspect) next.aspect_ratio = String(aspect).replace(":", "x");
    if (Number.isFinite(Number(megapixels))) next.megapixels = Number(megapixels);
    return next;
}
