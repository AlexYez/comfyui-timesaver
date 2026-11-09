// Working slider range lives in workflow properties, separate from server bounds.
export const MAX_MEGAPIXELS_PROPERTY = "max_megapixels";
export const DEFAULT_MAX_MEGAPIXELS = 4;
export const MIN_MEGAPIXELS = 0.1;
// Same technical FLOAT ceiling as TS Float Slider; this is not the UI default.
export const HARD_MAX_MEGAPIXELS = 10000;

function finiteNumber(value) {
    if (typeof value !== "number" && typeof value !== "string") return null;
    if (typeof value === "string" && !value.trim()) return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

export function normalizedMaximum(value) {
    const number = finiteNumber(value);
    if (number === null || number < MIN_MEGAPIXELS) return DEFAULT_MAX_MEGAPIXELS;
    return Math.min(HARD_MAX_MEGAPIXELS, number);
}

export function applyResolutionMaximum(node, widget, { preserveValue = false } = {}) {
    if (!widget) return;
    node.properties ||= {};
    let maximum = normalizedMaximum(node.properties[MAX_MEGAPIXELS_PROPERTY]);
    const current = finiteNumber(widget.value);
    // A saved graph is authoritative: opening it must not reset an old render size.
    if (preserveValue && current !== null && current >= MIN_MEGAPIXELS && current <= HARD_MAX_MEGAPIXELS) {
        maximum = Math.max(maximum, current);
    }
    node.properties[MAX_MEGAPIXELS_PROPERTY] = maximum;
    // Widget options may have originated in shared node metadata. Keep instances independent.
    widget.options = { ...widget.options, min: MIN_MEGAPIXELS, max: maximum };
    if (!preserveValue && current !== null && current > maximum) {
        widget.value = maximum;
        widget.callback?.(maximum);
    }
    node.setDirtyCanvas?.(true, true);
}

export function registerResolutionMaximum(nodeType, getResolutionWidget) {
    nodeType[`@${MAX_MEGAPIXELS_PROPERTY}`] = { type: "number" };
    const proto = nodeType.prototype;
    const originalConfigure = proto.configure;
    proto.configure = function (info) {
        const wasConfiguring = this._tsResolutionConfiguring;
        this._tsResolutionConfiguring = true;
        const widget = getResolutionWidget(this);
        // Vue/native numeric widgets can clamp during restore, before onConfigure.
        // Temporarily allow the server range; shared hidden-widget migration then
        // restores the saved values, and only afterwards do we choose a UI range.
        if (widget) widget.options = { ...widget.options, max: HARD_MAX_MEGAPIXELS };
        try {
            return originalConfigure?.apply(this, arguments);
        } finally {
            this._tsResolutionConfiguring = wasConfiguring;
            this.properties ||= {};
            if (Object.hasOwn(info?.properties || {}, MAX_MEGAPIXELS_PROPERTY)) {
                this.properties[MAX_MEGAPIXELS_PROPERTY] = info.properties[MAX_MEGAPIXELS_PROPERTY];
            } else {
                delete this.properties[MAX_MEGAPIXELS_PROPERTY];
            }
            applyResolutionMaximum(this, getResolutionWidget(this), { preserveValue: true });
        }
    };

    const originalPropertyChanged = proto.onPropertyChanged;
    proto.onPropertyChanged = function (name, value) {
        const result = originalPropertyChanged?.apply(this, arguments);
        if (name === MAX_MEGAPIXELS_PROPERTY && !this._tsResolutionConfiguring) {
            this.properties ||= {};
            this.properties[MAX_MEGAPIXELS_PROPERTY] = value;
            applyResolutionMaximum(this, getResolutionWidget(this));
        }
        return result;
    };
}
