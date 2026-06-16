// Shared helpers for TS_FloatSlider / TS_Int_Slider extensions.
// Filename starts with `_` so the loader convention treats it as private,
// even though ComfyUI itself just serves any .js it finds under WEB_DIRECTORY.

export const TS_WIDGET_NAME = "value";
export const TS_PROPERTY_MIN = "min";
export const TS_PROPERTY_MAX = "max";
export const TS_PROPERTY_STEP = "step";
export const TS_PROPERTY_DEFAULT = "default";

export function tsGetWidget(tsNode) {
    return tsNode?.widgets?.find((tsWidget) => tsWidget?.name === TS_WIDGET_NAME) || null;
}

export function tsNormalizeNumber(tsValue) {
    const tsNumber = Number(tsValue);
    return Number.isFinite(tsNumber) ? tsNumber : null;
}

export function tsCountDecimals(tsValue) {
    const tsText = String(tsValue);
    if (tsText.includes("e-")) {
        const tsParts = tsText.split("e-");
        const tsExp = Number(tsParts[1]);
        return Number.isFinite(tsExp) ? tsExp : 0;
    }
    const tsIndex = tsText.indexOf(".");
    return tsIndex >= 0 ? tsText.length - tsIndex - 1 : 0;
}

export function tsReadRealStep(tsOptions, tsType) {
    // ComfyUI's V3 widget factory stores the step TWICE: `options.step2` is the
    // real step from the input spec, while `options.step` is litegraph's
    // deprecated 10x-inflated alias (factory: options.step = realStep * 10).
    // Read step2; fall back to step/10, then to a type default. Reading `step`
    // directly would seed the slider with a 10x step (0.1 -> 1.0, 8 -> 80) — the
    // bug that made a configured 0.1 step collapse to integer/0.01 behaviour.
    const tsStep2 = tsNormalizeNumber(tsOptions?.step2);
    if (tsStep2 !== null && tsStep2 > 0) return tsStep2;
    const tsStep = tsNormalizeNumber(tsOptions?.step);
    if (tsStep !== null && tsStep > 0) return tsStep / 10;
    return tsType === "int" ? 1 : 0.1;
}

export function tsSnapToStep(tsValue, tsMin, tsStep, tsType, tsAnchor) {
    if (!Number.isFinite(tsStep) || tsStep <= 0) return tsValue;
    // Anchor on default (when supplied) instead of min. With min=-1e9 and
    // step=1, snapping relative to min loses sub-step precision in float64
    // (default 0.5 collapses to 1). Snapping relative to default keeps the
    // default exact and lets neighbours grid by `step`. Falls back to min
    // when no usable anchor is provided.
    const anchor = Number.isFinite(tsAnchor) ? tsAnchor : tsMin;
    const tsSteps = Math.round((tsValue - anchor) / tsStep);
    let tsSnapped = anchor + tsSteps * tsStep;
    if (tsType === "int") {
        tsSnapped = Math.round(tsSnapped);
        return tsSnapped;
    }
    const tsDecimals = tsCountDecimals(tsStep);
    if (tsDecimals > 0) {
        tsSnapped = Number(tsSnapped.toFixed(tsDecimals));
    }
    return tsSnapped;
}

export function tsEnsureProperties(tsNode, tsWidget, tsType) {
    if (!tsNode || !tsWidget) return;
    tsNode.properties = tsNode.properties || {};
    const tsOptions = tsWidget.options || {};

    if (tsNode.properties[TS_PROPERTY_MIN] === undefined) {
        tsNode.properties[TS_PROPERTY_MIN] = tsOptions.min ?? (tsType === "int" ? 0 : 0.0);
    }
    if (tsNode.properties[TS_PROPERTY_MAX] === undefined) {
        tsNode.properties[TS_PROPERTY_MAX] = tsOptions.max ?? (tsType === "int" ? 1 : 1.0);
    }
    if (tsNode.properties[TS_PROPERTY_STEP] === undefined) {
        tsNode.properties[TS_PROPERTY_STEP] = tsReadRealStep(tsOptions, tsType);
    }
    if (tsNode.properties[TS_PROPERTY_DEFAULT] === undefined) {
        const tsValue = tsNormalizeNumber(tsWidget.value);
        if (tsValue !== null) {
            tsNode.properties[TS_PROPERTY_DEFAULT] = tsValue;
        } else {
            tsNode.properties[TS_PROPERTY_DEFAULT] = tsType === "int" ? 0 : 0.0;
        }
    }
}

export function tsGetSanitizedConfig(tsNode, tsWidget, tsType) {
    const tsOptions = tsWidget.options || {};
    let tsMin = tsNormalizeNumber(tsNode.properties?.[TS_PROPERTY_MIN]);
    let tsMax = tsNormalizeNumber(tsNode.properties?.[TS_PROPERTY_MAX]);
    let tsStep = tsNormalizeNumber(tsNode.properties?.[TS_PROPERTY_STEP]);
    let tsDefault = tsNormalizeNumber(tsNode.properties?.[TS_PROPERTY_DEFAULT]);

    let tsChanged = false;

    if (!Number.isFinite(tsMin)) {
        tsMin = tsNormalizeNumber(tsOptions.min) ?? (tsType === "int" ? 0 : 0.0);
        tsChanged = true;
    }
    if (!Number.isFinite(tsMax)) {
        tsMax = tsNormalizeNumber(tsOptions.max) ?? (tsType === "int" ? 1 : 1.0);
        tsChanged = true;
    }
    if (!Number.isFinite(tsStep) || tsStep <= 0) {
        tsStep = tsReadRealStep(tsOptions, tsType);
        tsChanged = true;
    }

    if (tsType === "int") {
        tsMin = Math.round(tsMin);
        tsMax = Math.round(tsMax);
        tsStep = Math.max(1, Math.round(tsStep));
    }

    if (tsMin >= tsMax) {
        tsMax = tsMin + tsStep;
        tsChanged = true;
    }

    if (!Number.isFinite(tsDefault)) {
        tsDefault = tsNormalizeNumber(tsWidget.value);
        if (!Number.isFinite(tsDefault)) {
            tsDefault = tsMin;
        }
        tsChanged = true;
    }

    if (tsType === "int") {
        tsDefault = Math.round(tsDefault);
    }

    if (tsDefault < tsMin || tsDefault > tsMax) {
        tsDefault = Math.min(tsMax, Math.max(tsMin, tsDefault));
        tsChanged = true;
    }

    return { min: tsMin, max: tsMax, step: tsStep, default: tsDefault, changed: tsChanged };
}

export function tsApplyConfig(tsNode, tsWidget, tsType, tsConfig, tsForceDefault) {
    if (!tsNode || !tsWidget) return;

    const tsOptions = tsWidget.options || (tsWidget.options = {});
    tsOptions.min = tsConfig.min;
    tsOptions.max = tsConfig.max;
    // `step2` is the real step the modern widget reads; `step` is litegraph's
    // legacy 10x-inflated alias. Keep both in the factory's convention so every
    // code path (legacy canvas + Vue) recovers the same real step — and so
    // tsReadRealStep reads back exactly what we wrote on the next sync.
    tsOptions.step = tsConfig.step * 10;
    tsOptions.step2 = tsConfig.step;

    if (tsType === "int") {
        tsOptions.precision = 0;
    } else {
        // Float slider rounding must track the CONFIGURED step (the `step`
        // property the user edits), NOT the node's fixed schema `round`. Two
        // layers round the live value and both key off these options:
        // litegraph snaps to `options.round` then cleans float noise via
        // toFixed(precision), and the Vue number widget formats to `precision`
        // fraction digits. Deriving both from the step makes the slider grid by
        // exactly the step that was set — a 0.1 step gives 0.1 increments.
        // (Previously precision was max(step, round)-decimals while round stayed
        // pinned to the schema's 0.01, so a user-set 0.1 step still snapped to
        // 0.01.) tsConfig.step is the REAL step (see tsReadRealStep), so its
        // decimal count is correct with no `round` fallback needed.
        tsOptions.precision = tsCountDecimals(tsConfig.step);
        tsOptions.round = tsConfig.step;
    }

    let tsValue = tsNormalizeNumber(tsWidget.value);
    const tsDefault = tsConfig.default;

    if (tsForceDefault) {
        tsValue = tsDefault;
    } else if (tsValue === null) {
        tsValue = tsDefault;
    } else if (tsValue < tsConfig.min || tsValue > tsConfig.max) {
        tsValue = tsDefault;
    }

    if (tsValue === null) {
        tsValue = tsConfig.min;
    }

    tsValue = Math.min(tsConfig.max, Math.max(tsConfig.min, tsValue));
    tsValue = tsSnapToStep(tsValue, tsConfig.min, tsConfig.step, tsType, tsConfig.default);

    if (tsValue !== tsWidget.value) {
        tsWidget.value = tsValue;
        tsWidget.callback?.(tsValue);
    }

    tsNode.setDirtyCanvas?.(true, true);
}

export function tsSyncFromProperties(tsNode, tsType, tsForceDefault = false) {
    const tsWidget = tsGetWidget(tsNode);
    if (!tsWidget) return;

    tsEnsureProperties(tsNode, tsWidget, tsType);

    const tsConfig = tsGetSanitizedConfig(tsNode, tsWidget, tsType);
    if (tsConfig.changed) {
        tsNode.properties[TS_PROPERTY_MIN] = tsConfig.min;
        tsNode.properties[TS_PROPERTY_MAX] = tsConfig.max;
        tsNode.properties[TS_PROPERTY_STEP] = tsConfig.step;
        tsNode.properties[TS_PROPERTY_DEFAULT] = tsConfig.default;
    }

    tsApplyConfig(tsNode, tsWidget, tsType, tsConfig, tsForceDefault);
}

export function tsRegisterSliderExtension(app, { extensionId, nodeName, sliderType }) {
    app.registerExtension({
        name: extensionId,
        async beforeRegisterNodeDef(tsNodeType, tsNodeData) {
            if (tsNodeData?.name !== nodeName) return;

            tsNodeType[`@${TS_PROPERTY_MIN}`] = { type: "number" };
            tsNodeType[`@${TS_PROPERTY_MAX}`] = { type: "number" };
            tsNodeType[`@${TS_PROPERTY_STEP}`] = { type: "number" };
            tsNodeType[`@${TS_PROPERTY_DEFAULT}`] = { type: "number" };

            const tsOnNodeCreated = tsNodeType.prototype.onNodeCreated;
            tsNodeType.prototype.onNodeCreated = function () {
                const tsResult = tsOnNodeCreated ? tsOnNodeCreated.apply(this, arguments) : undefined;
                tsSyncFromProperties(this, sliderType, true);
                return tsResult;
            };

            const tsOnConfigure = tsNodeType.prototype.onConfigure;
            tsNodeType.prototype.onConfigure = function () {
                const tsResult = tsOnConfigure ? tsOnConfigure.apply(this, arguments) : undefined;
                tsSyncFromProperties(this, sliderType, false);
                return tsResult;
            };

            const tsOnPropertyChanged = tsNodeType.prototype.onPropertyChanged;
            tsNodeType.prototype.onPropertyChanged = function (tsPropName) {
                const tsResult = tsOnPropertyChanged ? tsOnPropertyChanged.apply(this, arguments) : undefined;
                if (
                    tsPropName === TS_PROPERTY_MIN ||
                    tsPropName === TS_PROPERTY_MAX ||
                    tsPropName === TS_PROPERTY_STEP
                ) {
                    tsSyncFromProperties(this, sliderType, false);
                } else if (tsPropName === TS_PROPERTY_DEFAULT) {
                    tsSyncFromProperties(this, sliderType, true);
                }
                return tsResult;
            };
        },
        loadedGraphNode(tsNode) {
            if (tsNode?.type !== nodeName) return;
            tsSyncFromProperties(tsNode, sliderType, false);
        },
        nodeCreated(tsNode) {
            if (tsNode?.type !== nodeName) return;
            tsSyncFromProperties(tsNode, sliderType, false);
        },
    });
}
