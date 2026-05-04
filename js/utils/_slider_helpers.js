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

export function tsSnapToStep(tsValue, tsMin, tsStep, tsType) {
    if (!Number.isFinite(tsStep) || tsStep <= 0) return tsValue;
    const tsSteps = Math.round((tsValue - tsMin) / tsStep);
    let tsSnapped = tsMin + tsSteps * tsStep;
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
        tsNode.properties[TS_PROPERTY_STEP] = tsOptions.step ?? (tsType === "int" ? 1 : 0.1);
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
        tsStep = tsNormalizeNumber(tsOptions.step) ?? (tsType === "int" ? 1 : 0.1);
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
    tsOptions.step = tsConfig.step;
    tsOptions.step2 = tsConfig.step;

    if (tsType === "int") {
        tsOptions.precision = 0;
    } else {
        tsOptions.precision = tsCountDecimals(tsConfig.step);
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
    tsValue = tsSnapToStep(tsValue, tsConfig.min, tsConfig.step, tsType);

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
