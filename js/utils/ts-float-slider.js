// Slider settings binding for TS_FloatSlider.
//
// Extension ID changed from `ts.slider-settings` (was shared with the int
// slider) to `ts.float-slider` as part of the one-node-one-file split.

import { app } from "/scripts/app.js";

import { tsRegisterSliderExtension } from "./_slider_helpers.js";

tsRegisterSliderExtension(app, {
    extensionId: "ts.float-slider",
    nodeName: "TS_FloatSlider",
    sliderType: "float",
});
