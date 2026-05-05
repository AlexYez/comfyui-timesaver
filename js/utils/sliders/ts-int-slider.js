// Slider settings binding for TS_Int_Slider.
//
// Extension ID changed from `ts.slider-settings` (was shared with the float
// slider) to `ts.int-slider` as part of the one-node-one-file split.

import { app } from "/scripts/app.js";

import { tsRegisterSliderExtension } from "./_slider_helpers.js";

tsRegisterSliderExtension(app, {
    extensionId: "ts.int-slider",
    nodeName: "TS_Int_Slider",
    sliderType: "int",
});
