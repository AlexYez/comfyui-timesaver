# TS Edit Align

Aligns an edited image with its source canvas. Use it for restoration, object removal
or any edit you want to overlay or compare with a wipe.

Connect VAE Decode (or TS VAE Fix if your graph uses it) to `image`, and connect
the source canvas to `source`. The node measures shift and scale along each axis
using matching regions. The status line reports the measured correction.
Alignment corrects geometry. Colors and details remain model-generated.
Without a `source`, the image passes through for text-to-image.

**`output_size`.** The default `source` returns the edit at the source size for
overlaying or blending. `edit` preserves the rendered size and resizes the source
for measurement. `max_shift` limits the search in output pixels and defaults to 24.

**MiniMax H3 / P-H3 Image.** Use the same node as for Qwen Image Edit or Flux
Kontext. If the graph fits the original image to a TS Resolution Selector
canvas, feed the prepared frame that goes into Conditioning's `image_1` into
`source`, for example the output of ResizeAndPadImage. Set `output_size=edit`
to preserve the selected resolution. Disable alignment for intended camera or
composition changes. A full redraw or sparse texture can make the measurement unreliable.
