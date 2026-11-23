# TS Deflicker

Reduces short brightness fluctuations in an ordered IMAGE batch. Works on
decoded video or a sequence of still frames. The node estimates one neutral
brightness correction per frame from small image regions and applies it to that
frame. It preserves alpha and uses no learned weights.

- **segments_info** limits correction to the joins described by TS Latent
  Upscale, including a few adjacent frames where a join can affect decoding.
  Without this input the node searches for brief fluctuations throughout each shot.
- **reference** is a synchronized draft with the same frame count and aspect
  ratio. It can be smaller. The node stabilizes the brightness difference from
  this draft, which retains the intended exposure trajectory. Use this input for
  deliberate flashes or changes in illumination.
- **strength** adjusts the amount of correction. Zero passes the input through.
- **radius** is the number of neighboring frames on each side for short
  fluctuation detection. Guided seam correction uses the supplied window ranges.

For Lipsync H3, connect the upscaled VAE Decode to images, the draft VAE Decode to
reference, and Latent Upscale's segments_info to the corresponding input.
Keep the audio on its existing connection to the video saver. When shooting or
generating without a reference, a brief real flash can resemble an exposure error.
A single frame passes through.
