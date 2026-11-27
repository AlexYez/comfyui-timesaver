# TS Video Seam Repair

Connect frames straight from VAE Decode to **images** and the matching draft before upscale to **reference**. Match frame count, framing and order. Connect the result to Video Saver. Apply before cropping, resizing or further upscaling so decoder coordinates remain valid.

By default the node uses the Fast V2 attention grid. For the classic H3 decoder, connect the same Load VAE output to **vae**. The node then obtains the loaded model’s partition for the actual frame width and height. The hybrid VAE with a Fast V2 decoder also supports this input.

Across sample frames, the node compares candidate horizontal and vertical boundaries with the draft. A sustained extra brightness step across a substantial part of a boundary triggers correction. Weak evidence leaves that boundary alone. No manual selection is needed.

The node checks overlapping 32-pixel-high sections for short vertical seams, including both ends of the frame. A recurring step can trigger repair on a decoder boundary. Between boundaries, strong evidence must persist at the same position in at least three quarters of the sampled frames. The correction follows the measured strength at each height in each frame.

Near the ends of a detected vertical seam, the node compares the step with neighbouring pixels and the aligned draft. This lets the repair reach a horizontal junction even when it lies at the edge of a detection window.

After composing both axes, the node smooths any remaining horizontal step in a narrow band. Its strength depends on how pronounced the original line was against neighbouring rows and the matching draft.

The correction tapers within 32 pixels on either side of the seam and preserves alpha. Process each camera shot separately. A faint line or a step visible in only one frame can fall below the threshold.
