# TS DLSS Upscaler

Upscale a picture — or a whole video batch — with **NVIDIA DLSS 5 Neural Rendering**: the same feature games use, running here on your frames. `IMAGE` in, `IMAGE` out, so a single still and a batch of decoded video frames both go straight in.

**It brings its own runtime.** On the first run the node downloads the NVIDIA/ReShade files it needs (~481 MB, once) into `models/DLSS` and lays them out the way the runtime expects — the download has a progress bar, and so does the processing that follows. Nothing is bundled with the pack: those binaries are NVIDIA's and RenoDX's, and their licence texts are saved next to them. A file deleted later is fetched again on the next run; `download_if_missing` turns the fetching off if you would rather place them by hand.

**Five modes, and 1× is not a no-op.** `2× (Performance)` is the default; `1.5×`, `1.724×` and `3×` change how much is invented. `1× (DLAA)` does not resize at all — the network re-renders the picture at its own size, which is the cleanest thing this feature does to footage that is already big enough. The output is capped at 7680×4320, and asking for more names the largest factor that fits instead of failing vaguely.

**Temporal, which is where the quality comes from.** For a batch of consecutive frames the node estimates motion vectors (DIS optical flow) and hands them over, so DLSS carries detail from frame to frame instead of treating each one as a still; a hard cut resets that history rather than smearing across it. **Switch `temporal` off for a batch of unrelated pictures** — otherwise each one drags the previous one's detail behind it.

**The 8-bit pipe is handled, not ignored.** The worker takes 8-bit RGBA, and a ComfyUI IMAGE is float. Rounding straight down turns smooth gradients into steps *before* the network sees them, and the network then sharpens the steps; `dither` (on by default) spends those bits as blue noise instead. `source_curve` is for pictures that are not display-referred SDR — log footage, PQ/HLG, a linear EXR render: they are converted to SDR before the network and converted back after, by the same curve, so the node changes the size and not the colour. Leave it at SDR for ordinary graph output.

**It says whether DLSS really ran.** After the batch the node reads the runtime's own log: when neural rendering silently fell back to a plain resize (an old driver, usually), that is a warning in the console rather than a picture that merely looks disappointing.

**Windows and an NVIDIA RTX card only.** The work is done by NVIDIA's signed D3D12 runtime and there is no other implementation of it; RTX 40/50 are official, RTX 30 works on the experimental path and wants a current driver. Anywhere else the node says so instead of failing obscurely.

> **Licensing (read before the first run).** This pack **hosts and redistributes none of the runtime**, and it is **not affiliated with or endorsed by NVIDIA, ReShade, RenoDX or the upstream project**. What the node downloads on your behalf is a third-party release, and the pieces inside it belong to other people: `nvngx_dlssnr.dll` and `nvngx_dlss.dll` are NVIDIA's, proprietary, under the [NVIDIA RTX SDKs License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt); `dxgi.dll` is ReShade (BSD-3-Clause); `renodx-dlss5.addon64` is the RenoDX add-on under its own terms; `nvngx.dll` is the upstream project's own worker. Their licence texts are written into `models/DLSS` next to the binaries and are meant to stay there.
>
> The `download_if_missing` switch **is your agreement to fetch those components** — the node prints the whole notice, with the source URL and every licence, in the log before it touches the network. **Install only components you are authorised to use, from sources their licences permit.** Turn the switch off and the node downloads nothing: place the files under `models/DLSS/host/` and `models/DLSS/dlss/` yourself.

**Use when:** upscaling footage or stills and you have an RTX card — especially video, where the temporal path beats a still-image upscaler run frame by frame.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
