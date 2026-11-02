# Locked .tsmodel files

Three pack loaders read `.tsmodel` diffusion models, checkpoints and model-only LoRAs. They provide separate file selectors and pass the resulting state dictionaries to ComfyUI loaders. [Node catalog](../nodes/README.md).

## Format and protection limits

The reader validates the TSMODEL magic, format version and trailer. The header may be zlib-compressed or stored in raw mode; its length and CRC are checked. Tensor data retains its original offsets.

This is not encryption, a password scheme or DRM. An ordinary safetensors reader cannot interpret this header without an adapter, but someone with the source can implement one or reverse the transformation. Do not promise protection against extracting a loaded model.

## Loading

Architecture and precision support depends on the installed ComfyUI and the dtypes understood by the reader. Delegating to stock loaders does not guarantee support for all future formats. The fast path integrates with ComfyUI's loading machinery; a fallback mmap reader is available. That fallback does not make damaged or unsupported files loadable.

Files are selected from ComfyUI model folders. The `.tsmodel` extension is not added to stock loader file lists. Model conversion belongs to the separate model-converter project; this pack contains reading support, not a general model converter.

Format source: [_mclock.py](../../nodes/loaders/_mclock.py); file discovery: [_locked.py](../../nodes/loaders/_locked.py). [Guide index](README.md).
