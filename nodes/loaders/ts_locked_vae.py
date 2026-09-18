"""TS Load VAE (locked) — штатный VAELoader для файлов `.tsmodel`."""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._locked import load, options

CATEGORY_FOLDER = "vae"


class TS_LockedVAE(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LockedVAE",
            display_name="TS Load VAE",
            category="TS/Loaders",
            description=(
                "Load a locked .tsmodel VAE. The state dict goes into the same comfy.sd.VAE "
                "the stock loader builds, so anything ComfyUI reads works here."
            ),
            inputs=[
                IO.Combo.Input(
                    "vae_name",
                    options=options(CATEGORY_FOLDER),
                    tooltip="A .tsmodel file from models/vae.",
                ),
            ],
            outputs=[IO.Vae.Output(display_name="vae")],
            search_aliases=["locked", "tsmodel", "vae loader"],
        )

    @classmethod
    def execute(cls, vae_name: str) -> IO.NodeOutput:
        import comfy.sd

        state, metadata, _path = load(CATEGORY_FOLDER, vae_name)
        try:
            vae = comfy.sd.VAE(sd=state, metadata=metadata)
        except TypeError:                   # ComfyUI до появления metadata у VAE
            vae = comfy.sd.VAE(sd=state)
        vae.throw_exception_if_invalid()
        return IO.NodeOutput(vae)


NODE_CLASS_MAPPINGS = {"TS_LockedVAE": TS_LockedVAE}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedVAE": "TS Load VAE"}
