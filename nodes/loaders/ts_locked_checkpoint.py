"""TS Load Checkpoint (locked) — штатный CheckpointLoaderSimple для `.tsmodel`."""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._locked import load, options
from ._mclock import LOG_PREFIX

CATEGORY_FOLDER = "checkpoints"


class TS_LockedCheckpoint(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LockedCheckpoint",
            display_name="TS Load Checkpoint",
            category="TS/Loaders",
            description=(
                "Load a locked .tsmodel checkpoint and get model, CLIP and VAE out of it, "
                "the way CheckpointLoaderSimple does. The architecture is guessed by "
                "ComfyUI itself, from the very same state dict."
            ),
            inputs=[
                IO.Combo.Input(
                    "ckpt_name",
                    options=options(CATEGORY_FOLDER),
                    tooltip="A .tsmodel file from models/checkpoints.",
                ),
            ],
            outputs=[
                IO.Model.Output(display_name="model"),
                IO.Clip.Output(display_name="clip"),
                IO.Vae.Output(display_name="vae"),
            ],
            search_aliases=["locked", "tsmodel", "checkpoint loader"],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> IO.NodeOutput:
        import comfy.sd
        import folder_paths

        state, metadata, path = load(CATEGORY_FOLDER, ckpt_name)
        out = comfy.sd.load_state_dict_guess_config(
            state, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            metadata=metadata,
        )
        if out is None:
            raise RuntimeError(
                f"{LOG_PREFIX} Could not detect the model type of {path}. The file opened "
                "fine, so this is the checkpoint itself, not the lock."
            )
        return IO.NodeOutput(*out[:3])


NODE_CLASS_MAPPINGS = {"TS_LockedCheckpoint": TS_LockedCheckpoint}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedCheckpoint": "TS Load Checkpoint"}
