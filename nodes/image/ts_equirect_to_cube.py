import numpy as np
import torch

from comfy_api.v0_0_2 import IO

from ...ts_dependency_manager import TSDependencyManager

py360convert = TSDependencyManager.import_optional("py360convert")


class TS_EquirectangularToCubemapFacesNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS Equirectangular to Cube",
            display_name="TS Equirectangular to Cube",
            category="TS/Image",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("cube_size", default=512, min=64, max=4096, step=64),
            ],
            outputs=[
                IO.Image.Output(display_name="front"),
                IO.Image.Output(display_name="right"),
                IO.Image.Output(display_name="back"),
                IO.Image.Output(display_name="left"),
                IO.Image.Output(display_name="top"),
                IO.Image.Output(display_name="bottom"),
            ],
        )

    @staticmethod
    def image_to_tensor(img_array):
        img_float32 = np.array(img_array).astype(np.float32) / 255.0
        return torch.from_numpy(img_float32)

    @classmethod
    def execute(cls, image, cube_size) -> IO.NodeOutput:
        if py360convert is None:
            raise RuntimeError(
                "[TS Equirectangular to Cube] Missing dependency 'py360convert'. "
                "Install it to enable 360 conversion."
            )

        # .cpu(): the upstream node may hand us a CUDA tensor; .float(): keep
        # the uint8 math safe for fp16 inputs. Process every frame — the old
        # squeeze(0) silently broke on batches.
        batch_np = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
        faces: dict[str, list[torch.Tensor]] = {k: [] for k in "FRBLUD"}
        for frame_np in batch_np:
            frame_uint8 = (frame_np * 255).astype(np.uint8)
            cubemap_dict = py360convert.e2c(frame_uint8, face_w=cube_size, cube_format='dict')
            for key in faces:
                faces[key].append(cls.image_to_tensor(cubemap_dict[key]))

        stacked = {key: torch.stack(tensors, dim=0) for key, tensors in faces.items()}
        return IO.NodeOutput(
            stacked['F'], stacked['R'], stacked['B'],
            stacked['L'], stacked['U'], stacked['D'],
        )


NODE_CLASS_MAPPINGS = {
    "TS Equirectangular to Cube": TS_EquirectangularToCubemapFacesNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Equirectangular to Cube": "TS Equirectangular to Cube",
}
