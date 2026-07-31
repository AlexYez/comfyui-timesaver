import numpy as np
import torch
from comfy_api.v0_0_2 import IO

from .._deps import TSDependencyManager

py360convert = TSDependencyManager.import_optional("py360convert")


class TS_CubemapFacesToEquirectangularNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS Cube to Equirectangular",
            display_name="TS Cube to Equirectangular",
            category="TS/Image",
            description="Stitch six cube faces into one equirectangular 360 panorama.",
            inputs=[
                IO.Image.Input("front", tooltip="Front cube face (+Z)."),
                IO.Image.Input("right", tooltip="Right cube face (+X)."),
                IO.Image.Input("back", tooltip="Back cube face (-Z)."),
                IO.Image.Input("left", tooltip="Left cube face (-X)."),
                IO.Image.Input("top", tooltip="Top cube face (+Y)."),
                IO.Image.Input("bottom", tooltip="Bottom cube face (-Y)."),
                IO.Int.Input("output_width", default=2048, min=64, max=8192, step=64, tooltip="Width of the output equirectangular panorama. Usually twice the height (2:1)."),
                IO.Int.Input("output_height", default=1024, min=32, max=4096, step=32, tooltip="Height of the output equirectangular panorama. Usually half the width (2:1)."),
            ],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @staticmethod
    def tensor_to_image(tensor_in, index):
        # .cpu(): upstream may hand us CUDA tensors; index selects the batch
        # frame (the old squeeze(0) silently broke on batches).
        frame = tensor_in[index].detach().cpu().float().clamp(0.0, 1.0).numpy()
        return (frame * 255).astype(np.uint8)

    @staticmethod
    def image_to_tensor(img_array):
        img_float32 = np.array(img_array).astype(np.float32) / 255.0
        return torch.from_numpy(img_float32)

    @classmethod
    def execute(cls, front, right, back, left, top, bottom, output_width, output_height) -> IO.NodeOutput:
        if py360convert is None:
            raise RuntimeError(
                "[TS Cube to Equirectangular] Missing dependency 'py360convert'. "
                "Install it to enable 360 conversion."
            )

        faces = {'F': front, 'R': right, 'B': back, 'L': left, 'U': top, 'D': bottom}
        batch = min(int(t.shape[0]) for t in faces.values())
        frames = []
        for index in range(batch):
            cubemap_dict = {
                key: cls.tensor_to_image(tensor, index) for key, tensor in faces.items()
            }
            equirectangular_img = py360convert.c2e(
                cubemap_dict, h=output_height, w=output_width, mode='bilinear'
            )
            frames.append(cls.image_to_tensor(equirectangular_img))

        return IO.NodeOutput(torch.stack(frames, dim=0))


NODE_CLASS_MAPPINGS = {
    "TS Cube to Equirectangular": TS_CubemapFacesToEquirectangularNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Cube to Equirectangular": "TS Cube to Equirectangular",
}
