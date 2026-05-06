import numpy as np
import torch

from comfy_api.v0_0_2 import IO

from ...ts_dependency_manager import TSDependencyManager

py360convert = TSDependencyManager.import_optional("py360convert")


class TS_CubemapFacesToEquirectangularNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS Cube to Equirectangular",
            display_name="TS Cube to Equirectangular",
            category="TS/Image",
            inputs=[
                IO.Image.Input("front"),
                IO.Image.Input("right"),
                IO.Image.Input("back"),
                IO.Image.Input("left"),
                IO.Image.Input("top"),
                IO.Image.Input("bottom"),
                IO.Int.Input("output_width", default=2048, min=64, max=8192, step=64),
                IO.Int.Input("output_height", default=1024, min=32, max=4096, step=32),
            ],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @staticmethod
    def tensor_to_image(tensor_in):
        img_np = tensor_in.squeeze(0).numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        return img_uint8

    @staticmethod
    def image_to_tensor(img_array):
        img_float32 = np.array(img_array).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float32).unsqueeze(0)
        return tensor

    @classmethod
    def execute(cls, front, right, back, left, top, bottom, output_width, output_height) -> IO.NodeOutput:
        if py360convert is None:
            raise RuntimeError(
                "[TS Cube to Equirectangular] Missing dependency 'py360convert'. "
                "Install it to enable 360 conversion."
            )

        cubemap_dict = {
            'F': cls.tensor_to_image(front), 'R': cls.tensor_to_image(right),
            'B': cls.tensor_to_image(back), 'L': cls.tensor_to_image(left),
            'U': cls.tensor_to_image(top), 'D': cls.tensor_to_image(bottom),
        }

        equirectangular_img = py360convert.c2e(cubemap_dict, h=output_height, w=output_width, mode='bilinear')

        equirectangular_tensor = cls.image_to_tensor(equirectangular_img)
        return IO.NodeOutput(equirectangular_tensor)


NODE_CLASS_MAPPINGS = {
    "TS Cube to Equirectangular": TS_CubemapFacesToEquirectangularNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Cube to Equirectangular": "TS Cube to Equirectangular",
}
