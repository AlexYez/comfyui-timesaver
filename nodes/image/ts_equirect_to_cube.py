import numpy as np
import torch

from comfy_api.latest import IO

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
        tensor = torch.from_numpy(img_float32).unsqueeze(0)
        return tensor

    @classmethod
    def execute(cls, image, cube_size) -> IO.NodeOutput:
        if py360convert is None:
            raise RuntimeError(
                "[TS Equirectangular to Cube] Missing dependency 'py360convert'. "
                "Install it to enable 360 conversion."
            )

        image_np = image.squeeze(0).numpy()
        image_uint8 = (image_np * 255).astype(np.uint8)
        cubemap_dict = py360convert.e2c(image_uint8, face_w=cube_size, cube_format='dict')

        front = cls.image_to_tensor(cubemap_dict['F'])
        right = cls.image_to_tensor(cubemap_dict['R'])
        back = cls.image_to_tensor(cubemap_dict['B'])
        left = cls.image_to_tensor(cubemap_dict['L'])
        top = cls.image_to_tensor(cubemap_dict['U'])
        bottom = cls.image_to_tensor(cubemap_dict['D'])
        return IO.NodeOutput(front, right, back, left, top, bottom)


NODE_CLASS_MAPPINGS = {
    "TS Equirectangular to Cube": TS_EquirectangularToCubemapFacesNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Equirectangular to Cube": "TS Equirectangular to Cube",
}
