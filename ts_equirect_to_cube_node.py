# Third-party imports
import numpy as np
import py360convert
import torch

class TS_EquirectangularToCubemapFacesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cube_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}), # Increased max
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "right", "back", "left", "top", "bottom")
    FUNCTION = "convert"
    CATEGORY = "Tools/TS_Image"

    def convert(self, image, cube_size):
        image_np = image.squeeze(0).numpy()
        image_uint8 = (image_np * 255).astype(np.uint8)
        # py360convert.e2c expects face_w as the width of each cube face
        cubemap_dict = py360convert.e2c(image_uint8, face_w=cube_size, cube_format='dict')
        
        front = self.image_to_tensor(cubemap_dict['F'])
        right = self.image_to_tensor(cubemap_dict['R'])
        back = self.image_to_tensor(cubemap_dict['B'])
        left = self.image_to_tensor(cubemap_dict['L'])
        top = self.image_to_tensor(cubemap_dict['U'])
        bottom = self.image_to_tensor(cubemap_dict['D'])
        return (front, right, back, left, top, bottom)

    def image_to_tensor(self, img_array):
        img_float32 = np.array(img_array).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float32).unsqueeze(0)
        return tensor

NODE_CLASS_MAPPINGS = {
    "TS Equirectangular to Cube": TS_EquirectangularToCubemapFacesNode # Ключ оставлен оригинальным
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Equirectangular to Cube": "TS Equirectangular to Cube"
}