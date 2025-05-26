# Third-party imports
import numpy as np
import py360convert
import torch

class TS_CubemapFacesToEquirectangularNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("IMAGE",), "right": ("IMAGE",),
                "back": ("IMAGE",), "left": ("IMAGE",),
                "top": ("IMAGE",), "bottom": ("IMAGE",),
                "output_width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "output_height": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "Tools/TS_Image"

    def convert(self, front, right, back, left, top, bottom, output_width, output_height):
        cubemap_dict = {
            'F': self.tensor_to_image(front), 'R': self.tensor_to_image(right),
            'B': self.tensor_to_image(back), 'L': self.tensor_to_image(left),
            'U': self.tensor_to_image(top), 'D': self.tensor_to_image(bottom)
        }
        # Assuming all faces are square and have the same dimensions as 'front'
        # face_h, face_w, _ = cubemap_dict['F'].shape 
        
        # py360convert.c2e takes the cubemap dictionary, height, and width of the output equirectangular
        equirectangular_img = py360convert.c2e(cubemap_dict, h=output_height, w=output_width, mode='bilinear')
        
        equirectangular_tensor = self.image_to_tensor(equirectangular_img)
        return (equirectangular_tensor,)

    def tensor_to_image(self, tensor_in):
        img_np = tensor_in.squeeze(0).numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        return img_uint8
    
    def image_to_tensor(self, img_array):
        img_float32 = np.array(img_array).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float32).unsqueeze(0)
        return tensor

NODE_CLASS_MAPPINGS = {
    "TS Cube to Equirectangular": TS_CubemapFacesToEquirectangularNode # Ключ оставлен оригинальным
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Cube to Equirectangular": "TS Cube to Equirectangular"
}