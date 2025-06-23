import torch
import numpy as np
import cv2
import os
import requests
import folder_paths
from comfy.utils import ProgressBar
from torch import nn
from torch.nn import functional as F

class RIFE(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        from .rife.warplayer import warp
        self.warp = warp
        self.load_model(model_path)
        
    def load_model(self, model_path):
        try:
            from .rife.IFNet_HDv3 import IFNet
            self.ifnet = IFNet().cuda()
            state_dict = torch.load(model_path)
            self.ifnet.load_state_dict(state_dict)
            self.ifnet.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load RIFE model: {str(e)}")

    def forward(self, img0, img1, timestep=0.5):
        with torch.no_grad():
            img0 = (img0 / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
            img1 = (img1 / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
            
            img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
            img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
            
            flow, mask = self.ifnet(torch.cat((img0, img1), dim=1), None)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])
            
            merged = warped_img0 * mask + warped_img1 * (1 - mask) if mask is not None else (warped_img0 + warped_img1) * 0.5
            merged = F.interpolate(merged, scale_factor=2.0, mode="bilinear", align_corners=False)
            
            return (merged.clamp(0, 1) * 255).cpu().numpy().squeeze().transpose(1, 2, 0).astype(np.uint8)

class TS_DeflickerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["temporal_median", "temporal_gaussian", "adaptive_histogram", "rife_interpolation"], {"default": "temporal_gaussian"}),
                "window_size": ("INT", {"default": 3, "min": 3, "max": 15, "step": 2}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preserve_details": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "deflicker"
    CATEGORY = "Video PostProcessing"

    def __init__(self):
        self.rife_model = None
        self.model_loaded = False

    def download_model(self, url, save_path):
        """Скачивает модель с указанного URL"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            print(f"Downloading RIFE model from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False

    def get_model_path(self):
        """Проверяет и скачивает модель при необходимости"""
        model_dir = folder_paths.models_dir
        rife_dir = os.path.join(model_dir, "rife")
        os.makedirs(rife_dir, exist_ok=True)
        
        model_path = os.path.join(rife_dir, "rife49.pth")
        
        if not os.path.exists(model_path):
            print("RIFE model not found, attempting to download...")
            download_url = "https://huggingface.co/hfmaster/models/resolve/main/rife/rife49.pth"
            if not self.download_model(download_url, model_path):
                raise FileNotFoundError("Failed to download RIFE model. Please download it manually.")
        
        return model_path

    def temporal_median_filter(self, window, center_idx):
        return np.median(window, axis=0).astype(np.uint8)

    def temporal_gaussian_filter(self, window, center_idx, preserve_details):
        length = window.shape[0]
        weights = np.array([np.exp(-(i-center_idx)**2/(2*(length/3)**2)) for i in range(length)])
        weights /= weights.sum()
        
        weighted = np.zeros_like(window[0], dtype=np.float32)
        for i in range(length):
            weighted += window[i].astype(np.float32) * weights[i]
        
        result = np.clip(weighted, 0, 255).astype(np.uint8)
        
        if preserve_details:
            edges = cv2.Canny(window[center_idx], 50, 150)
            result = np.where(edges[...,None]>0, window[center_idx], result)
            
        return result

    def adaptive_histogram_matching(self, window, center_idx, preserve_details):
        target = window[center_idx]
        reference = np.mean(window, axis=0).astype(np.uint8)
        target = target.astype(np.uint8)
        
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
        
        matched_lab = np.zeros_like(target_lab)
        for i in range(3):
            matched_lab[:,:,i] = self.match_histograms(
                target_lab[:,:,i], 
                reference_lab[:,:,i]
            )
        
        matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
        
        if preserve_details:
            edges = cv2.Canny(target, 50, 150)
            matched = np.where(edges[...,None]>0, target, matched)
            
        return matched

    def match_histograms(self, source, template):
        src_hist, _ = np.histogram(source, bins=256, range=(0,255))
        tpl_hist, _ = np.histogram(template, bins=256, range=(0,255))

        src_hist = src_hist / source.size
        tpl_hist = tpl_hist / template.size

        src_cdf = src_hist.cumsum()
        tpl_cdf = tpl_hist.cumsum()

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.argmin(np.abs(tpl_cdf - src_cdf[i]))
        
        return lut[source]

    def rife_deflicker(self, window, center_idx, preserve_details):
        if center_idx == 0:
            return window[0]
        elif center_idx == len(window)-1:
            return window[-1]
        
        prev_frame = window[center_idx-1]
        next_frame = window[center_idx+1]
        current_frame = window[center_idx]
        
        interpolated = self.rife_model(prev_frame, next_frame)
        
        if preserve_details:
            edges = cv2.Canny(current_frame, 50, 150)
            blend_strength = np.where(edges[...,None]>0, 0.1, 0.3)
            mixed = current_frame*(1-blend_strength) + interpolated*blend_strength
        else:
            mixed = cv2.addWeighted(current_frame, 0.7, interpolated, 0.3, 0)
        
        return mixed.astype(np.uint8)

    def deflicker(self, images, method="temporal_gaussian", window_size=3, intensity=0.5, preserve_details=True):
        if len(images) < 3:
            return (images,)
            
        np_images = (images.cpu().numpy() * 255).astype(np.uint8)
        processed_frames = []
        progress_bar = ProgressBar(len(images))
        
        for i in range(len(images)):
            progress_bar.update(1)
            
            if i == 0 or i == len(images) - 1:
                processed_frames.append(np_images[i])
                continue
                
            start = max(0, i - window_size // 2)
            end = min(len(images), i + window_size // 2 + 1)
            window = np_images[start:end]
            
            try:
                if method == "temporal_median":
                    processed = self.temporal_median_filter(window, i-start)
                elif method == "temporal_gaussian":
                    processed = self.temporal_gaussian_filter(window, i-start, preserve_details)
                elif method == "adaptive_histogram":
                    processed = self.adaptive_histogram_matching(window, i-start, preserve_details)
                elif method == "rife_interpolation":
                    if not self.model_loaded:
                        model_path = self.get_model_path()
                        self.rife_model = RIFE(model_path)
                        self.model_loaded = True
                    processed = self.rife_deflicker(window, i-start, preserve_details)
                else:
                    processed = np_images[i]
                
                if intensity < 1.0:
                    processed = cv2.addWeighted(
                        np_images[i], 1.0 - intensity,
                        processed, intensity,
                        0
                    )
                
                processed_frames.append(processed)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                processed_frames.append(np_images[i])
        
        processed_frames = np.stack(processed_frames)
        return (torch.from_numpy(processed_frames.astype(np.float32) / 255.0),)

NODE_CLASS_MAPPINGS = {
    "TS_DeflickerNode": TS_DeflickerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_DeflickerNode": "TS Advanced Video Deflicker"
}
