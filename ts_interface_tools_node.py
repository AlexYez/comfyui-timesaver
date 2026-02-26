import sys
import traceback
import os
import time
import uuid
import subprocess
import wave
from collections.abc import Mapping
import torch
import folder_paths

try:
    import imageio
except Exception:
    imageio = None

# ==============================================================================
# TS Logger (Utility Class)
# ==============================================================================
class TS_Logger:
    """
    Утилита для логирования.
    """
    @staticmethod
    def log(node_name, message, color="cyan"):
        print(f"[TS {node_name}] {message}")

    @staticmethod
    def error(node_name, message):
        TS_Logger.log(node_name, message, "red")

# ==============================================================================
# Node 1: TS Float Slider
# ==============================================================================
class TS_FloatSlider:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.5, "min": -1000000000.0, "max": 1000000000.0, "step": 0.1,
                    "display": "slider", "round": 0.01
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_value",)
    FUNCTION = "get_value"
    CATEGORY = "TS Tools/Sliders"
    DESCRIPTION = "Float slider (0.0 - 1.0)"

    def get_value(self, value):
        TS_Logger.log("FloatSlider", f"Value: {value:.2f}")
        return (float(value),)

# ==============================================================================
# Node 2: TS Int Slider
# ==============================================================================
class TS_Int_Slider:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 512, "min": -2147483648, "max": 2147483647, "step": 8,
                    "display": "slider" 
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_value",)
    FUNCTION = "get_value"
    CATEGORY = "TS Tools/Sliders"
    DESCRIPTION = "Int slider (320 - 2048)"

    def get_value(self, value):
        TS_Logger.log("IntSlider", f"Value: {value}")
        return (int(value),)

# ==============================================================================
# Node 3: TS Smart Switch (Any Type)
# ==============================================================================
class TS_Smart_Switch:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_type": (["images", "video", "audio", "mask", "string", "int", "float"],),
                "switch": ("BOOLEAN", {"default": True, "label_on": "Input 1", "label_off": "Input 2"}),
            },
            "optional": {
                "input_1": ("*",), 
                "input_2": ("*",),
            }
        }

    RETURN_TYPES = ("*",) 
    RETURN_NAMES = ("output",)
    FUNCTION = "smart_switch"
    CATEGORY = "TS Tools/Logic"
    DESCRIPTION = "Smart switch for ANY data. Auto-failover if one input is missing."

    def _is_valid_image(self, data):
        return isinstance(data, torch.Tensor) and data.ndim == 4

    def _is_valid_mask(self, data):
        if not isinstance(data, torch.Tensor):
            return False
        return data.ndim == 3 or (data.ndim == 4 and data.shape[-1] == 1)

    def _is_valid_video(self, data):
        if isinstance(data, torch.Tensor):
            return data.ndim == 5
        if isinstance(data, (list, tuple)) and data:
            return all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in data)
        return False

    def _is_valid_audio(self, data):
        if isinstance(data, dict) and "waveform" in data:
            return isinstance(data["waveform"], torch.Tensor)
        return False

    def _is_valid_scalar(self, data, data_type):
        if data_type == "string":
            return isinstance(data, str)
        if data_type == "int":
            return isinstance(data, int) and not isinstance(data, bool)
        if data_type == "float":
            return isinstance(data, float)
        return False

    def _is_valid_by_type(self, data, data_type):
        if data is None:
            return False
        if data_type == "images":
            return self._is_valid_image(data)
        if data_type == "video":
            return self._is_valid_video(data)
        if data_type == "audio":
            return self._is_valid_audio(data)
        if data_type == "mask":
            return self._is_valid_mask(data)
        if data_type in ("string", "int", "float"):
            return self._is_valid_scalar(data, data_type)
        return False

    def smart_switch(self, switch, data_type="images", input_1=None, input_2=None):
        selected_source = "None"
        result = None
        status_msg = ""

        try:
            valid_1 = self._is_valid_by_type(input_1, data_type)
            valid_2 = self._is_valid_by_type(input_2, data_type)

            if input_1 is not None and not valid_1:
                TS_Logger.log("SmartSwitch", f"Input 1 ignored: type mismatch for data_type={data_type}", "yellow")
            if input_2 is not None and not valid_2:
                TS_Logger.log("SmartSwitch", f"Input 2 ignored: type mismatch for data_type={data_type}", "yellow")

            has_input_1 = valid_1
            has_input_2 = valid_2

            if has_input_1 and has_input_2:
                if switch:
                    result = input_1
                    selected_source = "Input 1"
                    status_msg = "(Switch: ON)"
                else:
                    result = input_2
                    selected_source = "Input 2"
                    status_msg = "(Switch: OFF)"
            elif has_input_1:
                result = input_1
                selected_source = "Input 1"
                status_msg = "(Auto-Failover)"
            elif has_input_2:
                result = input_2
                selected_source = "Input 2"
                status_msg = "(Auto-Failover)"
            else:
                TS_Logger.error("SmartSwitch", "Warning: Both inputs are None.")
                return (None,)

            # Log info
            info = "Unknown"
            if hasattr(result, 'shape'): info = f"Tensor {result.shape}"
            elif isinstance(result, (int, float, str)): info = str(result)
            
            TS_Logger.log("SmartSwitch", f"Selected: {selected_source} {status_msg} | {info}")
            return (result,)

        except Exception as e:
            TS_Logger.error("SmartSwitch", f"Error: {str(e)}")
            return (None,)

# ==============================================================================
# Node 4: TS Math Int
# ==============================================================================
class TS_Math_Int:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "b": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "operation": ([
                    "add (+)",
                    "subtract (-)",
                    "multiply (*)",
                    "divide (/)",
                    "floor_divide (//)",
                    "modulo (%)",
                    "power (**)",
                    "min",
                    "max"
                ],),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "calculate"
    CATEGORY = "TS/Math"
    DESCRIPTION = "Integer math operations"

    def calculate(self, a, b, operation):
        try:
            if operation == "add (+)":
                result = a + b
            elif operation == "subtract (-)":
                result = a - b
            elif operation == "multiply (*)":
                result = a * b
            elif operation == "divide (/)":
                if b == 0:
                    TS_Logger.error("MathInt", "Division by zero")
                    result = 0
                else:
                    result = int(a / b)
            elif operation == "floor_divide (//)":
                if b == 0:
                    TS_Logger.error("MathInt", "Division by zero")
                    result = 0
                else:
                    result = a // b
            elif operation == "modulo (%)":
                if b == 0:
                    TS_Logger.error("MathInt", "Modulo by zero")
                    result = 0
                else:
                    result = a % b
            elif operation == "power (**)":
                result = int(pow(a, b))
            elif operation == "min":
                result = min(a, b)
            elif operation == "max":
                result = max(a, b)
            else:
                TS_Logger.error("MathInt", f"Unknown operation: {operation}")
                result = 0

            TS_Logger.log("MathInt", f"{a} {operation} {b} = {result}")
            return (int(result),)
        except Exception as e:
            TS_Logger.error("MathInt", f"Error: {str(e)}")
            return (0,)

# ==============================================================================
# Node 5: TS Animation Preview
# ==============================================================================
class TS_Animation_Preview:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "TS/Interface Tools"
    DESCRIPTION = "Create a looping H.265 preview video from an image batch."

    @classmethod
    def IS_CHANGED(cls, images, fps, audio=None):
        return time.time()

    def preview(self, images, fps, audio=None):
        node_name = "AnimationPreview"

        if images is None:
            TS_Logger.error(node_name, "No images provided.")
            return {"ui": {"ts_animation_preview": []}}

        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("TS Animation Preview expects IMAGE tensor [B,H,W,C].")

        frame_count, height, width, channels = images.shape
        TS_Logger.log(node_name, f"Input images shape: {tuple(images.shape)} | fps={fps}")

        fps_value = max(0.1, float(fps))
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        filename = f"ts_animation_preview_{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(temp_dir, filename)

        audio_path = None
        audio_waveform = None
        audio_sample_rate = None
        audio_payload = self._normalize_audio_input(audio)
        if audio_payload is not None:
            audio_waveform, audio_sample_rate = self._prepare_audio(audio_payload, frame_count, fps_value)
            TS_Logger.log(
                node_name,
                f"Audio waveform shape: {tuple(audio_waveform.shape)} | sample_rate={audio_sample_rate}"
            )
            audio_path = self._write_audio_wav(temp_dir, audio_waveform, audio_sample_rate)

        codec_used = self._write_h265_video(filepath, images, fps_value)

        if audio_path is not None:
            mux_path = os.path.join(temp_dir, f"ts_animation_preview_mux_{uuid.uuid4().hex}.mp4")
            duration_seconds = frame_count / fps_value if fps_value > 0 else 0.0
            try:
                self._mux_audio_into_video(filepath, audio_path, mux_path, duration_seconds)
                os.replace(mux_path, filepath)
            finally:
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
                if os.path.exists(mux_path):
                    try:
                        os.remove(mux_path)
                    except Exception:
                        pass

        TS_Logger.log(
            node_name,
            f"Output video: {filename} | {width}x{height} | frames={frame_count} | fps={fps_value} | codec={codec_used}"
        )

        payload = {
            "filename": filename,
            "subfolder": "",
            "type": "temp",
            "format": "video/mp4",
            "width": int(width),
            "height": int(height),
            "frames": int(frame_count),
            "fps": fps_value,
        }

        return {"ui": {"ts_animation_preview": [payload]}}

    def _write_h265_video(self, filepath, images, fps_value):
        if imageio is None:
            raise RuntimeError("imageio is not available. Please install it to enable TS Animation Preview.")
        codec_candidates = ["libx265", "hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc"]
        last_error = None

        for codec in codec_candidates:
            writer = None
            try:
                output_params = ["-pix_fmt", "yuv420p"]
                if codec == "libx265":
                    output_params += ["-crf", "28", "-preset", "medium"]
                writer = imageio.get_writer(
                    filepath,
                    fps=fps_value,
                    codec=codec,
                    output_params=output_params,
                )
                for frame in images:
                    writer.append_data(self._to_uint8_rgb(frame))
                writer.close()
                return codec
            except Exception as e:
                last_error = e
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass

        raise RuntimeError(f"Failed to encode H.265 preview video. Last error: {last_error}")

    def _to_uint8_rgb(self, frame):
        if not isinstance(frame, torch.Tensor) or frame.ndim != 3:
            raise ValueError("Frame must be a torch.Tensor with shape [H,W,C].")

        frame = frame.detach()
        channels = frame.shape[-1]
        if channels == 1:
            frame = frame.repeat(1, 1, 3)
        elif channels < 3:
            raise ValueError("Frame must have at least 3 channels.")

        frame = frame[..., :3]
        frame = torch.clamp(frame, 0.0, 1.0)
        frame = (frame * 255.0).to(torch.uint8).cpu().numpy()
        return frame

    def _normalize_audio_input(self, audio):
        if audio is None:
            return None

        if hasattr(audio, "model_dump") and callable(audio.model_dump):
            try:
                audio = audio.model_dump()
            except Exception:
                pass
        elif hasattr(audio, "dict") and callable(audio.dict):
            try:
                audio = audio.dict()
            except Exception:
                pass

        if isinstance(audio, (list, tuple)):
            if len(audio) == 2 and isinstance(audio[0], torch.Tensor) and isinstance(audio[1], (int, float)):
                return {"waveform": audio[0], "sample_rate": int(audio[1])}
            for item in audio:
                normalized = self._normalize_audio_input(item)
                if normalized is not None:
                    return normalized
            return None

        if isinstance(audio, dict):
            if "audio" in audio:
                return self._normalize_audio_input(audio["audio"])
            if "data" in audio:
                return self._normalize_audio_input(audio["data"])
            if "value" in audio:
                return self._normalize_audio_input(audio["value"])

            waveform = audio.get("waveform", audio.get("WAVEFORM", audio.get("samples", None)))
            sample_rate = audio.get(
                "sample_rate",
                audio.get("sampler_rate", audio.get("sampleRate", audio.get("sr", audio.get("SAMPLE_RATE", None))))
            )
            if waveform is not None and sample_rate is not None:
                return {"waveform": waveform, "sample_rate": sample_rate}

            TS_Logger.log(
                "AnimationPreview",
                f"Audio input ignored: missing waveform/sample_rate keys. keys={list(audio.keys())}",
                "yellow",
            )
            return None

        if isinstance(audio, Mapping):
            try:
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
            except KeyError:
                TS_Logger.log(
                    "AnimationPreview",
                    "Audio input ignored: missing waveform/sample_rate keys.",
                    "yellow",
                )
                return None
            except Exception as e:
                TS_Logger.log(
                    "AnimationPreview",
                    f"Audio input ignored: failed to load audio from mapping. {str(e)}",
                    "yellow",
                )
                return None

            return {"waveform": waveform, "sample_rate": sample_rate}

        waveform = getattr(audio, "waveform", None)
        sample_rate = getattr(audio, "sample_rate", getattr(audio, "sampler_rate", None))
        if waveform is not None and sample_rate is not None:
            return {"waveform": waveform, "sample_rate": sample_rate}

        TS_Logger.log("AnimationPreview", f"Audio input ignored: unsupported type {type(audio)}.", "yellow")
        return None

    def _prepare_audio(self, audio, frame_count, fps_value):
        if not isinstance(audio, dict):
            raise ValueError("Audio input must be a dict with keys 'waveform' and 'sample_rate'.")

        waveform = audio.get("waveform", None)
        sample_rate = audio.get("sample_rate", None)

        if waveform is None or sample_rate is None:
            raise ValueError("Audio input missing waveform or sample_rate.")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError("Audio waveform must be a torch.Tensor with shape [B,C,T].")
        if waveform.shape[0] < 1:
            raise ValueError("Audio waveform batch is empty.")

        if waveform.shape[0] > 1:
            TS_Logger.log("AnimationPreview", f"Audio batch size {waveform.shape[0]} detected. Using first item.", "yellow")

        waveform = waveform[0].detach().cpu().float()
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Audio sample_rate must be > 0.")

        target_samples = int(round((frame_count / fps_value) * sample_rate)) if fps_value > 0 else 0
        if target_samples > 0:
            current_samples = int(waveform.shape[1])
            if current_samples < target_samples:
                pad_amount = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            elif current_samples > target_samples:
                waveform = waveform[:, :target_samples]

        return waveform, sample_rate

    def _write_audio_wav(self, temp_dir, waveform, sample_rate):
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2:
            raise ValueError("Prepared audio waveform must be a torch.Tensor with shape [C,T].")

        channels, samples = waveform.shape
        if channels < 1 or samples < 1:
            raise ValueError("Prepared audio waveform is empty.")

        audio_int16 = torch.clamp(waveform, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767.0).to(torch.int16)

        if channels == 1:
            interleaved = audio_int16[0]
        else:
            interleaved = audio_int16.transpose(0, 1).contiguous().view(-1)

        filename = f"ts_animation_preview_audio_{uuid.uuid4().hex}.wav"
        filepath = os.path.join(temp_dir, filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(int(channels))
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(interleaved.cpu().numpy().tobytes())

        return filepath

    def _get_ffmpeg_exe(self):
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def _mux_audio_into_video(self, video_path, audio_path, output_path, duration_seconds):
        ffmpeg_exe = self._get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
        ]
        if duration_seconds and duration_seconds > 0:
            cmd += ["-t", f"{duration_seconds:.6f}"]
        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"FFmpeg mux failed. {error_msg}")

# ==============================================================================
# Node Registration
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "TS_FloatSlider": TS_FloatSlider,
    "TS_Int_Slider": TS_Int_Slider,
    "TS_Smart_Switch": TS_Smart_Switch,
    "TS_Math_Int": TS_Math_Int,
    "TS_Animation_Preview": TS_Animation_Preview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FloatSlider": "TS Float Slider",
    "TS_Int_Slider": "TS Int Slider",
    "TS_Smart_Switch": "TS Smart Switch",
    "TS_Math_Int": "TS Math Int",
    "TS_Animation_Preview": "TS Animation Preview"
}
