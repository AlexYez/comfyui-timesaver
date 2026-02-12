import asyncio
import io
import json
import hashlib
import os
import re
import shutil
import subprocess
import urllib.parse
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

import server
from aiohttp import web
import folder_paths
from comfy.utils import common_upscale
try:
    from send2trash import send2trash
except Exception:
    send2trash = None
try:
    import cv2
except Exception:
    cv2 = None

try:
    import av
    from comfy_api.latest._input.video_types import VideoInput
    from comfy_api.latest._util.video_types import VideoComponents
except Exception:
    av = None
    VideoComponents = None

    class VideoInput:
        pass


class TS_VideoFromFile(VideoInput):
    def __init__(self, file_path: str):
        self._file_path = file_path

    def get_stream_source(self):
        return self._file_path

    def _get_video_stream(self, container):
        for stream in container.streams:
            if stream.type == "video":
                return stream
        raise ValueError(f"No video stream found in file: {self._file_path}")

    def get_dimensions(self):
        with av.open(self._file_path, mode="r") as container:
            video_stream = self._get_video_stream(container)
            return (video_stream.width, video_stream.height)

    def get_frame_rate(self):
        with av.open(self._file_path, mode="r") as container:
            video_stream = self._get_video_stream(container)
            if video_stream.average_rate:
                return Fraction(video_stream.average_rate)
            return Fraction(1)

    def get_frame_count(self):
        with av.open(self._file_path, mode="r") as container:
            video_stream = self._get_video_stream(container)
            if video_stream.frames and video_stream.frames > 0:
                return int(video_stream.frames)
            if container.duration and video_stream.average_rate:
                duration_seconds = float(container.duration / av.time_base)
                return int(round(duration_seconds * float(video_stream.average_rate)))
            return 0

    def get_components(self):
        if av is None or VideoComponents is None:
            raise RuntimeError("Video support is not available (PyAV is missing).")

        frames = []
        with av.open(self._file_path, mode="r") as container:
            video_stream = self._get_video_stream(container)
            for frame in container.decode(video=video_stream.index):
                img = frame.to_ndarray(format="rgb24")
                frames.append(torch.from_numpy(img).float() / 255.0)
            images = torch.stack(frames) if frames else torch.zeros(0, 3, 0, 0)
            frame_rate = Fraction(video_stream.average_rate) if video_stream.average_rate else Fraction(1)
            return VideoComponents(images=images, frame_rate=frame_rate, audio=None, metadata=container.metadata)

    def save_to(self, path, format=None, codec=None, metadata=None):
        if isinstance(path, io.BytesIO):
            with open(self._file_path, "rb") as src:
                path.write(src.read())
            path.seek(0)
            return
        shutil.copyfile(self._file_path, path)


class TS_FileBrowser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "selection": ("STRING", {"default": "[]", "multiline": True, "forceInput": True}),
                "gallery_unique_id_widget": ("STRING", {"default": "", "multiline": False}),
                "current_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "MASK", "STRING")
    RETURN_NAMES = ("image", "video", "audio", "mask", "path")
    FUNCTION = "get_selected_media"
    CATEGORY = "TS/Input"

    @classmethod
    def IS_CHANGED(cls, selection="[]", current_path="", **kwargs):
        hasher = hashlib.sha256()
        hasher.update(str(selection).encode())
        hasher.update(str(current_path).encode())

        try:
            selections_list = json.loads(selection)
            input_dir = folder_paths.get_input_directory()
            for item in selections_list:
                path = item.get("path")
                if not path or not os.path.exists(path):
                    continue
                try:
                    mtime = os.path.getmtime(path)
                    hasher.update(str(mtime).encode())
                except Exception:
                    continue

                if item.get("type") == "image":
                    filename = os.path.basename(path)
                    name, _ = os.path.splitext(filename)
                    mask_filename = f"{name}_mask.png"
                    input_mask_path = os.path.join(input_dir, mask_filename)
                    if os.path.exists(input_mask_path):
                        hasher.update(str(os.path.getmtime(input_mask_path)).encode())
                        hasher.update(str(input_mask_path).encode())
                    original_mask_path = os.path.join(os.path.dirname(path), mask_filename)
                    if os.path.exists(original_mask_path):
                        hasher.update(str(os.path.getmtime(original_mask_path)).encode())
                        hasher.update(str(original_mask_path).encode())
        except Exception:
            pass

        return hasher.hexdigest()

    def get_selected_media(self, unique_id, selection="[]", current_path="", gallery_unique_id_widget=""):
        try:
            selections_list = json.loads(selection)
            if not isinstance(selections_list, list):
                selections_list = []
        except Exception:
            selections_list = []

        image_paths = [item.get("path") for item in selections_list if item.get("type") == "image"]
        video_paths = [item.get("path") for item in selections_list if item.get("type") == "video"]
        audio_paths = [item.get("path") for item in selections_list if item.get("type") == "audio"]

        image_tensor = self._load_images(image_paths)
        mask_tensor = self._load_mask_for_images(image_paths, image_tensor)

        video_output = None
        if video_paths and av is not None and VideoComponents is not None:
            first_video = next((p for p in video_paths if p and os.path.exists(p)), None)
            if first_video:
                video_output = TS_VideoFromFile(first_video)
        elif video_paths:
            print("[TS File Browser] video output unavailable (PyAV missing).")

        audio_output = self._load_audio(audio_paths[0]) if audio_paths else self._empty_audio()

        path_out = ""
        if len(selections_list) == 1:
            single_path = selections_list[0].get("path")
            if single_path and os.path.exists(single_path):
                path_out = single_path

        video_info = "empty"
        if video_output is not None:
            try:
                w, h = video_output.get_dimensions()
                video_info = f"{w}x{h}"
            except Exception:
                video_info = "set"

        print(f"[TS File Browser] image={tuple(image_tensor.shape)} mask={tuple(mask_tensor.shape)} "
              f"audio={tuple(audio_output['waveform'].shape)} video={video_info} path={'set' if path_out else 'empty'}")

        return (image_tensor, video_output, audio_output, mask_tensor, path_out)

    def _empty_audio(self):
        return {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

    def _load_audio(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return self._empty_audio()

        args = ["ffmpeg", "-i", file_path, "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "44100", "-ac", "2", "-"]

        try:
            proc = subprocess.run(args, capture_output=True, check=True)
            info_str = proc.stderr.decode("utf-8", "replace")

            sample_rate = 44100
            channels = 2

            sr_match = re.search(r"(\\d+)\\s+Hz", info_str)
            if sr_match:
                sample_rate = int(sr_match.group(1))

            ch_match = re.search(r"Hz,\\s+(mono|stereo)", info_str)
            if ch_match:
                channels = 1 if ch_match.group(1) == "mono" else 2

            waveform = torch.from_numpy(np.frombuffer(proc.stdout, dtype=np.float32))
            waveform = waveform.reshape(-1, channels).permute(1, 0)

            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except subprocess.CalledProcessError as e:
            print(f"[TS File Browser] audio extract failed: {e.stderr.decode('utf-8', 'replace')}")
            return self._empty_audio()
        except Exception as e:
            print(f"[TS File Browser] audio extract error: {e}")
            return self._empty_audio()

    def _load_images(self, image_paths):
        valid_paths = [p for p in image_paths if p and os.path.exists(p)]
        if not valid_paths:
            return torch.zeros(1, 64, 64, 3)

        sizes = {}
        batch_has_alpha = False
        for media_path in valid_paths:
            try:
                with Image.open(media_path) as img:
                    sizes[img.size] = sizes.get(img.size, 0) + 1
                    if not batch_has_alpha and (img.mode == "RGBA" or (img.mode == "P" and "transparency" in img.info)):
                        batch_has_alpha = True
            except Exception:
                continue

        if not sizes:
            return torch.zeros(1, 64, 64, 3)

        target_width, target_height = max(sizes.items(), key=lambda x: x[1])[0]
        target_mode = "RGBA" if batch_has_alpha else "RGB"

        image_tensors = []
        for media_path in valid_paths:
            try:
                with Image.open(media_path) as img:
                    img_out = img.convert(target_mode)
                    if img.size[0] != target_width or img.size[1] != target_height:
                        img_array_pre = np.array(img_out).astype(np.float32) / 255.0
                        tensor_pre = torch.from_numpy(img_array_pre)[None,].permute(0, 3, 1, 2)
                        tensor_post = common_upscale(tensor_pre, target_width, target_height, "lanczos", "center")
                        img_array = tensor_post.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
                    else:
                        img_array = np.array(img_out).astype(np.float32) / 255.0

                    image_tensor = torch.from_numpy(img_array)[None,]
                    image_tensors.append(image_tensor)
            except Exception as e:
                print(f"[TS File Browser] image load error: {media_path} ({e})")

        if not image_tensors:
            return torch.zeros(1, 64, 64, 3)

        final_image_tensor = torch.cat(image_tensors, dim=0)
        if final_image_tensor.shape[-1] == 4 and torch.min(final_image_tensor[:, :, :, 3]) > 0.9999:
            final_image_tensor = final_image_tensor[:, :, :, :3]
        return final_image_tensor

    def _load_mask_for_images(self, image_paths, image_tensor):
        if image_tensor is None or image_tensor.nelement() == 0:
            return torch.zeros(1, 64, 64)

        h, w = image_tensor.shape[1], image_tensor.shape[2]
        if not image_paths:
            return torch.zeros(image_tensor.shape[0], h, w, dtype=torch.float32)

        first_image = next((p for p in image_paths if p and os.path.exists(p)), None)
        if not first_image or not os.path.exists(first_image):
            return torch.zeros(image_tensor.shape[0], h, w, dtype=torch.float32)

        filename = os.path.basename(first_image)
        name, _ = os.path.splitext(filename)
        input_dir = folder_paths.get_input_directory()
        mask_filename = f"{name}_mask.png"

        input_mask_path = os.path.join(input_dir, mask_filename)
        original_dir_mask_path = os.path.join(os.path.dirname(first_image), mask_filename)

        mask_path = None
        if os.path.exists(input_mask_path):
            mask_path = input_mask_path
        elif os.path.exists(original_dir_mask_path):
            mask_path = original_dir_mask_path

        if mask_path:
            try:
                with Image.open(mask_path) as mask_img:
                    if mask_img.width != w or mask_img.height != h:
                        mask_img = mask_img.resize((w, h), Image.NEAREST)
                    if "A" in mask_img.getbands():
                        mask_data = mask_img.split()[-1]
                    else:
                        mask_data = mask_img.convert("L")
                    mask_np = np.array(mask_data).astype(np.float32) / 255.0
                    return torch.from_numpy(mask_np).unsqueeze(0)
            except Exception as e:
                print(f"[TS File Browser] mask load error: {e}")

        try:
            with Image.open(first_image) as img:
                if img.mode == "RGBA":
                    alpha = np.array(img.split()[-1]).astype(np.float32) / 255.0
                    inverted_alpha = 1.0 - alpha
                    return torch.from_numpy(inverted_alpha).unsqueeze(0)
        except Exception:
            pass

        return torch.zeros(image_tensor.shape[0], h, w, dtype=torch.float32)


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_STATE_FILE = os.path.join(NODE_DIR, "tsfb_ui_state.json")
CACHE_DIR = os.path.join(NODE_DIR, ".cache")
THUMBNAIL_CACHE_DIR = os.path.join(CACHE_DIR, "tsfb_thumbnails")

os.makedirs(THUMBNAIL_CACHE_DIR, exist_ok=True)


def _normalize_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _input_root() -> str:
    return _normalize_path(folder_paths.get_input_directory())


def _is_within_directory(path: str, base: str) -> bool:
    try:
        path_norm = os.path.normcase(os.path.normpath(path))
        base_norm = os.path.normcase(os.path.normpath(base))
        return os.path.commonpath([path_norm, base_norm]) == base_norm
    except ValueError:
        return False


def _clamp_directory(path: str) -> str:
    if not path:
        return _input_root()
    normalized = _normalize_path(path)
    if os.path.isabs(normalized) and os.path.isdir(normalized):
        return normalized
    fallback = _input_root()
    relative_candidate = _normalize_path(os.path.join(fallback, normalized))
    if os.path.isdir(relative_candidate):
        return relative_candidate
    return fallback


def _load_ui_state() -> dict:
    if not os.path.exists(UI_STATE_FILE):
        return {}
    try:
        with open(UI_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ui_state(data: dict) -> None:
    try:
        with open(UI_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[TS File Browser] UI state save error: {e}")


def _get_thumbnail_cache_path(filepath: str) -> str:
    filename = hashlib.md5(filepath.encode("utf-8")).hexdigest()
    return os.path.join(THUMBNAIL_CACHE_DIR, filename + ".webp")


def _get_directory_items(directory: str) -> list[dict]:
    items = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                try:
                    if entry.is_dir():
                        items.append({
                            "path": _normalize_path(entry.path),
                            "name": entry.name,
                            "mtime": entry.stat().st_mtime,
                            "type": "dir",
                        })
                        continue

                    if not entry.is_file():
                        continue

                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in SUPPORTED_IMAGE_EXTENSIONS:
                        ftype = "image"
                    elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                        ftype = "video"
                    elif ext in SUPPORTED_AUDIO_EXTENSIONS:
                        ftype = "audio"
                    else:
                        continue

                    items.append({
                        "path": _normalize_path(entry.path),
                        "name": entry.name,
                        "mtime": entry.stat().st_mtime,
                        "type": ftype,
                    })
                except (PermissionError, FileNotFoundError):
                    continue
    except Exception as e:
        print(f"[TS File Browser] scan error: {e}")
    return items


prompt_server = server.PromptServer.instance


@prompt_server.routes.get("/ts_file_browser/images")
async def tsfb_list_images(request):
    directory = _clamp_directory(request.query.get("directory", ""))
    show_images = request.query.get("show_images", "true").lower() == "true"
    show_videos = request.query.get("show_videos", "true").lower() == "true"
    show_audio = request.query.get("show_audio", "true").lower() == "true"
    sort_by = request.query.get("sort_by", "name")
    sort_order = request.query.get("sort_order", "asc")
    page = int(request.query.get("page", 1))
    per_page = int(request.query.get("per_page", 60))
    selected_paths = request.query.getall("selected_paths", [])

    if not directory or not os.path.isdir(directory):
        return web.json_response({"error": "Directory not found or is invalid."}, status=404)

    all_items = []
    for item in _get_directory_items(directory):
        if item["type"] == "dir":
            all_items.append(item)
            continue
        if item["type"] == "image" and show_images:
            all_items.append(item)
        elif item["type"] == "video" and show_videos:
            all_items.append(item)
        elif item["type"] == "audio" and show_audio:
            all_items.append(item)

    reverse_order = sort_order == "desc"
    if sort_by == "date":
        all_items.sort(key=lambda x: x["mtime"], reverse=reverse_order)
    elif sort_by == "type":
        all_items.sort(key=lambda x: x["type"], reverse=reverse_order)
    else:
        all_items.sort(key=lambda x: x["name"].lower(), reverse=reverse_order)

    all_items.sort(key=lambda x: x["type"] != "dir")

    if selected_paths:
        pinned = []
        remaining = []
        selected_set = set(selected_paths)
        for item in all_items:
            if item["path"] in selected_set:
                pinned.append(item)
            else:
                remaining.append(item)
        all_items = pinned + remaining

    parent_directory = os.path.dirname(directory)
    if not parent_directory or parent_directory == directory or not os.path.isdir(parent_directory):
        parent_directory = None

    start = max(0, (page - 1) * per_page)
    end = start + per_page
    paginated_items = all_items[start:end]

    return web.json_response({
        "items": paginated_items,
        "total_pages": (len(all_items) + per_page - 1) // per_page,
        "current_page": page,
        "current_directory": directory,
        "parent_directory": parent_directory,
    })


@prompt_server.routes.get("/ts_file_browser/thumbnail")
async def tsfb_thumbnail(request):
    filepath = request.query.get("filepath")
    if not filepath:
        return web.Response(status=400)

    filepath = urllib.parse.unquote(filepath)
    if not os.path.exists(filepath):
        return web.Response(status=404)

    if not os.path.isfile(filepath):
        return web.Response(status=404)

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS and ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return web.Response(status=415)

    cache_path = _get_thumbnail_cache_path(filepath)
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(filepath):
        return web.FileResponse(cache_path)

    try:
        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            if cv2 is None:
                return web.Response(status=501)
            video_cap = cv2.VideoCapture(filepath)
            if not video_cap.isOpened():
                raise IOError("Cannot open video file")
            try:
                ret, frame = video_cap.read()
                if not ret:
                    raise ValueError("Cannot read frame from video")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                video_cap.release()
        else:
            img = Image.open(filepath)

        img.thumbnail([320, 320], Image.LANCZOS)
        img.save(cache_path, "WEBP", quality=85)
        return web.FileResponse(cache_path)
    except Exception as e:
        print(f"[TS File Browser] thumbnail error: {e}")
        return web.Response(status=500)


@prompt_server.routes.get("/ts_file_browser/view")
async def tsfb_view(request):
    filepath = request.query.get("filepath")
    if not filepath:
        return web.Response(status=400)

    filepath = urllib.parse.unquote(filepath)
    if not os.path.exists(filepath):
        return web.Response(status=404)

    if not os.path.isfile(filepath):
        return web.Response(status=404)

    try:
        return web.FileResponse(filepath)
    except Exception:
        return web.Response(status=500)


@prompt_server.routes.post("/ts_file_browser/delete_files")
async def tsfb_delete_files(request):
    try:
        data = await request.json()
        filepaths = data.get("filepaths", [])
        if not isinstance(filepaths, list):
            return web.json_response({"status": "error", "message": "Invalid data format."}, status=400)

        if send2trash is None:
            return web.json_response({"status": "error", "message": "send2trash is not available"}, status=501)

        for filepath in filepaths:
            if not filepath or not os.path.isabs(filepath):
                continue
            normalized = _normalize_path(filepath)
            if not os.path.isfile(normalized):
                continue
            try:
                send2trash(os.path.normpath(normalized))
            except Exception as e:
                print(f"[TS File Browser] delete error: {e}")

        return web.json_response({"status": "ok", "message": "Delete operation completed."})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


@prompt_server.routes.post("/ts_file_browser/set_ui_state")
async def tsfb_set_ui_state(request):
    try:
        data = await request.json()
        node_id = str(data.get("node_id"))
        gallery_id = str(data.get("gallery_id"))
        state = data.get("state", {})
        if not node_id or not gallery_id:
            return web.json_response({"status": "error", "message": "node_id or gallery_id is required"}, status=400)

        node_key = f"{gallery_id}_{node_id}"
        ui_states = _load_ui_state()
        if node_key not in ui_states:
            ui_states[node_key] = {}
        ui_states[node_key].update(state)
        _save_ui_state(ui_states)
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


@prompt_server.routes.get("/ts_file_browser/get_ui_state")
async def tsfb_get_ui_state(request):
    try:
        node_id = request.query.get("node_id")
        gallery_id = request.query.get("gallery_id")
        if not node_id or not gallery_id:
            return web.json_response({"error": "node_id or gallery_id is required"}, status=400)

        node_key = f"{gallery_id}_{node_id}"
        ui_states = _load_ui_state()

        default_state = {
            "current_path": _input_root(),
            "selection": [],
            "sort_by": "name",
            "sort_order": "asc",
            "show_images": True,
            "show_videos": True,
            "show_audio": True,
        }

        node_saved_state = ui_states.get(node_key, {})
        final_state = {**default_state, **node_saved_state}
        final_state["current_path"] = _clamp_directory(final_state.get("current_path"))
        final_state["input_root"] = _input_root()

        return web.json_response(final_state)
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


@prompt_server.routes.get("/ts_file_browser/pick_file")
async def tsfb_pick_file(request):
    try:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            return web.json_response({"status": "error", "message": f"tkinter unavailable: {e}"}, status=501)

        def _pick():
            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            patterns = [f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS)]
            filetypes = [("Media Files", " ".join(patterns)), ("All Files", "*.*")]
            path = filedialog.askopenfilename(title="Select media file", filetypes=filetypes)
            try:
                root.destroy()
            except Exception:
                pass
            return path

        loop = asyncio.get_running_loop()
        selected_path = await loop.run_in_executor(None, _pick)
        if not selected_path:
            return web.json_response({"path": ""})
        return web.json_response({"path": _normalize_path(selected_path)})
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


NODE_CLASS_MAPPINGS = {
    "TS_FileBrowser": TS_FileBrowser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FileBrowser": "TS File Browser",
}
