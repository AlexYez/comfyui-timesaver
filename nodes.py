# Standard library imports
import os
import re
import shutil
import subprocess
import uuid
import zipfile

# Third-party imports
import numpy as np
import py360convert
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import hashlib # Kept for potential future checksum use
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Local imports
import folder_paths

class DownloadFilesNode:
    """
    A ComfyUI node to download files from a list of URLs with support for resume,
    retries, size verification, and internet connection check.
    """
    # Required class attributes for ComfyUI
    RETURN_TYPES = ()
    FUNCTION = "execute_downloads" # The main execution method called by ComfyUI
    OUTPUT_NODE = True
    CATEGORY = "Tools/IO"

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types for the node."""
        return {
            "required": {
                "file_list": ("STRING", {
                    "default": "https://example.com/file1.zip /path/to/save/dir1\nhttps://another.example/large_file.safetensors /path/to/models",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "description": "List of files to download. Each line: URL /path/to/save_directory",
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "description": "Skip download if the final file already exists and its size matches (if verification enabled).",
                 }),
                "verify_size": ("BOOLEAN", {
                     "default": True,
                     "description": "Verify the downloaded file size against the server's Content-Length header.",
                 }),
                "chunk_size_kb": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 65536, # Max 64MB chunks
                    "step": 1,
                    "description": "Download chunk size in Kilobytes (KB). Smaller values might be better for unstable connections."
                }),
            },
        }

    # --- Internal Helper Methods ---

    def _check_internet_connection(self, timeout=5):
        """Checks for a basic internet connection. Internal method."""
        try:
            # Use HEAD request for speed, disable retries for this specific check
            # Use a known reliable host like google or cloudflare
            print("[INFO] Checking internet connection...")
            requests.head("https://1.1.1.1", timeout=timeout, allow_redirects=True)
            # requests.head("https://www.google.com", timeout=timeout, allow_redirects=True)
            print("[INFO] Internet connection check: OK")
            return True
        except (requests.ConnectionError, requests.Timeout):
            print("[ERROR] Internet connection check: FAILED. No internet connection detected.")
            return False
        except requests.RequestException as e:
            print(f"[WARN] Internet connection check: Error ({e}). Assuming connection might be unstable.")
            # Proceed cautiously if it's not a clear connection error
            return True

    def _create_session_with_retries(self):
        """Creates a requests Session with retry logic. Internal method."""
        session = requests.Session()
        retries = Retry(
            total=5,          # Total number of retries
            backoff_factor=1, # Wait 1s, 2s, 4s, 8s, 16s between retries
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these status codes
            allowed_methods=frozenset(['HEAD', 'GET']) # Retry only for safe methods
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        # Add a default User-Agent to avoid potential blocks
        session.headers.update({'User-Agent': 'ComfyUI-DownloadNode/1.0'})
        return session

    def _parse_file_list(self, file_list_text):
        """Parse the file list from the multiline text input. Internal method."""
        files = []
        lines = file_list_text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines and comments
                continue
            # Split only on the first space(s) to allow spaces in paths
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                url, target_path = parts[0].strip(), parts[1].strip()
                if not url.startswith(('http://', 'https://')):
                    print(f"[WARN] Skipping line {i+1}: Invalid URL format '{url}'")
                    continue
                # Treat target_path as a directory where the file will be saved
                files.append({'url': url, 'target_dir': target_path})
            else:
                print(f"[WARN] Skipping line {i+1}: Invalid format. Expected 'URL /path/to/save_directory'. Found: '{line}'")
        return files

    def _download_single_file(self, session, url, target_dir, skip_existing=True, verify_size=True, chunk_size_bytes=8192):
        """Downloads a single file with resume, progress, and integrity check. Internal method."""
        try:
            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            print(f"[ERROR] Error creating directory {target_dir}: {e}")
            return False # Cannot proceed

        filename = os.path.basename(url.split('?')[0].split('#')[0]) # More robust filename extraction
        if not filename:
            filename = f"downloaded_file_{int(time.time())}"
            print(f"[WARN] Could not determine filename from URL '{url}'. Using '{filename}'.")

        local_file_path = os.path.join(target_dir, filename)
        temp_file_path = local_file_path + ".part"

        try:
            # 1. Get file info (size)
            print(f"[INFO] Fetching file info for: {filename}...")
            response_head = session.head(url, allow_redirects=True, timeout=(10, 20)) # (connect, read)
            response_head.raise_for_status()

            remote_file_size = int(response_head.headers.get('content-length', -1))
            accept_ranges = response_head.headers.get('accept-ranges', 'none').lower()
            can_resume = (accept_ranges == 'bytes') and (remote_file_size > 0)

            if remote_file_size == -1:
                print(f"[WARN] Server did not provide Content-Length for {filename}. Cannot verify size or reliably resume.")
                can_resume = False
                verify_size = False # Disable verification if size is unknown

            print(f"[INFO] Remote file size: {'{:,}'.format(remote_file_size) if remote_file_size > 0 else 'Unknown'}. Resumable: {can_resume}")

            # 2. Check if final file already exists
            local_size = 0 # Initialize local_size
            if skip_existing and os.path.exists(local_file_path):
                local_size = os.path.getsize(local_file_path)
                if verify_size and remote_file_size > 0:
                    if local_size == remote_file_size:
                        print(f"[OK] File '{filename}' already exists and size matches. Skipping.")
                        return True
                    else:
                        print(f"[WARN] File '{filename}' exists but size mismatch (local: {local_size}, remote: {remote_file_size}). Re-downloading.")
                        # Overwrite logic handles deletion if needed
                else:
                    # If not verifying size or size unknown, skip based on existence only
                    print(f"[OK] File '{filename}' already exists. Skipping (size check disabled or not possible).")
                    return True

            # 3. Handle resume logic
            resume_byte_pos = 0
            headers = {}
            file_mode = 'wb' # Default: write binary, overwrite

            if can_resume and os.path.exists(temp_file_path):
                resume_byte_pos = os.path.getsize(temp_file_path)
                if resume_byte_pos < remote_file_size:
                    print(f"[INFO] Resuming download for '{filename}' from byte {resume_byte_pos}")
                    headers["Range"] = f"bytes={resume_byte_pos}-"
                    file_mode = 'ab' # Switch to append binary
                elif resume_byte_pos == remote_file_size:
                     print(f"[INFO] Partial file '{filename}.part' found and seems complete. Verifying and finalizing.")
                     # Proceed to verification/rename step without download
                     pass
                else: # resume_byte_pos > remote_file_size
                    print(f"[WARN] Partial file '{filename}.part' ({resume_byte_pos} bytes) is larger than remote file ({remote_file_size} bytes). Restarting download.")
                    try: os.remove(temp_file_path)
                    except OSError as e: print(f"[WARN] Could not remove oversized partial file: {e}")
                    resume_byte_pos = 0
                    file_mode = 'wb'
            elif os.path.exists(temp_file_path):
                 # Temp file exists but cannot resume (e.g. server changed, size unknown)
                 print(f"[WARN] Found existing partial file '{filename}.part' but cannot resume. Restarting download.")
                 try: os.remove(temp_file_path)
                 except OSError as e: print(f"[WARN] Could not remove old partial file: {e}")
                 resume_byte_pos = 0
                 file_mode = 'wb'

            # 4. Perform download if needed
            needs_download = True
            # Re-check conditions for skipping download now that resume state is known
            if skip_existing and os.path.exists(local_file_path):
                 # We checked size match earlier if verify_size was True
                 if not (verify_size and remote_file_size > 0 and local_size != remote_file_size):
                      needs_download = False # Skip if exists and size matches (or size check not needed)

            if can_resume and os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) == remote_file_size:
                needs_download = False # .part file is already complete, verified later

            if needs_download:
                print(f"[INFO] Downloading '{filename}' to '{target_dir}'...")
                response_get = session.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=(15, 60), # Increased read timeout for large files
                    allow_redirects=True
                )
                response_get.raise_for_status()

                # Check if server ignored Range header (sent 200 OK instead of 206 Partial Content)
                if resume_byte_pos > 0 and response_get.status_code == 200:
                    print(f"[WARN] Server did not honor Range request for '{filename}' (Status 200). Restarting download from beginning.")
                    resume_byte_pos = 0
                    file_mode = 'wb'

                # Determine total size for progress bar
                total_size_for_tqdm = remote_file_size
                if total_size_for_tqdm <= 0:
                     total_size_for_tqdm = int(response_get.headers.get('content-length', 0))

                try:
                    with open(temp_file_path, file_mode) as file, tqdm(
                        total=total_size_for_tqdm if total_size_for_tqdm > 0 else None,
                        unit='B',
                        unit_scale=True,
                        desc=f"'{filename}'", # Keep filename in description
                        initial=resume_byte_pos,
                        mininterval=0.5,
                        ncols=100,
                        unit_divisor=1024,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]', # Standard bar format
                        disable=False # Ensure progress bar is enabled
                    ) as progress_bar:

                        if file_mode == 'wb' and progress_bar.n > 0:
                             progress_bar.reset(total=total_size_for_tqdm if total_size_for_tqdm > 0 else None)

                        for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                            if chunk:
                                file.write(chunk)
                                progress_bar.update(len(chunk))

                        # Check if download ended prematurely
                        if total_size_for_tqdm > 0 and progress_bar.n < total_size_for_tqdm:
                             print(f"[WARN] Download stream ended prematurely for '{filename}'. Expected {total_size_for_tqdm}, got {progress_bar.n}.")
                             return False # Keep .part file for potential resume

                except (requests.exceptions.ChunkedEncodingError, requests.exceptions.RequestException) as e_stream:
                    print(f"[ERROR] Error during download stream for '{filename}': {e_stream}")
                    print(f"[INFO] Partial file saved to '{temp_file_path}' for potential resume.")
                    return False
                except IOError as e_io:
                    print(f"[ERROR] IO Error writing file chunk for '{filename}': {e_io}")
                    return False
                except Exception as e_inner:
                    print(f"[ERROR] Unexpected error during download stream/write for '{filename}': {e_inner}")
                    traceback.print_exc()
                    return False

            # 5. Final verification and rename
            if os.path.exists(temp_file_path):
                print(f"[INFO] Verifying and finalizing '{filename}'...")
                current_size = os.path.getsize(temp_file_path)

                if verify_size and remote_file_size > 0:
                    if current_size != remote_file_size:
                        print(f"[ERROR] Size mismatch after download! Temp file '{temp_file_path}' size ({current_size}) != Expected size ({remote_file_size}). Download failed.")
                        return False

                try:
                    # Remove final destination *only* if we are overwriting (file_mode was 'wb' initially or reset)
                    # AND if it still exists (it might have been deleted earlier if size mismatch was detected)
                    if file_mode == 'wb' and os.path.exists(local_file_path):
                         print(f"[INFO] Removing existing destination file '{local_file_path}' before renaming.")
                         os.remove(local_file_path)

                    os.rename(temp_file_path, local_file_path)
                    print(f"[OK] Download complete and verified: '{local_file_path}'")
                    # Final check after rename
                    if verify_size and remote_file_size > 0:
                         final_size = os.path.getsize(local_file_path)
                         if final_size != remote_file_size:
                              print(f"[ERROR] CRITICAL: Size mismatch AFTER rename! '{local_file_path}' size ({final_size}) != Expected ({remote_file_size}). File may be corrupt.")
                              return False
                    return True

                except OSError as e_rename:
                    print(f"[ERROR] Error renaming temporary file '{temp_file_path}' to '{local_file_path}': {e_rename}")
                    return False
            elif os.path.exists(local_file_path):
                 # File existed and was skipped successfully
                 return True
            else:
                 print(f"[ERROR] Unknown state: Neither temporary nor final file exists for '{filename}' after processing.")
                 return False

        # Catch exceptions during HEAD request or other setup phases
        except requests.exceptions.Timeout as e:
            print(f"[ERROR] Timeout error processing {filename}: {e}")
            print("   Check connection or increase timeout settings.")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Connection error processing {filename}: {e}")
            print("   Check network connection and URL validity.")
            return False
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error processing {filename}: {e.response.status_code} {e.response.reason}")
            print(f"   URL: {url}")
            try:
                print(f"   Response: {e.response.text[:500]}...")
            except Exception: pass
            return False
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] General network error processing {filename}: {e}")
            return False
        except IOError as e:
             print(f"[ERROR] File System Error related to {filename} or {target_dir}: {e}")
             return False
        except Exception as e: # Catch-all for unexpected errors
             print(f"[ERROR] An unexpected error occurred processing {filename}: {type(e).__name__} - {e}")
             traceback.print_exc()
             return False

    # --- Main Execution Method ---

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 8):
        """Main function called by ComfyUI to parse list and download files."""
        print(f"\n--- Starting Download Node (Skip Existing: {skip_existing}, Verify Size: {verify_size}) ---")

        chunk_size_bytes = chunk_size_kb * 1024

        # 1. Check internet connection
        if not self._check_internet_connection():
             print("[ERROR] Aborting downloads due to lack of internet connection.")
             return {}

        # 2. Parse the input list
        files_to_download = self._parse_file_list(file_list)
        if not files_to_download:
            print("[INFO] No valid file URLs/paths found in the input list.")
            return {}

        # 3. Create a session for connection reuse and retries
        session = self._create_session_with_retries()

        # 4. Iterate and download each file
        success_count = 0
        failure_count = 0
        total_files = len(files_to_download)
        print(f"[INFO] Preparing to download {total_files} file(s)...")

        for i, file_info in enumerate(files_to_download):
            url = file_info['url']
            target_dir = file_info['target_dir']
            base_filename = os.path.basename(url.split('?')[0].split('#')[0]) or "unknown_file"
            print(f"\n--- Processing file {i+1}/{total_files}: {base_filename} ---")
            print(f"URL: {url}")
            print(f"Target Dir: {target_dir}")

            if self._download_single_file(session, url, target_dir, skip_existing, verify_size, chunk_size_bytes):
                success_count += 1
            else:
                failure_count += 1
                # Error message already printed by _download_single_file

        # 5. Print summary and finish
        print("\n--- Download Summary ---")
        print(f"Total files attempted: {total_files}")
        print(f"Successful/Skipped: {success_count}")
        print(f"Failed: {failure_count}")
        print("--- Download Node Finished ---")

        # Output node requires returning a dictionary
        return {}

class EDLToYouTubeChapters:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edl_file_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("youtube_chapters",)
    FUNCTION = "convert_edl_to_youtube_chapters"

    CATEGORY = "Tools"

    def convert_edl_to_youtube_chapters(self, edl_file_path):
        # Debugging: Print the input file path
        print("Input EDL File Path:", edl_file_path)

        # Remove any surrounding quotation marks from the file path
        edl_file_path = edl_file_path.strip('"')

        # Check if the file exists
        if not os.path.exists(edl_file_path):
            raise ValueError(f"File not found: {edl_file_path}")

        # Read the .edl file content
        with open(edl_file_path, "r", encoding="utf-8") as file:
            edl_text = file.read()

        # Debugging: Print the input text
        print("Input EDL Text:\n", edl_text)

        # Split the EDL text into lines
        lines = edl_text.splitlines()

        # Initialize an empty list to store the chapters
        chapters = []

        # Regular expression to match the timecode (first line of an event)
        timecode_pattern = re.compile(
            r"^\d+\s+\d+\s+V\s+C\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}"
        )

        # Regular expression to match the title (second line of an event)
        title_pattern = re.compile(r"\s*\|C:.*?\|M:(.*?)\|D:.*")

        # Define the start timecode offset (01:00:00:00)
        start_timecode = "01:00:00:00"
        start_hours, start_minutes, start_seconds, start_frames = map(int, start_timecode.split(':'))

        # Iterate through the lines
        i = 0
        while i < len(lines):
            # Skip blank lines and header lines
            if not lines[i].strip() or lines[i].startswith(("TITLE:", "FCM:")):
                print(f"Skipping Line: {lines[i]}")
                i += 1
                continue

            # Check if the current line is a timecode line
            timecode_match = timecode_pattern.match(lines[i])
            if timecode_match:
                timecode = timecode_match.group(1)
                print(f"Matched Timecode Line: {lines[i]}")
                print(f"Extracted Timecode: {timecode}")

                # Check if the next line is a metadata line
                if i + 1 < len(lines):
                    title_match = title_pattern.match(lines[i + 1])
                    if title_match:
                        title = title_match.group(1).strip()
                        print(f"Matched Metadata Line: {lines[i + 1]}")
                        print(f"Extracted Title: {title}")

                        # Convert the timecode to total seconds
                        hours, minutes, seconds, frames = map(int, timecode.split(':'))
                        total_seconds = (
                            (hours - start_hours) * 3600 +  # Subtract start hours
                            (minutes - start_minutes) * 60 +  # Subtract start minutes
                            (seconds - start_seconds)         # Subtract start seconds
                        )

                        # Convert total seconds to YouTube format (MM:SS or HH:MM:SS)
                        if total_seconds >= 3600:
                            # If the video is longer than 1 hour, use HH:MM:SS
                            youtube_timecode = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
                        else:
                            # If the video is shorter than 1 hour, use MM:SS
                            youtube_timecode = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

                        # Append the chapter to the list
                        chapters.append(f"{youtube_timecode} {title}")
                    else:
                        print(f"No Title Match for Line: {lines[i + 1]}")
                else:
                    print("No Metadata Line Found After Timecode Line")

                # Skip the next line (metadata line) since we've already processed it
                i += 2
            else:
                # Skip lines that don't match the timecode pattern
                print(f"Skipping Line (No Timecode Match): {lines[i]}")
                i += 1

        # Join the chapters into a single string with newlines
        youtube_chapters = "\n".join(chapters)

        # Debugging: Print the output to the console
        print("YouTube Chapters Output:\n", youtube_chapters)

        return (youtube_chapters,)
    
class EquirectangularToCubemapFaces:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cube_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "right", "back", "left", "top", "bottom")
    FUNCTION = "convert"
    CATEGORY = "Tools"

    def convert(self, image, cube_size):
        # Convert the input tensor to a numpy array
        image = image.squeeze(0).numpy()  # Remove batch dimension
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit image

        # Convert equirectangular to cubemap using py360convert
        cubemap = py360convert.e2c(image, cube_size, cube_format='dict')

        # Convert each face to a tensor
        front = self.image_to_tensor(cubemap['F'])
        right = self.image_to_tensor(cubemap['R'])
        back = self.image_to_tensor(cubemap['B'])
        left = self.image_to_tensor(cubemap['L'])
        top = self.image_to_tensor(cubemap['U'])
        bottom = self.image_to_tensor(cubemap['D'])

        return (front, right, back, left, top, bottom)

    def image_to_tensor(self, image):
        # Convert numpy array to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
        return image

class CubemapFacesToEquirectangular:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("IMAGE",),
                "right": ("IMAGE",),
                "back": ("IMAGE",),
                "left": ("IMAGE",),
                "top": ("IMAGE",),
                "bottom": ("IMAGE",),
                "output_width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "output_height": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "Tools"

    def convert(self, front, right, back, left, top, bottom, output_width, output_height):
        # Convert input tensors to numpy arrays
        front = self.tensor_to_image(front)
        right = self.tensor_to_image(right)
        back = self.tensor_to_image(back)
        left = self.tensor_to_image(left)
        top = self.tensor_to_image(top)
        bottom = self.tensor_to_image(bottom)

        # Combine faces into a single numpy array in dice format
        face_size = front.shape[0]  # Assuming all faces are square and the same size
        cubemap = np.zeros((face_size * 3, face_size * 4, 3), dtype=np.uint8)

        # Arrange faces in dice format
        cubemap[face_size:2*face_size, :face_size] = left       # Left
        cubemap[face_size:2*face_size, face_size:2*face_size] = front  # Front
        cubemap[face_size:2*face_size, 2*face_size:3*face_size] = right  # Right
        cubemap[face_size:2*face_size, 3*face_size:4*face_size] = back  # Back
        cubemap[:face_size, face_size:2*face_size] = top       # Top
        cubemap[2*face_size:3*face_size, face_size:2*face_size] = bottom  # Bottom

        # Convert cubemap to equirectangular using py360convert
        equirectangular = py360convert.c2e(cubemap, output_height, output_width, cube_format="dice")

        # Convert the equirectangular image back to a tensor
        equirectangular = np.array(equirectangular).astype(np.float32) / 255.0
        equirectangular = torch.from_numpy(equirectangular).unsqueeze(0)  # Add batch dimension

        return (equirectangular,)

    def tensor_to_image(self, tensor):
        # Convert tensor to numpy array
        tensor = tensor.squeeze(0).numpy()  # Remove batch dimension
        tensor = (tensor * 255).astype(np.uint8)  # Convert to 8-bit image
        return tensor
    
class Qwen2:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                    ],
                    {"default": "Qwen2.5-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
    ):
        if not prompt.strip():
            return ("Error: Prompt input is empty.",)

        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.tokenizer
                del self.model
                self.tokenizer = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return result
