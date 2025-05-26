# Standard library imports
import os
import time
import traceback

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Local imports (ComfyUI specific)
# import folder_paths

class TS_DownloadFilesNode:
    """
    A ComfyUI node to download files from a list of URLs with support for resume,
    retries, size verification, and internet connection check.
    """
    RETURN_TYPES = ()
    FUNCTION = "execute_downloads"
    OUTPUT_NODE = True
    CATEGORY = "Tools/TS_IO"

    @classmethod
    def INPUT_TYPES(cls):
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
                    "max": 65536,
                    "step": 1,
                    "description": "Download chunk size in Kilobytes (KB). Smaller values might be better for unstable connections."
                }),
            },
        }

    def _check_internet_connection(self, timeout=5):
        try:
            print("[INFO] TS_DownloadNode: Checking internet connection...")
            requests.head("https://1.1.1.1", timeout=timeout, allow_redirects=True)
            print("[INFO] TS_DownloadNode: Internet connection check: OK")
            return True
        except (requests.ConnectionError, requests.Timeout):
            print("[ERROR] TS_DownloadNode: Internet connection check: FAILED. No internet connection detected.")
            return False
        except requests.RequestException as e:
            print(f"[WARN] TS_DownloadNode: Internet connection check: Error ({e}). Assuming connection might be unstable.")
            return True

    def _create_session_with_retries(self):
        session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['HEAD', 'GET'])
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({'User-Agent': 'ComfyUI-TS_DownloadNode/1.0'})
        return session

    def _parse_file_list(self, file_list_text):
        files = []
        lines = file_list_text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                url, target_path = parts[0].strip(), parts[1].strip()
                if not url.startswith(('http://', 'https://')):
                    print(f"[WARN] TS_DownloadNode: Skipping line {i+1}: Invalid URL format '{url}'")
                    continue
                files.append({'url': url, 'target_dir': target_path})
            else:
                print(f"[WARN] TS_DownloadNode: Skipping line {i+1}: Invalid format. Expected 'URL /path/to/save_directory'. Found: '{line}'")
        return files

    def _download_single_file(self, session, url, target_dir, skip_existing=True, verify_size=True, chunk_size_bytes=8192):
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            print(f"[ERROR] TS_DownloadNode: Error creating directory {target_dir}: {e}")
            return False

        filename = os.path.basename(url.split('?')[0].split('#')[0])
        if not filename:
            filename = f"downloaded_file_{int(time.time())}"
            print(f"[WARN] TS_DownloadNode: Could not determine filename from URL '{url}'. Using '{filename}'.")

        local_file_path = os.path.join(target_dir, filename)
        temp_file_path = local_file_path + ".part"

        try:
            print(f"[INFO] TS_DownloadNode: Fetching file info for: {filename}...")
            response_head = session.head(url, allow_redirects=True, timeout=(10, 20))
            response_head.raise_for_status()
            remote_file_size = int(response_head.headers.get('content-length', -1))
            accept_ranges = response_head.headers.get('accept-ranges', 'none').lower()
            can_resume = (accept_ranges == 'bytes') and (remote_file_size > 0)

            if remote_file_size == -1:
                print(f"[WARN] TS_DownloadNode: Server did not provide Content-Length for {filename}. Cannot verify size or reliably resume.")
                can_resume = False; verify_size = False
            print(f"[INFO] TS_DownloadNode: Remote file size: {'{:,}'.format(remote_file_size) if remote_file_size > 0 else 'Unknown'}. Resumable: {can_resume}")

            local_size = 0
            if skip_existing and os.path.exists(local_file_path):
                local_size = os.path.getsize(local_file_path)
                if verify_size and remote_file_size > 0:
                    if local_size == remote_file_size:
                        print(f"[OK] TS_DownloadNode: File '{filename}' already exists and size matches. Skipping.")
                        return True
                    else: print(f"[WARN] TS_DownloadNode: File '{filename}' exists but size mismatch (local: {local_size}, remote: {remote_file_size}). Re-downloading.")
                else:
                    print(f"[OK] TS_DownloadNode: File '{filename}' already exists. Skipping (size check disabled or not possible).")
                    return True

            resume_byte_pos = 0; headers = {}; file_mode = 'wb'
            if can_resume and os.path.exists(temp_file_path):
                resume_byte_pos = os.path.getsize(temp_file_path)
                if resume_byte_pos < remote_file_size:
                    print(f"[INFO] TS_DownloadNode: Resuming download for '{filename}' from byte {resume_byte_pos}"); headers["Range"] = f"bytes={resume_byte_pos}-"; file_mode = 'ab'
                elif resume_byte_pos == remote_file_size: print(f"[INFO] TS_DownloadNode: Partial file '{filename}.part' found and seems complete. Verifying and finalizing.")
                else:
                    print(f"[WARN] TS_DownloadNode: Partial file '{filename}.part' ({resume_byte_pos} bytes) is larger than remote file ({remote_file_size} bytes). Restarting download.")
                    try: os.remove(temp_file_path)
                    except OSError as e: print(f"[WARN] TS_DownloadNode: Could not remove oversized partial file: {e}")
                    resume_byte_pos = 0; file_mode = 'wb'
            elif os.path.exists(temp_file_path):
                 print(f"[WARN] TS_DownloadNode: Found existing partial file '{filename}.part' but cannot resume. Restarting download.")
                 try: os.remove(temp_file_path)
                 except OSError as e: print(f"[WARN] TS_DownloadNode: Could not remove old partial file: {e}")
                 resume_byte_pos = 0; file_mode = 'wb'

            needs_download = True
            if skip_existing and os.path.exists(local_file_path):
                 if not (verify_size and remote_file_size > 0 and local_size != remote_file_size): needs_download = False
            if can_resume and os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) == remote_file_size: needs_download = False

            if needs_download:
                print(f"[INFO] TS_DownloadNode: Downloading '{filename}' to '{target_dir}'...")
                response_get = session.get(url, stream=True, headers=headers, timeout=(15, 60), allow_redirects=True)
                response_get.raise_for_status()
                if resume_byte_pos > 0 and response_get.status_code == 200:
                    print(f"[WARN] TS_DownloadNode: Server did not honor Range request for '{filename}' (Status 200). Restarting download from beginning.")
                    resume_byte_pos = 0; file_mode = 'wb'
                total_size_for_tqdm = remote_file_size if remote_file_size > 0 else int(response_get.headers.get('content-length', 0))
                try:
                    with open(temp_file_path, file_mode) as file, tqdm(
                        total=total_size_for_tqdm if total_size_for_tqdm > 0 else None, unit='B', unit_scale=True, desc=f"'{filename}'",
                        initial=resume_byte_pos, mininterval=0.5, ncols=100, unit_divisor=1024,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]', disable=False
                    ) as progress_bar:
                        if file_mode == 'wb' and progress_bar.n > 0: progress_bar.reset(total=total_size_for_tqdm if total_size_for_tqdm > 0 else None)
                        for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                            if chunk: file.write(chunk); progress_bar.update(len(chunk))
                        if total_size_for_tqdm > 0 and progress_bar.n < total_size_for_tqdm:
                             print(f"[WARN] TS_DownloadNode: Download stream ended prematurely for '{filename}'. Expected {total_size_for_tqdm}, got {progress_bar.n}.")
                             return False
                except (requests.exceptions.ChunkedEncodingError, requests.exceptions.RequestException) as e_stream:
                    print(f"[ERROR] TS_DownloadNode: Error during download stream for '{filename}': {e_stream}\n[INFO] ...Partial file saved to '{temp_file_path}' for potential resume.")
                    return False
                except IOError as e_io: print(f"[ERROR] TS_DownloadNode: IO Error writing file chunk for '{filename}': {e_io}"); return False
                except Exception as e_inner: print(f"[ERROR] TS_DownloadNode: Unexpected error during download stream/write for '{filename}': {e_inner}"); traceback.print_exc(); return False

            if os.path.exists(temp_file_path):
                print(f"[INFO] TS_DownloadNode: Verifying and finalizing '{filename}'...")
                current_size = os.path.getsize(temp_file_path)
                if verify_size and remote_file_size > 0 and current_size != remote_file_size:
                    print(f"[ERROR] TS_DownloadNode: Size mismatch after download! Temp file '{temp_file_path}' size ({current_size}) != Expected size ({remote_file_size}). Download failed.")
                    return False
                try:
                    if file_mode == 'wb' and os.path.exists(local_file_path):
                         print(f"[INFO] TS_DownloadNode: Removing existing destination file '{local_file_path}' before renaming.")
                         os.remove(local_file_path)
                    os.rename(temp_file_path, local_file_path)
                    print(f"[OK] TS_DownloadNode: Download complete and verified: '{local_file_path}'")
                    if verify_size and remote_file_size > 0:
                         final_size = os.path.getsize(local_file_path)
                         if final_size != remote_file_size:
                              print(f"[ERROR] TS_DownloadNode: CRITICAL: Size mismatch AFTER rename! '{local_file_path}' size ({final_size}) != Expected ({remote_file_size}). File may be corrupt.")
                              return False
                    return True
                except OSError as e_rename: print(f"[ERROR] TS_DownloadNode: Error renaming temporary file '{temp_file_path}' to '{local_file_path}': {e_rename}"); return False
            elif os.path.exists(local_file_path): return True
            else: print(f"[ERROR] TS_DownloadNode: Unknown state: Neither temporary nor final file exists for '{filename}' after processing."); return False
        except requests.exceptions.Timeout as e: print(f"[ERROR] TS_DownloadNode: Timeout error processing {filename}: {e}"); return False
        except requests.exceptions.ConnectionError as e: print(f"[ERROR] TS_DownloadNode: Connection error processing {filename}: {e}"); return False
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] TS_DownloadNode: HTTP error processing {filename}: {e.response.status_code} {e.response.reason} (URL: {url})")
            try: print(f"   Response: {e.response.text[:500]}...")
            except Exception: pass
            return False
        except requests.exceptions.RequestException as e: print(f"[ERROR] TS_DownloadNode: General network error processing {filename}: {e}"); return False
        except IOError as e: print(f"[ERROR] TS_DownloadNode: File System Error related to {filename} or {target_dir}: {e}"); return False
        except Exception as e: print(f"[ERROR] TS_DownloadNode: An unexpected error occurred processing {filename}: {type(e).__name__} - {e}"); traceback.print_exc(); return False

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 8):
        print(f"\n--- Starting TS_DownloadNode (Skip Existing: {skip_existing}, Verify Size: {verify_size}) ---")
        chunk_size_bytes = chunk_size_kb * 1024
        if not self._check_internet_connection():
             print("[ERROR] TS_DownloadNode: Aborting downloads due to lack of internet connection.")
             return {}
        files_to_download = self._parse_file_list(file_list)
        if not files_to_download: print("[INFO] TS_DownloadNode: No valid file URLs/paths found in the input list."); return {}
        session = self._create_session_with_retries()
        success_count = 0; failure_count = 0; total_files = len(files_to_download)
        print(f"[INFO] TS_DownloadNode: Preparing to download {total_files} file(s)...")
        for i, file_info in enumerate(files_to_download):
            url, target_dir = file_info['url'], file_info['target_dir']
            base_filename = os.path.basename(url.split('?')[0].split('#')[0]) or "unknown_file"
            print(f"\n--- TS_DownloadNode: Processing file {i+1}/{total_files}: {base_filename} ---")
            print(f"URL: {url}\nTarget Dir: {target_dir}")
            if self._download_single_file(session, url, target_dir, skip_existing, verify_size, chunk_size_bytes): success_count += 1
            else: failure_count += 1
        print(f"\n--- TS_DownloadNode: Download Summary ---\nTotal files attempted: {total_files}\nSuccessful/Skipped: {success_count}\nFailed: {failure_count}\n--- TS_DownloadNode Finished ---")
        return {}

NODE_CLASS_MAPPINGS = {
    "TS Files Downloader": TS_DownloadFilesNode # Ключ оставлен оригинальным
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Files Downloader": "TS Files Downloader"
}