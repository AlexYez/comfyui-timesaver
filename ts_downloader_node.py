# Standard library imports
import os
import time
import traceback
import re # Added for Content-Disposition parsing

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
try:
    from requests.utils import unquote as requests_unquote
except ImportError: # Fallback for older requests versions if any, though unquote should be there
    from urllib.parse import unquote as requests_unquote


# Local imports (ComfyUI specific)
# import folder_paths

class TS_DownloadFilesNode:
    """
    A ComfyUI node to download files from a list of URLs with support for resume,
    retries, size verification, internet connection check, and Hugging Face token.
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
                    "default": 4096, # Default changed to 4096KB (4MB)
                    "min": 1,
                    "max": 65536, # Max 64MB chunk
                    "step": 1,
                    "description": "Download chunk size in Kilobytes (KB). Larger values (e.g., 1024-8192KB) for stable/fast connections & large files. Smaller for unstable."
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "Optional Hugging Face token (read or write) for private/gated models. Leave empty if not needed."
                }),
            }
        }

    def _check_internet_connection(self, timeout=5):
        try:
            print("[INFO] TS_DownloadNode: Checking internet connection...")
            # Using a reliable public DNS server for the check
            requests.head("https://1.1.1.1", timeout=timeout, allow_redirects=True)
            print("[INFO] TS_DownloadNode: Internet connection check: OK")
            return True
        except (requests.ConnectionError, requests.Timeout):
            print("[ERROR] TS_DownloadNode: Internet connection check: FAILED. No internet connection detected.")
            return False
        except requests.RequestException as e:
            print(f"[WARN] TS_DownloadNode: Internet connection check: Error ({e}). Assuming connection might be unstable.")
            return True # Proceed with caution

    def _create_session_with_retries(self, hf_token=None):
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1, # e.g., sleeps for 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504], # HTTP status codes to retry on
            allowed_methods=frozenset(['HEAD', 'GET']) # Only retry for idempotent methods
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({'User-Agent': 'ComfyUI-TS_DownloadNode/1.1'}) # Version bump for clarity
        if hf_token and hf_token.strip():
            print("[INFO] TS_DownloadNode: Hugging Face token provided. Adding to session headers.")
            session.headers.update({'Authorization': f'Bearer {hf_token.strip()}'})
        return session

    def _parse_file_list(self, file_list_text):
        files = []
        lines = file_list_text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines and comments
                continue
            parts = line.split(maxsplit=1) # Split only on the first space
            if len(parts) == 2:
                url, target_path = parts[0].strip(), parts[1].strip()
                if not url.startswith(('http://', 'https://')):
                    print(f"[WARN] TS_DownloadNode: Skipping line {i+1}: Invalid URL format '{url}'. Must start with http:// or https://.")
                    continue
                if not target_path:
                    print(f"[WARN] TS_DownloadNode: Skipping line {i+1}: Target directory path is empty for URL '{url}'.")
                    continue
                files.append({'url': url, 'target_dir': target_path})
            else:
                print(f"[WARN] TS_DownloadNode: Skipping line {i+1}: Invalid format. Expected 'URL /path/to/save_directory'. Found: '{line}'")
        return files

    def _download_single_file(self, session, url, target_dir, skip_existing=True, verify_size=True, chunk_size_bytes=4194304): # Default chunk_size_bytes updated to 4MB (4096 * 1024)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            print(f"[ERROR] TS_DownloadNode: Error creating directory {target_dir}: {e}")
            return False

        # --- Phase 1: Get File Info (HEAD request) ---
        filename_from_url = os.path.basename(requests_unquote(url.split('?')[0].split('#')[0]))
        final_filename = None # Will be determined after HEAD request

        try:
            print(f"[INFO] TS_DownloadNode: Fetching file info for URL: {url}...")
            response_head = session.head(url, allow_redirects=True, timeout=(10, 30)) # connect_timeout=10s, read_timeout=30s
            response_head.raise_for_status() # Check for HTTP errors (4xx, 5xx)

            remote_file_size = int(response_head.headers.get('content-length', -1))
            accept_ranges = response_head.headers.get('accept-ranges', 'none').lower()
            can_resume = (accept_ranges == 'bytes') and (remote_file_size > 0)

            # Determine filename: 1. Content-Disposition, 2. URL, 3. Fallback
            content_disposition = response_head.headers.get('content-disposition')
            if content_disposition:
                # Example: "attachment; filename=\"fname.ext\"; filename*=UTF-8''fname.ext"
                # More robust regex for filename extraction, preferring filename*
                fn_match_utf8 = re.search(r"filename\*=\s*UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
                if fn_match_utf8:
                    final_filename = requests_unquote(fn_match_utf8.group(1).strip('" '))
                else:
                    fn_match_plain = re.search(r'filename="?([^"]+)"?', content_disposition, re.IGNORECASE)
                    if fn_match_plain:
                        final_filename = requests_unquote(fn_match_plain.group(1).strip('" '))
            
            if not final_filename:
                final_filename = filename_from_url

            if not final_filename or final_filename == "/": # If URL ends with / or filename is empty
                final_filename = f"downloaded_file_{int(time.time())}"
                print(f"[WARN] TS_DownloadNode: Could not determine a valid filename from URL or Content-Disposition. Using '{final_filename}'.")
            
            print(f"[INFO] TS_DownloadNode: Determined filename: '{final_filename}'")
            local_file_path = os.path.join(target_dir, final_filename)
            temp_file_path = local_file_path + ".part"
            temp_file_basename = os.path.basename(temp_file_path) # For logging

            if remote_file_size == -1:
                print(f"[WARN] TS_DownloadNode: Server did not provide Content-Length for '{final_filename}'. Cannot verify size or reliably resume.")
                can_resume = False
                verify_size = False # Cannot verify if server doesn't tell us the size
            print(f"[INFO] TS_DownloadNode: Remote file size: {'{:,}'.format(remote_file_size) if remote_file_size > 0 else 'Unknown'}. Resumable: {can_resume}")

            # --- Phase 2: Check Existing Final File ---
            if skip_existing and os.path.exists(local_file_path):
                local_size = os.path.getsize(local_file_path)
                if verify_size and remote_file_size > 0: # verify_size can be False if Content-Length was missing
                    if local_size == remote_file_size:
                        print(f"[OK] TS_DownloadNode: File '{final_filename}' already exists and size matches. Skipping.")
                        return True
                    else:
                        print(f"[WARN] TS_DownloadNode: File '{final_filename}' exists but size mismatch (local: {local_size}, remote: {remote_file_size}). Re-downloading.")
                else: # Size verification disabled or not possible
                    print(f"[OK] TS_DownloadNode: File '{final_filename}' already exists. Skipping (size check disabled or not possible).")
                    return True
            
            # --- Phase 3: Prepare for Download (Resume Logic) ---
            resume_byte_pos = 0
            file_mode = 'wb' # Default to write new file

            if can_resume and os.path.exists(temp_file_path):
                resume_byte_pos = os.path.getsize(temp_file_path)
                if resume_byte_pos < remote_file_size:
                    print(f"[INFO] TS_DownloadNode: Resuming download for '{final_filename}' from byte {resume_byte_pos}.")
                    file_mode = 'ab' # Append to existing .part file
                elif resume_byte_pos == remote_file_size:
                    print(f"[INFO] TS_DownloadNode: Partial file '{temp_file_basename}' found and seems complete. Will verify and finalize.")
                    # No network download needed, proceed to finalization
                else: # temp_file_path is larger than remote file
                    print(f"[WARN] TS_DownloadNode: Partial file '{temp_file_basename}' ({resume_byte_pos} bytes) is larger than remote file ({remote_file_size} bytes). Restarting download.")
                    try: os.remove(temp_file_path)
                    except OSError as e_rm: print(f"[WARN] TS_DownloadNode: Could not remove oversized partial file '{temp_file_basename}': {e_rm}")
                    resume_byte_pos = 0
            elif os.path.exists(temp_file_path): # Cannot resume, but .part exists
                print(f"[WARN] TS_DownloadNode: Found existing partial file '{temp_file_basename}' but cannot resume (e.g. server doesn't support it or size unknown). Restarting download.")
                try: os.remove(temp_file_path)
                except OSError as e_rm: print(f"[WARN] TS_DownloadNode: Could not remove old partial file '{temp_file_basename}': {e_rm}")
                resume_byte_pos = 0 # Ensure we start from scratch

            # --- Phase 4: Perform Network Download if Needed ---
            perform_network_download = True
            if file_mode == 'ab' and resume_byte_pos == remote_file_size and remote_file_size > 0: # Already resumed and complete
                perform_network_download = False
            elif os.path.exists(temp_file_path) and remote_file_size > 0 and os.path.getsize(temp_file_path) == remote_file_size:
                 # This case handles if .part was complete but not through 'ab' mode (e.g. can_resume was false initially but .part was there)
                print(f"[INFO] TS_DownloadNode: Full temporary file '{temp_file_basename}' found matching remote size. Skipping network ops.")
                perform_network_download = False
            
            if perform_network_download:
                print(f"[INFO] TS_DownloadNode: Starting download of '{final_filename}' to '{target_dir}' (mode: {file_mode}, resume_pos: {resume_byte_pos})...")
                headers_get = {}
                if resume_byte_pos > 0 and file_mode == 'ab': # Only send Range if we are actually resuming
                    headers_get["Range"] = f"bytes={resume_byte_pos}-"

                # Adjust timeouts for GET request (connect_timeout=15s, read_timeout=300s i.e. 5 minutes)
                # This is longer for read_timeout to accommodate large files and slow server starts for a chunk
                response_get = session.get(url, stream=True, headers=headers_get, timeout=(15, 300), allow_redirects=True) 
                
                # Check if server honored Range request if we made one
                if resume_byte_pos > 0 and file_mode == 'ab' and response_get.status_code == 200: # Expected 206 Partial Content
                    print(f"[WARN] TS_DownloadNode: Server did not honor Range request for '{final_filename}' (Status 200). Restarting download from beginning.")
                    resume_byte_pos = 0
                    file_mode = 'wb' # Switch to overwrite mode
                    # Need to reopen file in 'wb' mode, or handle this by seeking if file object already open (not the case here)
                    if os.path.exists(temp_file_path): # Remove previous .part to write fresh
                        try: os.remove(temp_file_path)
                        except OSError as e_rm_part: print(f"[WARN] TS_DownloadNode: Could not remove partial file for restart: {e_rm_part}")
                
                response_get.raise_for_status() # Check for HTTP errors on GET

                # Determine total size for tqdm, preferring HEAD's content-length
                # If HEAD didn't provide it, try GET's content-length (less reliable for progress before download starts)
                total_size_for_tqdm = remote_file_size if remote_file_size > 0 else int(response_get.headers.get('content-length', 0))
                
                try:
                    with open(temp_file_path, file_mode) as file:
                        # If we switched to 'wb' after a failed resume attempt, ensure initial is 0
                        current_initial_tqdm = resume_byte_pos if file_mode == 'ab' else 0
                        
                        with tqdm(
                            total=total_size_for_tqdm if total_size_for_tqdm > 0 else None,
                            unit='B', unit_scale=True, desc=f"'{final_filename}'",
                            initial=current_initial_tqdm, mininterval=0.5, ncols=100, unit_divisor=1024,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                            disable=False # or set to a global verbosity flag
                        ) as progress_bar:
                            for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                                if chunk: # filter out keep-alive new chunks
                                    file.write(chunk)
                                    progress_bar.update(len(chunk))
                            
                            # After loop, check if downloaded amount matches expected (if known)
                            if total_size_for_tqdm > 0 and progress_bar.n < total_size_for_tqdm:
                                print(f"[WARN] TS_DownloadNode: Download stream for '{final_filename}' ended prematurely. Expected {total_size_for_tqdm}, got {progress_bar.n}. File may be incomplete.")
                                # For now, rely on size check in Phase 5.
                except (requests.exceptions.ChunkedEncodingError, requests.exceptions.RequestException) as e_stream:
                    print(f"[ERROR] TS_DownloadNode: Error during download stream for '{final_filename}': {e_stream}")
                    print(f"[INFO] ...Partial file saved to '{temp_file_path}' for potential resume.")
                    return False
                except IOError as e_io:
                    print(f"[ERROR] TS_DownloadNode: IO Error writing file chunk for '{final_filename}': {e_io}")
                    return False
                except Exception as e_inner:
                    print(f"[ERROR] TS_DownloadNode: Unexpected error during download stream/write for '{final_filename}': {e_inner}")
                    traceback.print_exc()
                    return False
            
            # --- Phase 5: Finalize Download (Verify and Rename) ---
            if not os.path.exists(temp_file_path):
                print(f"[ERROR] TS_DownloadNode: Temporary file '{temp_file_basename}' not found after download/processing for '{final_filename}'. This indicates a problem.")
                if os.path.exists(local_file_path) and verify_size and remote_file_size > 0 and os.path.getsize(local_file_path) == remote_file_size:
                    print(f"[WARN] TS_DownloadNode: ...However, final file '{final_filename}' exists and matches size. Assuming it's okay (unexpected).")
                    return True
                return False

            print(f"[INFO] TS_DownloadNode: Verifying and finalizing '{final_filename}' from '{temp_file_basename}'...")
            current_temp_size = os.path.getsize(temp_file_path)

            if verify_size and remote_file_size > 0: # verify_size can be false if Content-Length was missing
                if current_temp_size != remote_file_size:
                    print(f"[ERROR] TS_DownloadNode: Size mismatch after download! Temp file '{temp_file_basename}' size ({current_temp_size}) != Expected remote size ({remote_file_size}). Download failed.")
                    return False
            elif remote_file_size == -1 and current_temp_size == 0 : # No remote size, and we downloaded nothing
                 print(f"[WARN] TS_DownloadNode: Remote file size unknown and downloaded file '{temp_file_basename}' is empty. This might be an error or an empty remote file.")
            
            try:
                # Ensure destination path is clear before renaming
                if os.path.exists(local_file_path):
                    print(f"[INFO] TS_DownloadNode: Removing existing destination file '{local_file_path}' before renaming.")
                    try:
                        os.remove(local_file_path)
                    except OSError as e_rem_final:
                        print(f"[ERROR] TS_DownloadNode: Could not remove existing final file '{local_file_path}': {e_rem_final}. Rename will likely fail.")
                        return False # Critical, cannot proceed with rename if target is blocked
                
                os.rename(temp_file_path, local_file_path)
                print(f"[OK] TS_DownloadNode: Download complete and finalized: '{local_file_path}'")

                # Final check of the renamed file (paranoid mode)
                if verify_size and remote_file_size > 0:
                    final_local_size = os.path.getsize(local_file_path)
                    if final_local_size != remote_file_size:
                        print(f"[ERROR] TS_DownloadNode: CRITICAL: Size mismatch AFTER RENAME! Final file '{local_file_path}' size ({final_local_size}) != Expected remote size ({remote_file_size}). File may be corrupt.")
                        return False
                return True
            except OSError as e_rename:
                print(f"[ERROR] TS_DownloadNode: Error renaming temporary file '{temp_file_basename}' to '{local_file_path}': {e_rename}")
                return False

        # --- Exception Handling for _download_single_file ---
        except requests.exceptions.HTTPError as e_http:
            # response_head or response_get .raise_for_status() failed
            status_code = e_http.response.status_code
            reason = e_http.response.reason
            err_url = e_http.request.url # URL that caused the error
            print(f"[ERROR] TS_DownloadNode: HTTP error processing '{final_filename or filename_from_url}': {status_code} {reason} (URL: {err_url})")
            try:
                # Try to print some of the response body if it's a client error and not too large
                if 400 <= status_code < 500 and e_http.response.text:
                     print(f"   Response body (partial): {e_http.response.text[:500]}...")
            except Exception: pass # Ignore errors trying to print response body
            if status_code == 401 or status_code == 403:
                print(f"[HINT] TS_DownloadNode: This may be a private/gated model. Ensure you have access and provide a Hugging Face token if required.")
            elif status_code == 404:
                 print(f"[HINT] TS_DownloadNode: File not found. Check the URL for typos or if the file was removed/moved.")
            return False
        except requests.exceptions.Timeout as e_timeout:
            print(f"[ERROR] TS_DownloadNode: Timeout error processing '{final_filename or filename_from_url}': {e_timeout}")
            return False
        except requests.exceptions.ConnectionError as e_conn:
            print(f"[ERROR] TS_DownloadNode: Connection error processing '{final_filename or filename_from_url}': {e_conn}")
            return False
        except requests.exceptions.RequestException as e_req: # Catch-all for other requests errors
            print(f"[ERROR] TS_DownloadNode: General network error processing '{final_filename or filename_from_url}': {e_req}")
            return False
        except IOError as e_io_fs: # Filesystem errors not caught during stream
            print(f"[ERROR] TS_DownloadNode: File System Error related to '{final_filename or filename_from_url}' or '{target_dir}': {e_io_fs}")
            return False
        except Exception as e_unexpected: # Catch any other unexpected errors
            print(f"[ERROR] TS_DownloadNode: An unexpected error occurred processing '{final_filename or filename_from_url}': {type(e_unexpected).__name__} - {e_unexpected}")
            traceback.print_exc()
            return False

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 4096, hf_token: str = ""): # Default chunk_size_kb changed to 4096
        print(f"\n--- Starting TS_DownloadNode (Skip Existing: {skip_existing}, Verify Size: {verify_size}, Chunk: {chunk_size_kb}KB) ---")
        
        # Convert chunk size from KB to Bytes, ensure at least 1KB
        chunk_size_bytes = max(1024, chunk_size_kb * 1024) 

        if not self._check_internet_connection():
            print("[ERROR] TS_DownloadNode: Aborting downloads due to lack of internet connection.")
            # ComfyUI expects a dictionary, even for output nodes that don't "return" data via sockets
            return {"ui": {"errors": ["No internet connection detected. Download aborted."]}} 

        files_to_download = self._parse_file_list(file_list)
        if not files_to_download:
            print("[INFO] TS_DownloadNode: No valid file URLs/paths found in the input list.")
            return {} # No files to process

        # Create a session with retry logic and optional HF token
        session = self._create_session_with_retries(hf_token)
        
        success_count = 0
        failure_count = 0
        total_files = len(files_to_download)
        
        print(f"[INFO] TS_DownloadNode: Preparing to download {total_files} file(s)...")

        for i, file_info in enumerate(files_to_download):
            url = file_info['url']
            target_dir = file_info['target_dir']
            
            # Try to get a preliminary filename for logging before HEAD request
            prelim_filename = os.path.basename(requests_unquote(url.split('?')[0].split('#')[0])) or "unknown_file"
            
            print(f"\n--- TS_DownloadNode: Processing file {i+1}/{total_files}: '{prelim_filename}' (from URL: {url}) ---")
            print(f"Target Dir: {target_dir}")
            
            # Pass the calculated chunk_size_bytes to the download method
            if self._download_single_file(session, url, target_dir, skip_existing, verify_size, chunk_size_bytes):
                success_count += 1
            else:
                failure_count += 1
        
        print(f"\n--- TS_DownloadNode: Download Summary ---")
        print(f"Total files attempted: {total_files}")
        print(f"Successful (or skipped): {success_count}")
        print(f"Failed: {failure_count}")
        print(f"--- TS_DownloadNode Finished ---")
        
        return {}

# ComfyUI Mappings
NODE_CLASS_MAPPINGS = {
    "TS Files Downloader": TS_DownloadFilesNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Files Downloader": "TS Files Downloader"
}