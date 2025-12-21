# Standard library imports
import os
import time
import traceback
import re
import socket
import zipfile

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

try:
    from requests.utils import unquote as requests_unquote
except ImportError:
    from urllib.parse import unquote as requests_unquote

# ComfyUI Imports
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

class TS_DownloadFilesNode:
    """
    A ComfyUI node to download files.
    Features: 
    - Auto-Unzip (Safe Overwrite)
    - UI Progress Bar & TQDM
    - Resume & Retries
    - Dropbox / Mirrors / Proxies support
    - Smart Size Detection
    """
    RETURN_TYPES = ()
    FUNCTION = "execute_downloads"
    OUTPUT_NODE = True
    CATEGORY = "Tools/TS_IO"

    @classmethod
    def INPUT_TYPES(cls):
        # ВАЖНО: Порядок полей сохранен.
        return {
            "required": {
                "file_list": ("STRING", {
                    "default": "https://www.dropbox.com/sh/example_folder?dl=0 /path/to/models\nhttps://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors /path/to/checkpoints",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "description": "List of files. Format: URL /path/to/save_dir",
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "description": "Skip download if file exists and size matches.",
                 }),
                "verify_size": ("BOOLEAN", {
                     "default": True,
                     "description": "Verify file size against Content-Length header.",
                 }),
                "chunk_size_kb": ("INT", {
                    "default": 4096, 
                    "min": 1,
                    "max": 65536, 
                    "step": 1,
                    "description": "Download chunk size in KB (4096 = 4MB)."
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "Hugging Face token (starts with hf_)."
                }),
                "hf_domain": ("STRING", {
                    "default": "huggingface.co, hf-mirror.com",
                    "multiline": False,
                    "description": "List of HF mirrors separated by comma."
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "Proxy URL (e.g., http://127.0.0.1:7890)."
                }),
                "modelscope_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "ModelScope Access Token."
                }),
                # Новый параметр в конце
                "unzip_after_download": ("BOOLEAN", {
                     "default": False,
                     "description": "If true, unzips .zip file to target_dir (merges/overwrites) and deletes archive.",
                 }),
            }
        }

    def _create_session_with_retries(self, proxy_url=None):
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['HEAD', 'GET'])
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })

        if proxy_url and proxy_url.strip():
            print(f"[INFO] TS_DownloadNode: Proxy enabled: {proxy_url}")
            session.proxies = {
                'http': proxy_url.strip(),
                'https': proxy_url.strip(),
            }
        return session

    def _check_internet_connection(self, proxy=None, timeout=5):
        proxies = {"http": proxy, "https": proxy} if proxy else None
        targets = ["https://8.8.8.8", "https://1.1.1.1", "https://www.modelscope.cn"]
        for url in targets:
            try:
                requests.head(url, timeout=timeout, proxies=proxies)
                return True
            except requests.RequestException:
                continue
        return False

    def _select_best_mirror(self, session, domain_list_str):
        if not domain_list_str: return "huggingface.co"
        domains = [d.strip() for d in domain_list_str.split(',') if d.strip()]
        if not domains: return "huggingface.co"
        if len(domains) == 1: return domains[0]

        print(f"[INFO] Checking mirrors: {domains}")
        for domain in domains:
            clean_domain = domain.replace("https://", "").replace("http://", "").strip("/")
            test_url = f"https://{clean_domain}"
            try:
                response = session.head(test_url, timeout=3, allow_redirects=True)
                if response.status_code < 500:
                    print(f"[INFO] Mirror '{clean_domain}' is ACTIVE.")
                    return clean_domain
            except requests.RequestException:
                continue
        return domains[0]

    def _get_headers_for_url(self, url, hf_token, ms_token):
        headers = {}
        if "huggingface.co" in url or "hf-mirror.com" in url or "hf-" in url:
            if hf_token and hf_token.strip():
                headers["Authorization"] = f"Bearer {hf_token.strip()}"
        elif "modelscope.cn" in url:
            headers["Referer"] = "https://www.modelscope.cn/"
            if ms_token and ms_token.strip():
                headers["Authorization"] = f"Bearer {ms_token.strip()}"
        return headers

    def _replace_hf_domain(self, url, target_domain):
        if not target_domain or target_domain.strip() == "huggingface.co":
            return url
        clean_domain = target_domain.replace("https://", "").replace("http://", "").strip("/")
        pattern = r"(https?://)(www\.)?huggingface\.co"
        if re.search(pattern, url):
            return re.sub(pattern, f"\\1{clean_domain}", url)
        return url

    def _process_dropbox_url(self, url):
        if "dropbox.com" in url:
            if "dl=0" in url:
                return url.replace("dl=0", "dl=1")
            if "dl=" not in url:
                return url + ("&dl=1" if "?" in url else "?dl=1")
        return url

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
                    print(f"[WARN] Line {i+1}: Invalid URL.")
                    continue
                target_path = os.path.abspath(os.path.expanduser(target_path))
                files.append({'url': url, 'target_dir': target_path})
            else:
                print(f"[WARN] TS_DownloadNode: Line {i+1}: Invalid format.")
        return files

    def _get_filename_from_headers(self, response):
        content_disposition = response.headers.get('content-disposition')
        if not content_disposition: return None
        fn_match_utf8 = re.search(r"filename\*=\s*UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if fn_match_utf8: return requests_unquote(fn_match_utf8.group(1).strip('" '))
        fn_match_plain = re.search(r'filename="?([^"]+)"?', content_disposition, re.IGNORECASE)
        if fn_match_plain: return requests_unquote(fn_match_plain.group(1).strip('" '))
        return None

    def _extract_zip(self, zip_path, extract_to):
        """Extracts a zip file with error handling for permissions."""
        print(f"[INFO] Auto-Unzip: Extracting '{os.path.basename(zip_path)}' to '{extract_to}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # zipfile.extractall overwrites existing files safely
                zip_ref.extractall(extract_to)
            
            print(f"[OK] Extraction complete. Deleting archive.")
            try:
                os.remove(zip_path)
            except OSError as e:
                print(f"[WARN] Could not delete archive '{zip_path}' after extraction: {e}")
            return True
        
        except zipfile.BadZipFile:
            print(f"[ERROR] Extraction failed: The file is corrupted or not a valid zip.")
            return False
        except PermissionError:
            print(f"[ERROR] Extraction failed: Permission denied. Some files in '{extract_to}' might be in use (loaded model?).")
            return False
        except Exception as e:
            print(f"[ERROR] Extraction unexpected error: {e}")
            return False

    def _download_single_file(self, session, url, target_dir, skip_existing, verify_size, chunk_size_bytes, hf_domain_active, hf_token, ms_token, unzip_after_download):
        try:
            processed_url = self._replace_hf_domain(url, hf_domain_active)
            processed_url = self._process_dropbox_url(processed_url)

            os.makedirs(target_dir, exist_ok=True)
            domain_headers = self._get_headers_for_url(processed_url, hf_token, ms_token)
            
            # --- Phase 1: HEAD ---
            filename_from_url = os.path.basename(requests_unquote(url.split('?')[0].split('#')[0]))
            final_filename = None

            print(f"[INFO] Connecting to: {processed_url}")
            try:
                response_head = session.head(processed_url, headers=domain_headers, allow_redirects=True, timeout=(10, 30))
                response_head.raise_for_status()
            except requests.RequestException as e:
                print(f"[ERROR] Connection failed: {e}")
                return False

            final_direct_url = response_head.url 
            remote_file_size = int(response_head.headers.get('content-length', -1))
            accept_ranges = response_head.headers.get('accept-ranges', 'none').lower()
            can_resume = (accept_ranges == 'bytes') and (remote_file_size > 0)

            final_filename = self._get_filename_from_headers(response_head)
            if not final_filename: final_filename = filename_from_url
            if not final_filename or final_filename == "/": final_filename = f"downloaded_file_{int(time.time())}"
            final_filename = re.sub(r'[<>:"/\\|?*]', '_', final_filename)
            
            local_file_path = os.path.join(target_dir, final_filename)
            temp_file_path = local_file_path + ".part"

            print(f"[INFO] File: '{final_filename}' | Size: {remote_file_size if remote_file_size > 0 else 'Unknown'}")

            # --- Phase 2: Check Existing (Early) ---
            if remote_file_size > 0:
                if skip_existing and os.path.exists(local_file_path):
                    if verify_size:
                        if os.path.getsize(local_file_path) == remote_file_size:
                            print(f"[OK] Verified existing file. Skipping.")
                            # Check unzip if user requests it even for existing files
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                self._extract_zip(local_file_path, target_dir)
                            return True
                        else:
                            print(f"[WARN] Size mismatch. Redownloading.")
                    else:
                        print(f"[OK] File exists. Skipping.")
                        return True
            else:
                 if skip_existing and os.path.exists(local_file_path) and not verify_size:
                     print(f"[WARN] Size unknown, skipping.")
                     return True
            
            # --- Phase 3: Resume Logic ---
            resume_byte_pos = 0
            file_mode = 'wb'
            if can_resume and os.path.exists(temp_file_path) and remote_file_size > 0:
                resume_byte_pos = os.path.getsize(temp_file_path)
                if resume_byte_pos < remote_file_size:
                    print(f"[INFO] Resuming from {resume_byte_pos} bytes.")
                    file_mode = 'ab'
                elif resume_byte_pos == remote_file_size:
                    print(f"[INFO] Temp file seems complete.")
                else:
                    os.remove(temp_file_path); resume_byte_pos = 0
            elif os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            # --- Phase 4: Download ---
            perform_network_download = True
            if file_mode == 'ab' and resume_byte_pos == remote_file_size and remote_file_size > 0:
                perform_network_download = False
            
            if perform_network_download:
                req_headers = domain_headers.copy()
                if resume_byte_pos > 0 and file_mode == 'ab':
                    req_headers["Range"] = f"bytes={resume_byte_pos}-"

                response_get = session.get(final_direct_url, stream=True, headers=req_headers, timeout=(15, 300), allow_redirects=True)
                
                if resume_byte_pos > 0 and file_mode == 'ab' and response_get.status_code == 200:
                    print(f"[WARN] Server rejected resume. Restarting.")
                    resume_byte_pos = 0; file_mode = 'wb'
                    if os.path.exists(temp_file_path): os.remove(temp_file_path)
                
                response_get.raise_for_status()
                
                # Late size check
                total_size = remote_file_size if remote_file_size > 0 else int(response_get.headers.get('content-length', 0))
                
                if remote_file_size == -1 and total_size > 0:
                    print(f"[INFO] Size detected via GET: {total_size} bytes.")
                    if skip_existing and os.path.exists(local_file_path):
                        if verify_size:
                            if os.path.getsize(local_file_path) == total_size:
                                print(f"[OK] File verified (Late Check). Skipping.")
                                response_get.close()
                                if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                    self._extract_zip(local_file_path, target_dir)
                                return True

                comfy_pbar = None
                if ProgressBar and total_size > 0:
                    comfy_pbar = ProgressBar(total_size)
                
                downloaded_since_update = 0
                ui_update_threshold = 1 * 1024 * 1024 

                with open(temp_file_path, file_mode) as f, tqdm(
                    total=total_size if total_size > 0 else None,
                    unit='B', unit_scale=True, desc=f"DL: {final_filename}",
                    initial=resume_byte_pos if file_mode == 'ab' else 0,
                    mininterval=1.0, ncols=100, unit_divisor=1024
                ) as pbar:
                    if comfy_pbar and resume_byte_pos > 0: comfy_pbar.update(resume_byte_pos)

                    for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                        if chunk:
                            f.write(chunk)
                            chunk_len = len(chunk)
                            pbar.update(chunk_len)
                            if comfy_pbar:
                                downloaded_since_update += chunk_len
                                if downloaded_since_update >= ui_update_threshold:
                                    comfy_pbar.update(downloaded_since_update)
                                    downloaded_since_update = 0
                    if comfy_pbar and downloaded_since_update > 0: comfy_pbar.update(downloaded_since_update)

            # --- Phase 5: Finalize ---
            if not os.path.exists(temp_file_path):
                if os.path.exists(local_file_path) and verify_size and (remote_file_size > 0 or total_size > 0):
                     target_size = remote_file_size if remote_file_size > 0 else total_size
                     if os.path.getsize(local_file_path) == target_size:
                         return True
                return False

            final_expected_size = remote_file_size if remote_file_size > 0 else (total_size if 'total_size' in locals() else -1)
            if verify_size and final_expected_size > 0:
                 if os.path.getsize(temp_file_path) != final_expected_size:
                     print(f"[ERROR] Incomplete download."); return False
            
            if os.path.exists(local_file_path):
                try: os.remove(local_file_path)
                except OSError: pass 
            
            os.rename(temp_file_path, local_file_path)
            print(f"[OK] Saved: {local_file_path}")

            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                self._extract_zip(local_file_path, target_dir)

            return True

        except Exception as e:
            print(f"[ERROR] Download failed: {e}"); return False

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 4096, hf_token: str = "", hf_domain: str = "huggingface.co, hf-mirror.com", proxy_url: str = "", modelscope_token: str = "", unzip_after_download: bool = False):
        print(f"\n--- TS_DownloadNode v2.10 Started ---")
        chunk_size_bytes = max(1024, chunk_size_kb * 1024) 

        if not self._check_internet_connection(proxy=proxy_url if proxy_url else None):
            print("[ERROR] No internet connection detected.")
        
        files_to_download = self._parse_file_list(file_list)
        if not files_to_download: return ()

        session = self._create_session_with_retries(proxy_url)
        active_mirror = self._select_best_mirror(session, hf_domain)
        print(f"[INFO] Using HF Mirror: '{active_mirror}'")
        
        success = 0; failed = 0
        for file_info in files_to_download:
            if self._download_single_file(session, file_info['url'], file_info['target_dir'], skip_existing, verify_size, chunk_size_bytes, active_mirror, hf_token, modelscope_token, unzip_after_download):
                success += 1
            else:
                failed += 1
        
        print(f"\n--- Done. Success: {success}, Failed: {failed} ---")
        return ()

NODE_CLASS_MAPPINGS = {
    "TS Files Downloader": TS_DownloadFilesNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Files Downloader": "TS Files Downloader (Ultimate)"
}