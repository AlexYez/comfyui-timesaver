# Standard library imports
import os
import time
import traceback
import re
import socket
import zipfile
from urllib.parse import urlparse # Added for domain extraction

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
    - Offline Mode (Target-based check)
    - Enable/Disable Toggle
    - Auto-Unzip
    - UI Progress Bar
    - Resume / Mirrors / Proxies
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
                "unzip_after_download": ("BOOLEAN", {
                     "default": False,
                     "description": "If true, unzips .zip file to target_dir and deletes archive.",
                 }),
                "enable": ("BOOLEAN", {
                     "default": True,
                     "description": "Enable or disable this node.",
                 }),
            }
        }

    def _create_session_with_retries(self, proxy_url=None):
        session = requests.Session()
        retries = Retry(
            total=3, # Reduced retries for faster checks
            backoff_factor=0.5,
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

    def _replace_hf_domain(self, url, target_domain):
        if not target_domain or target_domain.strip() == "huggingface.co":
            return url
        clean_domain = target_domain.replace("https://", "").replace("http://", "").strip("/")
        pattern = r"(https?://)(www\.)?huggingface\.co"
        if re.search(pattern, url):
            return re.sub(pattern, f"\\1{clean_domain}", url)
        return url

    def _check_connectivity_to_targets(self, file_list_parsed, session, hf_domain_str):
        """
        Checks connectivity ONLY to the domains listed in the file_list.
        Returns True if at least one target is reachable.
        """
        # 1. Select best mirror first (just pick first one for the check logic)
        active_mirror = "huggingface.co"
        if hf_domain_str:
            domains = [d.strip() for d in hf_domain_str.split(',') if d.strip()]
            if domains: active_mirror = domains[0]

        # 2. Extract unique base URLs from the file list
        unique_bases = set()
        for item in file_list_parsed:
            url = item['url']
            # Apply mirror logic to check the ACTUAL target, not the theoretical one
            final_url = self._replace_hf_domain(url, active_mirror)
            try:
                parsed = urlparse(final_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                unique_bases.add(base_url)
            except Exception:
                continue

        if not unique_bases:
            return True # No URLs to check, assume OK to proceed (or fail later)

        print(f"[INFO] Checking connectivity to targets: {list(unique_bases)} ...")
        
        # 3. Check them
        is_online = False
        for base_url in unique_bases:
            try:
                # Fast timeout: 2 seconds connect, 2 seconds read
                session.head(base_url, timeout=(2, 2), allow_redirects=True)
                print(f"[INFO] Target '{base_url}' is REACHABLE.")
                is_online = True
                break # If at least one is up, we proceed
            except requests.RequestException:
                print(f"[WARN] Target '{base_url}' is UNREACHABLE.")
                continue
        
        return is_online

    def _select_best_mirror(self, session, domain_list_str):
        if not domain_list_str: return "huggingface.co"
        domains = [d.strip() for d in domain_list_str.split(',') if d.strip()]
        if not domains: return "huggingface.co"
        if len(domains) == 1: return domains[0]

        # Simplified check for mirrors (we already checked connectivity in general)
        for domain in domains:
            clean_domain = domain.replace("https://", "").replace("http://", "").strip("/")
            test_url = f"https://{clean_domain}"
            try:
                response = session.head(test_url, timeout=2, allow_redirects=True)
                if response.status_code < 500:
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
        print(f"[INFO] Auto-Unzip: Extracting '{os.path.basename(zip_path)}' to '{extract_to}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"[OK] Extraction complete. Deleting archive.")
            try:
                os.remove(zip_path)
            except OSError: pass
            return True
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return False

    def _download_single_file(self, session, url, target_dir, skip_existing, verify_size, chunk_size_bytes, hf_domain_active, hf_token, ms_token, unzip_after_download):
        try:
            processed_url = self._replace_hf_domain(url, hf_domain_active)
            processed_url = self._process_dropbox_url(processed_url)

            os.makedirs(target_dir, exist_ok=True)
            domain_headers = self._get_headers_for_url(processed_url, hf_token, ms_token)
            
            # Phase 1: HEAD
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

            # Phase 2: Check Existing (Early)
            if remote_file_size > 0:
                if skip_existing and os.path.exists(local_file_path):
                    if verify_size:
                        if os.path.getsize(local_file_path) == remote_file_size:
                            print(f"[OK] Verified existing file. Skipping.")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                self._extract_zip(local_file_path, target_dir)
                            return True
                    else:
                        print(f"[OK] File exists. Skipping.")
                        return True
            
            # Phase 3: Resume
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

            # Phase 4: Download
            req_headers = domain_headers.copy()
            if resume_byte_pos > 0 and file_mode == 'ab':
                req_headers["Range"] = f"bytes={resume_byte_pos}-"

            response_get = session.get(final_direct_url, stream=True, headers=req_headers, timeout=(15, 300), allow_redirects=True)
            if resume_byte_pos > 0 and file_mode == 'ab' and response_get.status_code == 200:
                resume_byte_pos = 0; file_mode = 'wb'
                if os.path.exists(temp_file_path): os.remove(temp_file_path)
            
            response_get.raise_for_status()
            
            total_size = remote_file_size if remote_file_size > 0 else int(response_get.headers.get('content-length', 0))
            
            # Late Size Check
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

            # Phase 5: Finalize
            os.rename(temp_file_path, local_file_path)
            print(f"[OK] Saved: {local_file_path}")

            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                self._extract_zip(local_file_path, target_dir)

            return True

        except Exception as e:
            print(f"[ERROR] Download failed: {e}"); return False

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 4096, hf_token: str = "", hf_domain: str = "huggingface.co, hf-mirror.com", proxy_url: str = "", modelscope_token: str = "", unzip_after_download: bool = False, enable: bool = True):
        
        if not enable:
            print(f"\n--- TS_DownloadNode v2.12 Skipped (Disabled) ---")
            return ()
            
        print(f"\n--- TS_DownloadNode v2.12 Started ---")
        chunk_size_bytes = max(1024, chunk_size_kb * 1024) 

        # 1. Parse files first
        files_to_download = self._parse_file_list(file_list)
        if not files_to_download: return ()

        # 2. Setup session
        session = self._create_session_with_retries(proxy_url)

        # 3. Check connectivity ONLY to targets found in file_list
        if not self._check_connectivity_to_targets(files_to_download, session, hf_domain):
            print("[WARN] All target servers are unreachable. Switching to OFFLINE MODE. Execution finished.")
            return ()

        # 4. Determine Active Mirror (if applicable)
        active_mirror = self._select_best_mirror(session, hf_domain)
        print(f"[INFO] Using HF Mirror: '{active_mirror}'")
        
        # 5. Start Downloads
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