# Standard library imports
import os
import time
import traceback
import re
import socket

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

try:
    from requests.utils import unquote as requests_unquote
except ImportError:
    from urllib.parse import unquote as requests_unquote

class TS_DownloadFilesNode:
    """
    A ComfyUI node to download files.
    Features: 
    - Resume & Retries
    - Multi-Mirror Support (Auto-switch)
    - HF/ModelScope distinct tokens
    - Proxy support
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
                    "default": "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors /path/to/checkpoints\nhttps://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/resolve/master/README.md /path/to/models/LLM",
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
                    "description": "Hugging Face token (starts with hf_). Optional."
                }),
                "hf_domain": ("STRING", {
                    "default": "hf-mirror.com, huggingface.co",
                    "multiline": False,
                    "description": "List of HF domains/mirrors separated by comma. Node will use the first working one."
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "Proxy URL (e.g., http://127.0.0.1:7890). Leave empty to disable."
                }),
                "modelscope_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "ModelScope Access Token. Optional."
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
        """Basic connectivity check."""
        proxies = {"http": proxy, "https": proxy} if proxy else None
        # Try generic reliable targets
        targets = ["https://8.8.8.8", "https://1.1.1.1", "https://www.modelscope.cn"]
        for url in targets:
            try:
                requests.head(url, timeout=timeout, proxies=proxies)
                return True
            except requests.RequestException:
                continue
        return False

    def _select_best_mirror(self, session, domain_list_str):
        """
        Parses comma-separated domains, checks availability from left to right,
        and returns the first working domain.
        """
        if not domain_list_str:
            return "huggingface.co"

        # Split by comma and clean up whitespace
        domains = [d.strip() for d in domain_list_str.split(',') if d.strip()]
        
        if not domains:
            return "huggingface.co"
        
        # If there's only one, just return it without checking (save time/requests)
        if len(domains) == 1:
            return domains[0]

        print(f"[INFO] TS_DownloadNode: Checking availability of mirrors: {domains}")
        
        for domain in domains:
            # Clean domain for URL construction
            clean_domain = domain.replace("https://", "").replace("http://", "").strip("/")
            test_url = f"https://{clean_domain}"
            
            try:
                # Fast timeout check (3 seconds)
                response = session.head(test_url, timeout=3, allow_redirects=True)
                if response.status_code < 500: # 200, 301, 404 (site is up, even if path wrong)
                    print(f"[INFO] Mirror '{clean_domain}' is ACTIVE. Selected.")
                    return clean_domain
            except requests.RequestException:
                print(f"[WARN] Mirror '{clean_domain}' is UNREACHABLE or Timed out.")
                continue
        
        # Fallback to the first one if all fail
        print(f"[WARN] All mirrors failed checks. Defaulting to first option: '{domains[0]}'")
        return domains[0]

    def _get_headers_for_url(self, url, hf_token, ms_token):
        headers = {}
        # HF Logic
        if "huggingface.co" in url or "hf-mirror.com" in url or "hf-" in url:
            if hf_token and hf_token.strip():
                headers["Authorization"] = f"Bearer {hf_token.strip()}"
        # ModelScope Logic
        elif "modelscope.cn" in url:
            headers["Referer"] = "https://www.modelscope.cn/"
            if ms_token and ms_token.strip():
                headers["Authorization"] = f"Bearer {ms_token.strip()}"
        return headers

    def _replace_hf_domain(self, url, target_domain):
        """Replaces huggingface.co in the URL with the selected target_domain."""
        if not target_domain or target_domain.strip() == "huggingface.co":
            return url
        
        # Clean the target domain just in case
        clean_domain = target_domain.replace("https://", "").replace("http://", "").strip("/")
        
        # Regex to find huggingface.co (with http/https/www)
        pattern = r"(https?://)(www\.)?huggingface\.co"
        
        if re.search(pattern, url):
            return re.sub(pattern, f"\\1{clean_domain}", url)
        
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

    def _download_single_file(self, session, url, target_dir, skip_existing, verify_size, chunk_size_bytes, hf_domain_active, hf_token, ms_token):
        try:
            # Apply the selected active mirror
            actual_url = self._replace_hf_domain(url, hf_domain_active)
            if actual_url != url:
                # Only log if verbose, otherwise it gets spammy.
                pass 

            os.makedirs(target_dir, exist_ok=True)
            domain_headers = self._get_headers_for_url(actual_url, hf_token, ms_token)
            
            # --- Phase 1: HEAD ---
            filename_from_url = os.path.basename(requests_unquote(url.split('?')[0].split('#')[0]))
            final_filename = None

            print(f"[INFO] Connecting to: {actual_url}")
            try:
                response_head = session.head(actual_url, headers=domain_headers, allow_redirects=True, timeout=(10, 30))
                response_head.raise_for_status()
            except requests.RequestException as e:
                print(f"[ERROR] Failed to fetch info from {actual_url}: {e}")
                return False

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

            if remote_file_size == -1: verify_size = False; can_resume = False

            # --- Phase 2: Check Existing ---
            if skip_existing and os.path.exists(local_file_path):
                if verify_size and remote_file_size > 0:
                    if os.path.getsize(local_file_path) == remote_file_size:
                        print(f"[OK] File verified. Skipping.")
                        return True
                    else:
                        print(f"[WARN] Size mismatch. Redownloading.")
                else:
                    print(f"[OK] File exists. Skipping.")
                    return True
            
            # --- Phase 3: Resume Logic ---
            resume_byte_pos = 0
            file_mode = 'wb'
            if can_resume and os.path.exists(temp_file_path):
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
            elif os.path.exists(temp_file_path) and remote_file_size > 0 and os.path.getsize(temp_file_path) == remote_file_size:
                 perform_network_download = False

            if perform_network_download:
                req_headers = domain_headers.copy()
                if resume_byte_pos > 0 and file_mode == 'ab':
                    req_headers["Range"] = f"bytes={resume_byte_pos}-"

                response_get = session.get(actual_url, stream=True, headers=req_headers, timeout=(15, 300), allow_redirects=True)
                if resume_byte_pos > 0 and file_mode == 'ab' and response_get.status_code == 200:
                    resume_byte_pos = 0; file_mode = 'wb'
                    if os.path.exists(temp_file_path): os.remove(temp_file_path)
                
                response_get.raise_for_status()
                total_size = remote_file_size if remote_file_size > 0 else int(response_get.headers.get('content-length', 0))

                with open(temp_file_path, file_mode) as f, tqdm(
                    total=total_size if total_size > 0 else None,
                    unit='B', unit_scale=True, desc=f"DL: {final_filename}",
                    initial=resume_byte_pos if file_mode == 'ab' else 0,
                    mininterval=1.0, ncols=100, unit_divisor=1024
                ) as pbar:
                    for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                        if chunk: f.write(chunk); pbar.update(len(chunk))

            # --- Phase 5: Finalize ---
            if not os.path.exists(temp_file_path):
                if os.path.exists(local_file_path) and verify_size and remote_file_size > 0 and os.path.getsize(local_file_path) == remote_file_size:
                    return True
                return False

            if verify_size and remote_file_size > 0 and os.path.getsize(temp_file_path) != remote_file_size:
                print(f"[ERROR] Incomplete download."); return False
            
            if os.path.exists(local_file_path):
                try: os.remove(local_file_path)
                except OSError: pass 
            
            os.rename(temp_file_path, local_file_path)
            print(f"[OK] Saved: {local_file_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Download failed: {e}"); return False

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 4096, hf_token: str = "", hf_domain: str = "huggingface.co", proxy_url: str = "", modelscope_token: str = ""):
        print(f"\n--- TS_DownloadNode v2.3 Started ---")
        chunk_size_bytes = max(1024, chunk_size_kb * 1024) 

        # Check basic connectivity
        if not self._check_internet_connection(proxy=proxy_url if proxy_url else None):
            print("[ERROR] No internet connection detected.")
        
        files_to_download = self._parse_file_list(file_list)
        if not files_to_download: return ()

        session = self._create_session_with_retries(proxy_url)
        
        # --- NEW LOGIC: Select the best mirror ONCE ---
        active_mirror = self._select_best_mirror(session, hf_domain)
        print(f"[INFO] Using HF Mirror: '{active_mirror}' for this session.")
        
        success = 0; failed = 0
        for file_info in files_to_download:
            # Pass the determined active_mirror to the downloader
            if self._download_single_file(session, file_info['url'], file_info['target_dir'], skip_existing, verify_size, chunk_size_bytes, active_mirror, hf_token, modelscope_token):
                success += 1
            else:
                failed += 1
        
        print(f"\n--- Done. Success: {success}, Failed: {failed} ---")
        return ()

NODE_CLASS_MAPPINGS = {
    "TS Files Downloader": TS_DownloadFilesNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Files Downloader": "TS Files Downloader (Multi-Mirror)"
}