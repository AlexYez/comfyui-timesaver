# Standard library imports
import os
import time
import traceback
import re
import socket
import zipfile
import json
import hashlib
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

try:
    import folder_paths
except ImportError:
    folder_paths = None

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
                "integrity_mode": (["hf_sha256_auto", "size_only"], {
                    "default": "hf_sha256_auto",
                    "description": "Integrity mode: hf_sha256_auto (HF SHA256 when available) or size_only."
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

    def _resolve_target_directory(self, target_path):
        """
        Resolves target paths using ComfyUI native folders when possible.
        Backward compatibility:
        - Absolute paths stay absolute.
        - "models/..." or "models\\..." always resolves under folder_paths.models_dir.
        - Other relative paths resolve from ComfyUI base_path (fallback: current cwd behavior).
        """
        if target_path is None:
            return None

        cleaned = target_path.strip().strip('"').strip("'")
        if not cleaned:
            return None

        expanded = os.path.expandvars(os.path.expanduser(cleaned))
        if os.path.isabs(expanded):
            return os.path.abspath(expanded)

        normalized = expanded.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]

        normalized_lower = normalized.lower()
        if normalized_lower == "models" or normalized_lower.startswith("models/"):
            models_root = None
            if folder_paths and getattr(folder_paths, "models_dir", None):
                models_root = folder_paths.models_dir
            if not models_root and folder_paths and getattr(folder_paths, "base_path", None):
                models_root = os.path.join(folder_paths.base_path, "models")
            if models_root:
                suffix = normalized[6:].lstrip("/")
                return os.path.abspath(os.path.join(models_root, suffix)) if suffix else os.path.abspath(models_root)

        if folder_paths and getattr(folder_paths, "base_path", None):
            return os.path.abspath(os.path.join(folder_paths.base_path, expanded))

        return os.path.abspath(expanded)

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
                target_path = self._resolve_target_directory(target_path)
                if not target_path:
                    print(f"[WARN] Line {i+1}: Invalid target path.")
                    continue
                files.append({'url': url, 'target_dir': target_path})
        return files

    def _get_filename_from_headers(self, response):
        return self._get_filename_from_header_map(response.headers)

    def _get_filename_from_header_map(self, headers):
        content_disposition = headers.get('content-disposition')
        if not content_disposition: return None
        fn_match_utf8 = re.search(r"filename\*=\s*UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if fn_match_utf8: return requests_unquote(fn_match_utf8.group(1).strip('" '))
        fn_match_plain = re.search(r'filename="?([^"]+)"?', content_disposition, re.IGNORECASE)
        if fn_match_plain: return requests_unquote(fn_match_plain.group(1).strip('" '))
        return None

    def _safe_int(self, value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalize_etag(self, etag_value):
        if not etag_value:
            return None
        etag = str(etag_value).strip()
        if etag.lower().startswith("w/"):
            etag = etag[2:].strip()
        etag = etag.strip().strip('"').strip("'")
        return etag or None

    def _extract_total_size_from_content_range(self, content_range):
        if not content_range:
            return -1
        match = re.search(r"/(\d+)$", str(content_range).strip())
        if not match:
            return -1
        return self._safe_int(match.group(1), -1)

    def _extract_remote_size_from_headers(self, headers):
        total_from_range = self._extract_total_size_from_content_range(headers.get("content-range"))
        if total_from_range > 0:
            return total_from_range

        size = self._safe_int(headers.get("x-linked-size"), -1)
        if size > 0:
            return size
        size = self._safe_int(headers.get("content-length"), -1)
        if size > 0:
            return size
        return -1

    def _is_hf_url(self, url):
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return ("huggingface.co" in host) or ("hf-mirror.com" in host)

    def _extract_hf_expected_sha256(self, remote_etag, final_url):
        if not remote_etag or not self._is_hf_url(final_url):
            return None
        if re.fullmatch(r"[0-9a-fA-F]{64}", remote_etag):
            return remote_etag.lower()
        return None

    def _read_json_file(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _write_json_file(self, path, payload):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False

    def _remove_file_silent(self, path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _is_partial_meta_compatible(self, meta, source_url, remote_size, remote_etag):
        if not isinstance(meta, dict):
            return True

        meta_url = meta.get("source_url")
        if meta_url and meta_url != source_url:
            return False

        meta_size = self._safe_int(meta.get("remote_size"), -1)
        if remote_size > 0 and meta_size > 0 and meta_size != remote_size:
            return False

        meta_etag = meta.get("remote_etag")
        if meta_etag and remote_etag and meta_etag != remote_etag:
            return False

        return True

    def _compute_sha256(self, file_path, chunk_size=8 * 1024 * 1024):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _probe_remote_file(self, session, processed_url, domain_headers):
        response = None
        used_get_probe = False

        try:
            response = session.head(
                processed_url,
                headers=domain_headers,
                allow_redirects=True,
                timeout=(10, 30),
            )
            response.raise_for_status()
        except requests.RequestException as head_error:
            if response is not None:
                response.close()
                response = None
            print(f"[WARN] HEAD probe failed: {head_error}. Trying lightweight GET probe...")
            probe_headers = domain_headers.copy()
            probe_headers["Range"] = "bytes=0-0"
            try:
                response = session.get(
                    processed_url,
                    stream=True,
                    headers=probe_headers,
                    timeout=(10, 30),
                    allow_redirects=True,
                )
                response.raise_for_status()
                used_get_probe = True
            except requests.RequestException as get_error:
                if response is not None:
                    response.close()
                print(f"[ERROR] Remote probe failed: {get_error}")
                return None

        try:
            remote_size = self._extract_remote_size_from_headers(response.headers)
            remote_etag = self._normalize_etag(response.headers.get("x-linked-etag") or response.headers.get("etag"))
            supports_ranges = "bytes" in str(response.headers.get("accept-ranges", "")).lower()
            if response.status_code == 206:
                supports_ranges = True

            final_url = response.url or processed_url
            hf_expected_sha256 = self._extract_hf_expected_sha256(remote_etag, final_url)

            return {
                "final_url": final_url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "remote_size": remote_size,
                "remote_etag": remote_etag,
                "supports_ranges": supports_ranges,
                "hf_expected_sha256": hf_expected_sha256,
                "used_get_probe": used_get_probe,
            }
        finally:
            response.close()

    def _extract_zip(self, zip_path, extract_to):
        print(f"[INFO] Auto-Unzip: Extracting '{os.path.basename(zip_path)}' to '{extract_to}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"[OK] Extraction complete. Deleting archive.")
            try:
                os.remove(zip_path)
            except OSError: pass
            self._remove_file_silent(zip_path + ".tsmeta.json")
            self._remove_file_silent(zip_path + ".part")
            self._remove_file_silent(zip_path + ".part.tsmeta.json")
            return True
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return False

    def _download_single_file(self, session, url, target_dir, skip_existing, verify_size, chunk_size_bytes, hf_domain_active, hf_token, ms_token, unzip_after_download, integrity_mode):
        response_get = None
        try:
            processed_url = self._replace_hf_domain(url, hf_domain_active)
            processed_url = self._process_dropbox_url(processed_url)

            os.makedirs(target_dir, exist_ok=True)
            domain_headers = self._get_headers_for_url(processed_url, hf_token, ms_token)

            print(f"[INFO] Connecting to: {processed_url}")
            remote_info = self._probe_remote_file(session, processed_url, domain_headers)
            if not remote_info:
                return False

            remote_file_size = remote_info["remote_size"]
            remote_etag = remote_info["remote_etag"]
            final_direct_url = remote_info["final_url"] or processed_url
            supports_ranges = remote_info["supports_ranges"]
            hf_expected_sha256 = remote_info["hf_expected_sha256"]
            use_hf_sha256 = verify_size and integrity_mode == "hf_sha256_auto" and bool(hf_expected_sha256)

            filename_from_url = os.path.basename(requests_unquote(url.split('?')[0].split('#')[0]))
            final_filename = self._get_filename_from_header_map(remote_info["headers"])
            if not final_filename:
                final_filename = filename_from_url
            if not final_filename or final_filename == "/":
                final_filename = f"downloaded_file_{int(time.time())}"
            final_filename = re.sub(r'[<>:"/\\|?*]', '_', final_filename)

            local_file_path = os.path.join(target_dir, final_filename)
            temp_file_path = local_file_path + ".part"
            temp_meta_path = temp_file_path + ".tsmeta.json"
            final_meta_path = local_file_path + ".tsmeta.json"

            size_label = remote_file_size if remote_file_size > 0 else "Unknown"
            etag_label = remote_etag if remote_etag else "n/a"
            print(f"[INFO] File: '{final_filename}' | Size: {size_label} | ETag: {etag_label} | Range: {'yes' if supports_ranges else 'no'} | Integrity: {integrity_mode}")

            # Phase 2: Validate Existing Complete File
            if skip_existing and os.path.exists(local_file_path):
                if not verify_size:
                    print(f"[OK] File exists. Skipping (size verification disabled).")
                    if unzip_after_download and local_file_path.lower().endswith('.zip'):
                        self._extract_zip(local_file_path, target_dir)
                    return True

                local_file_size = self._safe_int(os.path.getsize(local_file_path), -1)
                if remote_file_size > 0 and local_file_size == remote_file_size:
                    if use_hf_sha256:
                        cached_meta = self._read_json_file(final_meta_path) or {}
                        cached_sha = str(cached_meta.get("sha256", "")).lower()
                        if cached_sha == hf_expected_sha256:
                            print(f"[OK] Verified existing HF file by cached SHA256. Skipping.")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                self._extract_zip(local_file_path, target_dir)
                            return True

                        print(f"[INFO] Verifying existing HF file SHA256 (one-time check)...")
                        actual_sha256 = self._compute_sha256(local_file_path).lower()
                        if actual_sha256 == hf_expected_sha256:
                            print(f"[OK] Existing HF file SHA256 verified. Skipping.")
                            self._write_json_file(final_meta_path, {
                                "source_url": url,
                                "resolved_url": processed_url,
                                "final_url": final_direct_url,
                                "remote_size": remote_file_size,
                                "remote_etag": remote_etag,
                                "sha256": actual_sha256,
                                "verified_at": int(time.time()),
                            })
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                self._extract_zip(local_file_path, target_dir)
                            return True

                        print(f"[WARN] Existing HF file failed SHA256 verification. Re-downloading.")
                        self._remove_file_silent(local_file_path)
                    else:
                        print(f"[OK] Verified existing file by size. Skipping.")
                        self._write_json_file(final_meta_path, {
                            "source_url": url,
                            "resolved_url": processed_url,
                            "final_url": final_direct_url,
                            "remote_size": remote_file_size,
                            "remote_etag": remote_etag,
                            "sha256": None,
                            "verified_at": int(time.time()),
                        })
                        if unzip_after_download and local_file_path.lower().endswith('.zip'):
                            self._extract_zip(local_file_path, target_dir)
                        return True
                elif remote_file_size > 0 and local_file_size < remote_file_size and not os.path.exists(temp_file_path):
                    # Convert truncated final file to .part so we can resume.
                    try:
                        os.replace(local_file_path, temp_file_path)
                        self._write_json_file(temp_meta_path, {
                            "source_url": url,
                            "resolved_url": processed_url,
                            "remote_size": remote_file_size,
                            "remote_etag": remote_etag,
                            "updated_at": int(time.time()),
                        })
                        print(f"[INFO] Found truncated final file. Moved to .part for resume.")
                    except OSError as move_error:
                        print(f"[WARN] Could not promote truncated file to .part: {move_error}")
                elif remote_file_size > 0 and local_file_size > remote_file_size:
                    print(f"[WARN] Existing file is larger than remote. Re-downloading.")
                    self._remove_file_silent(local_file_path)

            # Phase 3: Decide Resume Strategy
            resume_byte_pos = 0
            file_mode = "wb"
            if os.path.exists(temp_file_path):
                temp_meta = self._read_json_file(temp_meta_path)
                if not self._is_partial_meta_compatible(temp_meta, url, remote_file_size, remote_etag):
                    print(f"[WARN] Existing .part belongs to a different file. Removing stale partial.")
                    self._remove_file_silent(temp_file_path)
                    self._remove_file_silent(temp_meta_path)
                else:
                    temp_file_size = self._safe_int(os.path.getsize(temp_file_path), -1)
                    if temp_file_size < 0:
                        self._remove_file_silent(temp_file_path)
                        self._remove_file_silent(temp_meta_path)
                    elif remote_file_size > 0 and temp_file_size > remote_file_size:
                        print(f"[WARN] .part is larger than remote file. Removing stale partial.")
                        self._remove_file_silent(temp_file_path)
                        self._remove_file_silent(temp_meta_path)
                    elif remote_file_size > 0 and temp_file_size == remote_file_size:
                        print(f"[INFO] .part size matches remote. Finalizing without re-download.")
                        part_is_valid = True
                        if use_hf_sha256:
                            print(f"[INFO] Verifying completed .part with HF SHA256...")
                            part_sha256 = self._compute_sha256(temp_file_path).lower()
                            if part_sha256 != hf_expected_sha256:
                                print(f"[ERROR] .part SHA256 mismatch. Removing corrupt partial.")
                                self._remove_file_silent(temp_file_path)
                                self._remove_file_silent(temp_meta_path)
                                part_is_valid = False
                        if part_is_valid:
                            os.replace(temp_file_path, local_file_path)
                            self._remove_file_silent(temp_meta_path)
                            self._write_json_file(final_meta_path, {
                                "source_url": url,
                                "resolved_url": processed_url,
                                "final_url": final_direct_url,
                                "remote_size": remote_file_size,
                                "remote_etag": remote_etag,
                                "sha256": hf_expected_sha256 if use_hf_sha256 else None,
                                "verified_at": int(time.time()),
                            })
                            print(f"[OK] Saved: {local_file_path}")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                self._extract_zip(local_file_path, target_dir)
                            return True
                    elif temp_file_size > 0:
                        resume_byte_pos = temp_file_size
                        file_mode = "ab"
                        print(f"[INFO] Resuming from {resume_byte_pos} bytes.")

            # Phase 4: Open Download Stream
            request_headers = domain_headers.copy()
            if resume_byte_pos > 0:
                request_headers["Range"] = f"bytes={resume_byte_pos}-"

            response_get = session.get(
                final_direct_url,
                stream=True,
                headers=request_headers,
                timeout=(15, 300),
                allow_redirects=True,
            )

            if resume_byte_pos > 0 and response_get.status_code == 416:
                # Requested range is invalid (typically stale/incompatible partial).
                response_get.close()
                response_get = None
                print(f"[WARN] Server rejected resume range (416). Restarting full download.")
                self._remove_file_silent(temp_file_path)
                self._remove_file_silent(temp_meta_path)
                resume_byte_pos = 0
                file_mode = "wb"
                response_get = session.get(
                    processed_url,
                    stream=True,
                    headers=domain_headers,
                    timeout=(15, 300),
                    allow_redirects=True,
                )
            elif resume_byte_pos > 0 and response_get.status_code != 206:
                # Range was ignored (200). Restart cleanly to avoid corruption.
                response_get.close()
                response_get = None
                print(f"[WARN] Server ignored resume request. Restarting full download.")
                self._remove_file_silent(temp_file_path)
                self._remove_file_silent(temp_meta_path)
                resume_byte_pos = 0
                file_mode = "wb"
                response_get = session.get(
                    processed_url,
                    stream=True,
                    headers=domain_headers,
                    timeout=(15, 300),
                    allow_redirects=True,
                )

            response_get.raise_for_status()

            # Refresh metadata from GET response (important after redirects/fallbacks).
            final_direct_url = response_get.url or final_direct_url
            get_reported_size = self._extract_remote_size_from_headers(response_get.headers)
            if remote_file_size <= 0 and get_reported_size > 0:
                remote_file_size = get_reported_size
            if response_get.status_code == 206:
                supports_ranges = True

            # Save partial metadata snapshot for safe future resume.
            self._write_json_file(temp_meta_path, {
                "source_url": url,
                "resolved_url": processed_url,
                "final_url": final_direct_url,
                "remote_size": remote_file_size,
                "remote_etag": remote_etag,
                "updated_at": int(time.time()),
            })

            total_size = remote_file_size if remote_file_size > 0 else None

            comfy_pbar = None
            if ProgressBar and total_size:
                comfy_pbar = ProgressBar(total_size)

            downloaded_since_update = 0
            ui_update_threshold = 1 * 1024 * 1024

            with open(temp_file_path, file_mode) as f, tqdm(
                total=total_size,
                unit='B', unit_scale=True, desc=f"DL: {final_filename}",
                initial=resume_byte_pos if file_mode == 'ab' else 0,
                mininterval=1.0, ncols=100, unit_divisor=1024
            ) as pbar:
                if comfy_pbar and resume_byte_pos > 0:
                    comfy_pbar.update(resume_byte_pos)
                for chunk in response_get.iter_content(chunk_size=chunk_size_bytes):
                    if not chunk:
                        continue
                    f.write(chunk)
                    chunk_len = len(chunk)
                    pbar.update(chunk_len)
                    if comfy_pbar:
                        downloaded_since_update += chunk_len
                        if downloaded_since_update >= ui_update_threshold:
                            comfy_pbar.update(downloaded_since_update)
                            downloaded_since_update = 0
                if comfy_pbar and downloaded_since_update > 0:
                    comfy_pbar.update(downloaded_since_update)

            # Phase 5: Integrity Verification
            temp_final_size = self._safe_int(os.path.getsize(temp_file_path), -1)
            if verify_size and remote_file_size > 0 and temp_final_size != remote_file_size:
                if temp_final_size < remote_file_size:
                    print(f"[WARN] Download incomplete ({temp_final_size}/{remote_file_size}). Keeping .part for resume.")
                else:
                    print(f"[ERROR] Downloaded file is larger than expected. Removing .part.")
                    self._remove_file_silent(temp_file_path)
                    self._remove_file_silent(temp_meta_path)
                return False

            verified_sha256 = None
            if use_hf_sha256:
                print(f"[INFO] Verifying HF SHA256...")
                verified_sha256 = self._compute_sha256(temp_file_path).lower()
                if verified_sha256 != hf_expected_sha256:
                    print(f"[ERROR] HF SHA256 mismatch. Removing corrupted file.")
                    self._remove_file_silent(temp_file_path)
                    self._remove_file_silent(temp_meta_path)
                    return False

            # Phase 6: Finalize
            os.replace(temp_file_path, local_file_path)
            self._remove_file_silent(temp_meta_path)
            self._write_json_file(final_meta_path, {
                "source_url": url,
                "resolved_url": processed_url,
                "final_url": final_direct_url,
                "remote_size": remote_file_size,
                "remote_etag": remote_etag,
                "sha256": verified_sha256,
                "verified_at": int(time.time()),
            })
            print(f"[OK] Saved: {local_file_path}")

            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                self._extract_zip(local_file_path, target_dir)

            return True

        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return False
        finally:
            if response_get is not None:
                try:
                    response_get.close()
                except Exception:
                    pass

    def execute_downloads(self, file_list: str, skip_existing: bool = True, verify_size: bool = True, chunk_size_kb: int = 4096, hf_token: str = "", hf_domain: str = "huggingface.co, hf-mirror.com", proxy_url: str = "", modelscope_token: str = "", unzip_after_download: bool = False, enable: bool = True, integrity_mode: str = "hf_sha256_auto"):
        
        if not enable:
            print(f"\n--- TS_DownloadNode v2.12 Skipped (Disabled) ---")
            return ()
            
        print(f"\n--- TS_DownloadNode v2.12 Started ---")
        chunk_size_bytes = max(1024, chunk_size_kb * 1024) 
        integrity_mode_value = str(integrity_mode).strip().lower()
        if integrity_mode_value not in {"hf_sha256_auto", "size_only"}:
            print(f"[WARN] Unknown integrity_mode '{integrity_mode}'. Fallback to 'hf_sha256_auto'.")
            integrity_mode_value = "hf_sha256_auto"

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
            if self._download_single_file(session, file_info['url'], file_info['target_dir'], skip_existing, verify_size, chunk_size_bytes, active_mirror, hf_token, modelscope_token, unzip_after_download, integrity_mode_value):
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
