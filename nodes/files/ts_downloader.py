# Standard library imports
import hashlib
import json
import logging
import os
import re
import time
import zipfile
from urllib.parse import urlparse

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

try:
    from requests.utils import unquote as requests_unquote
except ImportError:
    from urllib.parse import unquote as requests_unquote

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_downloader")
LOG_PREFIX = "[TS Downloader]"

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


class TS_DownloadFilesNode(IO.ComfyNode):
    """
    A ComfyUI node to download files.
    Features: Offline Mode (Target-based check), Enable/Disable Toggle,
    Auto-Unzip, UI Progress Bar, Resume / Mirrors / Proxies.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS Files Downloader",
            display_name="TS Files Downloader (Ultimate)",
            category="TS/Files",
            is_output_node=True,
            inputs=[
                IO.String.Input(
                    "file_list",
                    default="https://www.dropbox.com/sh/example_folder?dl=0 /path/to/models\nhttps://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors /path/to/checkpoints",
                    multiline=True,
                    dynamic_prompts=False,
                ),
                IO.Boolean.Input("skip_existing", default=True),
                IO.Boolean.Input("verify_size", default=True),
                IO.Int.Input("chunk_size_kb", default=4096, min=1, max=65536, step=1),
                IO.String.Input("hf_token", default="", multiline=False, optional=True),
                IO.String.Input("hf_domain", default="huggingface.co, hf-mirror.com", multiline=False, optional=True),
                IO.String.Input("proxy_url", default="", multiline=False, optional=True),
                IO.String.Input("modelscope_token", default="", multiline=False, optional=True),
                IO.Boolean.Input("unzip_after_download", default=False, optional=True),
                IO.Boolean.Input("enable", default=True, optional=True),
                IO.Combo.Input("integrity_mode", options=["hf_sha256_auto", "size_only"], default="hf_sha256_auto", optional=True),
            ],
            outputs=[],
        )

    @staticmethod
    def _create_session_with_retries(proxy_url=None):
        session = requests.Session()
        retries = Retry(
            total=3,
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
            logger.info(f"{LOG_PREFIX} TS_DownloadNode: Proxy enabled: {proxy_url}")
            session.proxies = {
                'http': proxy_url.strip(),
                'https': proxy_url.strip(),
            }
        return session

    @staticmethod
    def _replace_hf_domain(url, target_domain):
        if not target_domain or target_domain.strip() == "huggingface.co":
            return url
        clean_domain = target_domain.replace("https://", "").replace("http://", "").strip("/")
        pattern = r"(https?://)(www\.)?huggingface\.co"
        if re.search(pattern, url):
            return re.sub(pattern, f"\\1{clean_domain}", url)
        return url

    @classmethod
    def _check_connectivity_to_targets(cls, file_list_parsed, session, hf_domain_str):
        active_mirror = "huggingface.co"
        if hf_domain_str:
            domains = [d.strip() for d in hf_domain_str.split(',') if d.strip()]
            if domains:
                active_mirror = domains[0]

        unique_bases = set()
        for item in file_list_parsed:
            url = item['url']
            final_url = cls._replace_hf_domain(url, active_mirror)
            try:
                parsed = urlparse(final_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                unique_bases.add(base_url)
            except Exception as exc:
                logger.debug("%s Could not parse URL '%s': %s", LOG_PREFIX, final_url, exc)
                continue

        if not unique_bases:
            return True

        logger.info(f"{LOG_PREFIX} Checking connectivity to targets: {list(unique_bases)} ...")
        is_online = False
        for base_url in unique_bases:
            try:
                session.head(base_url, timeout=(2, 2), allow_redirects=True)
                logger.info(f"{LOG_PREFIX} Target '{base_url}' is REACHABLE.")
                is_online = True
                break
            except requests.RequestException:
                logger.warning(f"{LOG_PREFIX} Target '{base_url}' is UNREACHABLE.")
                continue

        return is_online

    @staticmethod
    def _select_best_mirror(session, domain_list_str):
        if not domain_list_str:
            return "huggingface.co"
        domains = [d.strip() for d in domain_list_str.split(',') if d.strip()]
        if not domains:
            return "huggingface.co"
        if len(domains) == 1:
            return domains[0]

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

    @staticmethod
    def _get_headers_for_url(url, hf_token, ms_token):
        headers = {}
        if "huggingface.co" in url or "hf-mirror.com" in url or "hf-" in url:
            if hf_token and hf_token.strip():
                headers["Authorization"] = f"Bearer {hf_token.strip()}"
        elif "modelscope.cn" in url:
            headers["Referer"] = "https://www.modelscope.cn/"
            if ms_token and ms_token.strip():
                headers["Authorization"] = f"Bearer {ms_token.strip()}"
        return headers

    @staticmethod
    def _process_dropbox_url(url):
        if "dropbox.com" in url:
            if "dl=0" in url:
                return url.replace("dl=0", "dl=1")
            if "dl=" not in url:
                return url + ("&dl=1" if "?" in url else "?dl=1")
        return url

    @staticmethod
    def _resolve_target_directory(target_path):
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

    @classmethod
    def _parse_file_list(cls, file_list_text):
        files = []
        lines = file_list_text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                url, target_path = parts[0].strip(), parts[1].strip()
                if not url.startswith(('http://', 'https://')):
                    logger.warning(f"{LOG_PREFIX} Line {i+1}: Invalid URL.")
                    continue
                target_path = cls._resolve_target_directory(target_path)
                if not target_path:
                    logger.warning(f"{LOG_PREFIX} Line {i+1}: Invalid target path.")
                    continue
                files.append({'url': url, 'target_dir': target_path})
        return files

    @staticmethod
    def _get_filename_from_header_map(headers):
        content_disposition = headers.get('content-disposition')
        if not content_disposition:
            return None
        fn_match_utf8 = re.search(r"filename\*=\s*UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if fn_match_utf8:
            return requests_unquote(fn_match_utf8.group(1).strip('" '))
        fn_match_plain = re.search(r'filename="?([^"]+)"?', content_disposition, re.IGNORECASE)
        if fn_match_plain:
            return requests_unquote(fn_match_plain.group(1).strip('" '))
        return None

    @staticmethod
    def _safe_int(value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_etag(etag_value):
        if not etag_value:
            return None
        etag = str(etag_value).strip()
        if etag.lower().startswith("w/"):
            etag = etag[2:].strip()
        etag = etag.strip().strip('"').strip("'")
        return etag or None

    @classmethod
    def _extract_total_size_from_content_range(cls, content_range):
        if not content_range:
            return -1
        match = re.search(r"/(\d+)$", str(content_range).strip())
        if not match:
            return -1
        return cls._safe_int(match.group(1), -1)

    @classmethod
    def _extract_remote_size_from_headers(cls, headers):
        total_from_range = cls._extract_total_size_from_content_range(headers.get("content-range"))
        if total_from_range > 0:
            return total_from_range

        size = cls._safe_int(headers.get("x-linked-size"), -1)
        if size > 0:
            return size
        size = cls._safe_int(headers.get("content-length"), -1)
        if size > 0:
            return size
        return -1

    @staticmethod
    def _is_hf_url(url):
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return ("huggingface.co" in host) or ("hf-mirror.com" in host)

    @classmethod
    def _extract_hf_expected_sha256(cls, remote_etag, final_url):
        if not remote_etag or not cls._is_hf_url(final_url):
            return None
        if re.fullmatch(r"[0-9a-fA-F]{64}", remote_etag):
            return remote_etag.lower()
        return None

    @staticmethod
    def _read_json_file(path):
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

    @staticmethod
    def _write_json_file(path, payload):
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

    @staticmethod
    def _remove_file_silent(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    @classmethod
    def _is_partial_meta_compatible(cls, meta, source_url, remote_size, remote_etag):
        if not isinstance(meta, dict):
            return True

        meta_url = meta.get("source_url")
        if meta_url and meta_url != source_url:
            return False

        meta_size = cls._safe_int(meta.get("remote_size"), -1)
        if remote_size > 0 and meta_size > 0 and meta_size != remote_size:
            return False

        meta_etag = meta.get("remote_etag")
        if meta_etag and remote_etag and meta_etag != remote_etag:
            return False

        return True

    @staticmethod
    def _compute_sha256(file_path, chunk_size=8 * 1024 * 1024):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def _probe_remote_file(cls, session, processed_url, domain_headers):
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
            logger.warning(f"{LOG_PREFIX} HEAD probe failed: {head_error}. Trying lightweight GET probe...")
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
                logger.error(f"{LOG_PREFIX} Remote probe failed: {get_error}")
                return None

        try:
            remote_size = cls._extract_remote_size_from_headers(response.headers)
            remote_etag = cls._normalize_etag(response.headers.get("x-linked-etag") or response.headers.get("etag"))
            supports_ranges = "bytes" in str(response.headers.get("accept-ranges", "")).lower()
            if response.status_code == 206:
                supports_ranges = True

            final_url = response.url or processed_url
            hf_expected_sha256 = cls._extract_hf_expected_sha256(remote_etag, final_url)

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

    @classmethod
    def _extract_zip(cls, zip_path, extract_to):
        logger.info(f"{LOG_PREFIX} Auto-Unzip: Extracting '{os.path.basename(zip_path)}' to '{extract_to}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"{LOG_PREFIX} Extraction complete. Deleting archive.")
            try:
                os.remove(zip_path)
            except OSError:
                pass
            cls._remove_file_silent(zip_path + ".tsmeta.json")
            cls._remove_file_silent(zip_path + ".part")
            cls._remove_file_silent(zip_path + ".part.tsmeta.json")
            return True
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Extraction failed: {e}")
            return False

    @classmethod
    def _download_single_file(cls, session, url, target_dir, skip_existing, verify_size, chunk_size_bytes, hf_domain_active, hf_token, ms_token, unzip_after_download, integrity_mode):
        response_get = None
        try:
            processed_url = cls._replace_hf_domain(url, hf_domain_active)
            processed_url = cls._process_dropbox_url(processed_url)

            os.makedirs(target_dir, exist_ok=True)
            domain_headers = cls._get_headers_for_url(processed_url, hf_token, ms_token)

            logger.info(f"{LOG_PREFIX} Connecting to: {processed_url}")
            remote_info = cls._probe_remote_file(session, processed_url, domain_headers)
            if not remote_info:
                return False

            remote_file_size = remote_info["remote_size"]
            remote_etag = remote_info["remote_etag"]
            final_direct_url = remote_info["final_url"] or processed_url
            supports_ranges = remote_info["supports_ranges"]
            hf_expected_sha256 = remote_info["hf_expected_sha256"]
            use_hf_sha256 = verify_size and integrity_mode == "hf_sha256_auto" and bool(hf_expected_sha256)

            filename_from_url = os.path.basename(requests_unquote(url.split('?')[0].split('#')[0]))
            final_filename = cls._get_filename_from_header_map(remote_info["headers"])
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
            logger.info(f"{LOG_PREFIX} File: '{final_filename}' | Size: {size_label} | ETag: {etag_label} | Range: {'yes' if supports_ranges else 'no'} | Integrity: {integrity_mode}")

            if skip_existing and os.path.exists(local_file_path):
                if not verify_size:
                    logger.info(f"{LOG_PREFIX} File exists. Skipping (size verification disabled).")
                    if unzip_after_download and local_file_path.lower().endswith('.zip'):
                        cls._extract_zip(local_file_path, target_dir)
                    return True

                local_file_size = cls._safe_int(os.path.getsize(local_file_path), -1)
                if remote_file_size > 0 and local_file_size == remote_file_size:
                    if use_hf_sha256:
                        cached_meta = cls._read_json_file(final_meta_path) or {}
                        cached_sha = str(cached_meta.get("sha256", "")).lower()
                        if cached_sha == hf_expected_sha256:
                            logger.info(f"{LOG_PREFIX} Verified existing HF file by cached SHA256. Skipping.")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                cls._extract_zip(local_file_path, target_dir)
                            return True

                        logger.info(f"{LOG_PREFIX} Verifying existing HF file SHA256 (one-time check)...")
                        actual_sha256 = cls._compute_sha256(local_file_path).lower()
                        if actual_sha256 == hf_expected_sha256:
                            logger.info(f"{LOG_PREFIX} Existing HF file SHA256 verified. Skipping.")
                            cls._write_json_file(final_meta_path, {
                                "source_url": url,
                                "resolved_url": processed_url,
                                "final_url": final_direct_url,
                                "remote_size": remote_file_size,
                                "remote_etag": remote_etag,
                                "sha256": actual_sha256,
                                "verified_at": int(time.time()),
                            })
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                cls._extract_zip(local_file_path, target_dir)
                            return True

                        logger.warning(f"{LOG_PREFIX} Existing HF file failed SHA256 verification. Re-downloading.")
                        cls._remove_file_silent(local_file_path)
                    else:
                        logger.info(f"{LOG_PREFIX} Verified existing file by size. Skipping.")
                        cls._write_json_file(final_meta_path, {
                            "source_url": url,
                            "resolved_url": processed_url,
                            "final_url": final_direct_url,
                            "remote_size": remote_file_size,
                            "remote_etag": remote_etag,
                            "sha256": None,
                            "verified_at": int(time.time()),
                        })
                        if unzip_after_download and local_file_path.lower().endswith('.zip'):
                            cls._extract_zip(local_file_path, target_dir)
                        return True
                elif remote_file_size > 0 and local_file_size < remote_file_size and not os.path.exists(temp_file_path):
                    try:
                        os.replace(local_file_path, temp_file_path)
                        cls._write_json_file(temp_meta_path, {
                            "source_url": url,
                            "resolved_url": processed_url,
                            "remote_size": remote_file_size,
                            "remote_etag": remote_etag,
                            "updated_at": int(time.time()),
                        })
                        logger.info(f"{LOG_PREFIX} Found truncated final file. Moved to .part for resume.")
                    except OSError as move_error:
                        logger.warning(f"{LOG_PREFIX} Could not promote truncated file to .part: {move_error}")
                elif remote_file_size > 0 and local_file_size > remote_file_size:
                    logger.warning(f"{LOG_PREFIX} Existing file is larger than remote. Re-downloading.")
                    cls._remove_file_silent(local_file_path)

            resume_byte_pos = 0
            file_mode = "wb"
            if os.path.exists(temp_file_path):
                temp_meta = cls._read_json_file(temp_meta_path)
                if not cls._is_partial_meta_compatible(temp_meta, url, remote_file_size, remote_etag):
                    logger.warning(f"{LOG_PREFIX} Existing .part belongs to a different file. Removing stale partial.")
                    cls._remove_file_silent(temp_file_path)
                    cls._remove_file_silent(temp_meta_path)
                else:
                    temp_file_size = cls._safe_int(os.path.getsize(temp_file_path), -1)
                    if temp_file_size < 0:
                        cls._remove_file_silent(temp_file_path)
                        cls._remove_file_silent(temp_meta_path)
                    elif remote_file_size > 0 and temp_file_size > remote_file_size:
                        logger.warning(f"{LOG_PREFIX} .part is larger than remote file. Removing stale partial.")
                        cls._remove_file_silent(temp_file_path)
                        cls._remove_file_silent(temp_meta_path)
                    elif remote_file_size > 0 and temp_file_size == remote_file_size:
                        logger.info(f"{LOG_PREFIX} .part size matches remote. Finalizing without re-download.")
                        part_is_valid = True
                        if use_hf_sha256:
                            logger.info(f"{LOG_PREFIX} Verifying completed .part with HF SHA256...")
                            part_sha256 = cls._compute_sha256(temp_file_path).lower()
                            if part_sha256 != hf_expected_sha256:
                                logger.error(f"{LOG_PREFIX} .part SHA256 mismatch. Removing corrupt partial.")
                                cls._remove_file_silent(temp_file_path)
                                cls._remove_file_silent(temp_meta_path)
                                part_is_valid = False
                        if part_is_valid:
                            os.replace(temp_file_path, local_file_path)
                            cls._remove_file_silent(temp_meta_path)
                            cls._write_json_file(final_meta_path, {
                                "source_url": url,
                                "resolved_url": processed_url,
                                "final_url": final_direct_url,
                                "remote_size": remote_file_size,
                                "remote_etag": remote_etag,
                                "sha256": hf_expected_sha256 if use_hf_sha256 else None,
                                "verified_at": int(time.time()),
                            })
                            logger.info(f"{LOG_PREFIX} Saved: {local_file_path}")
                            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                                cls._extract_zip(local_file_path, target_dir)
                            return True
                    elif temp_file_size > 0:
                        resume_byte_pos = temp_file_size
                        file_mode = "ab"
                        logger.info(f"{LOG_PREFIX} Resuming from {resume_byte_pos} bytes.")

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
                response_get.close()
                response_get = None
                logger.warning(f"{LOG_PREFIX} Server rejected resume range (416). Restarting full download.")
                cls._remove_file_silent(temp_file_path)
                cls._remove_file_silent(temp_meta_path)
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
                response_get.close()
                response_get = None
                logger.warning(f"{LOG_PREFIX} Server ignored resume request. Restarting full download.")
                cls._remove_file_silent(temp_file_path)
                cls._remove_file_silent(temp_meta_path)
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

            final_direct_url = response_get.url or final_direct_url
            get_reported_size = cls._extract_remote_size_from_headers(response_get.headers)
            if remote_file_size <= 0 and get_reported_size > 0:
                remote_file_size = get_reported_size
            if response_get.status_code == 206:
                supports_ranges = True

            cls._write_json_file(temp_meta_path, {
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

            temp_final_size = cls._safe_int(os.path.getsize(temp_file_path), -1)
            if verify_size and remote_file_size > 0 and temp_final_size != remote_file_size:
                if temp_final_size < remote_file_size:
                    logger.warning(f"{LOG_PREFIX} Download incomplete ({temp_final_size}/{remote_file_size}). Keeping .part for resume.")
                else:
                    logger.error(f"{LOG_PREFIX} Downloaded file is larger than expected. Removing .part.")
                    cls._remove_file_silent(temp_file_path)
                    cls._remove_file_silent(temp_meta_path)
                return False

            verified_sha256 = None
            if use_hf_sha256:
                logger.info(f"{LOG_PREFIX} Verifying HF SHA256...")
                verified_sha256 = cls._compute_sha256(temp_file_path).lower()
                if verified_sha256 != hf_expected_sha256:
                    logger.error(f"{LOG_PREFIX} HF SHA256 mismatch. Removing corrupted file.")
                    cls._remove_file_silent(temp_file_path)
                    cls._remove_file_silent(temp_meta_path)
                    return False

            os.replace(temp_file_path, local_file_path)
            cls._remove_file_silent(temp_meta_path)
            cls._write_json_file(final_meta_path, {
                "source_url": url,
                "resolved_url": processed_url,
                "final_url": final_direct_url,
                "remote_size": remote_file_size,
                "remote_etag": remote_etag,
                "sha256": verified_sha256,
                "verified_at": int(time.time()),
            })
            logger.info(f"{LOG_PREFIX} Saved: {local_file_path}")
            if unzip_after_download and local_file_path.lower().endswith('.zip'):
                cls._extract_zip(local_file_path, target_dir)

            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Download failed: {e}")
            return False
        finally:
            if response_get is not None:
                try:
                    response_get.close()
                except Exception as exc:
                    logger.debug("%s Closing GET probe response failed: %s", LOG_PREFIX, exc)

    @classmethod
    def execute(
        cls,
        file_list: str,
        skip_existing: bool = True,
        verify_size: bool = True,
        chunk_size_kb: int = 4096,
        hf_token: str = "",
        hf_domain: str = "huggingface.co, hf-mirror.com",
        proxy_url: str = "",
        modelscope_token: str = "",
        unzip_after_download: bool = False,
        enable: bool = True,
        integrity_mode: str = "hf_sha256_auto",
    ) -> IO.NodeOutput:
        if not enable:
            logger.info("%s Skipped (disabled).", LOG_PREFIX)
            return IO.NodeOutput()

        logger.info("%s Started.", LOG_PREFIX)
        chunk_size_bytes = max(1024, chunk_size_kb * 1024)
        integrity_mode_value = str(integrity_mode).strip().lower()
        if integrity_mode_value not in {"hf_sha256_auto", "size_only"}:
            logger.warning(f"{LOG_PREFIX} Unknown integrity_mode '{integrity_mode}'. Fallback to 'hf_sha256_auto'.")
            integrity_mode_value = "hf_sha256_auto"

        files_to_download = cls._parse_file_list(file_list)
        if not files_to_download:
            return IO.NodeOutput()

        session = cls._create_session_with_retries(proxy_url)

        if not cls._check_connectivity_to_targets(files_to_download, session, hf_domain):
            logger.warning(f"{LOG_PREFIX} All target servers are unreachable. Switching to OFFLINE MODE. Execution finished.")
            return IO.NodeOutput()

        active_mirror = cls._select_best_mirror(session, hf_domain)
        logger.info(f"{LOG_PREFIX} Using HF Mirror: '{active_mirror}'")

        success = 0
        failed = 0
        for file_info in files_to_download:
            if cls._download_single_file(session, file_info['url'], file_info['target_dir'], skip_existing, verify_size, chunk_size_bytes, active_mirror, hf_token, modelscope_token, unzip_after_download, integrity_mode_value):
                success += 1
            else:
                failed += 1

        logger.info("%s Done. Success: %d, Failed: %d", LOG_PREFIX, success, failed)
        return IO.NodeOutput()


NODE_CLASS_MAPPINGS = {
    "TS Files Downloader": TS_DownloadFilesNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Files Downloader": "TS Files Downloader (Ultimate)",
}
