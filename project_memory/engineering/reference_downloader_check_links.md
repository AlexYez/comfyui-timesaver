---
name: reference_downloader_check_links
description: "Anonymous Check Links button; bounded GET probes, public redirects, UI lifecycle and activation after restart"
metadata:
  type: reference
---

Anonymous Check Links button; bounded GET probes, public redirects, UI lifecycle and activation after restart

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# Files Downloader: Check Links, 31.10.2026

User requested a button to verify that workflow recipients can fetch all linked
files without Hugging Face registration or authorization on other sites.

Implemented private backend nodes/files/_downloader_links.py, explicitly imported
by ts_downloader.py, POST /ts_downloader/check_links. Schema/input order unchanged.
New frontend js/files/_downloader_links.js; mountActions puts Scan + Check Links
in one fixed row, Download + Settings in another. Both rows nowrap; this avoids
width-dependent height growth at320px. Button is DOM-only, not serialized.

Checks EVERY list URL even when local disk status is ready. No graph execution,
downloads to disk, file-list edits, token/settings changes or auto mirror fallback.
Frontend submits4lines/batch, backend accepts<=8, workers<=4, one batch at a time.
Close/Esc/removal abort browser request and prevent subsequent batches. Existing
batch may finish within its timeout; lock releases in the worker, shielded from
request cancellation. Concurrent/reopened check gets409 with a helpful retry message.
List changes discard stale results. Report uses shared fullscreen overlay/theme,
en/ru, explicit per-file statuses and progress, keyboard anchor protecting Ctrl+Z.

Each file: fresh requests.Session, trust_env=False (no .netrc/env proxies), cookies
cleared every hop, no tokens. Reject userinfo/token-bearing source URL without
contacting it. Source signed URLs get temporary warning; signed CDN redirects from
a public source are normal. Public-address guard at every hop, max5redirects,
20second budget, GET Range0:2047, stream=True, read<=2048bytes, immediate close.
Do not replace with HEAD-only: HTML/login/authorization failures may differ on GET.
Important requests trap: even allow_redirects=False builds Response.next through
resolve_redirects and consumes the3xx body. This isolated session overrides
resolve_redirects with an empty iterator; manual redirect handling avoids that.
Real local HTTP test uses a302 with advertised1GB body but sends none: check must
follow Location immediately, not consume/hang. .netrc Authorization never sent.

Status available requires readable nonempty bytes, acceptable file type, no HTML
or Git LFS pointer, valid start-of-file206 range. Distinguish gated by HF error
header,401 unavailable without sign-in,403,404/410,429,5xx,timeout/DNS/TLS,
unsupported folder, temporary source URL, credential URL, uncertain responses.
Never return CDN signed URLs or exception strings containing them.200 alone is
not success. Unusual encoding/ranges get unverified. Model JSON/text files remain
supported. Folder validation reuses strict download-button target resolution.

Scope is deliberately honest: anonymous direct access from this server now,
not proof of a later complete transfer or another user's network restrictions.
Tokens, cookies, mirrors and node proxy settings are ignored; UI/docs say so.

Verified42backend tests and12UI cases across classic/Vue (including stale list,
cancel, removal, no token transmission, all-ready local files still checked,
serialization equality, minwidth320).213other targeted backend tests passed;
the one old progress-browser test initially lacked bundled Chromium, passed
under an isolated Edge runner and again against installed frontend. No user
settings/workflow writes: browser POST/PUT/PATCH/DELETE blocked except mocked check.

Live external tests: public12,452,449,080byte Regenerate LoRA -> available206,
sample2048; FLUX.1-dev VAE -> gated401; missing HF file -> missing404; HF blob
page -> webpage200; raw GitHub LICENSE -> available. No model fully downloaded.
Records tests/.cache/check-links-live.json, check-links-final.log,
check-links-ui-extra.log, check-links-ui-restart.log, check-links-progress-installed.log.
Screenshots check-links-report/layout-{classic,vue}.png inspected after animation.

Installed source+docs to runtime with backups through
tests/.cache/install_downloader_check_links.py. Runtime serves new JS immediately.
Backend is NOT restarted: user's explicit feedback_never_restart_comfyui.md rule.
Old running ComfyUI returns405 (not404!) for the missing POST route due its catchall.
Both404/405 explain restart instead of displaying false results. Activate with
user restart + Ctrl+F5. This is NOT live-backend-route validation; backend tests
and live network probes run in isolated processes. Version remains12.13.3,
CHANGELOG Unreleased; no commit/push without a new explicit user command.
