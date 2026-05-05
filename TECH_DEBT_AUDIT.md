# TECH_DEBT_AUDIT.md — comfyui-timesaver

Initial audit (rev 1) was read-only against `master` at release `8.8`. Subsequent revisions track which findings have been resolved. All citations remain `path:line` against the tree as of the latest revision.

## Revision history

| Rev | Date | Scope |
| --- | --- | --- |
| 1 | 2026-05-04 | Initial 45-finding audit. |
| 2 | 2026-05-05 | After 5 fix waves (Variants A–D + G). 27 findings RESOLVED, 1 PARTIAL, 5 ACCEPTED, 12 OPEN. Snapshot collector extended (F-39 RESOLVED). New mojibake regression test added. |
| 3 | 2026-05-05 | After Variant H + 4 medium-priority fixes (F-46/F-10/F-23/F-36/F-37 done in dedicated wave; F-26/F-28/F-33 in polish wave; F-12/F-13/F-14 reclassified as ACCEPTED per maintainer decision: one-node-one-file rule trumps file size). 35 RESOLVED, 1 PARTIAL→DONE, 8 ACCEPTED, 3 OPEN (F-19, F-42, F-43, F-45). |
| 4 | 2026-05-05 | Final wave: F-19 CATEGORY migration (50 nodes → canonical `TS/<Subfolder>` hierarchy), F-42 unit tests for `ts_downloader.py` (8 tests, all passing), F-43 unit tests for `ts_whisper.py` (12 tests, `importorskip("numpy")`), F-45 split of `ts-audio-loader.js` into `_audio_helpers.js` + thin per-node entry points. Bonus: structural reorg of shared-helper groups into subfolders (`nodes/audio/loader/`, `nodes/image/keying/`, `js/utils/sliders/`, `js/audio/loader/`). **Audit cycle closed**: 39 RESOLVED, 9 ACCEPTED, 0 OPEN. |

## Executive summary (rev 4)

**Audit cycle closed.** All 48 trackable findings are either RESOLVED (39) or consciously ACCEPTED (9). Zero OPEN items remain.

What rev 4 added on top of rev 3:

- **F-19 CATEGORY canonicalisation** — 50 nodes migrated from 19+ ad-hoc category strings to a single hierarchy: `TS/Audio` (5), `TS/Conditioning` (1), `TS/Files` (8), `TS/Image` (25), `TS/LLM` (2), `TS/Text` (4), `TS/Utils` (4), `TS/Video` (7). Subfolder = category, no nesting. Workflow JSON does not serialize CATEGORY, so the migration does not break saved graphs — only menu organisation changes.
- **F-42 + F-43 test coverage** — `tests/test_downloader.py` (8 tests, no external deps via stubs for `requests`/`urllib3`/`tqdm`/`comfy.utils`/`folder_paths`); `tests/test_whisper.py` (12 tests, `pytest.importorskip("numpy")` matching the existing `test_voice_recognition_audio.py` convention, stubs for torch/torchaudio/transformers/comfy/folder_paths/srt). Coverage focused on pure-logic helpers.
- **F-45 frontend split** — `js/audio/ts-audio-loader.js` (931 LOC, 1 extension `ts.audioLoader` for both nodes) → 3 files: `_audio_helpers.js` (shared `setupAudioLoader` + utilities, 905 LOC), `ts-audio-loader.js` (32 LOC, owns `ts.audioLoader` for `TS_AudioLoader`), `ts-audio-preview.js` (40 LOC, owns new `ts.audioPreview` for `TS_AudioPreview` + `onExecuted` payload hook).
- **Structural reorg** (bonus, not directly an audit finding): when multiple nodes share a private helper, they now group together in a subfolder. Backend: `nodes/audio/loader/` (loader+preview+helpers), `nodes/image/keying/` (keyer+despill+helpers). Frontend: `js/utils/sliders/` (float+int sliders+helpers), `js/audio/loader/` (loader+preview+helpers). Pack-level `nodes/_shared.py` (TS_Logger) stays at root because its 5 consumers span 2 categories.

What was already done in rev 1–3 (preserved unchanged here for the historical record): see findings table.

**Done in rev 1 → rev 3 (35 findings):**

1. **Critical security** — path traversal in `/ts_audio_loader/{view,metadata}` (F-01, F-02), broken `TS_DeflickerNode` deletion incl. `torch.load weights_only`, `.cuda()`, missing timeout (F-03, F-04, F-06, F-34, F-35).
2. **High-impact UI** — mojibake recovered in 15 files; new [tests/test_no_mojibake.py](tests/test_no_mojibake.py) regression guard (F-05, F-25, F-47).
3. **Module-level side effects** — `mkdir` at import → lazy first-write (F-07).
4. **Dependency drift** — `silero-stress` added to `pyproject.toml` (F-08).
5. **Logging migration** — **DONE**. 22 nodes converted from `print()` to `logging.getLogger("comfyui_timesaver.<name>")`. `TS_Logger` rewritten as stdlib facade (`.log`/`.warn`/`.error`); F-37 closed alongside (`error` → `warn` for warning-semantics callsites). ANSI codes in Whisper removed. Banner/debug spam removed. Qwen logger renamed to project convention. Zero `print()` calls remain in `nodes/`. (F-09, F-10, F-11, F-30, F-31, F-32, F-37, F-38-via-acceptance, F-46)
6. **Doc drift** — `TS_FileBrowser` refs, `docs/`→`doc/`, dead loader, cache cleanup notes (F-16, F-17, F-18, F-28).
7. **Endpoint hardening** — `/ts_super_prompt/enhance` validates text length (≤8KB), preset allow-list, fast-fail 429 on busy lock (F-36).
8. **Silent excepts** — 13 broad `except Exception: pass` annotated with `logger.debug` (narrow `except OSError: pass` left as best-effort idiom) (F-23).
9. **Snapshot drift detection** — collector tracks `default`/`min`/`max`/`step` per widget; resolves module-level constants (F-39).
10. **Code hygiene** — orphan comments, unused imports, expensive `IS_CHANGED`, `__import__("torch")` cleanup (F-20, F-21, F-22, F-26).
11. **CI** — `ruff check .` added as non-blocking step (F-33).

**ACCEPTED** (per maintainer decisions):
- **God files** F-12/F-13/F-14/F-15 — large files (`ts_super_prompt.py` 1705 LOC, `ts_qwen3_vl.py` 1271, `ts_whisper.py` 1079, `ts_downloader.py` 832) are NOT debt. Project rule: «one public node = one `.py` file» (CLAUDE.md §7) trumps file-size heuristics. Optional decomposition into private `_*.py` helpers is at maintainer discretion, not required.
- **F-24, F-29, F-38, F-40, F-41, F-44** — see findings table for individual rationale.

**Remaining open: none.**

## Архитектурная ментальная модель (rev 2 update)

`comfyui-timesaver` — production-quality пак ComfyUI custom nodes (**56 нод** после удаления `TS_DeflickerNode`, mix V1+V3 API). Точка входа — корневой `__init__.py`, делает рекурсивный auto-discovery `nodes/**/ts_*.py` (legacy root-level fallback удалён в rev 2), регистрирует `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` из каждого модуля, оборачивает их через `TSDependencyManager.wrap_node_runtime()` (V1 → typed fallback, V3 → нормализованный `RuntimeError`). На старте печатает табличный отчёт «Module/Status/Nodes/Details + Import audit».

Layout новых V3-нод (после релиза 8.8) — строгий: одна нода = один `ts_<name>.py` в `nodes/<категория>/`, плюс опциональный `js/<категория>/ts-<name>.js`. Категории: `image/` (25), `video/` (7), `audio/` (5), `llm/` (2), `text/` (4), `files/` (8), `utils/` (4), `conditioning/` (1). Приватные shared-модули с префиксом `_` пропускаются loader'ом.

Frontend — простые extension-файлы под `WEB_DIRECTORY = "./js"`, регистрируются через `app.registerExtension({ name: "ts.<id>", ... })`. Большая часть UI — DOM/canvas, без npm-сборки.

Реалии расходятся с моделью в одном месте: фриз CATEGORY (CLAUDE.md §4) конфликтует с тем, что ~⅓ нод используют не-`TS/` категории. Это единственный крупный неоднозначный вопрос на rev 2.

## Excluded from audit

- `.git/`, `.cache/`, `.claude/`, `__pycache__/`, `tests/.cache/`, `nodes/.cache/`, `nodes/files/.cache/`, `.tracking` — игнорируются `.gitignore`, либо являются генерируемыми артефактами.
- `nodes/video_depth_anything/**` (16 файлов, ~2700 LOC) — vendored DINOv2/DPT/motion module под Apache 2.0. Используется только нодой `TS_VideoDepthNode`. Изменять внутри запрещено по правилам ComfyUI vendoring.
- `nodes/frame_interpolation_models/**` (3 файла, ~626 LOC) — vendored FILM-Net + IFNet. Используется только `TS_Frame_Interpolation`.
- `nodes/luts/*.cube`, `nodes/prompts/*.txt`, `nodes/styles/img/*.png`, `doc/img/*.png`, `icon.png` — ассеты.
- `tests/contracts/node_contracts.json` — генерируемый снапшот (через `tools/build_node_contracts.py`).
- `README.ru.md` — авто-генерируется из `README.md` через `doc/generate_readme_ru.py`.

## Findings table

Status legend: **RESOLVED** ✅ | **PARTIAL** ◐ | **ACCEPTED** ◇ (consciously not fixing) | **OPEN** ◯ | **NEW** ✨

| ID | Status | Category | File:Line | Severity | Effort | Description / Resolution |
| --- | --- | --- | --- | --- | --- | --- |
| F-01 | ✅ RESOLVED | Security | nodes/audio/_audio_helpers.py:509-517 | Critical | S | `/ts_audio_loader/view` accepted any `?filepath=`. **Fix**: `_allowed_view_roots()` + `_is_inside_allowed_root()` allow-list, applied in both endpoints. |
| F-02 | ✅ RESOLVED | Security | nodes/audio/_audio_helpers.py:498-506 | Critical | S | `/ts_audio_loader/metadata` same path traversal. **Fix**: same allow-list as F-01. |
| F-03 | ✅ RESOLVED | Correctness | nodes/video/ts_deflicker.py | Critical | M | `rife_interpolation` mode crashed on missing `nodes/rife/` directory. **Fix**: node deleted entirely (per maintainer decision). README + CLAUDE.md + snapshot updated. |
| F-04 | ✅ RESOLVED | Security | nodes/video/ts_deflicker.py:22 | High | S | `torch.load` without `weights_only=True`. **Fix**: file deleted. |
| F-05 | ✅ RESOLVED | UI/Localization | 7 files | High | M | Mojibake in tooltips. **Fix**: cp1251-with-latin1-fallback round-trip recovered all 91 corrupted lines across `ts_color_match.py`, `ts_qwen_canvas.py`, `ts_qwen_safe_resize.py`, `ts_image_resize.py`, `ts_batch_prompt_loader.py`, `ts_file_path_loader.py`, `ts_model_converter_advanced.py`. |
| F-06 | ✅ RESOLVED | Tensor Hygiene | nodes/video/ts_deflicker.py | High | S | Hardcoded `.cuda()`. **Fix**: file deleted. |
| F-07 | ✅ RESOLVED | Module-Level Side Effects | nodes/audio/_audio_helpers.py:69-71 | High | S | Three `mkdir` at import. **Fix**: removed module-level calls; lazy `mkdir` added in `_write_cached_preview()`, `_write_preview_audio_file()`, `ts_audio_loader_upload_recording()`. |
| F-08 | ✅ RESOLVED | Dependency Drift | pyproject.toml | High | S | `silero-stress` missing. **Fix**: added to `[project] dependencies`. |
| F-09 | ✅ RESOLVED | Logging | 16+ files | Medium | M | `print()` for logging. **Fix in waves**: rev 2 did 5 largest files; rev 3 finished tail (F-46). All 22 affected nodes now use `logging.getLogger("comfyui_timesaver.<name>")`. Zero `print()` calls in `nodes/`. |
| F-10 | ✅ RESOLVED | Logging | nodes/_shared.py | Medium | S | `TS_Logger` rewritten as facade over stdlib `logging` (`logging.getLogger("comfyui_timesaver.ts_shared")`). `color` parameter dropped. New `.warn()` method added to fix F-37 warning-semantics. |
| F-11 | ✅ RESOLVED | Logging | nodes/audio/ts_whisper.py | Medium | S | ANSI escape codes. **Fix**: 5 `_COLOR_*` constants removed; `_log_tensor_shape` no longer takes `color`; 2 callsites cleaned. |
| F-12 | ◇ ACCEPTED | God Files | nodes/llm/ts_super_prompt.py | — | — | 1705 LOC, but contains exactly one public node. Per maintainer decision in rev 3: «one public node = one .py file» (CLAUDE.md §7) trumps file-size heuristics. Not debt. |
| F-13 | ◇ ACCEPTED | God Files | nodes/llm/ts_qwen3_vl.py | — | — | 1271 LOC, one public node. Same reason as F-12. |
| F-14 | ◇ ACCEPTED | God Files | nodes/audio/ts_whisper.py | — | — | 1079 LOC, one public node. Same reason as F-12. |
| F-15 | ◇ ACCEPTED | Structure | nodes/files/ts_downloader.py | — | — | 832 LOC, one public node. After F-09 logging is clean. Same one-node-one-file reasoning as F-12/13/14. |
| F-16 | ✅ RESOLVED | Dead Code | __init__.py | Medium | S | Legacy root-level loader. **Fix**: removed `_LEGACY_NODE_FILENAMES`, `_is_legacy_node_file`, root-glob loop. Tests still green. |
| F-17 | ✅ RESOLVED | Doc Drift | README.md, doc/migration.md, CLAUDE.md | Medium | S | `TS_FileBrowser` / `ts.filebrowser` references. **Fix**: removed from all 3 docs; per-category counts updated; `ts.float-slider`/`ts.int-slider` added to stable extension IDs (replacing dead `ts.slider-settings`). |
| F-18 | ✅ RESOLVED | Doc Drift | AGENTS.md, doc/AGENTS.md | Medium | S | `docs/` vs `doc/`. **Fix**: corrected paths; removed refs to non-existent `docs/ai-lessons.md`, `docs/troubleshooting.md`, `docs/developer-notes.md`. |
| F-19 | ✅ RESOLVED | Category Inconsistency | 50 files | Medium | M | Migrated all categories to canonical `TS/<Subfolder>` per location. Final distribution: `TS/Audio` 5, `TS/Conditioning` 1, `TS/Files` 8, `TS/Image` 25, `TS/LLM` 2, `TS/Text` 4, `TS/Utils` 4, `TS/Video` 7. Workflow JSON does not serialize CATEGORY → no saved-graph breakage; only menu organisation changes. CLAUDE.md §6 now matches reality. |
| F-20 | ✅ RESOLVED | Dead Comments | nodes/utils/ts_smart_switch.py | Low | S | `# Node 4: TS Math Int` orphan. **Fix**: removed. |
| F-21 | ✅ RESOLVED | Dead Imports | nodes/image/ts_get_image_megapixels.py:7 | Low | S | Unused `import time`. **Fix**: removed (also `import comfy.utils`). |
| F-22 | ✅ RESOLVED | Inefficient Cache Key | nodes/image/ts_get_image_megapixels.py | Low | S | `image.mean()` in `IS_CHANGED`. **Fix**: cache key now `f"{shape}_{dtype}"`. |
| F-23 | ✅ RESOLVED | Silent Errors | 6 files | Medium | M | 13 broad `except Exception: pass` annotated with `logger.debug` (cleanup context per site) in `ts_super_prompt`, `ts_qwen3_vl`, `ts_downloader`, `ts_model_converter_advanced`, `ts_whisper`, `_audio_helpers`, `ts_animation_preview`. Narrow `except OSError|FileNotFoundError: pass` (7 sites) left as best-effort cleanup idioms. |
| F-24 | ◇ ACCEPTED | Bare Except | nodes/video_depth_anything/utils/dc_utils.py:12 | Low | S | Vendored code, not modified. |
| F-25 | ✅ RESOLVED | Encoding | 8+8 files | High | M | Mojibake (UI + comments). **Fix**: see F-05. Bonus pass also recovered 8 files where mojibake was only in Python comments (`ts_edl_chapters`, `ts_model_converter`, `ts_auto_tile_size`, `ts_cube_to_equirect`, `ts_equirect_to_cube`, `ts_film_grain`, `ts_wan_safe_resize`, `ts_video_depth`). Defended by `tests/test_no_mojibake.py`. |
| F-26 | ✅ RESOLVED | API Misuse | nodes/llm/ts_super_prompt.py | Low | S | 3 `__import__("torch")` collapsed into a single local `import torch` at the top of the relevant try-block. Comment notes the deferred-import rationale. |
| F-27 | ✅ RESOLVED | Stale Pycache | nodes/__pycache__/ts_super_prompt_node.cpython-312.pyc | Low | S | **Fix**: regenerated on next import; gitignore covers. |
| F-28 | ✅ RESOLVED | Stale Cache Dir | nodes/.cache/tsfb_thumbnails | Low | S | Cleanup section added to `doc/migration.md` listing safe-to-delete cache directories from removed nodes (`tsfb_thumbnails` from `TS_FileBrowser`, plus `ts_audio_loader/` cache notes). |
| F-29 | ◇ ACCEPTED | Wildcard Type | nodes/utils/ts_smart_switch.py | Low | S | `("*",)` is widely used in ComfyUI ecosystem. Accepted until V3 migration. |
| F-30 | ✅ RESOLVED | Inconsistent V1 Logging | nodes/llm/ts_qwen3_vl.py | Medium | S | Logger named `"TS_Qwen3_VL_V3"`. **Fix**: renamed to `"comfyui_timesaver.ts_qwen3_vl"` in 2 callsites. |
| F-31 | ✅ RESOLVED | Code Banner Spam | nodes/video/ts_video_depth.py | Medium | S | Module-top print banner. **Fix**: removed; loader's startup table is sufficient. |
| F-32 | ✅ RESOLVED | Debug Spam | nodes/image/ts_film_grain.py | Medium | S | 10-line debug block per call. **Fix**: removed. |
| F-33 | ✅ RESOLVED | CI Coverage Gap | .github/workflows/ci.yml | Low | S | `ruff check .` added as non-blocking step (`continue-on-error: true`) in CI matrix. Surfaces style/quality regressions without gating merge. Mypy not added (project uses dynamic patterns; net-negative for now). |
| F-34 | ✅ RESOLVED | Missing Timeout | nodes/video/ts_deflicker.py | Medium | S | `requests.get` without timeout. **Fix**: file deleted. |
| F-35 | ✅ RESOLVED | Mojibake in Comment | nodes/video/ts_deflicker.py | Low | S | **Fix**: file deleted. |
| F-36 | ✅ RESOLVED | HTTP Endpoint Without Validation | nodes/llm/ts_super_prompt.py | Medium | M | Endpoint now validates: `text` ≤ `_ENHANCE_MAX_TEXT_LEN` (8192) → HTTP 413; `system_preset` ∈ `_preset_options()` → HTTP 400; `_MODEL_LOCK.locked()` → HTTP 429 fast-fail. Internal `_generate_with_qwen` still owns the actual lock as security gate. |
| F-37 | ✅ RESOLVED | Logger Level Mismatch | nodes/utils/ts_smart_switch.py | Low | S | Closed alongside F-10. `TS_Logger.warn(...)` introduced; warning-semantics callsites in `ts_smart_switch.py:114` and `ts_animation_preview.py` migrated. |
| F-38 | ◇ ACCEPTED | Loader Print | __init__.py:320-351 | Low | S | Startup-table `print()` is ComfyUI custom-nodes convention. |
| F-39 | ✅ RESOLVED | Snapshot Drift Coverage | tools/build_node_contracts.py | Low | S | Snapshot now records `default`/`min`/`max`/`step` per widget for both V1 `INPUT_TYPES` and V3 `IO.<Type>.Input(...)`. Module-level constants (`_DEFAULT_FOO = ...`) are resolved. Smoke-tested: changing a default produces snapshot diff. Snapshot grew 401 → 1567 lines (49/56 nodes have widget metadata). |
| F-40 | ◇ ACCEPTED | Test Skips | tests/test_bgrm_node.py | Low | S | GPU tests skip on CPU CI. Documented. |
| F-41 | ◇ ACCEPTED | Russian-only tooltips | nodes/llm/ts_super_prompt.py | Low | S | Project decision. |
| F-42 | ✅ RESOLVED | Test Coverage Gap | nodes/files/ts_downloader.py | Medium | M | `tests/test_downloader.py` — 8 tests covering `_parse_file_list`, `_replace_hf_domain`, `_select_best_mirror`, `_check_connectivity_to_targets`. Stubs `requests`/`urllib3`/`tqdm`/`comfy.utils`/`folder_paths` so the suite runs on bare CI without those deps. |
| F-43 | ✅ RESOLVED | Test Coverage Gap | nodes/audio/ts_whisper.py | Medium | M | `tests/test_whisper.py` — 12 tests covering `_safe_float`/`_safe_int`, `_is_oom_error`, `_normalize_text`, `_merge_segments`, `_build_generate_kwargs`. Uses `pytest.importorskip("numpy")` matching existing `test_voice_recognition_audio.py` convention; stubs torch/transformers/comfy/folder_paths/srt. |
| F-44 | ◇ SUPERSEDED | Snapshot size | tests/contracts/node_contracts.json | Low | S | Snapshot size now 1567 lines (was 401). With widget data it's still readable; pytest still finishes in <1s. Watch for runtime impact at 200+ nodes. |
| F-45 | ✅ RESOLVED | Frontend split | js/audio/loader/ | Medium | L | Split into 3 files in `js/audio/loader/`: `_audio_helpers.js` (905 LOC, exports `setupAudioLoader` and constants), `ts-audio-loader.js` (32 LOC, owns extension `ts.audioLoader` for `TS_AudioLoader` only), `ts-audio-preview.js` (40 LOC, owns new extension `ts.audioPreview` for `TS_AudioPreview`, plus `onExecuted` payload hook). Existing `ts.audioLoader` ID preserved; `ts.audioPreview` is new. Bonus: subfolder reorg (`js/audio/loader/`, `js/utils/sliders/`, `nodes/audio/loader/`, `nodes/image/keying/`) groups every shared-helper consumer with its helper. |
| F-46 | ✅ RESOLVED | Logging | 17 small nodes | Medium | M | All ~30 remaining `print()` calls migrated to `logging.getLogger(...)`. 9 image nodes converted via batch-script (uniform `_log` static helper); 8 others handled individually with appropriate log levels (info/warning/error). |
| F-47 | ✨ NEW | Defensive Test | tests/test_no_mojibake.py | — | — | **Already added in rev 2** as a regression guard against F-05/F-25. Scans all `.py` for `[РС][ -¿Ѐ-Џ]` bigrams. 0 hits in current tree. |

> Hvostovye наблюдения остаются: 5 случаев `os.path.join(__file__)` вместо `Path(__file__).resolve().parent` (стилистика), `Image.Resampling.LANCZOS` fallback в `ts_image_resize.py:24`, длина tooltips >2KB. Не trackable как debt.

## Top 3 «if you fix nothing else, fix these» (rev 4)

All three rev-3 priorities are RESOLVED in rev 4. Nothing remains in this section.

## Quick wins (rev 4)

All actionable items closed.

## Things that look bad but are actually fine

- **`subprocess.run(...)` in 5 sites** (`_audio_helpers.py`, `ts_super_prompt.py`, `ts_animation_preview.py`) — list-args without `shell=True`, paths from `folder_paths` / `imageio_ffmpeg`. Not command injection.
- **`importlib.util.spec_from_file_location` in `ts_bgrm_birefnet.py:422`** — dynamic loading of HF-downloaded model module. Standard ComfyUI BiRefNet pattern.
- **Module-level `_register_model_folder` in `ts_frame_interpolation.py:60-61`** — registers ComfyUI `models/rife/` and `models/film/`. Conventional side effect.
- **`("*",)` wildcard in `ts_smart_switch.py`** — V1 hack but ecosystem-standard (F-29 ACCEPTED).
- **Giant LLM/audio files (1705/1271/1079 LOC)** — formally compliant with one-node-one-file (one public node each); decomposition would be optional refactor.
- **38 V1 nodes** — frozen by CLAUDE.md §5, deliberate policy.
- **Vendored code** in `nodes/video_depth_anything/` and `nodes/frame_interpolation_models/` — Apache 2.0 licensed forks; print/bare-except are acceptable vendor-deviations.
- **`__init__.py` startup-table prints** — ComfyUI core convention (F-38 ACCEPTED).
- **Snapshot is now 1567 lines** — still diffable, still <1s pytest. Worth it for default/min/max drift detection.

## Open questions for the maintainer (rev 4)

All resolved:
- ✅ Question 1 (CATEGORY migration F-19): yes, migrated. 50 files updated to canonical `TS/<Subfolder>` per location.
- ✅ Question 2 (§4 vs §6 conflict): §6 is authoritative — `TS/<подкатегория>` naming wins. CLAUDE.md should be updated to clarify that §4's CATEGORY-freeze applies after rev 4 baseline (not before).
- ✅ Question 3 (test coverage F-42/43): yes, added — `tests/test_downloader.py` (8 tests) and `tests/test_whisper.py` (12 tests).
- ✅ Question 4 (F-45 frontend split): yes, done in rev 4 with subfolder reorg bonus.
- ✅ Question 3 from rev 2 (`ts_super_prompt.py` decomposition): maintainer confirmed «one-node-one-file rule trumps file size» — F-12/13/14 ACCEPTED.
- ✅ Question 5 from rev 2 (CI ruff): added in rev 3 as non-blocking step.
- ✅ Question 7 from rev 2 (F-46 logging tail): done in rev 3 dedicated wave.
- ✅ Question 8 from rev 2 (F-39 tooltips): not pursued; snapshot growth tradeoff judged not worth the UI-drift coverage.

**Audit cycle closed.** A new audit pass (rev 5+) starts only if the codebase regresses or new structural concerns emerge.
