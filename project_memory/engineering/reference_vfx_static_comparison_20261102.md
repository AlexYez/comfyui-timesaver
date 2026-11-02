---
name: reference_vfx_static_comparison_20261102
description: "Статичное сравнение апскейлеров 2026-11-02; VFX 0.2 и DLSS v21 выпущены в 12.13.6."
metadata:
  type: reference
---

Статичное сравнение апскейлеров 2026-11-02; VFX 0.2 и DLSS v21 выпущены в 12.13.6.

**Текущая опора (2026-11-02):** Результаты изображений и HTML находятся в локальном ignored cache/dist, поэтому в чистом клоне их нет. Изменение API-зависимости NVIDIA VFX вошло в 12.13.6; код ноды и сравнения описаны ниже как зафиксированный эксперимент.

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Источник владельца: output/images/artius-flux-2-dev-33b/artius-flux-2-dev-33b-2026-11-02_150830_00001_.png,
1328x800 RGB. В сообщении путь был ошибочно разорван перед _.png; нашли точный файл.
Сравнение 2x 2656x1600 в dist/upscale-comparison-20261102/index.html.
HTML открыт у пользователя в Chrome, подтверждён AX: 7 rows, full pictures,
face/hair crops, SDK inference / full processing time, disk sizes, A/B slider.
Локальный HTTP 127.0.0.1:18765, PID33880; http.json в cache. Код генератора
и тестов в tests/.cache/upscale-comparison-20261102/. Встроенный IAB не смог
attach, open_in_codex queued; Chrome успешно показывает страницу. Не трогать
выбранные пользователем варианты (на проверке v21 vs SharpNext).

Состав: Lanczos; TS DLSS v9; Visual Enhancer21 NR+Standard/Lanczos4 2x;
NvidiaVFX ULTRA/HIGHBITRATE_ULTRA; SharpNext; TS RTX VFX0.2 (пиксели равны
официальному VFX ULTRA). NRv9/v21 имеют разные pipeline; это НЕ чистый тест
обновления весов. v9skin_structure0, Default,1pass,temporaloff. VFX RGB8,strength1.
Время: latest receipts JSON в dist. SDK run/process_float замерен отдельно от
node execute; CLI includes IO/startup and first AI model downloads. v21 NR4.585s,
SharpNext420.782s total incldownload,394.109s after startup from native log,
VisionNext103.309s total incldownload,22.043s postinit. Source Flux generation
runtime отсутствует в PNG (только prompt/workflow); не придумывать.

Владелец затем отказался от больших моделей и попросил размеры. VisionNext
исключена из HTML, RazorNext тоже. Последняя успела завершиться до остановки:
проверили отсутствие VE_CLI.exe, run_cli_models.py exited0. Это честно сообщили.
Большие веса УЖЕ скачаны приложением; не запускать снова и не удалять без просьбы.
Точные manifests: VisionNext3294045858 bytes (3.294GB/3.068GiB),
SharpNext611497094 bytes, RazorNext9003747105 bytes. Источники в runtime licenses:
RazorNext=RelayVSR; SharpNext=LPNSR V3; VisionNext=VOSR2. Большие модели не
интегрировали в наш пак. Взято из официального v21 zip824005214bytes,
изолированно распакован в tests/.cache/.../upstream/v21.
DLSSv9 используемые engine570880+caller104960+nvngx_dlssnr165830144 bytes =~166.5MB;
не считать оставленный unused v11engine571904 второй раз.
v21 DLSSNR276551680 bytes; DLSSSR59314800 bytes. NVIDIA VFX0.2 nvvfx installed
560964163 bytes, compressed wheel435.8MB, VSR DLL59120752 bytes; это общий SDK,
не суммировать один и тот же SDK на TS RTX и официальный VFX.

Конфликт: официальный comfyui_nvidia_rtx_nodes0.2.1 требует nvidia-vfx>=0.2,
а global0.1.0.1 не экспортировал nvvfx.QualityLevel (только effects.QualityLevel).
НАШ optional extra был ==0.1.0.1, мог мешать совместной установке. Наша
TS_RTX_Upscaler уже работает с обеими версиями, реальная проверка обоих API
прошла. Изменён extra на >=0.1.0.1,<0.3, инструкции README/en/ru/help/requirements
рекомендуют0.2.0.0 с NVIDIAindex, Unreleased в changelog. 10 cache/cancel tests
прошли; схемы/алгоритмы не менялись. DEV и runtime синхронизированы (7 files),
backup runtime-before-sync.zip. Версия12.13.5 не повышалась, commit/push НЕТ.

Сначала isolated0.2 в cache/vfx02. Global upgrade while server alive failed
WinError5 nppc64_12.dll; восстановили16 отсутствующих файлов из vfx01-backup,
проверили hashes old installation. По асинхронному разрешению пользователя
«Да, обнови и перезапусти» остановлен idle единственный ComfyUI, обновлён
ТОЛЬКО nvidia-vfx (no-deps), запущен PID39252 через CREATE_NO_WINDOW с прежними
argv: --max-upload-size5000 --preview-method latent2rgb --use-ck-attention
--enable-manager --port8188 плюс --disable-auto-launch. Python/PyTorch не меняли.
Временные ~vvfx и ~vidia_vfx-0.1.0.1.dist-info после неудачного pip перемещены
в cache/pip-failed-uninstall (не удалены), чтобы не было invalid distribution.

Live smoke history c3377897-05d7-487a-ba7a-36160d93324b status success,
официальная RTXVideoSuperResolution зарегистрирована, обе ноды сохранили новые
PNG в output/images/upscale-comparison-20261102. API DynamicCombo надо отправлять
плоско: resize_type='scale by multiplier', resize_type.scale=2.0, НЕ dict.
Первый наш API тест ошибся этим и получил missing resize_type; второй исправлен.
Изолированный direct execute принимает dict — не путать два формата.
