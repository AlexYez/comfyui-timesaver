---
name: reference_schedulers_injector
description: "Side-effect модули, регистрирующие доп. schedulers (beta57, bong_tangent) и samplers (res_2s) в comfy.samplers через публичный SchedulerHandler/SCHEDULER_HANDLERS/SCHEDULER_NAMES и k_diffusion.sampling/KSAMPLER_NAMES/SAMPLER_NAMES — невидимы в меню нод, активны в родных KSampler-дропдаунах"
metadata:
  type: reference
---

Side-effect модули, регистрирующие доп. schedulers (beta57, bong_tangent) и samplers (res_2s) в comfy.samplers через публичный SchedulerHandler/SCHEDULER_HANDLERS/SCHEDULER_NAMES и k_diffusion.sampling/KSAMPLER_NAMES/SAMPLER_NAMES — невидимы в меню нод, активны в родных KSampler-дропдаунах

Исходники темы: [ts_schedulers_injector.py](../../nodes/utils/ts_schedulers_injector.py), [ts_samplers_injector.py](../../nodes/utils/ts_samplers_injector.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# Injector pattern (no UI node, only side effect)

Для добавления собственных scheduler-ов / sampler-ов в **родные** ноды ComfyUI (KSampler, KSamplerAdvanced, BasicScheduler, KSamplerSelect) не нужен monkey-patch. ComfyUI core имеет публичный API:

- `comfy.samplers.SchedulerHandler(handler, use_ms=True)` — NamedTuple.
- `comfy.samplers.SCHEDULER_HANDLERS: dict[str, SchedulerHandler]` — реестр.
- `comfy.samplers.SCHEDULER_NAMES: list[str]` — список имён (KSampler.SCHEDULERS = SCHEDULER_NAMES, тот же объект).
- `comfy.samplers.calculate_sigmas(model_sampling, name, steps)` — диспетчер: `SCHEDULER_HANDLERS[name].handler(model_sampling, steps)` если `use_ms=True`, иначе `handler(n=steps, sigma_min=..., sigma_max=...)`.

`KSampler.SCHEDULERS` — **ссылка** на `SCHEDULER_NAMES`, поэтому `.append()` мгновенно виден всем родным нодам без перезагрузки.

## Pattern

Файл `nodes/utils/ts_<topic>_injector.py`:
- Module-level side effect: проверка `if name not in SCHEDULER_HANDLERS:` → регистрация.
- `NODE_CLASS_MAPPINGS = {}` (пустой dict, явно) — loader (`__init__.py:206-233`) вернёт 0 нод, ничего в UI не появится, но `importlib.import_module(...)` уже выполнил регистрацию.
- Try/except ImportError вокруг `from comfy.samplers import ...` — fail-soft + WARNING с `[TS ...Injector]` префиксом, чтобы не свалить весь пак на неожиданной версии ComfyUI.
- Идемпотентность: если другой пак (например RES4LYF) уже зарегистрировал то же имя — пропускаем с INFO log.

## Эталон

[nodes/utils/ts_schedulers_injector.py](../../nodes/utils/ts_schedulers_injector.py) + [tests/test_schedulers_injector.py](../../tests/test_schedulers_injector.py).

Конкретно `beta57` = `partial(comfy.samplers.beta_scheduler, alpha=0.5, beta=0.7)` с `use_ms=True`. Это **не** оригинальная реализация — её просто нет, в RES4LYF `beta57` тоже вызывает встроенную `comfy.samplers.beta_scheduler` с этими параметрами (см. их `sigmas.py:1420-1421` и `helper.py:get_res4lyf_scheduler_list`).

## Зарегистрировано (9.31)

- **Schedulers** (`nodes/utils/ts_schedulers_injector.py`): `beta57`; `bong_tangent` — двухстадийная arctangent-кривая сигм `((2/π)·atan(−slope·(x−pivot))+1)/2`, defaults RES4LYF (slope 0.2, pivot 0.6, middle 0.5), масштабируется на реальный `model_sampling.sigma_max` → работает и на eps (SDXL), и на flow (Flux/SD3.5).
- **Samplers** (`nodes/utils/ts_samplers_injector.py`): `res_2s` — детерминированный экспоненциальный Runge-Kutta 2-го порядка («RES») в домене λ=−log σ. Сводится к экспоненциальному Эйлеру (DDIM) когда обе оценки модели совпадают; иначе 2-й порядок через φ₂-коррекцию.
- Все три — clean-room по публичной математике RES4LYF (AGPL-3.0 + non-commercial), **без копирования кода**. Тесты: `test_schedulers_injector.py`, `test_res4lyf_inject.py` (unit) + live/browser проверки res_2s/bong_tangent в `test_comfyui_live_api.py` / `test_browser_smoke.py`.

## Attribution policy (light credit)

Имя `beta57` и параметры `alpha=0.5/beta=0.7` популяризовал [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) (ClownsharkBatwing, AGPL-3.0 + non-commercial clause). Юридически copyright охраняет **текст кода**, а не числа/имена/алгоритмы — поэтому пока мы не копируем ни строчки из RES4LYF, AGPL не задействован и формальная атрибуция не требуется. Тест `test_injector_does_not_depend_on_res4lyf` фиксирует автономность как контракт.

Policy для будущих заимствованных presets:

- **Если код реально скопирован/адаптирован** из стороннего пака — полная атрибуция в docstring + проверка лицензионной совместимости (AGPL/GPL копирайт-вирусны, в MIT-пак тащить нельзя). Это блокер.
- **Если используем только идею / имя / числа** — достаточно одной строки комментария-сноски у регистрации (`# preset "X" popularised by <pack> (<url>)`). Без длинных Attribution-блоков в docstring. Эталон — `_register_beta57` в `ts_schedulers_injector.py`.

## Why

- Не нарушает one-node-one-file: «injector» — не нода, а сайд-эффект модуля; правило §13 «нет side effects на module-level» относится к загрузке моделей / открытию файлов, а не к лёгкой регистрации в core registry (RES4LYF и десятки других паков делают то же самое для `bong_tangent` и т.д.).
- Не патчит `KSampler.INPUT_TYPES` и не дёргает приватный API.
- Контракт-тест полностью CPU-only: стабит `comfy.samplers` через `monkeypatch.setitem(sys.modules, ...)`, потому что обычный test Python не содержит torch/comfy core.

## How to apply

- Когда понадобится добавить ещё один scheduler — расширяй тот же файл (новая функция `_register_<name>()` + вызов внизу), либо заведи `ts_<feature>_injector.py` если логика принципиально иная.
- Sampler-инжектор уже реализован — `nodes/utils/ts_samplers_injector.py` (`res_2s`). Механизм: `setattr(comfy.k_diffusion.sampling, "sample_<name>", fn)` — `ksampler()` резолвит сэмплер через `getattr(k_diffusion_sampling, "sample_<name>")` (см. `comfy/samplers.py:1030`) — + append имени в `comfy.samplers.KSAMPLER_NAMES` И `SAMPLER_NAMES` (KSampler.SAMPLERS = SAMPLER_NAMES, тот же list-объект). Сигнатура: `sample_<name>(model, x, sigmas, extra_args, callback, disable, **kwargs) -> x`; `model(x, sigma)` возвращает denoised (x0). Новую функцию-сэмплер добавляй туда же или в новый `_register_<name>()`.
- Контракт-тест на любой новый scheduler обязателен (registered + idempotent + correct handler args).
