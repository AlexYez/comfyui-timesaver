---
name: ts-smart-inpaint-handoff
description: "История: Worklog/handoff по ноде TS Smart Inpaint — что сделано (A0/A1 perf) и что дальше (feather/blur, large-crop). Контекст для продолжения работы в новой сессии."
metadata:
  type: reference
  status: historical
---

История: Worklog/handoff по ноде TS Smart Inpaint — что сделано (A0/A1 perf) и что дальше (feather/blur, large-crop). Контекст для продолжения работы в новой сессии.

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

# TS Smart Inpaint — handoff (создан 2026-06-22)

Контекст для продолжения работы над `nodes/image/ts_smart_inpaint.py`. Архитектура ноды —
в [ts-smart-inpaint-node](../engineering/ts-smart-inpaint-node.md). Правило dev/runtime — в [feedback_runtime_only_locations](../owner/feedback_runtime_only_locations.md).

## ✅ Сделано в этой сессии: A0 + A1 (commit `deccb54`, на GitHub + C: + D:)

Нода — single-shot, **только IMAGE-выход**, порт интерактивной ноды Angelo (у той 3 выхода
IMAGE/LATENT/IMAGE + межкликовый латент-холст `state["history"]`). Поэтому в порте финальный
полнокадровый re-encode + latent-blend был **мёртвым кодом** (его `new_latent` отбрасывался).

- **A0** — убран пост-композитный `_vae_encode(new_pixels)` + latent-blend; функция возвращает
  `None` вместо латента. Выход (пиксельный композит) не изменился ни на пиксель.
- **A1** — убран стартовый `current = _vae_encode(vae, image)`. Геометрия латента берётся из
  `vae.spacial_compression_encode()` (ComfyUI центр-кропит до кратности перед энкодом → латент
  `== img_side // ratio` для `crop_input` VAE; фолбэк на честный энкод при `crop_input=False`).
  Полный латент нужен только ветке Refine-без-апскейла (`scale<=1.0`, Refine only) — там энкод
  теперь **ленивый**. Геометрия внутри ядра читается из `mask.shape`, не `current.shape`.

**Итог:** в Replace / Refine-с-апскейлом через VAE идёт ТОЛЬКО кроп, полный 6–8K кадр — никогда.
Output byte-identical для всех crop_input VAE (SD/SDXL/FLUX/Qwen/Wan). Алгоритм инпейнта (crop→
upscale→encode→reference/zero→sample→decode→composite) НЕ тронут.

**Verified:** py_compile; smoke-import + сборка схемы; прогон `execute()` со stubbed VAE/семплером —
в Replace полный кадр не кодируется (только кроп), ленивый энкод срабатывает лишь в Refine-shortcut.

## ⛔ Сделано в сессии 2 (commit `4c1288e`) — `hard_edge` — ПОЗЖЕ УДАЛЁН в сессии 3 (`4ddbbdb`)

> ⚠️ `hard_edge` БОЛЬШЕ НЕТ. Пользователь решил, что опция не даёт пользы — удалена целиком
> (код + инпут + helper `_dilate_mask_2d` + `sample_mask_override`) в commit `4ddbbdb`. НЕ возвращать.
> Раздел ниже оставлен для истории. Актуальное состояние — в секции «сессия 3».

Фикс «мыла» реализован как **opt-in переключатель** `hard_edge` (IO.Boolean, default `False`,
добавлен ПОСЛЕДНИМ инпутом перед опциональным `reference` → widgets_values append-only, старые
workflow не ломаются). Дефолт (`False`) — **байт-идентичен** старому поведению.

`hard_edge=True` разводит две маски (тот же приём, что 5D-ветка уже делала внутри, теперь и для 4D):
- **маска генерации** (masked-zero + noise_mask) → **жёсткая бинарная** (`mask_raw>0.01`, порог как у
  bbox) → модель полностью дорисовывает область, без soft-zero leak и недо-денойза тонких мазков;
- **feather → только в композите, наружу**: `_dilate_mask_2d(binary, ceil(3·σ_seam))` → `_gaussian_blur_2d(σ_seam)`,
  где `σ_seam = max(0.5, σ_latent/3)`. Внутри нарисованного `alpha=1` (полная замена), узкая рампа уходит
  в контекст наружу, не выедая мазок. `feather=0` → жёсткий рез без рампы.

Ядро Angelo X-Refine не тронуто: `_refine_with_fine_upscaling` получил опц. параметр
`sample_mask_override`; при `None` путь байт-идентичен. Новый helper `_dilate_mask_2d` (max_pool2d).
Refine-no-upscale shortcut (`scale<=1.0`, не Smart Inpaint) override НЕ применяет (там нет пиксельного
композита для рампы) — задокументировано в коде.

**Verified (под ComfyUI python `D:/AiApps/.../python/python.exe`):** py_compile; новый
`tests/test_smart_inpaint_hard_edge.py` (standalone, mocked VAE/sampler) — 14 проверок PASS:
дефолт=feathered+crop-only; hard=binary sample mask + полностью занулённый регион + crop-only; 5D
binary в обоих режимах; на тонком мазке outward-feather держит `alpha=1.000`, а legacy симметричный
blur даёт пик `0.213` (демонстрация «мыла», которое фикс убирает). Регенерён `node_contracts.json`
(теперь 60 нод, `TSSmartInpaint` с виджетом `hard_edge`), `--check` зелёный. `test_node_contracts`,
`test_pack_imports`, `test_static_invariants` — pass (1 пред-существующий fail в `_whisper_engine.py`
cuda-hardcode, НЕ связан). Полный suite не гонял — нужен live ComfyUI на :8188.

Runtime D: НЕ трогал — обновится `git pull` / ComfyUI Manager + рестарт ComfyUI (действие пользователя).

## ✅ Сделано в сессии 3 (commit `4ddbbdb`, 2026-06-22): цветокоррекция + консолидация опций

Три директивы пользователя (через AskUserQuestion):
1. **`color_correct` — НОВАЯ опция, default ON, ПОСЛЕДНИМ инпутом.** Fix «покраснения» Flux на
   сгенерированной области. Подход — **кольцо по периметру** (выбран пользователем):
   - сдвиг оценивается ТОЛЬКО по СОХРАНЁННОМУ кольцу вокруг маски (alpha<0.02 в композите) — там
     `refined` и `original` это ОДИН контент → их поканальная разница = чистый сдвиг Flux;
   - инверсия (робастный поканальный gain+offset, MAD-триминг выбросов, клампы a∈[0.5,2], b∈[-0.25,0.25])
     применяется ко всему патчу перед композитом — helper `_color_correct_patch`;
   - **устойчивость:** оценка по неизменному окружению → не зависит от того, сколько/какого нового
     контента нарисовано, и НЕ перекрашивает объект под цвет окружения (красный на синем фоне остаётся
     красным — это SHIFT-оценка по парным пикселям, НЕ «match к окружению»);
   - чтобы кольцо всегда было, при ON `context_pad` поднимается до `_CC_ANALYSIS_MARGIN_PX=32` (лёгкая
     связь с генерацией — задокументирована); крошечное кольцо → no-op.
2. **`hard_edge` — УДАЛЁН целиком** (см. ⛔ выше).
3. **`limit_resolution` — сделан БЕЗУСЛОВНЫМ, тумблер убран.** `_fine_upscale_factor` всегда трактует
   `megapixels` как истинный потолок (мелкие апскейл/cap `max_linear`, большие downscale). Удалены
   мёртвый whole-frame Refine-shortcut и кламп `scale=max(1.0,scale)`.

**`color_correct=OFF` = прежний shipped-дефолт (downscale-on + soft feather) байт-в-байт.** Ядро
Angelo (helpers + crop→encode→sample→decode→composite) не тронуто.

**Verified (ComfyUI python):** compileall; `tests/test_smart_inpaint.py` (заменил старый
`test_smart_inpaint_hard_edge.py`) — все проверки PASS: схема (color_correct last, default True;
hard_edge/limit_resolution отсутствуют); безусловный downscale (1728px→1000px@1МП); `_color_correct_patch`
точно снимает известный gain+offset на нарисованной области (err 0.0000), красный остаётся красным
(оценка ring-only), tiny ring → no-op, вход не мутируется; execute дёргает CC при ON и пропускает при OFF
(4D и 5D). Контракт регенерён (`color_correct{default:true}`), `--check` зелёный; `test_pack_imports`,
`test_node_contracts` pass. Runtime D: обновлён `git pull` → `4ddbbdb`, blob C==D. Нужен рестарт ComfyUI.

**Параметры CC (если потребуется тюнинг):** `_CC_ANALYSIS_MARGIN_PX=32` (ширина кольца, image-px);
в `_color_correct_patch` — `alpha_eps=0.02`, `min_samples=256`, клампы gain/offset, `strength=1.0`
(не вынесен в UI — пользователь просил «просто вкл/выкл»). Альтернативный источник оценки (extra-decode
всего кропа) НЕ реализован — выбрали кольцо.

## ✅ Сделано в сессии 4 (commit `b920d41`, 2026-06-22): feather/context → проценты

Пользователь заметил: абсолютные px у `feather`/`context_pad` дают разный относительный эффект на
разных масках (15px = ~20% маленькой маски → переразмытие/призрак, ~1% большой). Решение (согласовано
через AskUserQuestion: **rename в `*_pct` + дефолт context ~8%**):
- `feather` (Int 15px) → **`feather_pct`** (Float %, default `3.0`); `context_pad` (Int 0px) →
  **`context_pct`** (Float %, default `8.0`). Проценты от **короткой стороны tight-bbox маски** (image px),
  через helper `_pct_to_px` с клампами (`_FEATHER_FLOOR_PX=2`, `_FEATHER_CEIL_PX=256`, `_CONTEXT_CEIL_PX=1024`).
- Якорь на маску, НЕ на процессинг-разрешение напрямую: т.к. кроп нормализован к `megapixels`,
  «% от маски» == «% от процессинг-кропа», но без циклической зависимости (context влияет на размер кропа).
- **RENAME, а не reinterpret in-place** — старое сохранённое px-значение (15) прочитается как явно
  неверный процент (15%), а не «тихо сломается». Маппинг widgets_values по позиции, имена те же слоты.
- Вся логика в `execute` (считает short side bbox → feather_px/context_px); ядро Angelo не тронуто.
  CC-кольцо держит свой отдельный floor 32px (`max(context_px, _CC_ANALYSIS_MARGIN_PX)`).
- **Бонус:** это чинит «мыло на тонких мазках» в корне (маленькая маска → маленькая кайма) — фундаментальнее
  удалённого `hard_edge`.

**Verified (ComfyUI python):** compileall; `tests/test_smart_inpaint.py` все PASS (схема: feather_pct=3.0/
context_pct=8.0, старые px-имена отсутствуют; `_pct_to_px`: 3%×500=15px, 3%×100=3px, 3%×5000=150px, floor
2px, ceil 256px; downscale/CC/5D по-прежнему ок). Контракт регенерён, `--check` зелёный; pack_imports/
node_contracts pass. D: → `b920d41`, blob C==D. Нужен рестарт ComfyUI.

## ✅ Сделано в сессии 5 (commits `92a0a77` + `d7d5b37`, 2026-06-22): UI-declutter + фикс тёмной линии feather

1. **UI-declutter (`92a0a77`):** `max_linear` и `color_correct` убраны из UI → константы вверху файла
   `_MAX_LINEAR=3.0` / `_COLOR_CORRECT=True` (CC всегда вкл). Меняются только в коде. Ядро не тронуто
   (константы прокидываются в его параметры). Текущий UI: replace, denoise, megapixels, context_pct,
   feather_pct, resize_method, seed, steps, cfg, sampler_name, scheduler, (+ optional reference).

2. **Фикс «тёмной мягкой линии по периметру маски» (`d7d5b37`)** — пользователь заметил тёмную полосу
   именно в зоне feather. Две причины, обе устранены (согласовано «оба фикса»):
   - **Гамма-композит:** блендинг `refined·α+orig·(1−α)` шёл в sRGB → кросс-фейд темнеет. Теперь композит
     в **линейном свете** (`_srgb_to_linear`/`_linear_to_srgb`, точная piecewise, fp32). Round-trip при
     α=0/1 → меняется только feather-полоса (тест: midpoint 0.2↔0.8 поднялся 0.5→0.6).
   - **Мягкое зануление латента:** в feather-рампе `sample_mask≈0.5` → латент полузануляется → тянется к
     среднему (≈серый у Flux), частично дорисовывается → тёмно-мутная полоса ДО композита. Теперь маска
     семплинга **жёсткая бинарная для 4D И 5D** (`mask_crop_up>=0.5`) — регион генерится полностью,
     контекст сохраняется, feather только в (мягком) композите. Это рецепт, который 5D уже использовал;
     раньше была развилка 4D-soft/5D-hard — убрана. Это и есть ядро удалённого `hard_edge`, но **как
     безусловный дефолт и БЕЗ outward-сдвига**.
   - Меняет вывод Replace/Refine у краёв маски (это и есть фикс).

**Verified (ComfyUI python):** compileall; `tests/test_smart_inpaint.py` все PASS (схема без max_linear/
color_correct + константы; `_pct_to_px`; sRGB round-trip err 0.0000, midpoint 0.600>0.500; маска семплинга
бинарная для Replace и Refine; CC всегда вкл через `_COLOR_CORRECT`, OFF через флип константы). Контракт
зелёный (схема в раунде 2 не менялась). `test_pack_imports`/`test_node_contracts`/`test_static_invariants`
— **12 passed** (cuda-hardcode whisper починен в `f92a02d`, не моё). D: → `d7d5b37`, blob C==D. Рестарт ComfyUI.

## 🔜 Дальше (НЕ сделано, согласовано как опциональное, по умолчанию ничего не менять)

### C. ✅ СДЕЛАНО (commit `4c1288e`) — Мыло на новой области (была главная жалоба) — см. секцию выше
**Симптом:** свежесгенерированная область выглядит размытой/призрачной, особенно на тонких/
произвольных мазках кистью (не на больших прямоугольниках).

**Корень:** слишком широкий feather + три накладывающихся «мягких» механизма в Replace (4D FLUX/Klein):
1. мягкое обнуление латента `latent_up = (1-mask)*latent_up` — у края старое «протекает»;
2. мягкий `noise_mask` при семплинге — край денойзится не на полную (смесь старого/нового);
3. мягкий пиксельный композит — кросс-фейд нового и оригинала.

`feather=15` (дефолт) при гауссе ±3σ даёт мягкую кайму ≈±45px (~90px шириной), причём симметрично —
**съедает маску внутрь**. Связь с формой: blur ∝ периметр/площадь. Большой прямоугольник — кайма
только по границе (внутри alpha=1, чётко). Тонкий мазок ⊂ каймы → alpha нигде не доходит до 1 →
новый контент НИКОГДА полностью не заменяет старый → мыло.

Важно: для 5D (Qwen/Wan) автор УЖЕ сделал правильно — жёсткая (бинарная) маска семплинга, feather
только в композите. Нужно распространить этот приём на 4D-Replace.

**План фикса (как опция, дефолт сохранить):**
- развести «маску генерации» (обнуление + noise_mask) → жёсткая/почти жёсткая, чтобы модель
  полностью дорисовывала новое (как 5D-ветка);
- feather → только в композите, узкий, смещённый НАРУЖУ (dilate→blur), чтобы внутри нарисованного
  alpha=1, а рампа уходила в контекст, не выедая нарисованное;
- снизить дефолт feather и/или ограничить ширину каймы долей от меньшей стороны маски;
- прямоугольная маска → минимальный/нулевой feather; произвольная → узкий feather наружу.
- РЕАЛИЗОВАТЬ как переключатель (напр. `hard_edge` / «feather только в композите»), не меняя
  поведения для тех, кто привык к текущему.

### B. ✅ Закрыто опцией `limit_resolution` (commit `25dde3e`) — Тяжёлая ветка Refine-без-апскейла
`scale<=1.0` Refine семплил ВЕСЬ латент кадра и декодил целиком — дорого/вис на 6–8K. Теперь при
`limit_resolution=ON` (дефолт) этот shortcut пропускается → большой Refine-кроп идёт через
ограниченный crop+downscale путь (гейт `and not allow_downscale`). При OFF — legacy shortcut
байт-в-байт. Менять границы при ON — это и есть согласованное поведение опции.

### ✅ СДЕЛАНО (commit `25dde3e`, 2026-06-22) — гигантские маски: опция `limit_resolution` (default ON)
Была причина зависания на больших картинках (жалоба пользователя про 7000×7000, обувь ~1/5):
Smart Inpaint клампил `scale = max(1.0, scale)` и НИКОГДА не уменьшал → кроп в несколько МП кодился/
семплился в нативе → hang/OOM. `megapixels` работал как нижняя граница, не как цель. (НЕ регресс
A0/A1 — та логика не менялась; A1 наоборот убрал стартовый энкод полного кадра.)

**Фикс:** `limit_resolution` (IO.Boolean, **default `True`/ON** — согласовано флипнуть дефолт через
AskUserQuestion, т.к. старый дефолт вис) делает `megapixels` истинным потолком:
- `_fine_upscale_factor` получил `allow_downscale`: при `current_mp>=target` и ON → фактор `sqrt(target/current)<1` (downscale);
- кламп `scale=max(1.0,scale)` для Smart Inpaint снимается при ON;
- Refine no-upscale shortcut пропускается при ON (см. B);
- кроп даунскейлится до бюджета → encode/sample → апскейл обратно в натив для композита (output res не меняется, страдает только детализация обработки). Поднять `megapixels` (до 8) = больше деталей.
- OFF восстанавливает legacy натив **байт-в-байт** (escape hatch для макс. качества).
Инпут добавлен ПОСЛЕ `hard_edge` (widgets_values append-only). Тест [6] в `test_smart_inpaint_hard_edge.py`:
ON → кроп 1664px(2.77МП)→1000px(1.00МП ровно бюджет); OFF → натив 1664px; crop-only сохранён.

**Verified:** py_compile/compileall; 18 проверок в test_smart_inpaint_hard_edge.py PASS; контракт
регенерён (60 нод, `limit_resolution{default:true}`), `--check` зелёный; test_pack_imports/test_node_contracts pass.
Runtime D: обновлён `git pull` до `25dde3e`, blob C==D. Нужен рестарт ComfyUI пользователем.

## Справка: параметры
- **`megapixels`** — целевое разрешение обработки кропа.
- **`max_linear`** — потолок линейного апскейла кропа (`min(sqrt(target/current), max_linear)`).
  Защита: крошечная маска не раздувается в 8–10× (VRAM, галлюцинации, мыло при downscale обратно).

## Как тестировать (без реальной модели)
Embedded-python ComfyUI: `D:/AiApps/ComfyUI/comfyui/python/python.exe`. Паттерн — импорт через
синтетический `tspack` (см. [reference_tests](../engineering/reference_tests.md)) + `args.cpu = True` ДО импорта пака.
Логику `execute()` можно гонять, замокав модульные `_vae_encode`/`_vae_decode`/`_do_sample` и
`latent_preview.prepare_callback`, с фейковым `vae` (`.spacial_compression_encode()`, `.crop_input`).
В этой сессии так проверяли, что в Replace полный кадр не кодируется.

## Ограничения (не нарушать)
- Это byte-for-byte порт Angelo X-Refine (MIT, notice в шапке файла). Ядро инпейнта — sacred.
- Любая правка feather МЕНЯЕТ результат → только опционально, дефолт сохранить.
- Пользователь доволен качеством алгоритма; цель — улучшать, НЕ ломая X-Refine.
