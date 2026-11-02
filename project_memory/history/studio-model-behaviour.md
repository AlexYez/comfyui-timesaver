---
name: studio-model-behaviour
description: "История: Measured behaviour of each image family in TS Image Studio — which model suits which repaint job, and what each denoise range actually does"
metadata:
  type: reference
  status: historical
---

История: Measured behaviour of each image family in TS Image Studio — which model suits which repaint job, and what each denoise range actually does

TS Image Studio снята из текущего пака. Код до удаления доступен в [истории Git](https://github.com/AlexYez/comfyui-timesaver/tree/c1668b3cfa2161e36bf9b9fa91288b949b4b0b1f/nodes/image/studio); удаление зафиксировано [коммитом a2251b1](https://github.com/AlexYez/comfyui-timesaver/commit/a2251b1). Записанное решение владельца датировано 2026-09-27, коммит — 2026-09-28: это разные события.

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

Measured 2026-08-03 on identical tasks (same source, same mask, same prompt,
same seed) through the studio's own backends. `inside` = mean |diff| within the
mask, `outside` = outside it, both out of 255.

**Denoise ranges are universal across families:**
- 0.1-0.25 — refinement: the object stays itself, texture and fine detail
  improve. This is what "improve the quality" actually wants.
- 0.3-0.5 — visible rework: shape and pattern change.
- 0.6-1.0 — replacement: the mask fills with what the prompt describes.

**Per family** (replace / detail / quality tasks):
- **Flux 2 Klein** — deliberately gentle (0.1-0.5 by design); detail at 0.2
  measured 7.8, the softest tool in the pack. Full replacement lives on its
  Replace switch, and it is the FASTEST at it: ~42s vs 113-172s elsewhere.
- **Krea 2** — best composition on replacement (a whole pomegranate that sits
  in the scene); saturates past 0.65 (56.8 -> 56.9 at 1.0), so its useful
  range is roughly 0.2-0.65.
- **Z-Image** — the most linear knob: 27.9 / 57.3 / 90.3 at 0.35 / 0.65 / 1.0.
  Best when the amount of change has to be dialled precisely.
- **Qwen Image** — best "add detail without substituting the subject" (16.5 at
  0.35). Needs its 4-step Lightning LoRA, or a repaint takes minutes.
- **Ideogram 4** — built for graphics and type, weakest at photo retouch. Had
  NO strength control until this session (its sampler takes strength by
  trimming the sigma schedule — SplitSigmasDenoise).

**Bleed is not a thing**: outside the mask every family measured 0.01-0.08.
The blend ring reaches ~50px; past 64px the frames are bit-identical.

**Upscale**: SeedVR2 hits its scale exactly (2x, 4x) in 20-26s and adds real
detail on large sources. The diffusion second pass takes ~10 min for 4x to
3840px; fidelity to the source is 9.3 at denoise 0.30 and 21.4 at 0.45, and at
0.45 a fourfold enlargement invents ornament instead of resolving it.

See also [ideogram4-caption-format](../engineering/ideogram4-caption-format.md), [studio-hosts-node-editors](studio-hosts-node-editors.md).

## Времена сквозного прогона (2026-08-04, через интерфейс студии)

Прогон «каждое семейство в каждом режиме» на RTX (12 ГБ занято), источник
1008×1008. Цифры — от нажатия Run до файла в истории, включая загрузку модели
при первом обращении к семейству.

| Семейство | Генерация | Перерисовка | Заметка |
|---|---:|---:|---|
| Flux 2 Klein | 28 с | 33 с | самый быстрый на всех режимах |
| Z-Image | 43 с | 722 с | перерисовка была на 2.6 МП (наследство памяти параметров) |
| Krea 2 | 43 с | — | |
| Qwen Image | 83 с | 435 с | 20B, перерисовка честно долгая |
| Ideogram 4 | 136 с | **4824 с (80 мин)** | апскейл 756 с; самая тяжёлая связка пака, см. ниже |
| Qwen Image Edit | 144 с | — | с референсом в ячейке; без него честно отказывает |

**Отдельной вкладки Edit в студии нет.** `qwen-image-edit` — модель внутри
**Generate**, и в списках Inpaint/Upscale её нет вовсе. Правки делаются так:
Generate → qwen-image-edit → положить картинку в ячейку референса → промпт →
Run. Скрипт, который ищет вкладку «Edit», не найдёт ничего — вкладок ровно
шесть: Generate, Inpaint, Upscale, Settings, Packs, Help.

Каждая пара «семейство × режим», которую предлагает студия, доведена до файла
через интерфейс (2026-08-04). Пустые клетки выше — это то, чего в студии нет,
а не то, что не проверено.

**Перерисовка Ideogram — 80 минут на Turbo (12 шагов), 1008².** Это не сбой:
граф грузит **два** UNet (`ideogram4_fp8_scaled` + `ideogram4_unconditional`),
8-миллиардный текстовый энкодер (`qwen3vl_8b`) и VAE от Flux 2, а LanPaint на
каждый шаг делает ещё внутренние итерации — и всё это перерисовывает **кадр
целиком**, а не вырез по маске (вырез только у Klein). На 16 ГБ это означает
постоянную подкачку весов. GPU при этом ровно на 100 %, а прогресс сэмплер не
шлёт вовсе — по сокету за 40 секунд не приходит ни одного `progress`, так что
«завис» и «считает» снаружи неотличимы. Смотреть на `nvidia-smi`.

Практический вывод: в таком виде режим годится для «поставил и ушёл», а не для
работы. Если понадобится сделать его пригодным — путь тот же, что у Klein:
перерисовывать вырез по маске, а не весь кадр.

Апскейл 4x от 1008² на Flux 2 Klein — 284 с, результат 4032×4032.

Важно: время перерисовки сильно зависит от «разрешения обработки», а оно
запоминается между сеансами (см. [reference_studio_persistence](reference_studio_persistence.md)). Сравнивать
семейства имеет смысл только на одинаковом значении.

## Инпэйнт открывается в Cleanup, а не в Repaint

Полоса режимов над картинкой — Cleanup | Repaint, по умолчанию **Cleanup**.
В нём мазок не «копится в маску»: `onStrokeEnd` сразу гонит LaMa, а вернувшаяся
чистая картинка загружается в холст, а `loadImage` обнуляет маску и историю.
Поэтому «нарисовал → нажал Run» в Cleanup даёт «Paint a mask first»: маски
действительно уже нет.

Для перерисовки моделью сперва переключить сегмент на **Repaint**
(`.ts-inp__seg button` — второй), и только потом красить.

Чем это отравляет автотесты: отказ приходит с задержкой и выглядит плавающим —
если Run нажать раньше, чем LaMa успела вернуть картинку, маска ещё на месте
и запуск проходит. Из-за этого один и тот же прогон давал разный результат для
двух моделей подряд (2026-08-04: ideogram отказ, krea2 успех — разница была
только во времени, не в семействе). Любой скрипт, который красит маску, обязан
сначала выбрать Repaint.

Порядок отказов в `run()` тоже сбивает с толку: без промпта дело до маски не
доходит («Write a prompt or open the designer first»), поэтому отсутствие
жалобы на маску ещё не значит, что маска есть.

## Апскейл сцепляется с предыдущим результатом

В режиме апскейла источник — выбранная работа сессии. Свежий результат
становится выбранным, поэтому второй запуск подряд увеличивает уже увеличенное:
1008² → 4032² (4x) → 8064² (ещё 2x), и время растёт соответственно. В прогоне
2026-08-04 именно так набежали десятки минут на Qwen — увеличивалось 4032², а
не исходный кадр.

Это не ошибка, а следствие «источник = то, что на сцене». Чтобы мерить
семейства честно, перед каждым запуском выбирать исходник заново (перетащить
из библиотеки) — иначе сравниваются разные задачи.

Куда это приводит: в прогоне 2026-08-04 цепочка прошла 1008² → 4032² → 8064²
→ **16128²** (260 Мп, файл 168 МБ; Krea 2 такое вытянула), а SeedVR2 на 8064²
честно упал с `out of memory` на 16 ГБ — запрошено было увеличение уже
увеличенного. Ошибка ComfyUI, а не студии, но человек в неё
въезжает незаметно: он видит «апскейл», а не «апскейл того, что получилось
в прошлый раз». Размер источника виден в подписи под сценой — при разборе
жалоб «почему так долго / почему OOM» смотреть туда первым делом.
