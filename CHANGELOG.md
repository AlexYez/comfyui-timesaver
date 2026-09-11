# Changelog

Only changes a **user** can notice: what appeared, what changed behaviour, and —
above all — what stopped working and what to do about it.

Node ids, input names, their order and their defaults are frozen on purpose: a
workflow saved a year ago must open today. When that promise is broken, it is
broken here, in writing, with a way back.

---

## 12.7.2

### Старые workflow снова видят свои модели в TS Files Downloader

Воркфлоу, сохранённый до того, как в загрузчик добавили настройку `integrity_mode`,
открывался с пустым списком: вместо ваших ссылок нода показывала образец из схемы
(Dropbox/HuggingFace-заглушки) и на прогоне качала бы не те файлы. Причина — в
общем механизме совместимости: он определял формат сохранения по длине массива
значений и короткий старый массив принимал за «новый формат», не раскладывая его
по именам. Теперь формат определяется подгонкой типов, а не длиной, поэтому
сохранение, сделанное до любого добавления настройки в конец схемы, читается
верно.

Затронуты все ноды со скрытым интерфейсом, у которых схема пополнялась со
временем; список моделей загрузчика — самый заметный случай.

### TS Audio Loader: режим «запись» переживает перезагрузку

В старом графе нода с режимом `record` при открытии сбрасывалась в `load`. При
восстановлении пути к источнику внутренний сторож форсил режим и перезаписывал
остальные поля ещё не восстановленным состоянием. Теперь на время загрузки этот
сброс подавляется; выбор файла вручную по-прежнему переводит ноду в `load`.

---

## 12.7.1

### TS Frame Interpolation: возврат точной длины после подрезки

Апскейл нередко портит первые кадры. Их срезают, из 200 кадров остаётся 188, а
на выходе снова нужны 200 — иначе результат не ложится на исходный звук и не
сравнивается с оригиналом.

Новый режим `match_length` делает это одной нодой. Целевую длину нода берёт из
входа `reference` — подключите исходный батч, оттуда читается только число
кадров, — либо из поля `target_frames`, когда исходника в графе уже нет.

Чем добирать, решает `fill`:

- `hold_start` — повторяет первый кадр в начале. Уцелевшие кадры остаются на
  своих исходных местах, синхронизация с оригиналом сохраняется. Значение по
  умолчанию, и для подрезанного начала правильное именно оно.
- `hold_end` — то же самое с конца.
- `stretch` — пересчитывает весь клип моделью до нужной длины. Глаже, но
  смещает по времени ВСЕ кадры: 188 → 200 это замедление на 6.4%.

Оба `hold_*` работают без модели и без видеопамяти. Если целевая длина меньше
текущей, клип обрезается со стороны удерживаемого края.

Старые воркфлоу не задеты: три новых входа дописаны в конец списка, а значение
`match_length` — в конец списка режимов.

### ⚠️ TS Resolution Selector: другие границы разрешения

Было 0.5–4.0 мегапикселя, стало **0.1–3.0**.

Снизу стало свободнее. Сверху — теснее, и это ломающее изменение: **воркфлоу, где
стояло больше трёх мегапикселей, при открытии подрежется до 3.0**, и разрешение
на выходе у него молча изменится. Если такие графы есть, проверьте их глазами.
Значение по умолчанию прежнее — 1.5.

---

## 12.7.0

### TS Compare — новая нода

Шторка «до/после» прямо в графе, для картинок и для видео одинаково. На входе
два `IMAGE`: одиночные кадры или бачи. Нода сама решает, что собрать — пару PNG
или ролик, — и показывает результат внутри себя: тянешь границу мышью и видишь,
где именно версии разошлись.

Батчи собираются в **один** файл, где A лежит над B, а шторка вырезает из него
две половины. Так сделано намеренно: две отдельные дорожки разъезжаются по
времени, а второй декодер отнимает у генерации видеопамять.

Пока идёт сборка, на самой ноде видна полоска прогресса. Первый кадр показан
сразу, до нажатия Play. Есть полноэкранный режим — кадр в нём центрируется и
масштабируется, при возврате нода принимает прежний размер.

### Интерфейс перестал дёргать браузер на больших воркфлоу

Три постоянно работавших механизма убраны — они жили всё время, пока нода есть
на холсте, независимо от того, открыта она или нет:

- **TS Video Loader** держал обработчик колеса на всём документе, и притом
  блокирующий: браузер обязан был дождаться ответа ноды перед каждой прокруткой
  холста. Теперь обработчик появляется, только когда указатель над таймлайном.
  Это и был единственный пункт, способный реально подтормаживать интерфейс.
- **TS Audio Loader** и **TS Files Downloader** опрашивали свои виджеты таймером
  трижды в секунду. Теперь значение само сообщает о смене.

Поведение нод не изменилось; проверять в старых воркфлоу нечего.

---

## 12.6.1

### Русские подписи у семи нод

`TS Batch Source`, `TS Batch Write`, `TS Batch Load Image`, `TS Latent Upscale`,
`TS Super Prompt RT`, `TS Video Cut` и два входа `TS Film Emulation` выпускались
без русского перевода: на русском интерфейсе у них оставались английские
названия входов и пустые подсказки. Теперь переведены описания, входы и выходы —
66 записей.

### Нода апскейла латента больше не зависит от подвижного API

`TS Latent Upscale` импортировал `comfy_api.latest` — это alias, который
переезжает вместе с ComfyUI, и нода могла сломаться от чужого обновления. Теперь
она, как и весь остальной пак, закреплена на `comfy_api.v0_0_2`.

### Мелочь в TS Batch Write

Умолчание входа `name` в схеме и в коде расходились (`""` против `None`). Для
графа ничего не менялось — расхождение было видно только при вызове ноды мимо
ComfyUI, — но теперь их два одинаковых.

---

## 12.6.0

### TS DLSS Upscaler — новая нода

NVIDIA DLSS 5 Neural Rendering прямо в графе: на входе `IMAGE`, на выходе
`IMAGE`, поэтому и одна картинка, и видео пачкой кадров заходят как есть.

Рантайм нода привозит сама: при первом запуске скачивает нужные файлы в
`models/DLSS` и раскладывает их так, как того требует подписанный рантайм
NVIDIA. Полоса прогресса есть и у загрузки, и у обработки; кнопка «Стоп»
доходит до чужого процесса, а не оставляет его висеть.

Пять режимов (`1×` DLAA … `3×`), модель по умолчанию **M**, потолок 7680×4320 —
и если запрошено больше, нода называет наибольший подходящий множитель, а не
просто отказывается. Для пачки последовательных кадров считаются векторы
движения, и DLSS переносит детали из кадра в кадр; **для пачки не связанных
между собой картинок `temporal` надо выключить**, иначе каждая потянет за собой
детали предыдущей. Дизеринг на пути к 8 битам воркера включён по умолчанию:
без него плавные переходы приходят к сети ступенями, а она их подчёркивает.
`source_curve` переводит лог/HDR/линейные исходники в SDR перед сетью и
возвращает обратно той же кривой — нода меняет размер, а не цвет.

После прогона читается лог рантайма: если нейронная обработка свалилась в
обычное растягивание (обычно старый драйвер), это будет предупреждением в
консоли, а не молча разочаровывающей картинкой.

**Только Windows и NVIDIA RTX**: работу делает подписанный D3D12-рантайм NVIDIA,
другой реализации у него нет. На остальных системах нода честно об этом
сообщает.

⚠️ **О лицензиях.** Пак ничего из рантайма не хостит и не распространяет и не
аффилирован с NVIDIA, ReShade, RenoDX или исходным проектом. Перед тем как
обратиться в сеть, нода печатает в лог полное уведомление: адрес источника,
размер и кто чем владеет. Выключатель `download_if_missing` — и есть ваше
согласие на загрузку; выключенный, он не качает ничего, и файлы вы
раскладываете сами. Подробности — в разделе ноды в README.

### TS Files Downloader — новый интерфейс

Список моделей теперь вверху ноды, кнопки под ним, остальные параметры — под
кнопкой «Настройки», которая открывает собственную панель внутри ноды.

**Напротив каждой модели видно, как она стоит**: зелёная точка — файл на диске,
красная — нет, жёлтая — лежит недокачанный `.part`, серая — судить не по чему.
Проверка смотрит только диск и никогда не ходит в сеть; она идёт при появлении
ноды, при правке списка и после загрузки. Пока модель качается, полоса хода
рисуется в её собственной строке — на списке из десяти это отвечает на вопрос
«а эта уже пришла?» без пересчёта. Полосы двигаются и при прогоне воркфлоу, а не
только по нажатию кнопки.

Строка списка теперь читается как две вещи: адрес и папка, разделённые стрелкой.
Пробел остаётся законным разделителем — списки, набранные раньше, и списки,
пришедшие с чужим воркфлоу, читаются как читались.

Входы, их порядок и умолчания не менялись — те же одиннадцать. Воркфлоу,
сохранённый раньше, открывается со своими значениями; это проверено на настоящем
старом графе, а не предположено.

### Что перестало ломаться

**Кнопка «Download Models» падала с `'PromptServer' object has no attribute
'last_prompt_id'`.** Штатная полоса прогресса ComfyUI вне прогона графа не просто
бесполезна — она бросает исключение изнутри ядра, потому что берёт атрибут,
который появляется только когда очередь начала считать первый граф. Кнопка граф
не запускает. Теперь полоса заводится только внутри прогона, а у кнопки свой
канал; заодно никакая полоса больше не может оборвать загрузку на десять
гигабайт.

**Кнопка зависала на 0 %, когда все модели уже скачаны.** Из всех исходов
движка ровно этот — самый частый — не сообщал о завершении. Теперь сообщает:
загрузка закрывается тостом, а точки напротив моделей становятся зелёными.

**Модели, взятые из воркфлоу, не появлялись в списке** после «Дополнить» или
«Заменить», пока не переключишь текстовый режим туда-обратно. Список рисуется по
значению виджета, а о записи в него извне его никто не извещал.

**Проверка статусов больше не шумит в журнале.** У ноды с умолчательным
списком-примером каждая перерисовка добавляла по две строки «Invalid target
path» — предупреждение, уместное во время загрузки и бессмысленное при чтении
диска.

---

## 12.5.0

### Видео в ноде больше не отнимает видеокарту у генерации

Жалоба звучала так: локально в браузере через одну-две генерации следующая идёт
заметно медленнее, а если смотреть с другого компьютера — всё быстро. Это не
совпадение, а точный симптом: видео в браузере декодирует ТА ЖЕ карта, на
которой считает ComfyUI, и удалённый просмотр просто уносит декодирование на
чужой GPU.

TS Video Saver после каждой генерации запускал результат **в бесконечном
цикле** — немому видео браузер автозапуск разрешает. Ни одна нода пака при этом
не знала, что граф начал считать, а наблюдатель видимости был ровно у одного
загрузчика.

Теперь у всех медиа-нод общий сторож: на старте прогона воспроизведение
снимается, ушла нода за край экрана или свернулась вкладка — тоже. Возобновлять
сам он ничего не будет: это решение человека.

**Сейвер больше не зацикливает результат по умолчанию** — ролик проигрывается
один раз, и карта освобождается. Кнопка повтора на месте, а сохранённые графы
своё прежнее значение не теряют.

**Кэш ленты миниатюр считается в мегабайтах, а не в лентах.** Предел «96 лент»
означал то 18 МБ, то 313 МБ — замерено на живых спрайтах: при высоте дорожки
48 px лента весит 0,2 МБ в видеопамяти, при 200 px уже 3,3 МБ. Бюджет теперь
64 МБ, и он значит одно и то же на любой высоте.

⚠️ Утечки памяти в паке нет — это проверялось отдельно и не подтвердилось.
Python-тракт не растёт от повторной работы, blob-ссылки освобождаются,
`ImageBitmap` закрываются, подписки снимаются. Дело было только в том, что
воспроизведение никто не останавливал.

## 12.4.0

### The cursor turns into a hand over a button

Hovering a node's button left the cursor as a crosshair, saying nothing about
the fact that something happens on click. Classic nodes are drawn ON THE CANVAS,
so their buttons are not DOM elements and `cursor: pointer` from CSS never
reaches them — measured on a live canvas: empty canvas `default`, anywhere over
a node `crosshair`.

Buttons, toggles and dropdowns now show the hand; text fields deliberately do
not, because a hand over an input promises a click that is not there. The cursor
is restored to whatever it was on the way out — LiteGraph sets its own while you
drag a link or resize a node, and those must not be overwritten.

### TS Files Downloader gets a "download now" button

A second button on the node pulls the whole list immediately, without running
the graph — same engine, same tokens, mirrors and unzip settings. It reports
`3/10 · 42% · model.safetensors` as it goes, and pressing it again cancels; the
partial file is kept as `.part`, so the next attempt resumes.

This is what turns `enable` into a mode worth having: switch it off and the node
does nothing at all when the workflow runs, while the models are still one click
away when you actually want them. The button ignores `enable` deliberately —
with the switch off it is the only way left to download.

**The list is readable now.** Each line is `<url> → <folder>`. A long address
wraps in the field, and a folder pressed against its tail read as part of the
link; the arrow ends that. A plain space still parses, so older lists and lists
arriving with someone else's workflow keep working.

### Audio is fitted to the picture that was actually written

Audio shorter than the video has always been padded and longer audio cut — that
part held up under measurement. What did not: the length was computed from the
**declared** frame count, and for a video source that number is an estimate
(`duration * fps`). Measured on a 20-frame disagreement — declared 50, wrote 30
— the sound came out 0.8 s longer than the picture; the other way round, 0.8 s
short. The measure now comes off the video that was actually written.

Two more edge cases found while checking:

* **An unusual channel count crashed the save.** PyAV wants the layout to match
  the row count, so three-channel audio ended the run with "Expected planar
  array.shape[0] to equal 2 but got 3" — after the clip was already generated.
  Anything unforeseen is mixed down to stereo with a line in the log.
* **An empty track wrote a silent one.** A zero-length waveform now means "no
  audio", instead of being padded into silence the source never had.

### TS Video Saver accepts a video that has no file, and keeps its sound

Feeding the saver a VIDEO built in memory — what Create Video and friends hand
over — ended in "Nothing to save: connect either images or a video", after the
run had already spent minutes generating it. The saver only understood a video
backed by a file: it asked for `get_stream_source()` and gave up without one,
while `VideoFromComponents` does not have that method at all.

It now falls back to `get_components()`, which every VIDEO must implement. A
file is still tried first — streaming from disk is what keeps a long clip out
of memory.

**Sound travels with the video too.** Re-saving a clip used to produce a silent
file unless you wired the audio yourself. The saver now takes the source's own
track when the audio input is empty — read straight from the audio packets for a
file source, so the frames stay streamed. An audio input you connect always
wins: you chose that track deliberately.

### One control closes a fullscreen editor, not two

TS Video Loader put its own "full screen" button directly under the shared ×
in the top-right corner — two controls, same action (measured: the button at
top 49 / right 7, the × at top 6 / right 10). The button travels into the
overlay with the editor's toolbar, which is why it ended up there.

Hiding it is now the overlay's job, not each node's: `openFullscreenOverlay`
takes the control that opened it and puts it away for the duration. TS Video
Saver passes the same thing — its button merely landed somewhere less visible,
and one mechanism beats two.

### The sound track no longer floods the timeline on uncompressed audio

A `.mov` straight from a camera — H.264 with 24-bit PCM sound — drew white bars
across the whole timeline instead of a waveform, burying the filmstrip and the
ruler with it. The video was fine and the browser played the file; the envelope
was the broken part.

`frame.to_ndarray()` hands back whatever the decoder uses. Compressed codecs
(opus, aac) decode to floats in 0…1, so this never showed; uncompressed PCM
arrives as integers, and the measured peaks ran to 1,365,262,336 where the
timeline expects 1.0. Samples are now scaled by their type's full range, in both
places peaks are computed — the overview and the zoomed window — and the drawing
code clamps what it is given as a second line of defence.

⚠️ The probe cache version moved again, so a file you already opened is re-read
instead of answering with the old numbers.

### TS Video Loader finds the cuts

A button on the transport row walks the file and marks every place the shot
changes. Double-click a marker and the trim snaps to that shot — from this cut
to the next, with the last one running to the end. Double-clicking empty space
still resets the trim.

The threshold was chosen by looking at frames, not by picking a round number.
On a checked scene eight genuine cuts scored 0.13 to 0.54 while the most
conspicuous non-cut scored 0.05, so 0.2 — the value the loud group suggested —
would have silently dropped two real ones. A plain pixel difference cannot
separate them at all: 0.198 on a real cut against an average of 0.005.

The first press reads the whole file (4.7 s for 78 s of SD); what it measures is
cached, so pressing again — or moving the threshold — answers at once.

### TS Video Loader opens webm and mkv that carry no duration

A webm downloaded as a stream — from YouTube, or anything written to a pipe —
has no duration in its container: the field is filled in when a file is closed
on disk, and that never happened. The probe read the zeros literally and the
node saw a clip of zero length, so nothing loaded.

Worse, PyAV's `guessed_rate` handed back **1000** for such a file, which is the
1/1000 time base and not a frame rate at all.

The probe now reads the packet timestamps when the container says nothing:
0.04 s on a 50 MB VP9 file, against 4.3 s for a full decode that returned
exactly the same frame count. That file now reads as 78.612 s, 25 fps, 1965
frames, and loads.

⚠️ Probe results are cached on disk and survive an update, so the cache version
moved too — a file you already tried is re-read rather than answering with the
old zeros.

### A batch manager: three nodes, results on disk as they come

**TS Batch Source**, **TS Batch Load Image** and **TS Batch Write** turn a
folder, a text file or a plain count into a job list and run the graph below
once per item.

The part that matters: results are written **as each one finishes**. A batch
that dies at item 90 leaves 89 captions on disk instead of nothing, and
`start_at` resumes from where it stopped without wiping what is there.

Nothing here is a loop. ComfyUI already runs a node once per element of a list
input, and each of those runs is independent — which is exactly what a
captioning model needs: a fresh conversation per picture, not one context that
grows for a hundred images.

**⚠️ But that list runs breadth-first, and it matters more than it sounds.**
Measured on a live server: the loader logged items 1, 2, 3 and only then the
writer logged 1/3, 2/3, 3/3 — ComfyUI finishes every copy of one node before it
starts the next. So results reach disk only after the model has done all of
them, and TS Image Prompt Injector, which stamps the current prompt into the
saved metadata, gets overwritten by the last item before anything is saved:
every picture ends up carrying the same prompt.

Turn on **`one_per_run`** and set the queue's Batch count to the number of jobs.
Each run is then a full pass through the graph — generated, stamped, saved,
previewed — before the next job starts. That is the mode to use whenever you
want to watch results arrive or need honest per-image metadata.

**Watching it happen.** ComfyUI holds the previews of every iteration and shows
them in one go after the last one, so a long batch otherwise looks frozen and
then finishes all at once. Connect an image to TS Batch Write and the current
result goes through the progress bar instead — item 47 is on screen while it is
item 47.

Three layouts: one file of blank-line separated blocks (read back by TS Batch
Prompt Loader); **one line per item**, read back by TS Batch Source itself, which
turns a file of captions into a file of generation jobs with no conversion in
between; or one `.txt` per item named after its picture — the layout caption
datasets expect.

**A seed per item.** A seed widget holds one number for all hundred calls, so a
hundred iterations of the same task used to come back identical. TS Batch Source
hands every item its own derived seed — wire it into the model's seed input.

**Three ready-made templates** ship with the pack (Workflow → Browse Templates →
Timesaver): caption a folder into one file, thirty variations of one prompt, and
generate images from a file of prompts — the last one wiring TS Image Prompt
Injector so every saved picture carries its own prompt.

⚠️ The source emits **paths**, not pictures, and TS Batch Load Image reads them
one at a time. A hundred 4K frames passed along as images would sit in the
output cache — around 10 GB — before the first caption is written.

### Both Super Prompt buttons stopped repeating themselves

Pressing "AI prompt" a second time returned the **same text**, on both TS Super
Prompt and TS Super Prompt RT. The frontend never sent a seed, so the server
fell back to a fixed one; on the RT node the runtime was not given a seed at
all. Both now generate a fresh seed on every press. A request without one still
works and stays reproducible, as before.

## 12.3.0

### TS Video Cut — trim a clip without losing sync

A new node: frames off the start and the end, with the audio cut to match, from
one pair of numbers in frames. The frame boundary drives the audio boundary
through `fps`, so a fractional rate (23.976, 29.97) cannot pull the sound away
from the picture — measured drift after the cut is at most 0.01 ms across rates
and sample rates.

Audio longer or shorter than the video does not shift the cut; it is clamped to
what exists, with a warning past 50 ms. With no audio connected the output is
silence of exactly the trimmed length rather than nothing, and cutting away the
entire clip is refused with the numbers in the message.

### TS Smart Switch stops computing the branch you did not choose

The two inputs are lazy now. Before, ComfyUI had to evaluate **both** branches
before calling the node — it cannot know only one value will be used — so a heavy
VAE Decode on input 1 ran even with the switch on input 2. That hurts because
ComfyUI caches only the *latest* input configuration: working on the second
stage evicts the first stage's result, and going back re-runs the decode.

Measured on a live server with one expensive and one cheap branch: **0.30 s with
the cheap branch selected against 4.42 s with the expensive one** — the unused
branch is not computed at all. Auto-failover still works: if the selected input
is missing, the other one is evaluated instead.

### TS Super Prompt RT: a transcription prompt written for Russian

Russian in Cyrillic, technical terms and product names in Latin script
(`ComfyUI`, `workflow`, `LoRA`, `Stable Diffusion`), direct speech in quotation
marks, and a rule that protects unfamiliar names — measured on a real recording,
«Artius Diffusion» used to come back as «Artus».

### TS Latent Upscale: precision fallback, model-free upscaling, grouped upscale

**bf16 falls back to fp16 on cards that only emulate it** — the whole Turing
line. The trap is that `torch.cuda.is_bf16_supported()` answers `True` there
anyway, so the check asks for native support explicitly. Measured on the H3
checkpoint, fp16 is also the *more accurate* half-precision path (0.38% vs 2.67%
deviation from fp32), and converting bf16 weights loses 343 of 345,280,216.

**Upscaling without any model** is now a choice in the same list —
`Interpolation: bilinear / bicubic / area / nearest`, the mode the original
offered as a separate node.

**Chunks are upscaled in groups sized from free RAM**, so the diffusion model is
offloaded once per group rather than once per chunk.

Also: the target size is checked against the conditioning's keyframes and warns
on a mismatch instead of failing deep inside sampling; the redundancy cost of the
chosen chunking is logged; and a redundant tensor copy was removed from the
upscale path.

### TS Latent Upscale — three MMH3 nodes in one, and subfolders that finally work

A new node for re-sampling a denoised **MiniMax H3** audio+video latent at a
larger size, built from
[Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)
(MIT, bbaudio-2025). What were three nodes wired together — the pipeline, the
upscale-model settings and the temporal split settings — are one node with
plain inputs.

**Subfolders in `models/latent_upscale_models` are listed.** The original
scanned only the folder root and returned bare filenames; on a folder holding
four models it offered two. The list is now recursive (`subfolder/file.safetensors`),
covers every path from `extra_model_paths.yaml`, and refuses a name that climbs
out of its folder. Choosing an upscaler from another model family now says so
instead of failing with `Missing key(s) in state_dict`.

`chunk_length` and `temporal_overlap` are validated against the model's 17-frame
keyframe grid before the run starts.

**Spatial tiling was not carried over**, along with its input — the seams need
their own fade and blend settings, and shorter chunks serve the same purpose.

### TS Film Emulation: grain that behaves like grain, and clips that fit in memory

**The grain is now applied in log space**, the way film density actually
fluctuates. It therefore rides the signal — barely present in shadows, strongest
in the upper mid-tones, fading again at the shoulder — instead of sitting on the
picture at one constant strength. Measured across a grey ramp: 0.001 / 0.016 /
0.042 / 0.061 / 0.044, against a flat 0.049 / 0.060 / 0.049 before.

**Two new controls, both last in the list and optional**, so saved workflows keep
their values: `grain_speed` holds one grain pattern across several frames the way
professional grain plugins do (1.0 = new every frame, 0.25 = held for four), and
`grain_seed` makes a re-render reproduce the grain exactly.

**Clips are processed on the GPU in chunks sized from free VRAM.** Peak memory is
about 2.6 GB regardless of clip length — the first attempt at chunking peaked at
15 GB — and the output does not depend on how the clip was divided. Measured on
the same machine: 24 frames of 1080p with a LUT, 6.5 s → 0.8 s; eight 4K frames,
6.8 s → 0.8 s. The LUT file is now parsed once per run instead of once per chunk.

**The shadow/highlight saturation split is no longer a hard step at mid-grey.**
On stills the step was nearly invisible; on video, pixels drifting around the
threshold flickered between two saturations along gradients.

## 12.2.1

### TS Super Prompt RT switches to the abliterated Gemma 4 builds

The catalogue now points at
[`hfmaster/Gemma-4-RT`](https://huggingface.co/hfmaster/Gemma-4-RT) — the
**abliterated** E2B and E4B artefacts — instead of the stock `litert-community`
ones. Same sizes (2.41 GB and 3.41 GB), same speed, but without the refusal
behaviour, which matters for a node whose only job is writing prompts.

All fifteen presets and voice transcription were re-run on both new models
before the switch: 15/15 on each, transcription unchanged.

**Existing installs will download the new files on first use.** The old ones are
not deleted and not needed; they can be removed from `models/LLM/litert` by hand.



### TS Super Prompt RT: a forgotten Stop button is no longer expensive

The microphone stops itself after three minutes, counts down for the last
fifteen seconds and afterwards says why it stopped — including when the
recording was silence, which is what a forgotten microphone usually captures. A
recording arriving at the route by another path is cut at five minutes.

Three minutes is not a model limit: Google documents **30 seconds per audio
clip** at 25 tokens/s, and the node already transcribes in 30-second segments.
That boundary is now held on purpose rather than by luck — this runtime accepted
85 s here and only refused at 90 s with `4688 >= 4096`, and segmenting costs
nothing in completeness (141 words in two segments against 140 in one oversized
pass over the same minute).

### TS RTX Upscaler keeps its engine between runs

The NVIDIA VSR engine used to be built from scratch on every run of the node.
Measured on an RTX 3080 Ti, that cost ~730 ms — on eight frames to 1080p, 93% of
the node's entire work. It is now created once and merely reconfigured: changing
the output size on a live engine costs 6 ms, changing quality costs nothing.
**Repeat runs are 6.8× faster** (0.93 s → 0.14 s); long clips, where the engine
was already amortised, are unchanged.

The engine holds 162 MB, and unlike the LiteRT runtime that memory IS visible to
ComfyUI, so keeping it hides nothing from the sampler. Frame processing is
serialised behind a lock — the engine is one per process, and two graphs calling
into it at once would race inside native code.

Nothing about the node's inputs, outputs or results changed.

### Installing `nvidia-vfx` is documented at last

The package on PyPI is a 2.7 KB stub that fails to build; the real wheel lives
only on NVIDIA's index. The command is now in the README, in `requirements.txt`
and in the `rtx-upscaler` extra:

```
pip install nvidia-vfx==0.1.0.1 --no-build-isolation --index-url https://pypi.nvidia.com
```

## 12.2.0

### TS Super Prompt RT — the same work, on Google's on-device runtime

A new node beside TS Super Prompt, running **Gemma 4 through LiteRT-LM**
instead of transformers. Measured on the same machine and the same prompt:
**43 tok/s at ~1.7 GB of VRAM**, against 20 tok/s at 8.5 GB for the Qwen path.
The same model also transcribes speech, so there is no Whisper in this node at
all — one `high_quality` switch picks E2B or E4B for both jobs, because it is
one model doing both.

**It gives the card back when it is done.** LiteRT computes on WebGPU, not CUDA,
and ComfyUI cannot see that memory: `torch.cuda.mem_get_info` reads the same
whether Gemma is resident or not. A model left loaded is therefore memory
ComfyUI still believes it has. Unloading takes about a second, so it is the
default; `keep_loaded` turns it off and says what that costs.

Requires a separate install — `python -m pip install litert-lm==0.16.1` — and
**works on Windows and macOS only**, because LiteRT-LM publishes no Linux
wheels. On Linux the node loads and explains that instead of failing obscurely.
Models come from `litert-community` (Apache-2.0, no token) into
`models/LLM/litert` on first use.

Nothing about the existing TS Super Prompt changed.

### `Music Prompt Enhance` is now `Audio Prompt Enhance ACE-Step`

Same preset, new name and a new prompt written for **ACE-Step 1.5 XL**. It
produces the model's **Style** field: one line of comma-separated descriptors
covering the nine dimensions the official guide lists, with the separate fields
left alone — no invented BPM, key, time signature or duration, since each has a
box of its own and a number in the caption only argues with it.

**Old workflows keep working.** The former name is aliased to the new one, so a
graph saved with `Music Prompt Enhance` selects exactly this preset; the widget
shows the new label after you open and re-save it. Without that alias the node
would have fallen back to its default preset without a word — a working graph
quietly writing a different kind of prompt.

### Two new presets for audio: MiniMax Music 3 and Stable Audio 3

`Audio Prompt Enhance Minimax` turns an idea — in Russian if you like — into a
caption in MiniMax Music 3's own format: three headings, the labelled lines each
one wants, and the 250–450 words the official guide asks for. It also takes a
**picture** and writes the music that would score it, which is what the
`Application Scenarios & Imagery` line was made for.

`Audio Prompt Enhance Stable Audio SFX` writes for Stable Audio 3, for sound
effects and solo instruments. It opens with the dataset tag the model expects —
`TrackType: SFX` or `TrackType: Instrument` — then names the source, the action
and the recording, and never wanders into verses and choruses.

**A bug the new presets uncovered:** with a picture attached, every preset was
told, in the user turn, to "infer whether image or video generation is more
appropriate" and to "describe what is in it" — a leftover from when the node
only wrote picture prompts. Those two lines sit closer to the model than the
system prompt, so a music preset given a photograph returned an image prompt,
lens and depth of field included. The medium now follows the preset, and for an
audio preset a picture is announced as a mood board for the sound. Measured on
the same photo: 91 words of photo description became a 350-word music caption.
Quoted words are handled by medium too — drawn on screen for picture and video,
sung for music, and kept out of the caption where the model expects them in its
own lyrics field.

### The video presets stopped inventing dialogue

Give TS Super Prompt a picture and no words at all, and it used to hand back a
prompt in which people speak: MiniMax H3 got two riders described as talking in
warm and bright voices, LTX got an outright invented Russian line. Both models
make sound in the same pass as the picture, so an invented line becomes real
speech in the result.

The cause was not one bug but two. The H3 presets only ever said what to do **if**
the idea contains words in quotes — what to do when it contains none was written
nowhere, so the model filled the gap itself. The LTX preset did carry that rule,
but buried in the middle while every one of its examples contained speech, and a
2B model follows an example far more readily than a rule.

All three presets now state the prohibition in as many words and carry an
example in which nobody speaks. Measured across three seeds on the same picture:
no invented speech in any of them, and a quoted line still arrives verbatim
inside its `<d>` block.

**Reference mode takes its labels from your text.** You write `<Picture 1>`,
`<Video 1>`, `<Audio 1>` in the prompt field yourself, so the preset now copies
exactly those instead of guessing — it used to announce an `<Audio 1>` nobody
had attached, sending the model looking for a voice that did not exist. Its
answer also had room to finish: six fields did not fit in the old token budget
and were being cut off in the fifth.

Known limits, measured rather than assumed: with a picture *and* a spoken line,
the small model still borrows scenery from the preset's worked example, and it
sometimes omits the camera move on a still with no action. Six different prompt
rewrites were tried against three seeds each; none removed these without making
something else worse, and the bigger model does not fix them either. They are
cosmetic — the prompt is valid and the line is correct — and are better solved
by a stronger model than by more instructions.

---

## 12.1.0 — 22 Aug 2026

### New node: TS Angle Select

Point a camera at the subject and get the prompt that asks a model for exactly
that view. The node shows a small 3D preview — subject, orbit, camera — with
three controls under it: rotation, height and zoom.

The wording is not the node's. With the bundled **Qwen Multi-Angle** preset the
output is the trigger phrase the Multiple-Angles LoRA was trained on:
`<sks> back view elevated shot close-up`, and nothing more. It reads like a
fragment because it is one — the LoRA learned these exact words, and prettier
English breaks the conditioning.

Presets are plain JSON, one file per model in `nodes/text/angle_presets`, so a
new model is a new file rather than a code change. A preset missing a phrase is
skipped with a line in the log instead of quietly producing a prompt with a hole
where the angle should be.

Three.js ships with the pack for the preview and loads **only when the node
appears**. It is deliberately kept out of the web folder: ComfyUI imports every
script in there on page load, and 675 KB would otherwise be paid by everyone who
never places this node.

### TS Super Prompt asks before it downloads, and checks transformers first

Two halves of one complaint. The node used to go and fetch its language model
without a word — several gigabytes — and, on an older transformers, it did that
first and only then failed with *"Transformers does not recognize this
architecture `qwen3_5`"*. So the download was both invisible and, sometimes,
entirely wasted.

**It asks now.** When the model is not on the machine, pressing the enhance
button opens a dialog naming the model, the exact size and the folder it will
land in. *Not now*, Escape, or a click outside all mean no, and nothing is
downloaded. Agreeing is remembered for the session, so it asks once, not on
every press. The size is the real repository size read from the Hugging Face
API — no bytes are fetched to find it out.

**It checks the library first.** Before any download, the node asks the hub what
architecture the model is and compares it with what the installed transformers
knows. If the library cannot load it, the run stops immediately with the model
type, the installed version, and the command that fixes it — instead of
spending a few minutes and gigabytes to arrive at the same conclusion. The check
asks the library what it supports rather than comparing version strings, so a
build from git or a partial upgrade is judged on what it can actually do.

**Requirement raised: `transformers>=5.2.0`** (was 4.57.0). The default model is
a Qwen3.5, and 5.2.0 is the first release that knows the `qwen3_5` architecture
— verified against the repository tags, it is absent in 5.1.0. The floor now
says what the pack actually needs.

### TS Video Loader stopped refusing clips the machine could easily hold

The memory ceiling was a flat 8 GB written into the code. On a 64 GB machine
that turned down a **13-second 4K clip** — 323 frames of 4096x2160, about
31.9 GB as float32 — with a message telling you to trim the timeline. The
ceiling now comes from the machine: 60% of its RAM, never below the old 8 GB, and
`TS_VIDEO_MAX_BYTES` still overrides everything. That clip now loads in 51 s.

Free memory is only ever a warning, never a refusal: it moves with whatever
models ComfyUI happens to be holding, so refusing on it would make the same
graph pass one minute and fail the next. And if an allocation really does fail,
the message reads like the guard's instead of a bare MemoryError.

**New input `when_too_large`** (appended, optional, so saved graphs are
untouched). Left at `stop` it behaves as before. Set to `use disk`, the frames
go into a memory-mapped file in the ComfyUI temp folder: what comes out is an
ordinary IMAGE tensor, and the allocation cannot fail outright however long the
clip is. It is not free — decoding writes every frame once, so memory still
climbs while it runs, and the same 31.9 GB clip took 92 s against 51 s in RAM —
but those pages are backed by a real file the system can drop, and downstream
only the frames a node touches are read back. Leftover files are swept on the
next decode.

### Depth is now two nodes: TS Video Depth and TS Image Depth

A still and a clip want different models, and one node with a `mode` switch made
that hard to see: half the widgets were dead at any moment. They are two nodes
now.

**TS Video Depth** keeps its node id, its first thirteen inputs and their order,
so saved graphs open unchanged. What it lost is the `mode` switch and the two
single-image inputs, which existed only in unreleased builds.

**TS Image Depth** is new, on the `TS/Image/Depth` shelf. It runs **Depth
Anything V2 Large**, the model actually trained on stills, and shows only what
applies to one: no window, no flicker filter, no `input_size`.

It runs the reference pipeline and nothing else: trim to a multiple of 14, run
the model, normalize **each picture on its own** min/max, resize back
bilinearly. Denoise, dithering and the guided upscale are gone from it — they
were built for video, and on a still the guided filter put a halo on contours.
Measured against the reference implementation, the map now differs by **0.36 %**,
under one 8-bit level, at identical detail; the old single-image path differed
by **4.10 %** and looked soft.

Two things caused that. `input_size` belongs to the video model — it lifts the
short side to 518 px, so a 1600 px photo went into the model at 784×518. And
`percentile` normalization buys temporal stability by clipping 1 %/99 %, which a
single picture has no neighbours to need: the clip only burned the nearest and
farthest pixels to flat white and black. On the image node the resolution is
`max_res`, native by default, and normalization is per picture.

### Only safetensors are offered in the depth model lists

Both depth nodes now suggest fp16 safetensors only — half the download and an
order of magnitude quicker to read (0.01 s against 0.66 s, measured). The small
video model was converted and published alongside the large one, so nothing was
lost from the choice.

A saved workflow that still names an old `.pth` keeps working: the file resolves
and downloads exactly as before, it is simply no longer suggested in the list.

The old single-image path handed the video model one picture as 32 duplicated
frames and kept the first result. That is 32 runs of the model for one answer, and the answer was
worse: measured on a portrait, the duplicated window flattens the depth range
until the face and hair are one white shape. The dedicated engine keeps the
structure and takes **0.18 s instead of 6.1 s**, on 2.6 GB instead of 9.2 GB.

Short clips gained from the same correction. Anything that fits in one window is
now run as a single window of exactly its own length rather than being padded out
with copies of the last frame — an input the model never saw while training. A
one-frame clip in *video* mode dropped from about 6 s to 0.22 s.

Two new controls are exposed rather than guessed: `flicker_suppression` blends in
a temporal median of the depth — a median, not an average, so single-frame pops
disappear while real movement stays — with `flicker_radius` for its reach; and
`window_length` / `window_overlap` open up the sliding window itself.

Weights moved to fp16 safetensors: half the download, and they load in
hundredths of a second instead of two thirds of one. Measured against pure fp32,
the depth differs by 0.02 % of its range. The `.pth` files stay in the list.

Moving between the two nodes used to cost about four seconds each time, because
the previous model was thrown away. Both engines now stay cached and ComfyUI's
own model manager decides what to evict; a switch costs 0.2 s.

Under both nodes sits one shared engine (`nodes/_depth_core.py`), so the two
cannot drift apart: normalization, denoise, dithering and upscale are the same
code in both.

### TS Music Stems moves to RoFormer, and the stems finally add back up

`model_name` now picks the engine. **BS-RoFormer SW** is the default and
returns six stems — vocals, bass, drums, guitar, piano and everything else.
**Mel-Band RoFormer** returns only vocals and instrumental, which is exactly
why it is better at that split: it spends its whole capacity on one boundary.
Demucs is still there, unchanged, so a workflow saved last year still sounds
like it did last year.

Mask separation does not sum back to the mix, and a null test finds the gap
immediately. So one stem is no longer taken from the model: it is the mix minus
all the others, which makes the set exact by construction. Measured on real
music, `vocal + instrumental` nulls against the source at 161 dB and the six
stems at 144 dB — the floating-point floor, not a modelling error.

Three things that were wrong before are now right. The last chunk is
back-shifted instead of being padded with silence, because a transformer that
attends across the whole segment has never seen a synthetic tail. The
overlap-add window is a raised cosine across the chunk rather than a short
linear fade, so the frames with the least context are the ones suppressed. And
the progress bar counts real chunks instead of pulsing on a timer, which also
means the run can be cancelled.

`precision` chooses fp16 or fp32 for the RoFormer engines; fp16 is roughly
twice as fast on half the VRAM and its error was measured at -61 dBFS or below on
real music. bfloat16 is deliberately absent — `view_as_complex`, which these
models use to build their mask, does not accept it, so offering it would only
guarantee a crash on the first chunk.

Saved graphs are safe: every existing input and output kept its position, and
`guitar` and `piano` were appended after them. An output the chosen model
cannot produce returns an ExecutionBlocker, so that branch of the graph is
skipped rather than fed silence that would look like a broken model.

### TS Prompt Builder now works in packs, and the packs know what goes together

A pack is a folder of wildcards plus a semantic map. The map is what makes the
difference: it says what each wildcard is, where it belongs in the phrase, what
it excludes and what it pairs with — so a run stops producing a winter street in
a swimsuit, or a close-up with a full-body pose, or two incompatible scenes at
once. Assembly follows the eight steps written in the packs themselves.

Any packs combine, in any combination. Their roles interleave into one sentence
instead of one pack's output being glued onto another's, and wildcards are
namespaced by pack so two `face.txt` never collide. Where two packs both offer a
face, a light or a pose, one survives — drawn with a weight, not by rank, so a mix
of five reads as a blend instead of as the highest-priority pack talking over
everyone.

And the scene now holds together. Place lives in the words of a line, not in the
links between files, which is why the semantic map alone could not stop a pool, a
rainstorm and a kitchen from sharing one sentence — 22% of assemblies did exactly
that. The place is chosen first and every other line is read against it; the ones
that disagree about where or when we are get dropped. The same measurement
afterwards reads 3%, and those are metaphors rather than mistakes.

Drop a folder into `nodes/prompts/` and press Reload; no restart. The node groups
wildcards by role, dims the ones that will collapse to a single pick, lets you pin
one so it wins a collision, and previews the result live with the same code the
run uses. A second output, `info`, says what the map threw out and why.

Your saved graphs keep working: the node id and its inputs are unchanged, the old
flat block list is still read, and `seed = 0` still means a new prompt every run.
One deliberate change — a wildcard that appears in a pack later is added switched
**off**, so an author adding a file no longer shifts everyone's prompts.

### Both LTX HDR technologies, chosen by one dropdown

There are two of them and they are not variations of one thing. The native 2.5
path **preserves** the range an EXR already carried and works in ACEScct. The HDR
IC-LoRA **grows** range out of ordinary SDR and works in ARRI LogC3. Feed one
path's output through the other's inverse curve and the colour shifts.

`hdr_mode` on the settings node picks which, and everything downstream follows:
the decode applies the right inverse, the stats node reports against the right
ceiling (linear 222.86 for ACEScct, 55.08 for LogC3), and in IC-LoRA mode the
guide is an ordinary SDR frame with no EXR read at all. Our LogC3 inverse matches
the official `LTXVHDRDecodePostprocess` to within 1e-6, measured over 501 points —
and unlike theirs, writing the EXR needs neither OpenCV nor
`OPENCV_IO_ENABLE_OPENEXR`.

The IC-LoRA is validated by Lightricks on LTX 2.3; support for 2.5 is officially
in development.

### TS Video Saver — the quality dropdown speaks Russian now

Every sub-setting of the format dropdown — quality, the H.265 10-bit switch, the
ProRes profile — stayed English in a Russian interface, and had since the node
was written. The translation was there all along; it simply never matched. The
frontend builds an i18n key out of the widget name, and a sub-widget is called
`format.quality`, but a dot means "go one level deeper" to i18next — so the key
had to be spelled `format_quality`. Measured on a live server by feeding four
candidate spellings at once: only the underscore one reached the screen.

### Native HDR for LTX 2.5 — seven nodes and one checkbox

An EXR goes in with the sun still in it, and an EXR comes out with the sun still
in it. The path is the official one: ACEScct working space, guides prepared
separately for each stage, float32 only for the final decode, scene-linear
Rec.709 master. No HDR LoRA is involved and none is needed — native HDR is part
of the ordinary LTX 2.5 inference path.

**Off by default, and free while off.** With the switch down no EXR is read, no
float32 VAE is loaded, and the graph behaves exactly as before. Measured on the
live server: with HDR off, a run whose EXR loader pointed at a file that does not
exist still finished successfully, because the lazy branch is genuinely never
walked. The EXR saver produced nothing at all — not a black frame, not a stub
file.

Two things the ordinary nodes cannot do, which is why these exist: `Load Image`
flattens everything above 1.0 without saying so, and `LTXVPreprocess` pushes the
frame through H.264 and 8-bit bytes. The HDR branch bypasses both.

`TS LTX HDR Stats` is the one to reach for when something feels wrong: lost HDR
looks completely normal until someone tries to pull the sky back in the edit.

### TS Video Saver — EXR sequences, and ProRes 4444 confirmed

A fourth format: **EXR sequence**, one scene-linear file per frame in its own
folder, 32-bit float or 16-bit half. It reads from a new `hdr_image` socket,
because the ordinary `images` input is clamped to 0..1 long before the saver sees
it. No compression setting — this encoder does not offer one. Sequences carry no
audio; the small H.264 preview is written in the same pass over the frames, so a
streamed source is only read once.

ProRes **4444** and **4444 XQ** were already there and are now verified end to
end, alpha variant included.

Everything else about the node is unchanged: existing formats keep their order,
and the new socket is last in the schema, so old workflows open exactly as they
were saved.

### Windows: the console stops screaming about closed sockets

`ConnectionResetError: [WinError 10054]` from `_ProactorBasePipeTransport._call_connection_lost`
was 402 of 1865 lines in a live session's log — 22% of it — arriving in bursts
every time a websocket closed. It is a CPython bug (python/cpython#83191, open
since 2020), not a ComfyUI one, and updating Python does not help.

It was not only noise: the exception escapes before `self._sock.close()`, so the
socket stayed open until the garbage collector got to it. The pack now wraps
that one method and finishes the interrupted cleanup, for six socket-teardown
codes and nothing else — an unknown code is re-raised, because your protocol's
`connection_lost` runs through the same method. Measured over eight page reloads
and two jobs: 9 tracebacks before, 0 after. Kill switch:
`TS_DISABLE_PROACTOR_GUARD=1`.

### New on the canvas — "Tidy up"

Right-click the canvas or a node → **Tidy up** → **Tidy layout**: the selection
(or the whole graph, when nothing is selected) is arranged into columns that
follow the wiring, each node shrunk to the size its content asks for, everything
on the grid. No node to add — it is a canvas command, and both entries are in
the command palette too, so keys can be bound to them.

The link reroutes — ComfyUI's small round dots on a wire — are spread evenly
along the straight line between the sockets they connect, so a wire that ran as
a dogleg becomes a straight run; a dot shared by several links settles between
them. A wire whose straight line would cut through a node keeps its bend: a
detour that exists for a reason is not undone. **Align link dots only** does
just that part, leaving the nodes where they are.

**Tidy layout + route the wires** goes further: after the layout, every wire
that would cross somebody else's node gets dots that take it into the corridor
between columns, along one free lane, and back — and every dot that earns
nothing is dropped again. Wires that already have dots are left to their owner,
wires running against the flow are left alone, and running it twice changes
nothing.

**Pack as tiles** is the other way round: nodes packed as tightly as they go,
every node in a column the same width, a column holding several consecutive
layers rather than one, and the column height chosen so the whole schema lands
near 16:9 with no column left half empty. Nodes of a kind stay together; the
wires are not touched at all. On a real 32-node workflow: 17 columns and
5540×1308 become 5 columns and 1760×1128, with 76% of the area actually used
instead of 17%.

Columns come from the graph, not from where things happened to sit: a node's
column is its distance from the start of the flow, and the order inside a column
is chosen to keep links from crossing, starting from the order you already had.
Groups are laid out from the inside out — the nodes of a group are arranged
within it, the frame is fitted to them, and the group joins the outer layout as
one block. Pinned nodes are never moved. The top-left corner of the schema stays
where it was.

---

## 12.0.1 — 15 Aug 2026

### The size you gave a node is now the size it keeps

Every TS node that draws its own interface — Video Saver, Video Loader, LoRA
Loader, Super Prompt, Image Studio, Resolution Selector, Prompt Builder — used
to open at a different size than you left it. The workflow file had the right
numbers all along (measured: 520×640 saved, 520×480 restored); the layout
recomputes a node's height from its widgets *after* the graph is applied and
overwrote them. It could go either way — one node shrank to its minimum, another
grew by twenty pixels every time it was opened.

The height is now re-asserted once the layout has settled, on loading a workflow
and after a run finishes. TS Group Bypasser is deliberately excluded: its height
follows the number of groups in the graph, which is its own rule.

### TS Files Downloader — it now knows every loader your ComfyUI has

"Get models from workflow" missed models in `Load Latent Upscale Model`, and it
was not alone: the scanner matched loaders against a table written by hand, and
on the maintainer's machine **49 installed node types own a model widget that
table never heard of** — two of them from ComfyUI itself, reading a whole
category (`latent_upscale_models`) that was missing from it. A model in such a
node was reported as "no models found in this workflow".

The map is now derived from the running server. Every loader's dropdown is
filled from a `models/` folder, so the options themselves say where they came
from, and the answer is per WIDGET: a loader carrying both a text encoder and a
checkpoint sends each to its own folder. The download target is the directory
the files are actually read from, not the registry key — `models/clip` rather
than a folder named after `clip_gguf`. The old table stays as the fallback for a
category that has no files on this machine yet. Pressing `R` forgets the answer,
because that is the gesture of someone who just installed something.

### TS LoRA Loader — every row has a switch now

Turning a LoRA off without deleting it was already possible: you clicked the
row's name. Nothing said so — the row just faded — so the obvious move was to
delete the entry and add it back afterwards, losing its strength and its place
in the order. There is a visible switch on each row now, in the same shape the
group toggles use. The name still works, for whoever had learned it.

### TS LoRA Loader — `R` finally reaches it

A LoRA dropped into `models/loras` did not show up in the node's search until
the page was reloaded. ComfyUI refreshes stock dropdowns by walking a node's
widgets; this node draws its own picker, so the refresh went straight past it.
It now listens on the same hook ComfyUI offers for exactly this, and reads the
fresh list out of the refresh itself instead of asking the server again.

---

## 12.0.0 — 12 Aug 2026

### TS Super Prompt — the LTX preset now targets LTX 2.5

`Video Prompt Enhance LTX` learned the one thing that actually changed in 2.5:
the model can **cut inside a single generation** and hold the same person, place
and voice across the cut. The preset now explains how to write those cuts — name
the transition, re-establish the framing, repeat the character description
word-for-word, say what the sound does — while keeping a single continuous take
as the default and four cuts as the ceiling.

Also: camera moves must say where the subject ends up once the move is over
(that is what lets the model finish the motion), and length now scales with the
action instead of a fixed sentence count. Shot lists and scene headers stay
forbidden — LTX renders `INT. KITCHEN - DAY` as on-screen text rather than
reading it as a cut.

**A sign is no longer read aloud.** Quoted words that the idea calls a sign, a
label or a title are written into the shot as something the camera sees, and
nobody says them — the preset used to invent a person to speak a shop window,
and sometimes replaced the words while doing it. A title now appears *over* the
picture rather than hanging in the world on an invented board.

**Quoted text is no longer translated.** Anything you put in `" "`, `« »`, `' '`
or `( )` — a spoken line, a sign, a title — is copied character for character in
its own alphabet, and Russian stays in Cyrillic. It used to come back in English
on the 2B model: the rule sat in the middle of a long instruction, which is
exactly the part a small model drops. It is now the first line of the preset and
the last, worded as a mechanical instruction rather than an explanation, and it
covers text in the frame as well as speech. Sampling was tightened to match
(`temperature` 0.5 → 0.35): copying is not a task that benefits from invention.

The preset also follows the format LTX ships in its own ComfyUI pack — the
prompt opens with `Style: …`, verbs are present-progressive, the sound is woven
through the action instead of collected at the end, and camera movement appears
only when you asked for it.

The preset name is unchanged, so saved workflows keep working.

### New node — TS Smart Batch

Batches images the way core's **Batch Images** does, with two differences that
matter in practice:

- **Inputs grow.** Fill the last slot and the next one appears, up to 32.
- **Every slot is optional.** Core's two inputs are required, so muting or
  bypassing either side breaks the whole graph before it even runs.

Behaviour: several connected → one batch in slot order, gaps skipped; one
connected → it passes straight through on its own; none → a plain error saying
so, instead of a blank frame pretending to be a result.

Sizes and channels are reconciled exactly as core does — a missing alpha channel
is padded, a differently sized image is resized to the first one that actually
arrived — so it is a drop-in replacement. Built for first-frame / last-frame
pairs you want to switch on and off without rewiring.

### TS Video Saver understands date tokens

`filename_prefix` now expands `%date:yyyy-MM-dd%`, `%date:hhmmss%` and the rest
of the family (`yyyy yy MM M dd d hh h mm m ss s`) anywhere in the path, so
`videos/my-run/my-run-%date:yyyy-MM-dd%_%date:hhmmss%` works as written.

Worth knowing why it did not before: ComfyUI's own tooltips promise these tokens
on every saving node, but the **backend never expands them** — the frontend
rewrites the value before queueing, and only for its own nodes. Sent through the
API, even core's `SaveImage` fails with `OSError: Invalid argument`, because a
colon is not legal in a Windows filename. Doing it server-side means it now works
from the UI, from the API and from a script alike. ComfyUI's own `%year%`,
`%width%` and friends keep working as before.

### ⚠️ Breaking: downloads go only into registered model folders

`TS Files Downloader` used to accept any relative target that stayed inside the
ComfyUI folder — `input/downloads`, `user/…`, and also `custom_nodes/…`. That
last one is why it changed: a `file_list` line arrives inside someone else's
workflow, and combined with an archive carrying an `__init__.py` it meant
running a stranger's code on the next start.

A relative target must now name a registered model folder (optionally prefixed
with `models/`). Writing outside `models/` is still possible on purpose — give an
absolute path and set `TS_DOWNLOADER_ALLOW_EXTERNAL=1` on the machine, which is a
decision its owner makes outside any workflow.

### ⚠️ Breaking: 16 nodes were retired (11 Aug 2026)

A graph that used one of these opens with the node shown **in red** as missing.
Nothing else in the graph is affected, and the rest of the pack is unchanged.

| Node id | Was called | Was in |
| --- | --- | --- |
| `TS_QwenCanvas` | TS Qwen Canvas | TS/Image/Size |
| `TS_QwenSafeResize` | TS Qwen Safe Resize | TS/Image/Size |
| `TS_WAN_SafeResize` | TS WAN Safe Resize | TS/Image/Size |
| `TS_Color_Grade` | TS Color Grade | TS/Image/Color |
| `TS_FilmGrain` | TS Film Grain | TS/Image/Color |
| `TS_Keyer` | TS Keyer | TS/Image/Cutout |
| `TS_Despill` | TS Despill | TS/Image/Cutout |
| `TS Cube to Equirectangular` | TS Cube to Equirectangular | TS/Image/360 |
| `TS Equirectangular to Cube` | TS Equirectangular to Cube | TS/Image/360 |
| `TS_Video_Upscale_With_Model` | TS Video Upscale With Model | TS/Video |
| `TS_FilePathLoader` | TS File Path Loader | TS/Files |
| `TS_ModelScanner` | TS Model Scanner | TS/Files |
| `TS_ModelConverter` | TS Model Converter | TS/Files |
| `TS_ModelConverterAdvanced` | TS Model Converter Advanced | TS/Files |
| `TS_ModelConverterAdvancedDirect` | TS Model Converter Advanced Direct | TS/Files |
| `TS_CPULoraMerger` | TS CPU LoRA Merger | TS/Files |

**Getting one back.** Everything they need — code, help pages, README sections,
the Russian locale, tests, example workflows and a snapshot of their contracts —
is kept in `archive/removed-nodes-2026-08-11.zip`, together with instructions.
Restore from that archive rather than from memory: the contract snapshot inside
is what guarantees a restored node keeps the same inputs in the same order, so
old graphs still load.

### Renamed (display name only — node ids unchanged, graphs unaffected)

- `TS_Qwen3_VL_V3` is now shown as **TS Qwen 3**.
- `TS Files Downloader` dropped the "(Ultimate)" suffix.

### Security

- **The Hugging Face token no longer reaches mirrors.** It used to be computed
  once before the mirror loop, so the first failure of `huggingface.co` sent a
  private token — often with write access — to `hf-mirror.com`. Now every
  endpoint is asked whether it is the official origin. One rule for the whole
  pack, in `nodes/_hf_download.py`; the file downloader, the shared helper and
  the Qwen engine all use it.
- **Downloads go only into registered model folders.** A `file_list` line
  arrives inside someone else's workflow, and the node's own resolver still had
  a fallback that accepted any name under the ComfyUI root — `custom_nodes`
  included. Combined with an archive carrying an `__init__.py`, that meant
  running someone else's code on the next start. Absolute paths are unchanged:
  still allowed, still only with `TS_DOWNLOADER_ALLOW_EXTERNAL=1`.
- **Archives are checked member by member.** Executable names (`.py`, `.dll`,
  `.bat`, …) are refused inside a zip just as they are outside it, and there are
  now ceilings on unpacked size, member count and compression ratio. The same
  limits apply to studio content packs.
- **The download request re-derives its headers for the final URL** it actually
  fetches, instead of carrying the ones computed for the address before
  redirects.

### Fixed — work that used to be lost

- **TS Lama Cleanup:** opening a saved workflow wiped every retouch. State is
  now restored on load instead of being overwritten by the source poller.
- **TS SAM Media Loader:** after a reload the editor came up empty and the first
  click overwrote the saved points. Points, source and checkpoint are restored.
- **TS Prompt Builder:** loading a workflow replaced the block selection saved in
  the node with whatever this machine had configured. The node's own selection
  wins again; new files on disk are appended, switched off.
- **TS Ideogram Designer:** uploading a reference no longer overwrites a
  same-named file in `input/` — which another graph may well be using.
- **Studio queue:** reordering jobs cleared the queue and resubmitted them; one
  failed resubmit lost the rest silently. The remainder is now always sent and
  failures are reported.
- **Sliders:** a legacy 10× step stored in properties silently rewrote saved
  values on load.

### Fixed — correctness

- **Audio taken from a video** was handed on in the frame's raw format: 16-bit
  tracks arrived at ±32767 instead of ±1, and packed stereo arrived as distorted
  mono. Both are normalised now.
- **TS Image Tile Merger** drew black seams when feather was on but tiles did not
  overlap. Feather is now bounded by the overlap it has to live in.
- **TS Langevin Inpaint** crashed on a latent batch larger than one.
- **TS Smart Switch** rejected ComfyUI's own `VIDEO` object.
- **TS Qwen 3** honours a local model path, as its tooltip has always promised,
  and no longer lets two repositories with the same last name share one cache
  folder. An existing cache is moved, not re-downloaded.

### Fixed — responsiveness and memory

- **Cancel now stops long nodes.** Frame interpolation, BiRefNet and ViTMatte
  drew a progress bar but never asked whether the run had been cancelled, so
  pressing Cancel did nothing until they finished.
- **Studio pack and pass routes no longer freeze ComfyUI.** They ran blocking
  network calls directly on the event loop; with the host unreachable the whole
  interface — previews, progress, `/interrupt` — stopped for up to two minutes.
- **Whisper keeps one model in memory instead of every model ever loaded.**
- **The studio releases the microphone** when the prompt panel is torn down.
- Assorted leaks closed: preview frames, history frames, the stage itself.

### Documentation

- The built-in help for TS Video Loader described environment variables
  (`TS_VIDEO_EXTRA_ROOTS`, `TS_VIDEO_ALLOW_ANY_PATH`) that do not exist. The real
  names are `TS_MEDIA_EXTRA_ROOTS` and `TS_MEDIA_ALLOW_ANY_PATH`.
- `requirements.txt` was missing `comfyui-frontend-package`, which
  `pyproject.toml` declared.

### For maintainers

- The contract snapshot now records **input order and outputs**. It used to sort
  widgets by name, which meant the single most dangerous change in the pack —
  reordering inputs, because `widgets_values` is positional — was invisible to
  the guard.
- A broken internal import is reported as `ERROR`, not `SKIPPED`. `SKIPPED` is
  for a third-party package the user did not install; using it for our own typo
  meant a node could vanish and CI would call it normal.
- The CI smoke test asks `comfy_entrypoint()` for the node list and compares it
  against the snapshot by name. It used to read `NODE_CLASS_MAPPINGS`, which the
  pack deletes on any modern ComfyUI — so it checked nothing at all.
- One failing `define_schema` no longer takes the rest of the pack with it.
- `tools/preflight.py` also checks that the built-in help matches the README.
