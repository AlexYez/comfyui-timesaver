<div align="center">
<img src="icon.png" alt="Timesaver" width="100" />

# Timesaver Nodes for ComfyUI

Nodes for images, video, audio and prompts, plus tools for organizing workflows.

[Русский](#на-русском) · [Workflow examples](example_workflows/) · [License](LICENSE.txt)
</div>

## Installation

In ComfyUI Manager, search for **Timesaver**. For a manual installation, run these commands from the ComfyUI directory:

```sh
cd custom_nodes
git clone https://github.com/AlexYez/comfyui-timesaver
cd comfyui-timesaver
python -m pip install -r requirements.txt
```

Use the Python interpreter that runs ComfyUI. Restart ComfyUI, then search for `TS` on the canvas. No separate frontend build is required.

To update a manual installation, run `git pull --ff-only` from the pack directory, install updated requirements with the same interpreter and restart ComfyUI.

<a id="-node-reference"></a>
<a id="image"></a><a id="video"></a><a id="hdr"></a><a id="audio"></a><a id="llm"></a><a id="text"></a><a id="ideogram"></a><a id="files"></a><a id="loaders"></a><a id="utils"></a><a id="conditioning"></a>

## Using the nodes

- **Image** — resize and alignment, masks, inpainting and restoration.
- **Video** — loading, trimming, saving, upscaling and HDR tools.
- **Audio and prompts** — audio editing, speech tools, prompt construction and model-assisted rewriting.
- **Workflow tools** — sliders, switches, file downloads and graph layout commands.

Open a node’s help in ComfyUI for its inputs and options. The same [English and Russian reference](js/docs/) and [example workflows](example_workflows/) are available here.

Some nodes need model weights, an optional Python package or a separately installed application. Check the node's input tooltips and follow any dependency message in the ComfyUI console. If a node is missing, look for `[TS Timesaver]` in that console. Start ComfyUI with `TS_VERBOSE_STARTUP=1` for the full registration report. See [package metadata](pyproject.toml) for version and dependency details.

## На русском

Timesaver добавляет ноды обработки изображений, видео, звука и промптов, а также инструменты для работы с графом.

Подсказки по конкретным нодам доступны в ComfyUI и в [справочнике](js/docs/). Готовые графы собраны в [примерах workflow](example_workflows/).

В ComfyUI Manager найдите **Timesaver**. Для ручной установки клонируйте пак в `custom_nodes` и установите [зависимости](requirements.txt) командами выше. Используйте Python, которым запускается ComfyUI. После перезапуска найдите ноды по `TS` в поиске канваса.

Для ручного обновления выполните `git pull --ff-only` в папке пака, установите обновлённые зависимости тем же Python и перезапустите ComfyUI. Для отдельных нод могут понадобиться модели, дополнительные пакеты или внешние приложения. Смотрите подсказки к входам и сообщения в консоли ComfyUI. Если нода не появилась, найдите в консоли `[TS Timesaver]`. Флаг окружения `TS_VERBOSE_STARTUP=1` включает полный отчёт регистрации при запуске.

Код распространяется по лицензии [MIT](LICENSE.txt). У сторонних моделей, SDK и приложений собственные условия.
