<div align="center">
<img src="icon.png" alt="Timesaver" width="100" />

# Timesaver Nodes for ComfyUI

Nodes and frontend extensions for images, video, audio, prompts and workflow tools.

[Русский](README.ru.md) · [Version and requirements](pyproject.toml) · [Changes](CHANGELOG.md) · [MIT](LICENSE.txt)
</div>

## Start here

ComfyUI loads this pack from `custom_nodes`; there is no separate frontend build. Search for `TS` on the canvas after installation. Some ML features download weights or require optional dependencies: consult the individual node help.

| Task | Read |
|---|---|
| Install, run or update | [Development and installation](doc/operations/development.md). |
| Find a node and its controls | [Node reference, EN/RU](doc/nodes/README.md), [screenshots](doc/nodes/screenshots.md). |
| Arrange a graph or recreate a node | [Frontend commands](doc/user-guide/README.md). |
| Understand the internals | [Architecture](doc/architecture/README.md). |
| Change the project | [AGENTS.md](AGENTS.md), then [engineering rules](doc/engineering/README.md). |
| Validate a change | [Checks and CI limits](doc/operations/testing.md). |
| Release | [Release procedure](doc/operations/release.md). |
| Find configuration or logs | [Environment and data](doc/operations/environment.md). |
| Diagnose a failure | [Troubleshooting](doc/operations/troubleshooting.md). |
| Learn requirements and decisions | [Project memory](project_memory/MEMORY.md). |

Contributor documentation is maintained in Russian, the owner's working language. Runtime node help is available in English and Russian.

## Install

In ComfyUI Manager, search for `Timesaver` and install it. Or, from your ComfyUI directory:

```sh
cd custom_nodes
git clone https://github.com/AlexYez/comfyui-timesaver
cd comfyui-timesaver
python -m pip install -r requirements.txt
```

Use the interpreter that runs ComfyUI, then restart ComfyUI. For updates, use `git pull --ff-only`, install updated dependencies and arrange the restart with the owner. See [installation details](doc/operations/development.md).

## Repository map

```text
__init__.py          ComfyUI V3 loader and entrypoint
nodes/              Python nodes and shared ML/media helpers
js/                 frontend extensions and bundled js/docs help
locales/ru/         Russian interface translations
example_workflows/  example graphs
doc/                modular documentation and history
project_memory/     owner requirements, decisions, knowledge and history
pyproject.toml      canonical version, requirements and Registry metadata
```

## Compatibility

Saved workflows depend on node IDs, input/output names and order, defaults and categories. Compare changes against the [contract snapshot](tests/contracts/node_contracts.json). The [owner's original requirements](project_memory/owner/README.md) take precedence over summaries.

Documentation and memory are versioned. Credentials, model weights, outputs and the optional local test laboratory are not. [All documentation](doc/README.md).

## License and support

The pack's code is [MIT licensed](LICENSE.txt). Downloaded models, SDKs and external applications retain their own terms. [Credits](doc/CREDITS.md). Maintainer: [AlexYez](https://github.com/AlexYez); [issues](https://github.com/AlexYez/comfyui-timesaver/issues).

This pack hosts and redistributes none of the runtime and is not affiliated with or endorsed by NVIDIA. Install only components you are authorised to use.

<a id="-node-reference"></a>
<a id="image"></a><a id="video"></a><a id="hdr"></a><a id="audio"></a><a id="llm"></a><a id="text"></a><a id="ideogram"></a><a id="files"></a><a id="loaders"></a><a id="utils"></a><a id="conditioning"></a>
The former inline reference now lives in the [node catalog](doc/nodes/README.md).
