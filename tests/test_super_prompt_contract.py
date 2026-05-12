from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine_stubs import _DummyEngine, install_engine_module_stub  # noqa: E402


class _Input:
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.args = args
        self.kwargs = kwargs
        self.optional = bool(kwargs.get("optional", False))
        self.socketless = bool(kwargs.get("socketless", False))
        self.default = kwargs.get("default")
        self.options = kwargs.get("options", args[0] if args else None)


class _Output:
    def __init__(self, id=None, display_name=None, **kwargs):
        self.id = id
        self.display_name = display_name
        self.kwargs = kwargs


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *values, **kwargs):
        self.values = values
        self.kwargs = kwargs


class _ComfyType:
    Input = _Input
    Output = _Output


class _IO:
    ComfyNode = object
    Schema = _Schema
    NodeOutput = _NodeOutput
    String = _ComfyType
    Boolean = _ComfyType
    Combo = _ComfyType
    Int = _ComfyType
    Image = _ComfyType


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_PRESET_FIXTURE = (
    {
        "Prompts enhance": {
            "system_prompt": "Enhance prompt.",
            "gen_params": {"temperature": 0.8},
        }
    },
    ["Prompts enhance"],
)


def _fixture_load_presets():
    return _PRESET_FIXTURE


_ENGINE_SINGLETON = _DummyEngine()


def _fixture_get_qwen_engine() -> _DummyEngine:
    return _ENGINE_SINGLETON


def _install_stubs(monkeypatch):
    root = Path(__file__).resolve().parents[1]

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.v0_0_2")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(root / ".test_models")
    folder_paths.get_input_directory = lambda: str(root / ".test_input")
    folder_paths.get_annotated_filepath = lambda annotated: ""
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    aiohttp = types.ModuleType("aiohttp")
    web = types.SimpleNamespace(
        Request=object,
        StreamResponse=object,
        json_response=lambda data, status=200: {"data": data, "status": status},
    )
    aiohttp.web = web
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.web", web)

    # nodes.llm.ts_qwen3_vl — only the symbols super_prompt imports.
    qwen_module = types.ModuleType("nodes.llm.ts_qwen3_vl")
    qwen_module._load_presets = _fixture_load_presets
    qwen_module.TS_Qwen3_VL_V3 = _DummyEngine
    monkeypatch.setitem(sys.modules, "nodes.llm.ts_qwen3_vl", qwen_module)

    # nodes.llm._qwen_engine stub. The real engine pulls in torch and would
    # force tests to depend on a CUDA-capable Python install.
    engine_module = install_engine_module_stub(monkeypatch)
    engine_module.get_qwen_engine = _fixture_get_qwen_engine


def _load_module(monkeypatch):
    """Return the public shim that aggregates the super_prompt subpackage.

    To monkeypatch behaviour inside the subpackage, patch the originating
    submodule (`_helpers` / `_voice` / `_qwen`) — `monkeypatch.setattr(shim, …)`
    only mutates the shim's own dict and does not propagate back into the
    submodules that already imported the symbol by value.
    """
    _install_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    for cached in (
        "nodes.llm.ts_super_prompt",
        "nodes.llm.super_prompt",
        "nodes.llm.super_prompt._helpers",
        "nodes.llm.super_prompt._voice",
        "nodes.llm.super_prompt._qwen",
        "nodes.llm.super_prompt.ts_super_prompt",
    ):
        sys.modules.pop(cached, None)
    return importlib.import_module("nodes.llm.ts_super_prompt")


def _load_qwen():
    return importlib.import_module("nodes.llm.super_prompt._qwen")


def test_super_prompt_schema_contract(monkeypatch):
    module = _load_module(monkeypatch)

    schema = module.TS_SuperPrompt.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "TS_SuperPrompt"
    assert schema.display_name == "TS Super Prompt"
    assert schema.category == "TS/LLM"
    # "Your instruction" preset is intentionally hidden from this node.
    # The Image input has been replaced by the socketless ``attached_image``
    # string widget driven by the JS frontend's Attach button.
    assert list(inputs) == ["text", "high_quality", "system_preset", "attached_image"]
    assert inputs["text"].default == ""
    assert inputs["high_quality"].default is False
    assert inputs["system_preset"].options == ["Prompts enhance"]
    assert inputs["attached_image"].default == ""
    assert inputs["attached_image"].socketless is True
    assert "промпта" in inputs["text"].kwargs["tooltip"]
    assert "turbo" in inputs["high_quality"].kwargs["tooltip"]
    assert "пресет" in inputs["system_preset"].kwargs["tooltip"]
    assert "изображения" in inputs["attached_image"].kwargs["tooltip"]
    assert schema.outputs[0].display_name == "text"


def test_super_prompt_shim_does_not_double_register_node(monkeypatch):
    """The shim must not duplicate NODE_CLASS_MAPPINGS — see the snapshot tool.

    The real entry lives at ``nodes.llm.super_prompt.ts_super_prompt``. Both
    files registering the same id confused ``tools/build_node_contracts.py``
    into recording the shim as the node's source file with ``api='unknown'``.
    """
    module = _load_module(monkeypatch)
    real = importlib.import_module("nodes.llm.super_prompt.ts_super_prompt")

    assert module.TS_SuperPrompt is real.TS_SuperPrompt
    assert module.NODE_CLASS_MAPPINGS == {}
    assert real.NODE_CLASS_MAPPINGS == {"TS_SuperPrompt": real.TS_SuperPrompt}


def test_super_prompt_default_model_has_stock_qwen_option(monkeypatch):
    module = _load_module(monkeypatch)

    assert module.DEFAULT_MODEL_ID == "huihui-ai/Huihui-Qwen3.5-2B-abliterated"
    assert module.SUPER_PROMPT_MODEL_QWEN_2B == "Qwen/Qwen3.5-2B"
    assert module.SUPER_PROMPT_MODEL_QWEN_2B in module.SUPER_PROMPT_MODEL_OPTIONS


def test_qwen_download_monitor_scales_directory_progress(monkeypatch):
    module = _load_module(monkeypatch)
    qwen = _load_qwen()
    events = []

    # QwenDownloadProgressMonitor lives in _qwen.py and resolves
    # send_progress / directory_size from its own namespace (imported by
    # value from _helpers). Patches must target _qwen, not the public shim.
    monkeypatch.setattr(qwen, "send_progress", lambda op, text, percent=None: events.append((op, text, percent)))
    monkeypatch.setattr(qwen, "directory_size", lambda _path: 50)

    monitor = module.QwenDownloadProgressMonitor("op", "test/model", Path("unused"), 100, 20.0, 40.0)
    monitor._emit_progress()

    assert events[-1][0] == "op"
    assert "50 B" in events[-1][1]
    assert events[-1][2] == 30.0


def test_super_prompt_strips_thinking_output(monkeypatch):
    module = _load_module(monkeypatch)

    assert module._clean_model_output("<think>hidden</think>\nFinal prompt: \"A clean prompt\"") == "A clean prompt"


def test_super_prompt_disables_thinking_in_chat_template(monkeypatch):
    module = _load_module(monkeypatch)

    class Processor:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return {"input_ids": [[1, 2, 3]]}

    processor = Processor()
    result = module._apply_chat_template_no_thinking(_DummyEngine(), processor, [{"role": "user", "content": []}])

    assert result == {"input_ids": [[1, 2, 3]]}
    assert processor.kwargs["enable_thinking"] is False


def test_super_prompt_uses_structured_chat_for_processors(monkeypatch):
    module = _load_module(monkeypatch)

    messages = module._build_messages("System", "idea", "image", image=None, max_image_size=1024)

    assert isinstance(messages[0]["content"], list)
    assert isinstance(messages[1]["content"], list)
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["type"] == "text"
    assert "non-thinking" in messages[0]["content"][0]["text"]


def test_super_prompt_flattens_chat_for_text_only_template(monkeypatch):
    module = _load_module(monkeypatch)

    class TextOnlyProcessor:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            if isinstance(messages[0]["content"], list):
                raise TypeError("expected string content")
            self.messages = messages
            self.kwargs = kwargs
            return {"input_ids": [[4, 5, 6]]}

    processor = TextOnlyProcessor()
    messages = module._build_messages("System", "idea", "image", image=None, max_image_size=1024)
    result = module._apply_chat_template_no_thinking(_DummyEngine(), processor, messages)

    assert result == {"input_ids": [[4, 5, 6]]}
    assert isinstance(processor.messages[0]["content"], str)
    assert processor.kwargs["enable_thinking"] is False


def test_super_prompt_structured_chat_matches_transformers_processor(monkeypatch):
    module = _load_module(monkeypatch)

    class StructuredProcessor:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            for message in messages:
                for content in message["content"]:
                    _ = content["type"]
            self.kwargs = kwargs
            return {"input_ids": [[7, 8, 9]]}

    processor = StructuredProcessor()
    messages = module._build_messages("System", "idea", "image", image=None, max_image_size=1024)
    result = module._apply_chat_template_no_thinking(_DummyEngine(), processor, messages)

    assert result == {"input_ids": [[7, 8, 9]]}
    assert processor.kwargs["enable_thinking"] is False


def test_super_prompt_normalizes_openai_style_generation_params(monkeypatch):
    module = _load_module(monkeypatch)

    params = module._normalize_generation_params(
        {
            "max_tokens": 123,
            "temperature": 0.7,
            "stop": ["END"],
            "presence_penalty": 0.2,
        }
    )

    assert params["max_new_tokens"] == 123
    assert params["temperature"] == 0.7
    assert "max_tokens" not in params
    assert "stop" not in params
    assert "presence_penalty" not in params


def test_super_prompt_validate_accepts_unknown_preset(monkeypatch):
    """Legacy workflows referencing a renamed/removed preset must keep
    loading — validate_inputs accepts any string and _resolve_preset falls
    back to the canonical default at execute time."""
    module = _load_module(monkeypatch)

    assert module.TS_SuperPrompt.validate_inputs(
        text="hello",
        high_quality=False,
        system_preset="Some preset that was renamed in v9.6",
        attached_image="",
    ) is True


def test_super_prompt_resolve_preset_falls_back_on_unknown(monkeypatch):
    """If a workflow ships an unknown preset, _resolve_preset returns the
    DEFAULT_PRESET system_prompt + gen_params instead of empty/zero values."""
    module = _load_module(monkeypatch)

    system_prompt, gen_params = module._resolve_preset(
        "Some preset that was renamed", custom_system_prompt=None
    )
    assert system_prompt == "Enhance prompt."
    assert gen_params == {"temperature": 0.8}


def test_super_prompt_retries_without_unused_model_kwargs(monkeypatch):
    module = _load_module(monkeypatch)

    class Model:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "max_tokens" in kwargs:
                raise ValueError("The following `model_kwargs` are not used by the model: ['max_tokens']")
            return ["generated"]

    model = Model()
    result = module._generate_with_filtered_kwargs(
        model,
        {"input_ids": [1, 2, 3]},
        {"max_tokens": 12, "max_new_tokens": 16},
    )

    assert result == ["generated"]
    assert len(model.calls) == 2
    assert "max_tokens" not in model.calls[-1]
    assert model.calls[-1]["max_new_tokens"] == 16
