from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


class _Input:
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.args = args
        self.kwargs = kwargs
        self.optional = bool(kwargs.get("optional", False))
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


class _DummyQwen:
    @classmethod
    def _load_presets(cls):
        return {
            "Prompts enhance": {
                "system_prompt": "Enhance prompt.",
                "gen_params": {"temperature": 0.8},
            }
        }, ["Prompts enhance"]

    @classmethod
    def _is_bitsandbytes_available(cls):
        return False


def _install_stubs(monkeypatch):
    root = Path(__file__).resolve().parents[1]

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(root / ".test_models")
    folder_paths.get_input_directory = lambda: str(root / ".test_input")
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

    qwen_module = types.ModuleType("nodes.ts_qwen3_vl_v3_node")
    qwen_module.TS_Qwen3_VL_V3 = _DummyQwen
    monkeypatch.setitem(sys.modules, "nodes.ts_qwen3_vl_v3_node", qwen_module)


def _load_module(monkeypatch):
    _install_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.ts_super_prompt_node", None)
    return importlib.import_module("nodes.ts_super_prompt_node")


def test_super_prompt_schema_contract(monkeypatch):
    module = _load_module(monkeypatch)

    schema = module.TS_SuperPrompt.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "TS_SuperPrompt"
    assert schema.display_name == "TS Super Prompt"
    assert schema.category == "TS/LLM"
    assert list(inputs) == ["text", "high_quality", "system_preset", "image"]
    assert inputs["text"].default == ""
    assert inputs["high_quality"].default is False
    assert inputs["system_preset"].options == ["Prompts enhance", "Your instruction"]
    assert inputs["image"].optional is True
    assert "промпта" in inputs["text"].kwargs["tooltip"]
    assert "turbo" in inputs["high_quality"].kwargs["tooltip"]
    assert "пресет" in inputs["system_preset"].kwargs["tooltip"]
    assert "изображение" in inputs["image"].kwargs["tooltip"]
    assert schema.outputs[0].display_name == "text"


def test_super_prompt_default_model_has_stock_qwen_option(monkeypatch):
    module = _load_module(monkeypatch)

    assert module.DEFAULT_MODEL_ID == "huihui-ai/Huihui-Qwen3.5-2B-abliterated"
    assert module.SUPER_PROMPT_MODEL_QWEN_2B == "Qwen/Qwen3.5-2B"
    assert module.SUPER_PROMPT_MODEL_QWEN_2B in module.SUPER_PROMPT_MODEL_OPTIONS


def test_qwen_download_monitor_scales_directory_progress(monkeypatch):
    module = _load_module(monkeypatch)
    events = []

    monkeypatch.setattr(module, "_send_progress", lambda op, text, percent=None: events.append((op, text, percent)))
    monkeypatch.setattr(module, "_directory_size", lambda _path: 50)

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
    result = module._apply_chat_template_no_thinking(_DummyQwen(), processor, [{"role": "user", "content": []}])

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
    result = module._apply_chat_template_no_thinking(_DummyQwen(), processor, messages)

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
    result = module._apply_chat_template_no_thinking(_DummyQwen(), processor, messages)

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
