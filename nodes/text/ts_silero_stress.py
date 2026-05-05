"""TS Silero Stress — Russian stress marks and 'ё' restoration via silero-stress.

node_id: TS_SileroStress
"""

import importlib
import inspect
import logging
import os
import re
import shutil

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import IO


class TS_SileroStress(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_silero_stress")
    _LOG_PREFIX = "[TS SileroStress]"

    _RUN_DEVICES = ("cpu", "gpu")
    _MODEL_DIR_NAME = "silero-stress"
    _MODEL_FILE_NAME = "accentor.pt"
    _MODEL_PACKAGE = "silero_stress.data"
    _MODEL_PICKLE_PACKAGE = "accentor_models"
    _MODEL_PICKLE_NAME = "accentor"
    _COMBINING_ACUTE = "\u0301"
    _STRESS_MARKERS = ("unicode", "silero_plus")
    _STRESS_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_SileroStress",
            display_name="TS Silero Stress",
            category="TS/Text",
            description=(
                "Automatic stress marks and yo restoration via silero-stress. "
                "Outputs Unicode combining acute accents."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Russian text for automatic stress and yo restoration.",
                ),
                IO.Combo.Input(
                    "run_device",
                    options=list(cls._RUN_DEVICES),
                    default="cpu",
                    advanced=True,
                    tooltip="Execution device for silero-stress.",
                ),
                IO.Combo.Input(
                    "stress_marker",
                    options=list(cls._STRESS_MARKERS),
                    default="unicode",
                    tooltip="Stress mark output format: Unicode combining acute or native Silero plus sign.",
                ),
                IO.Boolean.Input(
                    "use_accentor",
                    default=True,
                    tooltip="Run the common accentor for non-homograph stress and yo placement.",
                ),
                IO.Boolean.Input(
                    "use_homosolver",
                    default=True,
                    tooltip="Run the homograph disambiguation model.",
                ),
                IO.Boolean.Input(
                    "put_stress",
                    default=True,
                    tooltip="Place stress marks in non-homograph words.",
                ),
                IO.Boolean.Input(
                    "put_yo",
                    default=True,
                    tooltip="Restore letter yo in non-homograph words where needed.",
                ),
                IO.Boolean.Input(
                    "put_stress_homo",
                    default=True,
                    tooltip="Place stress marks in homographs.",
                ),
                IO.Boolean.Input(
                    "put_yo_homo",
                    default=True,
                    tooltip="Restore letter yo in homographs where needed.",
                ),
                IO.Boolean.Input(
                    "stress_single_vowel",
                    default=True,
                    tooltip="Place stress marks even in words with a single vowel.",
                ),
                IO.String.Input(
                    "words_to_ignore",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip="Comma or newline separated words that should be skipped completely.",
                ),
            ],
            outputs=[IO.String.Output(display_name="text")],
            search_aliases=["silero stress", "stress", "yo", "accentor", "homograph"],
        )

    @classmethod
    def validate_inputs(cls, text, run_device, stress_marker, **kwargs) -> bool | str:
        if not isinstance(text, str):
            return "Text must be a string."
        if run_device not in cls._RUN_DEVICES:
            return f"Unsupported run_device '{run_device}'."
        if stress_marker not in cls._STRESS_MARKERS:
            return f"Unsupported stress_marker '{stress_marker}'."
        return True

    @classmethod
    def _stress_log_info(cls, message: str) -> None:
        cls._LOGGER.info("%s %s", cls._LOG_PREFIX, message)

    @classmethod
    def _stress_log_warning(cls, message: str) -> None:
        cls._LOGGER.warning("%s %s", cls._LOG_PREFIX, message)

    @classmethod
    def _resolve_stress_device(cls, run_device: str) -> str:
        if run_device == "cpu":
            return "cpu"

        target_device = comfy.model_management.get_torch_device()
        if target_device.type == "cpu":
            cls._stress_log_warning("GPU requested but not available. Falling back to CPU.")
            return "cpu"

        return str(target_device)

    @classmethod
    def _load_accentor_runtime(cls, device_name: str):
        try:
            importlib.import_module("silero_stress")
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'silero_stress'. Install package 'silero-stress' to enable TS Silero Stress."
            ) from exc

        model_path = cls._ensure_stress_model_exists()
        accentor = torch.package.PackageImporter(model_path).load_pickle(
            cls._MODEL_PICKLE_PACKAGE,
            cls._MODEL_PICKLE_NAME,
        )
        cls._restore_stress_weights(accentor)
        if hasattr(accentor, "to"):
            accentor.to(device=device_name)
        return accentor

    @classmethod
    def _stress_model_path(cls) -> str:
        model_dir = os.path.join(folder_paths.models_dir, cls._MODEL_DIR_NAME)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, cls._MODEL_FILE_NAME)

    @classmethod
    def _ensure_stress_model_exists(cls) -> str:
        model_path = cls._stress_model_path()
        if os.path.isfile(model_path):
            return model_path

        try:
            try:
                import importlib_resources as impresources
            except ImportError:
                from importlib import resources as impresources

            package_file = impresources.files(cls._MODEL_PACKAGE).joinpath(cls._MODEL_FILE_NAME)
            with impresources.as_file(package_file) as source_path:
                shutil.copyfile(str(source_path), model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare Silero Stress model at '{model_path}'."
            ) from exc

        cls._stress_log_info(f"Prepared model in ComfyUI models directory: {model_path}")
        return model_path

    @classmethod
    def _restore_stress_weights(cls, accentor) -> None:
        quantized_weight = accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data.clone()
        restored_weights = accentor.homosolver.model.bert.scale * (
            quantized_weight - accentor.homosolver.model.bert.zero_point
        )
        accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data = restored_weights

    @classmethod
    def _parse_words_to_ignore(cls, words_to_ignore: str) -> list[str]:
        if not isinstance(words_to_ignore, str) or not words_to_ignore.strip():
            return []
        parts = re.split(r"[,;\r\n]+", words_to_ignore)
        return [part.strip() for part in parts if part.strip()]

    @classmethod
    def _filter_callable_kwargs(cls, callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return kwargs

        return {key: value for key, value in kwargs.items() if key in params}

    @classmethod
    def _invoke_stress_processor(cls, processor, text: str, **kwargs) -> str:
        filtered_kwargs = cls._filter_callable_kwargs(processor, kwargs)
        result = processor(text, **filtered_kwargs)
        if not isinstance(result, str):
            raise RuntimeError("silero-stress returned non-string output.")
        return result

    @classmethod
    def _convert_stress_marks_to_unicode(cls, text: str) -> str:
        output: list[str] = []
        pending_stress = False

        for char in text:
            if char == "+":
                if pending_stress:
                    output.append("+")
                pending_stress = True
                continue

            if pending_stress:
                if char in cls._STRESS_VOWELS:
                    output.append(char)
                    output.append(cls._COMBINING_ACUTE)
                else:
                    output.append("+")
                    output.append(char)
                pending_stress = False
                continue

            output.append(char)

        if pending_stress:
            output.append("+")

        return "".join(output)

    @classmethod
    def execute(
        cls,
        text: str,
        run_device: str = "cpu",
        stress_marker: str = "unicode",
        use_accentor: bool = True,
        use_homosolver: bool = True,
        put_stress: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = True,
        put_yo_homo: bool = True,
        stress_single_vowel: bool = True,
        words_to_ignore: str = "",
    ) -> IO.NodeOutput:
        normalized_text = text if isinstance(text, str) else ""
        if not normalized_text.strip():
            return IO.NodeOutput("")

        try:
            if not use_accentor and not use_homosolver:
                return IO.NodeOutput(normalized_text)

            accentor = cls._load_accentor_runtime(cls._resolve_stress_device(run_device))
            ignore_words = cls._parse_words_to_ignore(words_to_ignore)

            common_kwargs = {}
            if ignore_words:
                common_kwargs["words_to_ignore"] = ignore_words

            if use_accentor and use_homosolver:
                stressed_text = cls._invoke_stress_processor(
                    accentor,
                    normalized_text,
                    put_stress=put_stress,
                    put_yo=put_yo,
                    put_stress_homo=put_stress_homo,
                    put_yo_homo=put_yo_homo,
                    stress_single_vowel=stress_single_vowel,
                    **common_kwargs,
                )
            elif use_accentor:
                accentor_processor = getattr(accentor, "accentor", None)
                if accentor_processor is None or not callable(accentor_processor):
                    raise RuntimeError("silero-stress accentor processor is not available.")
                stressed_text = cls._invoke_stress_processor(
                    accentor_processor,
                    normalized_text,
                    put_stress=put_stress,
                    put_yo=put_yo,
                    stress_single_vowel=stress_single_vowel,
                    **common_kwargs,
                )
            else:
                homosolver_processor = getattr(accentor, "homosolver", None)
                if homosolver_processor is None or not callable(homosolver_processor):
                    raise RuntimeError("silero-stress homosolver processor is not available.")
                stressed_text = cls._invoke_stress_processor(
                    homosolver_processor,
                    normalized_text,
                    put_stress_homo=put_stress_homo,
                    put_yo_homo=put_yo_homo,
                    **common_kwargs,
                )

            output_text = (
                stressed_text
                if stress_marker == "silero_plus"
                else cls._convert_stress_marks_to_unicode(stressed_text)
            )
            cls._stress_log_info(
                f"Processed text: device={run_device}, stress_marker={stress_marker}, use_accentor={use_accentor}, "
                f"use_homosolver={use_homosolver}, input_length={len(normalized_text)}, "
                f"output_length={len(output_text)}"
            )
            return IO.NodeOutput(output_text)
        except Exception as exc:
            cls._stress_log_warning(f"Execution fallback activated: {exc}")
            return IO.NodeOutput(normalized_text)



NODE_CLASS_MAPPINGS = {"TS_SileroStress": TS_SileroStress}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SileroStress": "TS Silero Stress"}
