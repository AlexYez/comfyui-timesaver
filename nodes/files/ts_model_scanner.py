"""TS Model Scanner — readable summary of a safetensors / ckpt file.

node_id: TS_ModelScanner
"""

import os

import folder_paths
import comfy.model_patcher
from comfy_api.latest import IO
from safetensors import safe_open


def _build_model_choices():
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    model_files = [f for f in diffusion_models if f.endswith(".safetensors")]
    if not model_files:
        model_files = ["No diffusion models found"]
    return sorted(model_files)


class TS_ModelScanner(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ModelScanner",
            display_name="TS Model Scanner",
            category="TS/Files",
            inputs=[
                IO.Combo.Input("model_name", options=_build_model_choices()),
                IO.Model.Input("model", optional=True),
                IO.Boolean.Input(
                    "summary_only",
                    default=False,
                    label_on="Summary Only",
                    label_off="Full Detail",
                    optional=True,
                ),
            ],
            outputs=[IO.String.Output(display_name="model_info")],
        )

    @classmethod
    def _scan_loaded_model(cls, model, summary_only=False):
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            real_model = model.model
        else:
            real_model = model

        output_lines = []
        stats = {}
        total_params = 0

        output_lines.append("=== MODEL SCAN REPORT ===")
        output_lines.append("Source: loaded MODEL")
        output_lines.append(f"Type: {type(real_model).__name__}")
        output_lines.append("-" * 60)

        try:
            iterator = real_model.named_parameters()
            if hasattr(real_model, "diffusion_model"):
                output_lines.append("Note: Scanning internal diffusion_model")
                iterator = real_model.diffusion_model.named_parameters()

            for name, param in iterator:
                shape_str = str(tuple(param.shape))
                dtype_str = str(param.dtype).replace("torch.", "")
                device_str = str(param.device).split(":")[0]
                num_params = param.numel()

                total_params += num_params
                if dtype_str not in stats:
                    stats[dtype_str] = 0
                stats[dtype_str] += num_params

                if not summary_only:
                    output_lines.append(f"{name:<50} | {shape_str:<20} | {dtype_str:<10} | {device_str:<6}")

        except Exception as e:
            return f"Error scanning model: {str(e)}"

        output_lines.append("-" * 60)
        output_lines.append("=== SUMMARY STATISTICS ===")
        output_lines.append(f"Total Params: {total_params:,}")
        for dtype, count in stats.items():
            percent = (count / total_params) * 100 if total_params > 0 else 0
            output_lines.append(f" - {dtype}: {count:,} ({percent:.2f}%)")

        return "\n".join(output_lines)

    @classmethod
    def _scan_safetensors_file(cls, model_path, summary_only=False):
        output_lines = []
        stats = {}
        total_params = 0

        output_lines.append("=== MODEL SCAN REPORT ===")
        output_lines.append("Source: diffusion_models (disk)")
        output_lines.append(f"File: {model_path}")
        output_lines.append("Note: Scanning safetensors file")
        output_lines.append("-" * 60)

        try:
            with safe_open(model_path, framework="pt", device="cpu") as f_in:
                for name in f_in.keys():
                    tensor = f_in.get_tensor(name)
                    shape_str = str(tuple(tensor.shape))
                    dtype_str = str(tensor.dtype).replace("torch.", "")
                    device_str = "cpu"
                    num_params = tensor.numel()

                    total_params += num_params
                    if dtype_str not in stats:
                        stats[dtype_str] = 0
                    stats[dtype_str] += num_params

                    if not summary_only:
                        output_lines.append(f"{name:<50} | {shape_str:<20} | {dtype_str:<10} | {device_str:<6}")

                    del tensor

        except Exception as e:
            return f"Error scanning safetensors file: {str(e)}"

        output_lines.append("-" * 60)
        output_lines.append("=== SUMMARY STATISTICS ===")
        output_lines.append(f"Total Params: {total_params:,}")
        for dtype, count in stats.items():
            percent = (count / total_params) * 100 if total_params > 0 else 0
            output_lines.append(f" - {dtype}: {count:,} ({percent:.2f}%)")

        return "\n".join(output_lines)

    @classmethod
    def execute(cls, model_name, model=None, summary_only=False) -> IO.NodeOutput:
        if model is not None:
            return IO.NodeOutput(cls._scan_loaded_model(model, summary_only))

        if not model_name or model_name == "No diffusion models found":
            return IO.NodeOutput("Error: No diffusion_models available for scanning.")

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        if not model_path or not os.path.exists(model_path):
            return IO.NodeOutput(f"Error: File not found: {model_path}")

        if not model_path.endswith(".safetensors"):
            return IO.NodeOutput(f"Error: Unsupported format for scanning: {model_path}")

        return IO.NodeOutput(cls._scan_safetensors_file(model_path, summary_only))


NODE_CLASS_MAPPINGS = {"TS_ModelScanner": TS_ModelScanner}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelScanner": "TS Model Scanner"}
