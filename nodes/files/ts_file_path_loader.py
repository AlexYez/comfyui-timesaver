import os
import glob

import folder_paths

from comfy_api.latest import IO


class TS_FilePathLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_FilePathLoader",
            display_name="TS File Path Loader",
            category="TS/Files",
            inputs=[
                IO.String.Input("folder_path", default="", multiline=False),
                IO.Int.Input("index", default=0, min=0, step=1),
            ],
            outputs=[
                IO.String.Output(display_name="file_path"),
                IO.String.Output(display_name="file_name"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder_path: str, index: int) -> str:
        return f"{folder_path}_{index}"

    @classmethod
    def execute(cls, folder_path: str, index: int) -> IO.NodeOutput:
        folder_path = os.path.normpath(folder_path.strip())

        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path '{folder_path}' does not exist or is not a directory")

        supported_extensions = folder_paths.supported_pt_extensions | {".mp4", ".mov"}

        files = []
        for ext in supported_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            files.extend(glob.glob(pattern, recursive=False))
        files = sorted(files)

        if not files:
            all_files = glob.glob(os.path.join(folder_path, "*"))
            raise ValueError(
                f"No supported files found in folder '{folder_path}'. "
                f"Supported extensions: {supported_extensions}. "
                f"Files in folder: {all_files if all_files else 'No files found'}"
            )

        if index >= len(files):
            index = index % len(files)

        file_path = os.path.normpath(files[index])
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        return IO.NodeOutput(file_path, file_name)


NODE_CLASS_MAPPINGS = {
    "TS_FilePathLoader": TS_FilePathLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FilePathLoader": "TS File Path Loader",
}
