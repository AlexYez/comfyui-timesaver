import hashlib
import os

import folder_paths

from comfy_api.v0_0_2 import IO


def _supported_extensions() -> tuple[str, ...]:
    return tuple(
        ext.lower() for ext in (folder_paths.supported_pt_extensions | {".mp4", ".mov"})
    )


def _list_supported_files(folder_path: str) -> list[str]:
    """List supported files in `folder_path`, sorted by name.

    Uses os.listdir + case-insensitive suffix matching instead of glob:
    glob treated `[`/`]` in the *folder path* as character classes (a real
    Windows folder like `D:\\models[old]` matched nothing), and lowercase
    glob patterns missed `.MP4`-style names on case-sensitive filesystems.
    """
    extensions = _supported_extensions()
    files = []
    for name in os.listdir(folder_path):
        if not name.lower().endswith(extensions):
            continue
        full = os.path.join(folder_path, name)
        if os.path.isfile(full):
            files.append(full)
    return sorted(files)


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
        # Include a digest of the folder's current listing so adding/removing
        # files re-triggers execution. The previous fingerprint was just
        # `folder_path_index`, identical to default input caching: a stale
        # cached path survived any change to the directory contents.
        normalized = os.path.normpath(str(folder_path or "").strip())
        try:
            files = _list_supported_files(normalized)
        except OSError:
            files = []
        digest = hashlib.sha256(
            "\n".join(os.path.basename(f) for f in files).encode("utf-8")
        ).hexdigest()
        return f"{normalized}|{index}|{digest}"

    @classmethod
    def execute(cls, folder_path: str, index: int) -> IO.NodeOutput:
        folder_path = os.path.normpath(folder_path.strip())

        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path '{folder_path}' does not exist or is not a directory")

        files = _list_supported_files(folder_path)

        if not files:
            all_files = sorted(os.listdir(folder_path))
            raise ValueError(
                f"No supported files found in folder '{folder_path}'. "
                f"Supported extensions: {sorted(_supported_extensions())}. "
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
