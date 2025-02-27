from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "TS Youtube Chapters": EDLToYouTubeChapters,
    "TS Files Downloader": DownloadFilesNode,
    "DownloadFilesNode": DownloadFilesNode,
    "TS Equirectangular to Cube": EquirectangularToCubemapFaces,
    "TS Cube to Equirectangular": CubemapFacesToEquirectangular,
    "TS Qwen2.5": Qwen2,
}


__all__ = ['NODE_CLASS_MAPPINGS']
