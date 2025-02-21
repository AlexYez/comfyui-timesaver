from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "Timesaver Youtube Chapters": EDLToYouTubeChapters,
    "Timesaver Files Downloader": DownloadFilesNode,
}


__all__ = ['NODE_CLASS_MAPPINGS']
