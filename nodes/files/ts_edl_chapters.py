import logging
import os
import re

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_edl_chapters")
LOG_PREFIX = "[TS YouTube Chapters]"


class TS_EDLToYouTubeChaptersNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS Youtube Chapters",
            display_name="TS YouTube Chapters",
            category="TS/Files",
            inputs=[
                IO.String.Input("edl_file_path", default="", multiline=False),
            ],
            outputs=[IO.String.Output(display_name="youtube_chapters")],
        )

    @classmethod
    def execute(cls, edl_file_path) -> IO.NodeOutput:
        logger.info("%s Input EDL File Path: %s", LOG_PREFIX, edl_file_path)
        edl_file_path = edl_file_path.strip('"')
        if not os.path.exists(edl_file_path):
            raise ValueError(f"TS_EDLToYouTubeChaptersNode: File not found: {edl_file_path}")
        with open(edl_file_path, "r", encoding="utf-8") as file:
            edl_text = file.read()
        lines = edl_text.splitlines()
        chapters = []
        timecode_pattern = re.compile(
            r"^\d+\s+\d+\s+V\s+C\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}"
        )
        title_pattern = re.compile(r"\s*\|C:.*?\|M:(.*?)\|D:.*")
        start_timecode = "01:00:00:00"
        start_hours, start_minutes, start_seconds, _ = map(int, start_timecode.split(':'))
        i = 0
        while i < len(lines):
            if not lines[i].strip() or lines[i].startswith(("TITLE:", "FCM:")):
                i += 1
                continue
            timecode_match = timecode_pattern.match(lines[i])
            if timecode_match:
                timecode = timecode_match.group(1)
                if i + 1 < len(lines):
                    title_match = title_pattern.match(lines[i + 1])
                    if title_match:
                        title = title_match.group(1).strip()
                        hours, minutes, seconds, _ = map(int, timecode.split(':'))
                        total_seconds = (hours - start_hours) * 3600 + (minutes - start_minutes) * 60 + (seconds - start_seconds)
                        if total_seconds < 0:
                            total_seconds = 0

                        if total_seconds >= 3600:
                            youtube_timecode = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
                        else:
                            youtube_timecode = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
                        chapters.append(f"{youtube_timecode} {title}")
                i += 2
            else:
                i += 1
        youtube_chapters_output = "\n".join(chapters)
        logger.info("%s YouTube Chapters Output:\n%s", LOG_PREFIX, youtube_chapters_output)
        return IO.NodeOutput(youtube_chapters_output)


NODE_CLASS_MAPPINGS = {
    "TS Youtube Chapters": TS_EDLToYouTubeChaptersNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Youtube Chapters": "TS YouTube Chapters",
}
