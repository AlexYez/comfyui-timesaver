# Standard library imports
import os
import re

class TS_EDLToYouTubeChaptersNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edl_file_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("youtube_chapters",)
    FUNCTION = "convert_edl_to_youtube_chapters"
    CATEGORY = "Tools/TS_Video"

    def convert_edl_to_youtube_chapters(self, edl_file_path):
        print("TS_EDLToYouTubeChaptersNode: Input EDL File Path:", edl_file_path)
        edl_file_path = edl_file_path.strip('"')
        if not os.path.exists(edl_file_path):
            raise ValueError(f"TS_EDLToYouTubeChaptersNode: File not found: {edl_file_path}")
        with open(edl_file_path, "r", encoding="utf-8") as file:
            edl_text = file.read()
        # print("TS_EDLToYouTubeChaptersNode: Input EDL Text:\n", edl_text) # Optional debug
        lines = edl_text.splitlines()
        chapters = []
        timecode_pattern = re.compile(
            r"^\d+\s+\d+\s+V\s+C\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}"
        )
        title_pattern = re.compile(r"\s*\|C:.*?\|M:(.*?)\|D:.*")
        start_timecode = "01:00:00:00" # Standard EDL start offset
        start_hours, start_minutes, start_seconds, _ = map(int, start_timecode.split(':')) # Frames not needed for offset calc
        i = 0
        while i < len(lines):
            if not lines[i].strip() or lines[i].startswith(("TITLE:", "FCM:")):
                i += 1; continue
            timecode_match = timecode_pattern.match(lines[i])
            if timecode_match:
                timecode = timecode_match.group(1)
                if i + 1 < len(lines):
                    title_match = title_pattern.match(lines[i + 1])
                    if title_match:
                        title = title_match.group(1).strip()
                        hours, minutes, seconds, _ = map(int, timecode.split(':')) # Frames not needed for YouTube time
                        total_seconds = (hours - start_hours) * 3600 + (minutes - start_minutes) * 60 + (seconds - start_seconds)
                        if total_seconds < 0: total_seconds = 0 # Ensure no negative time
                        
                        if total_seconds >= 3600:
                            youtube_timecode = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
                        else:
                            youtube_timecode = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
                        chapters.append(f"{youtube_timecode} {title}")
                i += 2 # Skip title line
            else:
                i += 1
        youtube_chapters_output = "\n".join(chapters)
        print("TS_EDLToYouTubeChaptersNode: YouTube Chapters Output:\n", youtube_chapters_output)
        return (youtube_chapters_output,)

NODE_CLASS_MAPPINGS = {
    "TS Youtube Chapters": TS_EDLToYouTubeChaptersNode # Ключ оставлен оригинальным
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Youtube Chapters": "TS YouTube Chapters"
}