#  Package Modules
import re
import os
import requests
from tqdm import tqdm

class DownloadFilesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_list": ("STRING", {
                    "default": "https://example.com/file1.txt /path/to/save\nhttps://example.com/file2.txt /path/to/save",
                    "multiline": True,
                    "dynamicPrompts": False,
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "download_files"
    OUTPUT_NODE = True
    CATEGORY = "custom_nodes"

    def parse_file_list(self, file_list_text):
        """Parse the file list from the multiline text input."""
        files = []
        for line in file_list_text.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) == 2:
                files.append((parts[0], parts[1]))
        return files

    def download_file(self, url, local_dir):
        """Download a file from a URL and save it to the specified directory with resume support."""
        os.makedirs(local_dir, exist_ok=True)  # Create directory if it doesn't exist
        filename = os.path.basename(url)  # Extract filename from URL
        local_file_path = os.path.join(local_dir, filename)
        temp_file_path = local_file_path + ".part"
        
        # Get remote file size with retry logic
        try:
            response = requests.head(url, allow_redirects=True)  # Allow redirects in case of URL redirection
            if response.status_code != 200:
                print(f"Failed to fetch file info for {filename}. Status code: {response.status_code}. Skipping.")
                return
            remote_file_size = int(response.headers.get('content-length', 0))
            if remote_file_size == 0:
                print(f"Unable to determine file size for {filename}. Skipping.")
                return
        except requests.RequestException as e:
            print(f"Error fetching file info for {filename}: {e}. Skipping.")
            return

        # Check if file already exists and is fully downloaded
        if os.path.exists(local_file_path):
            local_file_size = os.path.getsize(local_file_path)
            if local_file_size == remote_file_size:
                print(f"File {filename} already downloaded. Skipping.")
                return
        
        headers = {}
        resume_byte_pos = 0
        
        if os.path.exists(temp_file_path):
            resume_byte_pos = os.path.getsize(temp_file_path)
            headers["Range"] = f"bytes={resume_byte_pos}-"
        
        print(f"Downloading {filename} to {local_file_path}...")
        try:
            response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
            if response.status_code not in [200, 206]:
                print(f"Failed to download {filename}. Status code: {response.status_code}. Skipping.")
                return

            total_size = remote_file_size
            
            with open(temp_file_path, 'ab') as file, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filename, initial=resume_byte_pos
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            os.rename(temp_file_path, local_file_path)
            print(f"File saved to {local_file_path}\n")
        except requests.RequestException as e:
            print(f"Error downloading {filename}: {e}. Skipping.")

    def download_files(self, file_list):
        """Download files based on the provided text input."""
        files_to_download = self.parse_file_list(file_list)
        for file_url, target_dir in files_to_download:
            self.download_file(file_url, target_dir)
        
        return {}
    
class EDLToYouTubeChapters:
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

    CATEGORY = "text processing"

    def convert_edl_to_youtube_chapters(self, edl_file_path):
        # Debugging: Print the input file path
        print("Input EDL File Path:", edl_file_path)

        # Remove any surrounding quotation marks from the file path
        edl_file_path = edl_file_path.strip('"')

        # Check if the file exists
        if not os.path.exists(edl_file_path):
            raise ValueError(f"File not found: {edl_file_path}")

        # Read the .edl file content
        with open(edl_file_path, "r", encoding="utf-8") as file:
            edl_text = file.read()

        # Debugging: Print the input text
        print("Input EDL Text:\n", edl_text)

        # Split the EDL text into lines
        lines = edl_text.splitlines()

        # Initialize an empty list to store the chapters
        chapters = []

        # Regular expression to match the timecode (first line of an event)
        timecode_pattern = re.compile(
            r"^\d+\s+\d+\s+V\s+C\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}:\d{2}"
        )

        # Regular expression to match the title (second line of an event)
        title_pattern = re.compile(r"\s*\|C:.*?\|M:(.*?)\|D:.*")

        # Define the start timecode offset (01:00:00:00)
        start_timecode = "01:00:00:00"
        start_hours, start_minutes, start_seconds, start_frames = map(int, start_timecode.split(':'))

        # Iterate through the lines
        i = 0
        while i < len(lines):
            # Skip blank lines and header lines
            if not lines[i].strip() or lines[i].startswith(("TITLE:", "FCM:")):
                print(f"Skipping Line: {lines[i]}")
                i += 1
                continue

            # Check if the current line is a timecode line
            timecode_match = timecode_pattern.match(lines[i])
            if timecode_match:
                timecode = timecode_match.group(1)
                print(f"Matched Timecode Line: {lines[i]}")
                print(f"Extracted Timecode: {timecode}")

                # Check if the next line is a metadata line
                if i + 1 < len(lines):
                    title_match = title_pattern.match(lines[i + 1])
                    if title_match:
                        title = title_match.group(1).strip()
                        print(f"Matched Metadata Line: {lines[i + 1]}")
                        print(f"Extracted Title: {title}")

                        # Convert the timecode to total seconds
                        hours, minutes, seconds, frames = map(int, timecode.split(':'))
                        total_seconds = (
                            (hours - start_hours) * 3600 +  # Subtract start hours
                            (minutes - start_minutes) * 60 +  # Subtract start minutes
                            (seconds - start_seconds)         # Subtract start seconds
                        )

                        # Convert total seconds to YouTube format (MM:SS or HH:MM:SS)
                        if total_seconds >= 3600:
                            # If the video is longer than 1 hour, use HH:MM:SS
                            youtube_timecode = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
                        else:
                            # If the video is shorter than 1 hour, use MM:SS
                            youtube_timecode = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

                        # Append the chapter to the list
                        chapters.append(f"{youtube_timecode} {title}")
                    else:
                        print(f"No Title Match for Line: {lines[i + 1]}")
                else:
                    print("No Metadata Line Found After Timecode Line")

                # Skip the next line (metadata line) since we've already processed it
                i += 2
            else:
                # Skip lines that don't match the timecode pattern
                print(f"Skipping Line (No Timecode Match): {lines[i]}")
                i += 1

        # Join the chapters into a single string with newlines
        youtube_chapters = "\n".join(chapters)

        # Debugging: Print the output to the console
        print("YouTube Chapters Output:\n", youtube_chapters)

        return (youtube_chapters,)