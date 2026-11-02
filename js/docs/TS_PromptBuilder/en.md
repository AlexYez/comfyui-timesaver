# TS Prompt Builder

Assembles a prompt from wildcard files and your own text, in the order shown.
Each enabled TXT block contributes one random non-empty line. Each enabled
custom block contributes its text. Parts are joined with a comma and a space.

Click **Open interface** to arrange the blocks as tiles in the pack's shared
fullscreen editor. Drag a tile by its handle, or focus its toggle and use
**Alt + arrow keys**. Tile numbers show output order; the toggle enables or
disables that block. Close with **Esc** or the shared close button. The compact
node and fullscreen editor share the same state.

**+ Your text** adds an editable block. You can add several, place them before,
between or after file blocks, disable them, or remove them with ×. Text,
enabled flags and order are saved in the workflow's existing `config_json`;
no new node inputs are required. Free text is not written into the shared
default configuration. Executing a saved workflow does not overwrite defaults
for other workflows.

Wildcard files are `.txt` files directly inside `nodes/prompts/` (one choice per
non-empty line). Subfolders and semantic maps are not read. **Reload** refreshes
the file list without restarting ComfyUI, preserving your text and existing
choices. New files in a saved workflow are appended disabled. Files removed
from disk disappear from the list. Explicit file toggles and ordering can also
be used as defaults for new nodes; custom text always stays workflow-local.

`seed = 0` selects fresh random lines every run. A positive seed makes each
file's selection reproducible; moving or editing a custom block does not change
which line is picked from another file. Newlines inside custom text are kept.

[Full node reference](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
