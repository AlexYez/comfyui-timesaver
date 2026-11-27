# TS Audio Saver

Connect AUDIO and click the gear next to the colour swatch. Choose WAV (16-bit PCM at 44.1 kHz), FLAC (24-bit at the input sample rate) or MP3. Selecting MP3 reveals the bitrate control with 128, 192, 256 or 320 kbps (default). Run the workflow to save the full track on the server. The waveform selection sets the playback range.

Enter a prefix relative to output, without the extension, for example `stems/%date:yyyy-MM-dd%/vocals_%date:hh-mm-ss%`. In the UI, use `%Node name.widget_name%` as in Save Image to insert another widget's value. The node creates subfolders and appends a counter. Each batch item gets its own file. Use `%batch_num%` for the zero-based batch index. The player previews the first item.

For stems, connect a separate saver to each output and stack the nodes vertically. Click the coloured circle to choose one of eight presets or a custom colour. Each node stores its colour in the workflow. For finer range adjustments, open the fullscreen player with the button on the right.

To group stems by the track selected in TS Audio Loader, enter `stems/%TS_AudioLoader.file_stem%/vocals` as the filename prefix. Replace `vocals` with `drums`, `bass` or another name in each saver. Selecting `Song.mix.mp3` creates `output/stems/Song.mix/`. The token reads the file selected in the loader UI.
