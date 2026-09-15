# TS Song Creator

The lyrics and the style in one editor, and two ordinary strings on the way out — exactly what song models (**YuE 2**, **ACE-Step**) ask for.

**Section tags by button.** `[verse]`, `[chorus]`, `[bridge]` and the rest go in with a click and always land on their own line, which is how the model knows where a part begins. The list comes from the preset, so a model that reads different tags is a data change, not a code change.

**Stress by right-click.** Select a vowel (or just put the caret on it) and choose **Mark stressed**. The letter gets a combining accent — the one dictionaries use and the one a model reads as "the stress is here". Calling it again takes the mark off, and a separate item clears them all. Cyrillic has no precomposed accented letters at all, so by hand this means copying a character out of a character map — that single command is why the whole menu is drawn here, familiar Copy / Cut / Paste / Select all included.

**A style library.** 55 considered presets for YuE 2 — four kinds of blues, ten rock flavours, nine electronic, jazz and big band, hip-hop from old-school to drill, cinematic orchestral, a lullaby and chiptune — searchable, each with a one-line description. The names are translated; **the prompt itself deliberately stays English**, because the model was trained on those words and translating them does not translate the conditioning, it breaks it. A chosen style lands in the field and is yours to edit: a preset is a starting point, not a cage.

**The voice is its own choice.** Sex first — female, male or other — then the kinds and shades within it: clear, airy, powerful, husky, whispered, soulful, rap, falsetto, screamed, and under "other" a duet, a choir, gang vocals, vocal chops or no vocals at all. Twenty-six voices in all. A genre brings its usual voice along, but the one you pick **replaces** it rather than adding to it — three choices in a row will not leave three singers in the field. Your own edits survive: if you rewrote that part by hand, the new voice is simply appended.

**Full screen.** For long lyrics: the same fields across the whole window with the style library beside them. The content moves into the overlay and back, so the selection and the undo history survive opening it.

Presets are plain JSON in `nodes/text/song_presets/`, one file per model.

**Use when:** any song you hand to YuE or ACE-Step — especially a Russian one, where without stress marks the model sings the words with the wrong syllable stressed.


<a id="ideogram"></a>
### 🎨 Ideogram (1 node)

Design tools for the open-weight **Ideogram 4** image model.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
