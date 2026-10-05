"""Repair of the Ideogram 4 JSON captions a small Qwen writes.

Shared by two callers: the Super Prompt engine (every JSON preset — the
Ideogram editor's Generate / Read-from-image buttons go through its /enhance
route) and TS Ideogram Designer's Prompt mode. Pure stdlib, no ComfyUI imports.

⚠️ Why it exists (measured 05.10.2026, Huihui Qwen3.5 4B, "Ideogram Prompt
Enhance", 6 answers out of 6 unparseable):

- two elements MERGED into one object — the model forgets ``},{`` between them
  and closes with a spare ``}`` at the end: ``{"type":"obj",…,"type":"text",…}}]}``;
- a missing closing quote after a colour: ``"#7A9CB3]``;
- on-image text the idea never asked for (the preset's example "БАЙКАЛ" is
  copied onto a sleeping fox), or a requested one cut short
  ("КОФЕ И КНИГ" for «Кофе и книги»).

Nothing here invents content: structure is mended, requested text is restored
to what was typed, and text nobody asked for is dropped.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

logger = logging.getLogger("comfyui_timesaver.caption_json")
LOG_PREFIX = "[TS Caption JSON]"

_CLOSERS = {"}": "{", "]": "["}
_OPENERS = {"{": "}", "[": "]"}
# A colour whose closing quote went missing: `"#7A9CB3]` / `"#7A9CB3,`.
_OPEN_HEX = re.compile(r'"(#[0-9A-Fa-f]{3,8})(?=\s*[\],}])')
# Quoted spans in the idea: "…", «…», “…”, „…“.
_QUOTED = re.compile(r'"([^"\n]+)"|«([^»\n]+)»|“([^”\n]+)”|„([^“”\n]+)[“”]')
# How close a model's text must be to a quoted span to count as that span.
_MATCH_RATIO = 0.6


class _Merged(list):
    """Several elements the model wrote into one JSON object."""


def _pairs_hook(pairs: list[tuple[str, Any]]) -> Any:
    """Split an object that carries more than one ``type`` key into elements."""
    if sum(1 for key, _ in pairs if key == "type") < 2:
        return dict(pairs)
    parts: list[dict[str, Any]] = []
    for key, value in pairs:
        if key == "type" or not parts:
            parts.append({})
        parts[-1][key] = value
    return _Merged(parts)


def _flatten(value: Any) -> Any:
    """Put merged elements back into their array; a stray merge keeps its first part."""
    if isinstance(value, _Merged):
        return _flatten(value[0]) if value else {}
    if isinstance(value, dict):
        return {key: _flatten(item) for key, item in value.items()}
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            if isinstance(item, _Merged):
                out.extend(_flatten(part) for part in item)
            else:
                out.append(_flatten(item))
        return out
    return value


def _looks_like_element(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in ("bbox", "text", "type", "desc"))


def _clamp_bbox(value: Any) -> list[int] | None:
    """[y_min, x_min, y_max, x_max] as ints on the 0-1000 grid, or None if unusable."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        numbers = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    y0, x0, y1, x1 = (min(1000, max(0, item)) for item in numbers)
    y0, y1 = min(y0, y1), max(y0, y1)
    x0, x1 = min(x0, x1), max(x0, x1)
    if y1 - y0 < 1 or x1 - x0 < 1:
        return None
    return [y0, x0, y1, x1]


def _normalise_elements(elements: list[Any]) -> list[dict[str, Any]]:
    """One flat list of well-formed elements.

    ⚠️ Seen live on a long prompt (4B): the couple's element carried every
    other element as an array in its `desc`; texts typed as "obj"; a type
    missing; bboxes at 1050/1080; an empty ``{"type":"obj"}``.
    """
    flat: list[dict[str, Any]] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        nested: list[Any] = []
        for key in list(element):
            value = element[key]
            if isinstance(value, list) and value and all(_looks_like_element(v) for v in value):
                nested.extend(value)
                del element[key]
        text = element.get("text")
        if isinstance(text, str) and text.strip():
            element["type"] = "text"
        elif element.get("type") not in ("obj", "text"):
            element["type"] = "obj"
        if element["type"] == "obj":
            element.pop("text", None)
        bbox = _clamp_bbox(element.get("bbox"))
        if bbox is None:
            element.pop("bbox", None)
        else:
            element["bbox"] = bbox
        desc = element.get("desc")
        if not isinstance(desc, str):
            element.pop("desc", None)
        # Keep only what carries meaning: an obj needs words, a text its text
        # (a wordless text element may still be filled from the request).
        if element["type"] == "text" or str(element.get("desc") or "").strip():
            flat.append({key: element[key] for key in ("type", "bbox", "text", "desc", "color_palette")
                         if key in element})
        flat.extend(_normalise_elements(nested))
    return flat


def _balance(text: str) -> str:
    """Drop closers that match nothing, close what was left open (string-aware)."""
    out: list[str] = []
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(char)
        elif char in _CLOSERS:
            if not stack or stack[-1] != _CLOSERS[char]:
                continue  # a spare closer: the model's, not the structure's
            stack.pop()
            _strip_trailing_comma(out)
            out.append(char)
            if not stack:
                break  # the object is complete; whatever follows is chatter
            continue
        out.append(char)
    if in_string:
        out.append('"')
    while stack:
        _strip_trailing_comma(out)
        out.append(_OPENERS[stack.pop()])
    return "".join(out)


def _strip_trailing_comma(out: list[str]) -> None:
    index = len(out) - 1
    while index >= 0 and out[index].isspace():
        index -= 1
    if index >= 0 and out[index] == ",":
        del out[index]


_ELEMENT_KEYS = ("type", "bbox", "text", "desc", "color_palette")
# One element field: a string, or a flat list (bbox numbers, palette strings).
# A list holding objects does not match — the fields inside it are read one by one.
_ELEMENT_PAIR = re.compile(
    r'"(type|bbox|text|desc|color_palette)"\s*:\s*("(?:[^"\\]|\\.)*"|\[[^\[\]{}]*\])', re.S)
_ELEMENTS_KEY = re.compile(r'"elements"\s*:')


def _scan_elements(section: str) -> list[dict[str, Any]]:
    """Elements read as a SEQUENCE of fields, whatever nesting the model wrapped them in.

    ⚠️ Why not parse the array: on a long prompt the 4B model nests children
    under a "text" key of their parent, nests a text inside a text, loses a
    bracket level halfway and merges two objects with repeated keys (seen live,
    05.10.2026). The field ORDER survives all of that — the schema writes
    type → bbox → text → desc → color_palette — so a field that goes back in
    that order (or repeats) starts the next element.

    A fragment holding only a desc (the model's afterthought about the element
    before) is added to the nearest object above it.
    """
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for key, raw in _ELEMENT_PAIR.findall(section):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rank = _ELEMENT_KEYS.index(key)
        if current and (key == "type" or rank <= max(_ELEMENT_KEYS.index(k) for k in current)):
            elements.append(current)
            current = {}
        current[key] = value
    if current:
        elements.append(current)

    merged: list[dict[str, Any]] = []
    for element in elements:
        if set(element) == {"desc"} and isinstance(element["desc"], str):
            host = next((e for e in reversed(merged)
                         if e.get("type", "obj") == "obj" and not e.get("text")), None)
            if host is not None:
                host["desc"] = f"{str(host.get('desc') or '').rstrip(' .;')}; {element['desc']}"
                continue
        if not _absorbed_as_repeat(merged, element):
            merged.append(element)
    return merged


def _absorbed_as_repeat(elements: list[dict[str, Any]], element: dict[str, Any]) -> bool:
    """The model repeats a whole element; a copy already there keeps the fuller words.

    Same type, bbox and text, and one desc contains the other — the first copy
    may already carry an afterthought added to it, so they are not byte-equal.
    """
    desc = str(element.get("desc") or "")
    for existing in elements:
        if (existing.get("type"), existing.get("bbox"), existing.get("text")) != (
                element.get("type"), element.get("bbox"), element.get("text")):
            continue
        known = str(existing.get("desc") or "")
        if desc in known:
            return True
        if known in desc:
            existing["desc"] = desc
            return True
    return False


def _loads(text: str) -> Any:
    try:
        return _flatten(json.loads(text, object_pairs_hook=_pairs_hook))
    except json.JSONDecodeError:
        return None


def parse_caption(text: str) -> dict[str, Any] | None:
    """The first JSON object in ``text``, mended if it has to be; None if hopeless."""
    start = str(text or "").find("{")
    if start < 0:
        return None
    candidate = _OPEN_HEX.sub(r'"\1"', text[start:])
    marker = _ELEMENTS_KEY.search(candidate)
    if marker:
        # Everything before the elements (description, style, background) is
        # plain JSON the model rarely breaks; the elements are read field by field.
        head = _loads(_balance(candidate[:marker.start()]))
        if isinstance(head, dict):
            composition = head.get("compositional_deconstruction")
            if not isinstance(composition, dict):
                composition = head["compositional_deconstruction"] = {}
            composition["elements"] = _normalise_elements(
                _scan_elements(candidate[marker.end():]))
            return head
    for attempt in (candidate, _balance(candidate)):
        value = _loads(attempt)
        if isinstance(value, dict):
            composition = value.get("compositional_deconstruction")
            if isinstance(composition, dict) and isinstance(composition.get("elements"), list):
                composition["elements"] = _normalise_elements(composition["elements"])
            return value
    return None


def _quoted_spans(idea: str) -> list[str]:
    """Quoted spans in order, each once — a sign named twice is still one sign."""
    spans = [next(group for group in match if group).strip()
             for match in _QUOTED.findall(idea or "") if any(match)]
    seen: set[str] = set()
    return [span for span in spans if not (span.casefold() in seen or seen.add(span.casefold()))]


def keep_requested_text(caption: dict[str, Any], idea: str) -> dict[str, Any]:
    """Text elements must say what the idea asked for — and only that.

    Only for captions written from an idea alone: with a reference image, text
    read off the picture is legitimate and is left as it is.
    """
    composition = caption.get("compositional_deconstruction")
    elements = composition.get("elements") if isinstance(composition, dict) else None
    if not isinstance(elements, list):
        return caption
    quotes = _quoted_spans(idea)
    folded_idea = str(idea or "").casefold()
    used: set[str] = set()
    kept: list[Any] = []
    blanks: list[dict[str, Any]] = []
    for element in elements:
        if not isinstance(element, dict) or element.get("type") != "text":
            kept.append(element)
            continue
        written = str(element.get("text") or "").strip()
        if not written:
            # A text element with a place and a lettering but no words — the
            # model forgot the `text` field. Filled below from a request no
            # other element took.
            blanks.append(element)
            kept.append(element)
            continue
        # Several requested lines written as one text ("A\nB"): each line is
        # held to its own request, the lines stay together.
        lines = [line.strip() for line in written.split("\n") if line.strip()]
        matches = [_best_quote(line, quotes) for line in lines]
        if lines and all(match is not None for match in matches):
            element["text"] = "\n".join(
                quote.upper() if line.isupper() else quote
                for line, quote in zip(lines, matches))
            used.update(matches)
            kept.append(element)
        elif written.casefold() in folded_idea:
            kept.append(element)
        else:
            logger.info("%s Dropped on-image text the idea did not ask for: %r",
                        LOG_PREFIX, written)
    unclaimed = [quote for quote in quotes if quote not in used]
    for element in blanks:
        if unclaimed:
            element["text"] = unclaimed.pop(0)
        else:
            kept.remove(element)
            logger.info("%s Dropped a text element without words.", LOG_PREFIX)
    for quote in unclaimed:
        if not _lift_text_out_of_desc(kept, quote):
            _place_after_previous_request(kept, quote, quotes)
    composition["elements"] = kept
    return caption


def _place_after_previous_request(elements: list[Any], quote: str, quotes: list[str]) -> None:
    """A requested text the model dropped altogether.

    ⚠️ Seen live: «Ответственные жильцы» — the second line of the tablet, typed
    right after «Снимем избушку недорого» — vanished from the caption. Requests
    written one after another usually sit one under another, so it goes just
    below the previous request's element, half its height; with nothing to
    lean on it is added without a bbox (Ideogram then places it freely) rather
    than lost.
    """
    previous = quotes[:quotes.index(quote)]
    anchor = None
    for earlier in reversed(previous):
        anchor = next((e for e in elements if isinstance(e, dict) and e.get("type") == "text"
                       and earlier.casefold() in str(e.get("text") or "").casefold()
                       and isinstance(e.get("bbox"), list)), None)
        if anchor is not None:
            break
    placed: dict[str, Any] = {"type": "text"}
    if anchor is not None:
        y0, x0, y1, x1 = anchor["bbox"]
        height = max(1, (y1 - y0) // 2)
        top = min(999, y1 + 5)
        placed["bbox"] = [top, x0, min(1000, top + height), x1]
        placed["text"] = quote
        placed["desc"] = str(anchor.get("desc") or "clearly legible lettering") + ", smaller, below the line above"
        elements.insert(elements.index(anchor) + 1, placed)
    else:
        placed["text"] = quote
        placed["desc"] = "clearly legible lettering"
        elements.append(placed)
    logger.info("%s Restored the requested text %r the model left out.", LOG_PREFIX, quote)


def _best_quote(written: str, quotes: list[str]) -> str | None:
    best, ratio = None, 0.0
    for quote in quotes:
        score = difflib.SequenceMatcher(None, written.casefold(), quote.casefold()).ratio()
        if score > ratio:
            best, ratio = quote, score
    return best if ratio >= _MATCH_RATIO else None


def _lift_text_out_of_desc(elements: list[Any], quote: str) -> bool:
    """A requested text the model only mentioned in an object's desc.

    ⚠️ Seen live: «Timesaver» lived only inside the hut's desc ("a wooden
    'Timesaver' sign above the door"). Ideogram draws text it is given as a
    text element far more reliably, so the words get their own element — placed
    on the upper part of that object, lettered with the model's own clause.
    """
    folded = quote.casefold()
    for index, element in enumerate(elements):
        if not isinstance(element, dict) or element.get("type") != "obj":
            continue
        desc = str(element.get("desc") or "")
        if folded not in desc.casefold():
            continue
        clause = next((part.strip() for part in re.split(r"[,;]", desc)
                       if folded in part.casefold()), "")
        lifted: dict[str, Any] = {"type": "text"}
        bbox = element.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            y0, x0, y1, x1 = bbox
            height, width = y1 - y0, x1 - x0
            lifted["bbox"] = [int(y0 + 0.05 * height), int(x0 + 0.2 * width),
                              max(int(y0 + 0.25 * height), int(y0 + 0.05 * height) + 1),
                              max(int(x1 - 0.2 * width), int(x0 + 0.2 * width) + 1)]
        lifted["text"] = quote
        lifted["desc"] = f"clearly legible lettering, {clause}" if clause else "clearly legible lettering"
        elements.insert(index + 1, lifted)
        logger.info("%s Gave the requested text %r its own element.", LOG_PREFIX, quote)
        return True
    return False


def repair_caption_text(text: str, idea: str | None = None) -> str | None:
    """Mended compact caption JSON, or None when nothing usable was written.

    ``idea``: the user's own words when the caption was written from them alone
    (no reference image) — then text elements are held to it.
    """
    caption = parse_caption(text)
    if caption is None:
        return None
    if idea is not None:
        caption = keep_requested_text(caption, idea)
    # Compact separators + literal UTF-8: the serialization Ideogram 4 was
    # trained on (docs/prompting.md).
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
