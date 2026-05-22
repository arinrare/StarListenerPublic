import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ebooklib
import numpy as np
from bs4 import BeautifulSoup
from ebooklib import epub
import soundfile as sf
import torch

warnings.filterwarnings("ignore", message=".*dropout.*num_layers.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")

from kokoro import KPipeline

SAMPLE_RATE = 24000

try:
    from sl_utility import _clean_line_for_parsing, _preprocess_for_notes
except ModuleNotFoundError:
    from .sl_utility import _clean_line_for_parsing, _preprocess_for_notes  # type: ignore


def _env_bitrate() -> str:
    return str(os.environ.get("TTS_BITRATE", "128k")).strip().strip("'\"")


def _env_pronunciations() -> str:
    return str(os.environ.get("TTS_PRONUNCIATIONS", "pronunciations.json")).strip().strip("'\"")


def _env_word_limit() -> Optional[int]:
    raw = str(os.environ.get("TTS_WORD_LIMIT", "")).strip().strip("'\"")
    if raw and raw.isdigit():
        return int(raw)
    return None


def _extract_epub_text(epub_path: str, word_limit: Optional[int] = None) -> Tuple[str, int]:
    book = epub.read_epub(epub_path)

    all_docs = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    docs_by_id = {}
    for it in all_docs:
        try:
            docs_by_id[str(it.get_id())] = it
        except Exception:
            continue

    ordered_items = []
    for ent in getattr(book, "spine", []) or []:
        idref = ent[0] if isinstance(ent, tuple) else ent
        if not idref or str(idref).lower() == "nav":
            continue
        it = docs_by_id.get(str(idref))
        if it is None:
            try:
                it = book.get_item_with_id(idref)
            except Exception:
                continue
        if it is None:
            continue
        try:
            if it.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
        except Exception:
            continue
        ordered_items.append(it)

    if not ordered_items:
        ordered_items = all_docs

    parts = []
    for item in ordered_items:
        content = item.get_content()
        html_text = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        soup = BeautifulSoup(html_text, "html.parser")
        parts.append(soup.get_text("\n"))

    full_text = "\n\n".join(parts)
    words = full_text.split()
    total_words = len(words)

    if word_limit and word_limit > 0 and word_limit < total_words:
        words = words[:word_limit]
        full_text = " ".join(words)

    return full_text, min(total_words, word_limit) if word_limit else total_words


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"[tts] ffmpeg failed: {result.stderr.strip()}\n")
        sys.stderr.flush()
        return False
    return True


def _book_stem(epub_path: Optional[str]) -> str:
    if not epub_path:
        return "tts"
    stem = Path(epub_path).stem.strip() or "tts"
    cleaned = re.sub(r"[^a-zA-Z0-9_\-.,() ]", "_", stem)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("_") or "tts"


def _get_output_dir(epub_path: Optional[str]) -> Path:
    out = _get_project_root() / "output"
    if epub_path:
        name = re.sub(r"\s+", "_", Path(epub_path).name)
        out = out / name
    else:
        out = out / "tts"
    out.mkdir(parents=True, exist_ok=True)
    return out



def _split_natural_paragraphs(full_text: str) -> list:
    text = re.sub(r"\r\n?", "\n", full_text)
    lines = text.split("\n")

    paragraphs = []
    current = []
    blanks = 0

    for line in lines:
        if not line.strip():
            blanks += 1
            if blanks >= 1 and current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        blanks = 0
        current.append(line.strip())

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _load_pronunciations(filename: str) -> dict:
    path = _get_project_root() / "assets" / filename
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_pronunciations(text: str, pron_dict: dict) -> str:
    if not pron_dict:
        return text
    escaped = [re.escape(w) for w in pron_dict.keys()]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b")

    def replacer(match):
        matched = match.group(0)
        for key, phoneme in pron_dict.items():
            if key.lower() == matched.lower():
                return f"[{matched}](/{phoneme}/)"
        return matched

    return pattern.sub(replacer, text)


_PIPELINE_CACHE: dict = {}


def _get_pipeline(lang: str) -> KPipeline:
    lang_code = _lang_to_code(lang)
    if lang_code not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[lang_code] = KPipeline(lang_code=lang_code)
    return _PIPELINE_CACHE[lang_code]


def _lang_to_code(lang: str) -> str:
    if lang.startswith("en-us") or lang.startswith("en"):
        return "a"
    if lang.startswith("en-gb"):
        return "b"
    return "a"


def _generate_tts_from_paragraphs(text: str, lang: str, voice: str, speed: float, pron_dict: dict):
    pipeline = _get_pipeline(lang)
    paras = _split_natural_paragraphs(text)
    total = len(paras)
    all_audio = []
    all_word_ts = []
    all_para_ts = []
    cum_ms = 0.0
    sr = SAMPLE_RATE

    for pi, para in enumerate(paras):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        para = _apply_pronunciations(para, pron_dict)
        sys.stderr.write(json.dumps({"status": "paragraph", "value": f"{pi+1}/{total}"}) + "\n")
        sys.stderr.flush()

        para_samples = 0
        para_word_ts = []
        sub_offset_ms = 0.0

        for result in pipeline(para, voice=voice, speed=speed):
            audio_tensor = result.audio
            sub_samples = 0
            if audio_tensor is not None:
                audio_np = audio_tensor.cpu().numpy()
                sub_samples = len(audio_np)
                para_samples += sub_samples
                all_audio.append(audio_np)
            if hasattr(result, "tokens") and result.tokens:
                for t in result.tokens:
                    para_word_ts.append({
                        "word": t.text + (t.whitespace if t.whitespace else ""),
                        "start_ms": round(t.start_ts * 1000 + sub_offset_ms),
                        "end_ms": round(t.end_ts * 1000 + sub_offset_ms),
                    })
            sub_offset_ms += sub_samples / sr * 1000.0

        para_start_ms = cum_ms
        para_end_ms = para_start_ms + (para_samples / sr * 1000.0)

        if para_word_ts:
            for wt in para_word_ts:
                wt["start_ms"] = round(wt["start_ms"] + para_start_ms)
                wt["end_ms"] = round(wt["end_ms"] + para_start_ms)
            all_word_ts.extend(para_word_ts)
        all_para_ts.append({
            "text": para,
            "start_ms": round(para_start_ms),
            "end_ms": round(para_end_ms),
        })
        cum_ms = para_end_ms

    audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    return audio, sr, all_word_ts, all_para_ts


def _generate_segmented_audio(
    segments: list,
    main_voice: str,
    footnote_voice: str,
    speed: float,
    lang: str,
    pron_dict: dict,
):
    lang_code = _lang_to_code(lang)
    pipeline = _get_pipeline(lang)
    all_audio = []
    all_word_ts = []
    all_para_ts = []
    cum_ms = 0.0
    sr = SAMPLE_RATE
    total = len(segments)
    seg_idx = 0

    for voice_type, text in segments:
        seg_idx += 1
        text = re.sub(r"\s+", " ", text).strip()
        voice = footnote_voice if voice_type == "footnote" else main_voice
        if not text.strip():
            continue

        sys.stderr.write(json.dumps({"status": "paragraph", "value": f"{seg_idx}/{total}"}) + "\n")
        sys.stderr.flush()

        for para in _split_natural_paragraphs(text):
            para = re.sub(r"\s+", " ", para).strip()
            if not para:
                continue
            para = _apply_pronunciations(para, pron_dict)

            para_samples = 0
            para_word_ts = []
            sub_offset_ms = 0.0

            for result in pipeline(para, voice=voice, speed=speed):
                audio_tensor = result.audio
                sub_samples = 0
                if audio_tensor is not None:
                    audio_np = audio_tensor.cpu().numpy()
                    sub_samples = len(audio_np)
                    para_samples += sub_samples
                    all_audio.append(audio_np)
                if hasattr(result, "tokens") and result.tokens:
                    for t in result.tokens:
                        if t.start_ts is not None and t.end_ts is not None:
                            para_word_ts.append({
                                "word": t.text + (t.whitespace if t.whitespace else ""),
                                "start_ms": round(t.start_ts * 1000 + sub_offset_ms),
                                "end_ms": round(t.end_ts * 1000 + sub_offset_ms),
                            })
                sub_offset_ms += sub_samples / sr * 1000.0

            para_start_ms = cum_ms
            para_end_ms = para_start_ms + (para_samples / sr * 1000.0)

            if para_word_ts:
                for wt in para_word_ts:
                    wt["start_ms"] = round(wt["start_ms"] + para_start_ms)
                    wt["end_ms"] = round(wt["end_ms"] + para_start_ms)
                all_word_ts.extend(para_word_ts)
            all_para_ts.append({
                "text": para,
                "start_ms": round(para_start_ms),
                "end_ms": round(para_end_ms),
            })
            cum_ms = para_end_ms

    audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    return audio, sr, all_word_ts, all_para_ts


def _chunk_text(full_text: str) -> list:
    result = []
    for block in re.split(r"(?:\r?\n){2,}", full_text):
        block = block.strip()
        if not block:
            continue
        if len(block.split()) <= 120:
            result.append(block)
            continue
        sub_blocks = re.split(r"\n", block)
        current = []
        for line in sub_blocks:
            line = line.strip()
            if not line:
                continue
            if current and line and line[0].isupper() and len(line) > 40:
                result.append(" ".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            result.append(" ".join(current))

    final = []
    buf = []
    buf_wc = 0
    for p in result:
        wc = len(p.split())
        if wc > 100:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            merged_sent = []
            merged_wc = 0
            for s in sentences:
                swc = len(s.split())
                if merged_wc + swc > 100 and merged_sent:
                    final.append(" ".join(merged_sent))
                    merged_sent = [s]
                    merged_wc = swc
                else:
                    merged_sent.append(s)
                    merged_wc += swc
            if merged_sent:
                final.append(" ".join(merged_sent))
            continue
        if buf and buf_wc + wc <= 80:
            buf.append(p)
            buf_wc += wc
        elif wc < 40 and not buf:
            buf.append(p)
            buf_wc = wc
        else:
            if buf:
                final.append(" ".join(buf))
                buf = []
                buf_wc = 0
            final.append(p)
    if buf:
        final.append(" ".join(buf))
    return final


_NOTES_BLOCK_RE = re.compile(
    r"^\s*(?:FOOTNOTES|FOOTNOTES\s+AND\s+ENDNOTES|ENDNOTES|NOTES)\s*$",
    re.IGNORECASE,
)


def _remove_notes_block(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _NOTES_BLOCK_RE.match(line.strip()):
            return "\n".join(lines[:i]).strip()
    return text


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_anchor_in_raw(raw_text: str, prep_text: str, fn: dict) -> Optional[int]:
    pos = fn.get("position")
    if pos is None or not isinstance(pos, (int, float)) or pos < 0:
        return None

    pos = int(pos)

    # Clean prep_text to match the scanner's coordinate space.
    # The scanner applies _clean_line_for_parsing + rstrip("\r") per line,
    # and positions are character offsets into that cleaned text.
    cleaned_prep_lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in prep_text.split("\n")]
    cleaned_prep = "\n".join(cleaned_prep_lines)

    if pos >= len(cleaned_prep):
        return None

    ctx_start = max(0, pos - 70)
    ctx_end = min(len(cleaned_prep), pos + 10)
    ctx = cleaned_prep[ctx_start:ctx_end]
    ctx_norm = _normalize_ws(ctx)
    if len(ctx_norm) < 15:
        return None

    words = [w.strip("'\"-.,;:!?()[]") for w in ctx_norm.split()]
    words = [w for w in words if len(w) > 1 and all(c.isalpha() or c in "-'" for c in w)]
    if not words:
        return None

    # Use progressively fewer words from the middle of the context
    # (avoid edge words that are affected by preprocessing artifacts)
    mid_start = len(words) // 4
    mid_end = len(words) * 3 // 4
    core_words = words[mid_start:mid_end] if mid_end > mid_start else words
    if len(core_words) < 3:
        core_words = words

    raw_norm = _normalize_ws(raw_text)

    match = None
    for num_words in range(min(len(core_words), 8), 2, -1):
        search_words = core_words[-num_words:]
        pattern = r"\s+".join(re.escape(w) for w in search_words)
        try:
            matches = list(re.finditer(pattern, raw_norm))
        except re.error:
            continue
        if len(matches) == 1:
            match = matches[0]
            break
        elif len(matches) > 1:
            match = matches[-1]
            break

    if match is None:
        match = _find_anchor_by_marker_fallback(raw_text, fn)

    if match is None:
        return None

    raw_pos = 0
    norm_idx = 0
    for ch in raw_text:
        if norm_idx >= match.end():
            break
        if not ch.isspace():
            norm_idx += 1
        elif raw_pos == 0 or (raw_pos > 0 and not raw_text[raw_pos - 1].isspace()):
            norm_idx += 1
        raw_pos += 1

    return raw_pos


_SUPERSCRIPT_DIGITS = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
                      "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
                      "8": "\u2078", "9": "\u2079"}


def _find_anchor_by_marker_fallback(raw_text: str, fn: dict) -> Optional[Any]:
    marker = fn.get("marker")
    if not marker:
        return None

    marker_str = str(marker).strip()
    if not marker_str:
        return None

    raw_norm = _normalize_ws(raw_text)

    superscript_marker = "".join(_SUPERSCRIPT_DIGITS.get(ch, ch) for ch in marker_str)

    patterns = []
    for m in (marker_str, superscript_marker):
        patterns.append(re.escape("(" + m + ")"))
        patterns.append(re.escape("[" + m + "]"))
        patterns.append(re.escape("( " + m + " )"))
        patterns.append(re.escape("( " + m + ")"))
        patterns.append(re.escape("(" + m + " )"))
        patterns.append(re.escape("[ " + m + " ]"))

    best_match = None
    best_count = float("inf")

    for pattern in patterns:
        try:
            found = list(re.finditer(pattern, raw_norm))
        except re.error:
            continue
        if 0 < len(found) < best_count:
            best_count = len(found)
            best_match = found[-1]
        elif len(found) == 1 and best_count == 1:
            best_match = found[0]

    return best_match


def _find_sentence_end(text: str, pos: int) -> int:
    m = re.search(r"[.!?](?:\s|\n|$)", text[pos:])
    if m:
        return pos + m.start()
    nl = text.find("\n", pos)
    if nl >= 0:
        return nl
    return max(0, len(text) - 1)


def _build_voice_segments(
    raw_text: str,
    footnotes: list,
    mode: str,
) -> list:
    if mode == "as_is":
        return [("prose", raw_text)]

    if mode == "skip":
        return [("prose", _remove_notes_block(raw_text))]

    prep_text = _preprocess_for_notes(raw_text)

    anchors = []
    for fn in footnotes:
        pos = _find_anchor_in_raw(raw_text, prep_text, fn)
        if pos is not None:
            anchors.append((pos, fn))

    if not anchors:
        return [("prose", _remove_notes_block(raw_text))]

    anchors.sort(key=lambda x: x[0])

    segments = []
    last_pos = 0

    for pos, fn in anchors:
        sentence_end = _find_sentence_end(raw_text, pos)

        prose = raw_text[last_pos:sentence_end + 1]
        if prose.strip():
            segments.append(("prose", prose))

        marker = fn.get("marker", "")
        definition = fn.get("suggested_definition", "")
        if definition and str(definition).strip():
            segments.append(("footnote", f" Footnote {marker}: {definition}. End of footnote. "))

        last_pos = sentence_end + 1

    remaining = _remove_notes_block(raw_text[last_pos:])
    if remaining.strip():
        segments.append(("prose", remaining))

    return segments


def _extract_epub_items(epub_path: str) -> list:
    book = epub.read_epub(epub_path)

    all_docs = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    docs_by_id: Dict[str, Any] = {}
    for it in all_docs:
        try:
            docs_by_id[str(it.get_id())] = it
        except Exception:
            continue

    ordered_items = []
    for ent in getattr(book, "spine", []) or []:
        idref = ent[0] if isinstance(ent, tuple) else ent
        if not idref or str(idref).lower() == "nav":
            continue
        it = docs_by_id.get(str(idref))
        if it is None:
            try:
                it = book.get_item_with_id(idref)
            except Exception:
                continue
        if it is None:
            continue
        try:
            if it.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
        except Exception:
            continue
        ordered_items.append(it)

    if not ordered_items:
        ordered_items = all_docs

    result = []
    for chapter_index, item in enumerate(ordered_items):
        content = item.get_content()
        html_text = (
            content.decode("utf-8", errors="ignore")
            if isinstance(content, bytes)
            else str(content)
        )
        soup = BeautifulSoup(html_text, "html.parser")
        result.append((chapter_index, soup.get_text("\n")))

    return result


def generate_tts(
    text: Optional[str] = None,
    *,
    voice: str = "bf_emma",
    speed: float = 1.0,
    lang: str = "en-gb",
    output_path: Optional[str] = None,
    voice_segments: Optional[list] = None,
    footnote_voice: str = "bm_george",
    epub_path: Optional[str] = None,
    full_text: Optional[str] = None,
) -> dict:
    sys.stderr.write(json.dumps({"status": "provider", "value": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")}) + "\n")
    sys.stderr.flush()

    pron_dict = _load_pronunciations(_env_pronunciations())

    if voice_segments:
        sys.stderr.write(json.dumps({"status": "debug", "path": "segmented", "count": len(voice_segments)}) + "\n")
        sys.stderr.flush()
        samples, sample_rate, word_ts, para_ts = _generate_segmented_audio(
            voice_segments, voice, footnote_voice, speed, lang, pron_dict
        )
        display_text = full_text or " ".join(st for _, st in voice_segments)
    elif text:
        sys.stderr.write(json.dumps({"status": "debug", "path": "paragraphs", "len": len(text)}) + "\n")
        sys.stderr.flush()
        display_text = full_text or text
        samples, sample_rate, word_ts, para_ts = _generate_tts_from_paragraphs(
            display_text, lang, voice, speed, pron_dict
        )
    else:
        return {"error": "No text or voice segments provided"}

    if output_path:
        out_dir = _get_project_root() / output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(output_path).stem or "book"
    else:
        out_dir = _get_output_dir(epub_path)

        stem = _book_stem(epub_path) if epub_path else "tts"

    wav_file = str(out_dir / f"{stem}.wav")
    sf.write(wav_file, samples, sample_rate)

    mp3_file = str(out_dir / f"{stem}.mp3")
    bitrate = _env_bitrate()
    if _wav_to_mp3(wav_file, mp3_file, bitrate):
        if os.path.exists(wav_file):
            os.remove(wav_file)
        out_file = mp3_file
    else:
        out_file = wav_file

    ts_file = str(out_dir / f"{stem}_timestamps.json")
    with open(ts_file, "w", encoding="utf-8") as f:
        json.dump({"timestamps": word_ts, "paragraphs": para_ts, "full_text": display_text, "book_stem": stem}, f, ensure_ascii=False)

    duration_s = round(len(samples) / sample_rate, 2)
    return {
        "book_stem": stem,
        "output_path": str(Path(out_file).resolve()),
        "timestamps_path": str(Path(ts_file).resolve()),
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "word_timestamps": word_ts,
        "paragraph_timestamps": para_ts,
        "full_text": display_text,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if len(argv) < 1:
        sys.stdout.write(json.dumps({"error": "No JSON payload provided"}))
        sys.stdout.flush()
        return 0

    raw_arg = argv[0]

    if raw_arg == "--file" and len(argv) > 1:
        with open(argv[1], "r", encoding="utf-8") as _f:
            raw_arg = _f.read()

    text: Optional[str] = None
    voice = "bf_emma"
    speed = 1.0
    lang = "en-gb"
    output_filename: Optional[str] = None
    epub_path: Optional[str] = None
    output_path_override: Optional[str] = None
    word_count: Optional[int] = None
    footnotes: Optional[list] = None
    footnote_mode = "as_is"
    footnote_voice_name = "bm_george"
    voice_segments: Optional[list] = None

    if raw_arg.strip().startswith("{"):
        try:
            payload = json.loads(raw_arg)
            if isinstance(payload, dict):
                text = payload.get("text")
                voice = str(payload.get("voice", voice))
                speed = float(payload.get("speed", speed))
                lang = str(payload.get("lang", lang))
                output_filename = payload.get("output_filename") or payload.get("output")
                epub_path = payload.get("epub_path")
                output_path_override = payload.get("output_path")
                footnotes = payload.get("footnotes")
                footnote_mode = str(payload.get("footnote_mode", footnote_mode))
                footnote_voice_name = str(payload.get("footnote_voice", footnote_voice_name))
        except Exception:
            pass

    if epub_path and not text:
        word_limit = _env_word_limit()

        if footnotes and isinstance(footnotes, list) and footnote_mode in ("inline", "skip"):
            items = _extract_epub_items(epub_path)
            all_segments: list = []
            total_words = 0

            for chapter_index, raw_text in items:
                item_fns = [fn for fn in footnotes if fn.get("chapter_index") == chapter_index]
                if item_fns:
                    segs = _build_voice_segments(raw_text, item_fns, footnote_mode)
                elif footnote_mode == "skip":
                    segs = [("prose", _remove_notes_block(raw_text))]
                else:
                    segs = [("prose", raw_text)]

                for vt, st in segs:
                    wc = len(st.split())
                    if word_limit and total_words + wc > word_limit:
                        if vt == "footnote":
                            all_segments.append((vt, st))
                            continue
                        remaining = word_limit - total_words
                        if remaining > 0:
                            truncated = " ".join(st.split()[:remaining])
                            all_segments.append((vt, truncated))
                        total_words = word_limit
                        break
                    all_segments.append((vt, st))
                    total_words += wc

                if word_limit and total_words >= word_limit:
                    break

            voice_segments = all_segments
            word_count = total_words
        else:
            extracted, wc = _extract_epub_text(epub_path, word_limit)
            if not extracted.strip():
                sys.stdout.write(json.dumps({"error": "No text extracted from EPUB"}))
                sys.stdout.flush()
                return 0
            text = extracted
            word_count = wc

    if not text and not voice_segments:
        text = (
            "This is a test of the text-to-speech system."
        )

    result = generate_tts(
        text=text,
        voice=voice,
        speed=speed,
        lang=lang,
        output_path=output_path_override,
        voice_segments=voice_segments,
        footnote_voice=footnote_voice_name,
        epub_path=epub_path,
        full_text=(" ".join(st for _, st in voice_segments) if voice_segments else None),
    )

    if word_count is not None:
        result["word_count"] = word_count

    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
