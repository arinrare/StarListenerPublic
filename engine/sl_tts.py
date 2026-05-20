import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ebooklib
import numpy as np
from bs4 import BeautifulSoup
from ebooklib import epub
import soundfile as sf

_venv_root = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"
_capi = str(_venv_root / "onnxruntime" / "capi")
if os.path.isdir(_capi):
    os.environ["PATH"] = _capi + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("ORT_LOG_LEVEL", "4")
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as _rt
_rt.set_default_logger_severity(4)
_rt.preload_dlls()
if "CUDAExecutionProvider" in _rt.get_available_providers():
    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")


from kokoro_onnx import Kokoro

SAMPLE_RATE = 24000

try:
    from sl_utility import _preprocess_for_notes
except ModuleNotFoundError:
    from .sl_utility import _preprocess_for_notes  # type: ignore


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_model_path(model_name: str) -> str:
    return str(_get_project_root() / "assets" / model_name)


def _env_model() -> str:
    return str(os.environ.get("TTS_MODEL", "kokoro-v1.0.fp16-gpu.onnx")).strip().strip("'\"")


def _env_voices() -> str:
    return str(os.environ.get("TTS_VOICES", "voices-v1.0.bin")).strip().strip("'\"")


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


def _load_pronunciations(filename: str) -> dict:
    path = _get_project_root() / "assets" / filename
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_custom_phonemes(text: str, pron_dict: dict, lang: str, tokenizer) -> str:
    escaped = [re.escape(w) for w in pron_dict.keys()]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    parts = []
    last_end = 0

    for match in pattern.finditer(text):
        before = text[last_end:match.start()]
        if before:
            parts.append(tokenizer.phonemize(before, lang))

        matched = match.group(0)
        for key, phoneme in pron_dict.items():
            if key.lower() == matched.lower():
                parts.append(phoneme)
                break

        last_end = match.end()

    if last_end < len(text):
        parts.append(tokenizer.phonemize(text[last_end:], lang))

    return " ".join(parts)


# kokoro_onnx has an off-by-one bug: after truncating phonemes to MAX_PHONEME_LENGTH (510),
# it tries voice[len(tokens)] which crashes when len(tokens) == 510.
# We chunk to 509 to stay safely within bounds.
_PHONEME_CHUNK_MAX = 509


def _chunk_phonemes(phonemes: str, max_len: int = _PHONEME_CHUNK_MAX) -> list:
    if len(phonemes) <= max_len:
        return [phonemes]
    parts = phonemes.split(" ")
    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) + (1 if current else 0) > max_len:
            if current:
                chunks.append(current)
            while len(part) > max_len:
                chunks.append(part[:max_len])
                part = part[max_len:]
            current = part
        else:
            current = (current + " " + part) if current else part
    if current:
        chunks.append(current)
    return chunks


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



def _distribute_paragraph_timings(para_text: str, para_start_ms: float, para_end_ms: float) -> list:
    words = para_text.split()
    word_chars = [len(w) for w in words]
    total_word_chars = sum(word_chars)
    if total_word_chars == 0:
        return []

    para_duration = para_end_ms - para_start_ms
    ms_per_word_char = para_duration / total_word_chars

    word_ts = []
    cursor_ms = para_start_ms
    for w in words:
        dur = len(w) * ms_per_word_char
        word_ts.append({
            "word": w + " ",
            "start_ms": round(cursor_ms),
            "end_ms": round(cursor_ms + dur),
        })
        cursor_ms += dur

    return word_ts


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


def _group_word_ts_to_blocks(word_ts: list) -> list:
    CAP = 50
    MAX_SENTENCE = 80
    blocks = []
    block_start = 0
    sentence_start = 0
    block_wc = 0

    for i, wt in enumerate(word_ts):
        is_end = wt["word"].rstrip().endswith(('.', '!', '?'))
        if not is_end and i < len(word_ts) - 1:
            continue

        next_start = i + 1
        sent_len = next_start - sentence_start

        if sent_len > MAX_SENTENCE:
            words = [word_ts[j]["word"] for j in range(sentence_start, next_start)]
            mid = len(words) // 2
            for offset in range(mid, 0, -1):
                if words[offset - 1].rstrip().endswith((',', ';', ':', '.')):
                    mid = offset
                    break
            split = sentence_start + mid
            _flush_block(word_ts, block_start, split, blocks)
            block_start = sentence_start = split
            block_wc = next_start - split
        else:
            block_wc += sent_len
            sentence_start = next_start

        if block_wc > CAP:
            _flush_block(word_ts, block_start, next_start, blocks)
            block_start = next_start
            block_wc = 0

    if block_start < len(word_ts):
        _flush_block(word_ts, block_start, len(word_ts), blocks)
    return blocks


def _flush_block(word_ts: list, start: int, end: int, blocks: list) -> None:
    if start < end:
        text = "".join(wt["word"] for wt in word_ts[start:end]).strip()
        if text:
            blocks.append({
                "text": text,
                "start_ms": word_ts[start]["start_ms"],
                "end_ms": word_ts[end - 1]["end_ms"],
            })


def _flush_passage(words: list, start: int, end: int, passages: list) -> None:
    if start < end:
        text = " ".join(words[start:end]).strip()
        if text:
            passages.append(text)


def _split_text_into_passages(text: str, cap: int = 50, max_sentence: int = 80) -> list:
    paragraphs = _split_natural_paragraphs(text)
    passages = []

    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        words = para.split()
        if not words:
            continue

        block_start = 0
        sentence_start = 0
        block_wc = 0

        for i, word in enumerate(words):
            is_end = word.rstrip().endswith(('.', '!', '?'))
            if not is_end and i < len(words) - 1:
                continue

            next_start = i + 1
            sent_len = next_start - sentence_start

            if sent_len > max_sentence:
                mid = sent_len // 2
                split_pos = sentence_start + mid
                for offset in range(mid, 0, -1):
                    if words[sentence_start + offset - 1].rstrip().endswith((',', ';', ':', '.')):
                        split_pos = sentence_start + offset
                        break
                _flush_passage(words, block_start, split_pos, passages)
                block_start = sentence_start = split_pos
                block_wc = next_start - split_pos
            else:
                block_wc += sent_len
                sentence_start = next_start

            if block_wc > cap:
                _flush_passage(words, block_start, next_start, passages)
                block_start = next_start
                block_wc = 0

        if block_start < len(words):
            _flush_passage(words, block_start, len(words), passages)

    return passages


def _generate_tts_from_paragraphs(text: str, lang: str, kokoro: Kokoro, voice: str, speed: float, pron_dict: dict):
    paras = _split_natural_paragraphs(text)
    total = len(paras)
    all_audio = []
    all_para_ts = []
    cum_samples = 0
    sr = SAMPLE_RATE

    for pi, para in enumerate(paras):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        sys.stderr.write(json.dumps({"status": "paragraph", "value": f"{pi+1}/{total}"}) + "\n")
        sys.stderr.flush()

        if pron_dict:
            phonemes = _apply_custom_phonemes(para, pron_dict, lang, kokoro.tokenizer)
            chunks = _chunk_phonemes(phonemes)
            chunk_samples = []
            for chunk in chunks:
                s, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang, is_phonemes=True, trim=False)
                chunk_samples.append(s.flatten() if s.ndim > 1 else s)
            flat = np.concatenate(chunk_samples) if chunk_samples else np.array([], dtype=np.float32)
        else:
            samples, sr = kokoro.create(para, voice=voice, speed=speed, lang=lang, trim=False)
            flat = samples.flatten() if samples.ndim > 1 else samples

        para_duration_ms = len(flat) / sr * 1000.0
        para_start_ms = cum_samples / sr * 1000.0
        para_end_ms = para_start_ms + para_duration_ms

        all_audio.append(flat)
        all_para_ts.append({
            "text": para,
            "start_ms": round(para_start_ms),
            "end_ms": round(para_end_ms),
        })
        cum_samples += len(flat)

    audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    return audio, sr, [], all_para_ts


def _generate_segmented_audio(
    segments: list,
    main_voice: str,
    footnote_voice: str,
    kokoro: Kokoro,
    speed: float,
    lang: str,
    pron_dict: dict,
):
    all_audio = []
    all_para_ts = []
    cum_samples = 0
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

            if pron_dict:
                phonemes = _apply_custom_phonemes(para, pron_dict, lang, kokoro.tokenizer)
                chunks = _chunk_phonemes(phonemes)
                chunk_samples = []
                for chunk in chunks:
                    s, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang, is_phonemes=True, trim=False)
                    chunk_samples.append(s.flatten() if s.ndim > 1 else s)
                flat = np.concatenate(chunk_samples) if chunk_samples else np.array([], dtype=np.float32)
            else:
                samples, sr = kokoro.create(para, voice=voice, speed=speed, lang=lang, trim=False)
                flat = samples.flatten() if samples.ndim > 1 else samples

            para_duration_ms = len(flat) / sr * 1000.0
            para_start_ms = cum_samples / sr * 1000.0
            para_end_ms = para_start_ms + para_duration_ms

            all_audio.append(flat)
            all_para_ts.append({
                "text": para,
                "start_ms": round(para_start_ms),
                "end_ms": round(para_end_ms),
            })
            cum_samples += len(flat)

    audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    return audio, sr, [], all_para_ts


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
    ctx_start = max(0, pos - 70)
    ctx_end = min(len(prep_text), pos + 10)
    ctx = prep_text[ctx_start:ctx_end]
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
    else:
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
    model_path: Optional[str] = None,
    voices_path: Optional[str] = None,
    output_path: Optional[str] = None,
    voice_segments: Optional[list] = None,
    footnote_voice: str = "bm_george",
    epub_path: Optional[str] = None,
    full_text: Optional[str] = None,
) -> dict:
    model = model_path or _get_model_path(_env_model())
    voices = voices_path or _get_model_path(_env_voices())

    if not os.path.isfile(model):
        return {"error": f"Model file not found: {model}"}
    if not os.path.isfile(voices):
        return {"error": f"Voices file not found: {voices}"}

    kokoro = Kokoro(model, voices)
    sys.stderr.write(json.dumps({"status": "provider", "value": str(kokoro.sess.get_providers()[0])}) + "\n")
    sys.stderr.flush()
    pron_dict = _load_pronunciations(_env_pronunciations())

    if voice_segments:
        sys.stderr.write(json.dumps({"status": "debug", "path": "segmented", "count": len(voice_segments)}) + "\n")
        sys.stderr.flush()
        samples, sample_rate, word_ts, para_ts = _generate_segmented_audio(
            voice_segments, voice, footnote_voice, kokoro, speed, lang, pron_dict
        )
        display_text = full_text or " ".join(st for _, st in voice_segments)
    elif text:
        sys.stderr.write(json.dumps({"status": "debug", "path": "paragraphs", "len": len(text)}) + "\n")
        sys.stderr.flush()
        display_text = full_text or text
        samples, sample_rate, word_ts, para_ts = _generate_tts_from_paragraphs(
            display_text, lang, kokoro, voice, speed, pron_dict
        )
    else:
        return {"error": "No text or voice segments provided"}

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
