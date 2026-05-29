import re
from typing import Optional, List, Tuple, Any

from bs4 import BeautifulSoup

try:
    from sl_utility import _safe_text, _def_line_regex
except ModuleNotFoundError:  # pragma: no cover
    from .sl_utility import _safe_text, _def_line_regex  # type: ignore


_TRAILING_FOOTNOTE_MARK_RE = re.compile(
    r"\s*(?:\(\s*\d{1,3}\s*\)|\[\s*\d{1,3}\s*\]|[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}|[\*†‡§]+)\s*$",
    re.UNICODE,
)


def _strip_trailing_footnote_marker_from_heading(label: Optional[str]) -> Optional[str]:
    """Remove trailing footnote markers from headings.

    Example: 'PART TWO.(1)' -> 'PART TWO.'
    Kept intentionally conservative: only strips a *trailing* marker token.
    """

    t = _safe_text(label or "")
    if not t:
        return None

    # Remove one or more trailing marker tokens.
    prev = None
    while prev != t:
        prev = t
        t = _TRAILING_FOOTNOTE_MARK_RE.sub("", t).rstrip()

    return t or None


_NOTES_HEADER_LINE_RE = re.compile(
    r"^\s*(?:FOOTNOTES|FOOTNOTES\s+AND\s+ENDNOTES|ENDNOTES|NOTES)\s*$",
    re.IGNORECASE,
)


def _looks_like_abbrev_run(line: Optional[str]) -> bool:
    """Return True for lines that look like speaker-initial abbreviation runs.

    This guard must be *very* strict to avoid suppressing real chapter headings.

    We only match lines that are predominantly uppercase abbreviation tokens that
    end with a period, e.g.:
      "MGR. NC. PF. AAL. RD. WTJ. RS. JJ. JJR."
      "RD. PF. RS. MGR. NG."
    """

    t = _safe_text(line or "")
    if not t:
        return False

    raw_tokens = [x for x in re.split(r"\s+", t.strip()) if x]
    if len(raw_tokens) < 4:
        return False

    # Keep only tokens that *look* like initials/abbreviations.
    dot_tokens: List[str] = []
    for tok in raw_tokens:
        tok2 = re.sub(r"[^A-Za-z\.]", "", tok).strip()
        if not tok2:
            continue
        if not tok2.endswith("."):
            continue
        letters = re.sub(r"[^A-Za-z]", "", tok2)
        if not letters:
            continue
        if letters != letters.upper():
            continue
        # Typical initials are 1-4 letters (JJR., WTJ., MGR.).
        if not (1 <= len(letters) <= 4):
            continue
        dot_tokens.append(letters)

    if len(dot_tokens) < 4:
        return False

    # The line should be mostly these dot-tokens.
    if (len(dot_tokens) / max(1, len(raw_tokens))) < 0.80:
        return False

    # Avoid pathological matches.
    if sum(len(x) for x in dot_tokens) > 32:
        return False

    return True


def _is_notes_header_line(line: Optional[str]) -> bool:
    t = _safe_text(line or "")
    if not t:
        return False
    if len(t) > 120:
        return False
    return bool(_NOTES_HEADER_LINE_RE.match(t))


def _line_looks_like_heading_component(line: Optional[str]) -> bool:
    """Return True if a single line looks like it could be part of a multi-line heading.

    This is intentionally more permissive than `_looks_like_chapter_heading_text` so we can
    capture headings like:
      Leaves from
      THE TOKYO PAPERS.
      FOREWORD.
    """

    t = _safe_text(line or "")
    if not t:
        return False
    t = _strip_trailing_footnote_marker_from_heading(t) or t

    # Reject speaker-initial abbreviation lines.
    if _looks_like_abbrev_run(t):
        return False

    # Don't treat NOTES headers as headings.
    if _is_notes_header_line(t):
        return False

    if len(t) < 3 or len(t) > 90:
        return False

    # Reject obvious prose lines.
    if t.count(",") >= 2:
        return False

    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < 3:
        return False

    upper = sum(1 for ch in letters if ch.isupper())
    if (upper / max(1, len(letters))) >= 0.85:
        return True

    # Title-ish line: allow connector words to be lowercase.
    connector = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "to",
        "the",
        "upon",
        "with",
        "without",
    }

    words = [w for w in re.split(r"\s+", t.strip()) if w]
    alpha_words = []
    for w in words:
        w2 = re.sub(r"[^A-Za-z]", "", w)
        if w2:
            alpha_words.append(w2)

    if len(alpha_words) < 1:
        return False

    # Require at least one capital-starting word.
    if not any(w[0].isupper() for w in alpha_words):
        return False

    # All non-connector words must start with a capital.
    for w in alpha_words:
        wl = w.lower()
        if wl in connector:
            continue
        if not w[0].isupper():
            return False

    return True


def _infer_multiline_heading_from_lines(lines: List[str], start_index: int) -> Optional[Tuple[int, str]]:
    """Infer a heading by taking a short run of heading-like lines starting at start_index.

    Returns (line_index_of_heading_start, label) or None.
    """

    if not lines:
        return None
    i = max(0, int(start_index or 0))
    if i >= len(lines):
        return None

    parts: List[Tuple[int, str]] = []
    scanned = 0
    li = i
    while li < len(lines) and scanned < 80 and len(parts) < 6:
        scanned += 1
        raw = lines[li]
        t = _safe_text(raw or "")
        if not t:
            # Allow leading blanks before the heading.
            if not parts:
                li += 1
                continue

            # Also allow blank line(s) *between* heading components (common in
            # title-card layouts), but only if another heading-like line
            # appears shortly afterwards.
            try:
                look = li + 1
                while look < len(lines) and look <= li + 3 and not _safe_text(lines[look] or ""):
                    look += 1
                if look < len(lines):
                    cand = _safe_text(lines[look] or "")
                    cand2 = _strip_trailing_footnote_marker_from_heading(cand) or cand
                    if cand2 and not _is_notes_header_line(cand2) and _line_looks_like_heading_component(cand2):
                        li += 1
                        continue
            except Exception:
                pass

            break

        t2 = _strip_trailing_footnote_marker_from_heading(t) or t
        if _is_notes_header_line(t2):
            break

        if _line_looks_like_heading_component(t2):
            parts.append((li, t2))
            li += 1
            continue

        # Stop once we have started collecting parts.
        if parts:
            break
        li += 1

    if not parts:
        return None

    # De-dup while preserving order.
    seen = set()
    uniq: List[Tuple[int, str]] = []
    for idx, p in parts:
        key = _safe_text(p).upper()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append((idx, p))

    if not uniq:
        return None

    label = " ".join([p for _, p in uniq]).strip()
    label = _safe_text(label)
    if not label or len(label) < 4:
        return None

    # If this looks like a PART/BOOK heading but a strong CHAPTER/roman heading follows
    # shortly afterwards, prefer the strong heading (avoid polluting chapter grouping).
    try:
        head0 = _safe_text(uniq[0][1] if uniq else "")
        if re.match(r"^\s*\[?\s*PART\b", head0, re.IGNORECASE) or re.match(r"^\s*\[?\s*BOOK\b", head0, re.IGNORECASE):
            chapter_line_re = re.compile(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", re.IGNORECASE)
            roman_line_re = re.compile(r"^\s*[IVXLC]{1,12}\.?(?:\s+)?$")
            arabic_line_re = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")

            def _next_nonempty(idx: int) -> Optional[int]:
                for j in range(idx, min(idx + 40, len(lines))):
                    if _safe_text(lines[j] or ""):
                        return j
                return None

            start_scan = int(uniq[-1][0]) + 1
            for j in range(start_scan, min(start_scan + 40, len(lines))):
                t = _safe_text(lines[j] or "")
                if not t:
                    continue
                if _is_notes_header_line(t):
                    break
                if chapter_line_re.match(t):
                    return None
                if roman_line_re.match(t) or arabic_line_re.match(t):
                    k = _next_nonempty(j + 1)
                    if k is not None and _line_looks_like_heading_component(lines[k]):
                        return None
    except Exception:
        pass
    return (uniq[0][0], label)


def _find_notes_block_end_from_header(lines: List[str], header_idx: int) -> Optional[int]:
    """Given a NOTES/ENDNOTES header line index, find where that notes block ends.

    Returns an index into `lines` that likely starts the next section/prose.
    """

    if not lines:
        return None
    hi = int(header_idx or 0)
    if hi < 0 or hi >= len(lines):
        return None

    def_re = _def_line_regex()
    roman_re = re.compile(r"^\s*[IVXLC]{1,8}\.?(?:\s+)?$")
    arabic_re = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")
    chapter_re = re.compile(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", re.IGNORECASE)
    # Combined headings like "IX. CHAPTER NAME." (numeral + title on one line).
    roman_title_re = re.compile(r"^\s*([IVXLC]{1,12})\.\s+(.+?)\s*$")
    arabic_title_re = re.compile(r"^\s*(\d{1,3})\.\s+(.+?)\s*$")

    def _looks_like_allcaps_heading(s: str) -> bool:
        t = _safe_text(s or "").strip()
        if not t:
            return False
        t = _strip_trailing_footnote_marker_from_heading(t) or t
        if _is_notes_header_line(t):
            return False
        u = t.upper()
        # Avoid treating "NOTES TO CHAPTER ..." as an end marker.
        if "NOTE" in u:
            return False
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 6:
            return False
        upper_count = sum(1 for ch in letters if ch.isupper())
        return (upper_count / max(1, len(letters))) >= 0.90

    saw_defs = False
    def_count = 0
    in_def = False

    for j in range(hi + 1, len(lines)):
        line = lines[j] or ""
        stripped = line.strip()

        if not stripped:
            in_def = False
            continue

        # Another NOTES header: treat as a boundary.
        if _is_notes_header_line(stripped):
            return j

        # Chapter/section heading boundary must be checked *before* def_re.
        # Some valid chapter headings (e.g. "V. THE FIELD OF ...") also match
        # the generic definition-line regex (single-letter marker), and we do
        # not want to accidentally treat them as note definitions.
        if saw_defs and def_count >= 2:
            if chapter_re.match(stripped):
                return j

            # Standalone chapter numerals can appear as their own line.
            # Only treat them as a boundary if a heading-like line follows soon.
            if roman_re.match(stripped) or arabic_re.match(stripped):
                try:
                    look = j + 1
                    while look < len(lines) and look <= j + 6 and not _safe_text(lines[look] or ""):
                        look += 1
                    if look < len(lines):
                        cand = _safe_text(lines[look] or "")
                        cand2 = _strip_trailing_footnote_marker_from_heading(cand) or cand
                        if cand2 and not _is_notes_header_line(cand2) and _line_looks_like_heading_component(cand2):
                            return j
                except Exception:
                    pass

            # Numeral + title on one line.
            # IMPORTANT: require the *title* to be a strong heading, otherwise we
            # will incorrectly treat manuscript-variant markers like "C. The passage ..."
            # (C is a Roman numeral) as a chapter boundary.
            m_rt = roman_title_re.match(stripped)
            if m_rt:
                title = _safe_text(m_rt.group(2) or "")
                if title and _looks_like_allcaps_heading(title):
                    return j
            m_at = arabic_title_re.match(stripped)
            if m_at:
                title = _safe_text(m_at.group(2) or "")
                if title and _looks_like_allcaps_heading(title):
                    return j

            # Strong ALL-CAPS heading line (FOREWORD/INTRODUCTION/TITLE CARD etc.).
            if _looks_like_allcaps_heading(stripped):
                return j

        if def_re.match(stripped):
            saw_defs = True
            def_count += 1
            in_def = True
            continue

        # Keep consuming continuation lines inside a definition.
        if in_def:
            continue

    return len(lines)


def _infer_logical_chapter_label(lines: List[str]) -> Optional[str]:
    """Infer a stable human-readable chapter label from text.

    Many EPUBs split a single real chapter across multiple HTML files. We try to
    detect headings like:
      II.
      CHAPTERNAME.

    Returns a label like "II. CHAPTERNAME.".
    """

    if not lines:
        return None

    # Optional fallback: multi-line title-card headings (e.g., PART/FOREWORD blocks).
    # IMPORTANT: do not let this override real numeral/CHAPTER headings; we only
    # return it if the numeric scan below finds nothing.
    ml_fallback: Optional[str] = None
    try:
        chapter_line_re0 = re.compile(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", re.IGNORECASE)
        roman_line_re0 = re.compile(r"^\s*[IVXLC]{1,12}\.?(?:\s+)?$")
        arabic_line_re0 = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")
        window0 = lines[: min(len(lines), 220)]

        def _next_nonempty(idx: int) -> Optional[int]:
            for j in range(idx, min(idx + 40, len(window0))):
                if _safe_text(window0[j] or ""):
                    return j
            return None

        strong_early = False
        for i, ln in enumerate(window0[:120]):
            t = _safe_text(ln or "")
            if not t:
                continue
            if chapter_line_re0.match(t):
                strong_early = True
                break
            if roman_line_re0.match(t) or arabic_line_re0.match(t):
                j = _next_nonempty(i + 1)
                if j is not None and _line_looks_like_heading_component(window0[j]):
                    strong_early = True
                    break

        if not strong_early:
            ml = _infer_multiline_heading_from_lines(lines, 0)
            if ml is not None:
                _, label = ml
                if label:
                    ml_fallback = label
    except Exception:
        ml_fallback = None

    def _is_probable_heading(s: str) -> bool:
        t = _safe_text(s)
        if not t:
            return False
        # Reject glossary/notes-like headings.
        if ":" in t or ";" in t or "$" in t:
            return False
        if re.search(r"\bsee\b", t, re.IGNORECASE):
            return False
        t_upper = t.upper().strip(".")
        # Avoid notes/frontmatter headers.
        if t_upper in {"NOTES", "FOOTNOTES", "ENDNOTES", "CONTENTS"}:
            return False

        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 6:
            return False
        upper = sum(1 for ch in letters if ch.isupper())
        ratio = upper / max(1, len(letters))
        if ratio < 0.85:
            return False
        if len(t) > 120:
            return False
        return True

    def _is_title_case_heading(s: str) -> bool:
        t = _safe_text(s)
        if not t:
            return False
        # Reject glossary/notes-like headings.
        if ":" in t or ";" in t or "$" in t:
            return False
        if re.search(r"\bsee\b", t, re.IGNORECASE):
            return False
        # Avoid common non-chapter headers.
        t_upper = t.upper().strip(".")
        if t_upper in {"NOTES", "FOOTNOTES", "ENDNOTES", "CONTENTS"}:
            return False
        # Many extracts include a trailing period even for headings.
        had_trailing_period = t.endswith(".")
        if had_trailing_period and t[:-1].count(".") >= 1:
            return False
        if had_trailing_period:
            t = t.rstrip(".").rstrip()

        if len(t) < 8 or len(t) > 80:
            return False
        words = [w for w in re.split(r"\s+", t) if w]
        if len(words) < 2:
            return False

        connector = {
            "a",
            "an",
            "and",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "into",
            "of",
            "on",
            "to",
            "the",
            "upon",
            "with",
            "without",
        }

        # Require most words to start with a capital (Title Case-ish).
        starts_caps = 0
        alpha_words = 0
        for w in words:
            w2 = re.sub(r"[^A-Za-z]", "", w)
            if not w2:
                continue
            wl = w2.lower()
            if wl in connector:
                continue
            alpha_words += 1
            if w2[0].isupper():
                starts_caps += 1
        if alpha_words < 2:
            return False
        if (starts_caps / max(1, alpha_words)) < 0.7:
            return False
        return True

    roman_line_re = re.compile(r"^\s*[IVXLC]{1,8}\.?(?:\s+)?$")
    arabic_line_re = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")
    paren_roman_line_re = re.compile(r"^\s*\(\s*([IVXLC]{1,8})\s*\)\s*$")
    paren_arabic_line_re = re.compile(r"^\s*\(\s*(\d{1,3})\s*\)\s*$")
    inline_re = re.compile(r"^\s*([IVXLC]{1,8}|\d{1,3})\.?(?:\s+)(.+?)\s*$")
    tight_inline_re = re.compile(r"^\s*([IVXLC]{1,8}|\d{1,3})\.?([A-Z][A-Za-z].{5,120})\s*$")
    chapter_line_re = re.compile(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\s*[:\.]?\s*$", re.IGNORECASE)
    note_heading_re = re.compile(r"^\s*NOTE\s+ON\s+THE\s+CHRONOLOGY\.?\s*$", re.IGNORECASE)

    # Scan the document for a number + heading pair.
    # Some extractors hard-wrap text into many short lines, pushing the real
    # chapter numeral/title beyond the very beginning of the file.
    # Keep a cap to avoid pathological O(n) work on extremely large inputs.
    window = lines[: min(len(lines), 20000)]

    def _allow_arabic_heading(line_idx: int) -> bool:
        return int(line_idx) <= 120

    def _is_roman_token(token: str) -> bool:
        probe = _safe_text(token or "").strip().rstrip(".")
        return bool(re.fullmatch(r"[IVXLC]{1,12}", probe, re.IGNORECASE))

    for i, line in enumerate(window):
        t = _safe_text(line)
        if not t:
            continue

        # Some headings carry trailing footnote markers in broken extracts
        # (e.g., "IX.(1)"), which can prevent numeral detection.
        t = _strip_trailing_footnote_marker_from_heading(t) or t

        # 0) Standalone note heading (no numeral prefix)
        if note_heading_re.match(t):
            return _safe_text(t)

        # 0) "CHAPTER IV" / "Chapter 4" + optional heading line
        m_ch = chapter_line_re.match(t)
        if m_ch:
            chap_num = _safe_text(m_ch.group(1)).rstrip(".")
            chap_num = chap_num + "."
            # Look ahead for a title line.
            for j in range(i + 1, min(i + 40, len(window))):
                cand = window[j]
                if not _safe_text(cand):
                    continue
                if _is_probable_heading(cand) or _is_title_case_heading(cand):
                    head_parts = [_safe_text(cand)]
                    # Append additional heading component lines (common in broken EPUB extracts).
                    look = j + 1
                    while look < min(j + 10, len(window)) and len(head_parts) < 3:
                        nxt = _safe_text(window[look] or "")
                        if not nxt:
                            look += 1
                            continue
                        nxt2 = _strip_trailing_footnote_marker_from_heading(nxt) or nxt
                        if _line_looks_like_heading_component(nxt2) and not _is_notes_header_line(nxt2):
                            head_parts.append(nxt2)
                            look += 1
                            continue
                        break
                    head = " ".join([p for p in head_parts if p]).strip()
                    return f"{chap_num} {head}".strip()
                break
            return f"{chap_num}"

        # 1) Inline pattern: "III. CHAPTERNAME. / CHAPTERNAME.." (dot optional in some extracts)
        m_inline = inline_re.match(t)
        if m_inline:
            token = _safe_text(m_inline.group(1)).rstrip(".")
            allow_mid = _is_roman_token(token) or _allow_arabic_heading(i)
            if allow_mid and (_is_probable_heading(m_inline.group(2)) or _is_title_case_heading(m_inline.group(2))):
                num = token + "."
                head = _safe_text(m_inline.group(2))
                return f"{num} {head}".strip()

        # 1b) Tight inline pattern: "IIThe Story ..." (missing whitespace between numeral and title)
        m_tight = tight_inline_re.match(t)
        if m_tight:
            head = _safe_text(m_tight.group(2))
            token = _safe_text(m_tight.group(1)).rstrip(".")
            allow_mid = _is_roman_token(token) or _allow_arabic_heading(i)
            if head and allow_mid and (_is_probable_heading(head) or _is_title_case_heading(head)):
                num = token + "."
                return f"{num} {head}"

        # 2) Two-line pattern: numeral line then heading line (possibly after blanks)
        m_paren_rom = paren_roman_line_re.match(t)
        m_paren_ara = paren_arabic_line_re.match(t)
        if roman_line_re.match(t) or arabic_line_re.match(t) or m_paren_rom or m_paren_ara:
            num_src = (
                _safe_text((m_paren_rom or m_paren_ara).group(1))
                if (m_paren_rom or m_paren_ara)
                else _safe_text(t)
            )
            allow_mid = _is_roman_token(num_src) or _allow_arabic_heading(i)
            for j in range(i + 1, min(i + 80, len(window))):
                cand = window[j]
                if not _safe_text(cand):
                    continue
                if allow_mid and (_is_probable_heading(cand) or _is_title_case_heading(cand)):
                    num = _safe_text(num_src).rstrip(".") + "."
                    head_parts = [_safe_text(cand)]
                    look = j + 1
                    while look < min(j + 12, len(window)) and len(head_parts) < 3:
                        nxt = _safe_text(window[look] or "")
                        if not nxt:
                            look += 1
                            continue
                        nxt2 = _strip_trailing_footnote_marker_from_heading(nxt) or nxt
                        if _line_looks_like_heading_component(nxt2) and not _is_notes_header_line(nxt2):
                            head_parts.append(nxt2)
                            look += 1
                            continue
                        break
                    head = " ".join([p for p in head_parts if p]).strip()
                    return f"{num} {head}".strip()
                # If we hit non-heading prose, bail.
                break

    # Fallback if no numeric heading was found.
    return ml_fallback


def _label_is_plausible_chapter_label(label: Optional[str]) -> bool:
    """Return True if label looks like a real chapter/section heading.

    This prevents notes/glossary definition lines (e.g. "4. the High: ... see")
    from overwriting chapter context and scrambling grouping.
    """

    t = _safe_text(label or "")
    if not t:
        return False

    # Reject speaker-initial abbreviation runs.
    if _looks_like_abbrev_run(t):
        return False

    upper = t.upper().strip().strip(".")
    if upper in {"NOTES", "FOOTNOTES", "ENDNOTES", "FOOTNOTES AND ENDNOTES", "CONTENTS"}:
        return False

    # Numeric heading followed by lowercase is commonly a glossary entry, not a chapter.
    if re.match(r"^\s*\d{1,3}\.\s+[a-z]", t):
        return False

    # Glossary/definition-y punctuation and editor references.
    if ":" in t or ";" in t or "$" in t:
        return False
    if re.search(r"\bsee\b", t, re.IGNORECASE):
        return False

    return True


def _find_chapter_headings_in_text(text: str) -> List[Tuple[int, str]]:
    """Return a list of (char_offset, label) chapter headings found in text.

    Used for malformed EPUB files where multiple chapters may appear within one
    spine document. We assign each anchor to the nearest preceding chapter
    heading based on its character offset ("position").
    """

    if not text:
        return []

    def _is_probable_heading(s: str) -> bool:
        t = _safe_text(s)
        if not t:
            return False
        if _looks_like_abbrev_run(t):
            return False
        t_upper = t.upper().strip(".")
        if t_upper in {"NOTES", "FOOTNOTES", "ENDNOTES", "CONTENTS"}:
            return False
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 6:
            return False
        upper = sum(1 for ch in letters if ch.isupper())
        if (upper / max(1, len(letters))) < 0.85:
            return False
        if len(t) > 120:
            return False
        return True

    def _is_title_case_heading(s: str) -> bool:
        t = _safe_text(s)
        if not t:
            return False
        if _looks_like_abbrev_run(t):
            return False
        t_upper = t.upper().strip(".")
        if t_upper in {"NOTES", "FOOTNOTES", "ENDNOTES", "CONTENTS"}:
            return False
        had_trailing_period = t.endswith(".")
        if had_trailing_period and t[:-1].count(".") >= 1:
            return False
        if had_trailing_period:
            t = t.rstrip(".").rstrip()

        if len(t) < 8 or len(t) > 80:
            return False
        words = [w for w in re.split(r"\s+", t) if w]
        if len(words) < 2:
            return False

        connector = {
            "a",
            "an",
            "and",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "into",
            "of",
            "on",
            "to",
            "the",
            "upon",
            "with",
            "without",
        }

        starts_caps = 0
        alpha_words = 0
        for w in words:
            w2 = re.sub(r"[^A-Za-z]", "", w)
            if not w2:
                continue
            wl = w2.lower()
            if wl in connector:
                continue
            alpha_words += 1
            if w2[0].isupper():
                starts_caps += 1
        if alpha_words < 2:
            return False
        if (starts_caps / max(1, alpha_words)) < 0.7:
            return False
        return True

    roman_line_re = re.compile(r"^\s*[IVXLC]{1,8}\.?(?:\s+)?$")
    arabic_line_re = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")
    paren_roman_line_re = re.compile(r"^\s*\(\s*([IVXLC]{1,8})\s*\)\s*$")
    paren_arabic_line_re = re.compile(r"^\s*\(\s*(\d{1,3})\s*\)\s*$")
    inline_re = re.compile(r"^\s*([IVXLC]{1,8}|\d{1,3})\.?(?:\s+)(.+?)\s*$")
    tight_inline_re = re.compile(r"^\s*([IVXLC]{1,8}|\d{1,3})\.?([A-Z][A-Za-z].{5,120})\s*$")
    chapter_line_re = re.compile(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\s*[:\.]?\s*$", re.IGNORECASE)
    note_heading_re = re.compile(r"^\s*NOTE\s+ON\s+THE\s+CHRONOLOGY\.?\s*$", re.IGNORECASE)
    # Some extractors hard-wrap and can embed a heading after a paragraph-ending period
    # with many spaces before the numeral, e.g. "... 'Four'.     V. CHAPTERNAME.".
    embedded_inline_re = re.compile(
        r"(?:^|[\.\!\?]\s{2,}|\s{6,})([IVXLC]{1,8}|\d{1,3})\.\s+([A-Z][A-Z\s]{4,80}?)(?:\.|\s*$)",
        re.UNICODE,
    )

    lines = text.split("\n")
    headings: List[Tuple[int, str]] = []

    # When we infer a numeric heading like "II." + "THE ..." by looking ahead,
    # record the title/component line indices that were consumed so we don't
    # later emit a duplicate title-only heading at that same location.
    consumed_heading_part_lines: set[int] = set()

    # Precompute line offsets so we can place synthetic headings precisely.
    line_offsets: List[int] = []
    off0 = 0
    for ln in lines:
        line_offsets.append(off0)
        off0 += len(ln) + 1

    def _prepend_prefix_title(line_idx: int, label: str) -> str:
        label_text = _safe_text(label or "")
        if not label_text:
            return label_text
        if not re.match(r"^\s*(?:[IVXLC]{1,8}|\d{1,3})\.\s+", label_text):
            return label_text
        for k in range(int(line_idx) - 1, max(-1, int(line_idx) - 8), -1):
            prev_raw = _safe_text(lines[k] or "")
            if not prev_raw:
                continue
            prev_clean = _strip_trailing_footnote_marker_from_heading(prev_raw) or prev_raw
            if (
                not prev_clean
                or _is_notes_header_line(prev_clean)
                or chapter_line_re.match(prev_clean)
                or roman_line_re.match(prev_clean)
                or arabic_line_re.match(prev_clean)
                or paren_roman_line_re.match(prev_clean)
                or paren_arabic_line_re.match(prev_clean)
                or inline_re.match(prev_clean)
                or tight_inline_re.match(prev_clean)
                or _def_line_regex().match(prev_clean)
            ):
                break
            if not (_is_title_case_heading(prev_clean) or _is_probable_heading(prev_clean)):
                break
            prev_norm = re.sub(r"\s+", " ", prev_clean).strip(" .")
            label_norm = re.sub(r"\s+", " ", label_text).strip()
            if prev_norm and label_norm:
                if label_norm.upper().startswith(prev_norm.upper()):
                    return label_text
                prefix = f"{prev_norm}." if not prev_clean.rstrip().endswith(".") else prev_clean.rstrip()
                return f"{prefix} {label_norm}".strip()
            break
        return label_text

    note_boundary_lines: List[int] = []
    try:
        note_headers = [i for i, ln in enumerate(lines) if _is_notes_header_line(ln)]
        for hi in note_headers[:80]:
            end_idx = _find_notes_block_end_from_header(lines, hi)
            if isinstance(end_idx, int) and 0 <= end_idx < len(lines):
                note_boundary_lines.append(int(end_idx))
    except Exception:
        note_boundary_lines = []

    def _is_near_notes_boundary_line(line_idx: int) -> bool:
        li = int(line_idx)
        for b in note_boundary_lines:
            if abs(li - int(b)) <= 40:
                return True
        return False

    def _allow_arabic_heading(line_idx: int) -> bool:
        li = int(line_idx)
        return li <= 120 or _is_near_notes_boundary_line(li)

    def _is_roman_token(token: str) -> bool:
        probe = _safe_text(token or "").strip().rstrip(".")
        return bool(re.fullmatch(r"[IVXLC]{1,12}", probe, re.IGNORECASE))

    def _is_strong_middoc_heading_label(label: str) -> bool:
        t = _safe_text(label or "")
        if not t:
            return False
        if re.match(r"^\s*\[?\s*(?:PART|BOOK)\b", t, re.IGNORECASE):
            return True
        if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", t, re.IGNORECASE):
            return True
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) >= 6:
            upper = sum(1 for ch in letters if ch.isupper())
            if (upper / max(1, len(letters))) >= 0.90:
                return True
        return False

    # 0) Synthetic heading at start of file (multi-line titles like PART/FOREWORD blocks).
    # Only do this if we *don't* see a strong CHAPTER/roman heading early.
    try:
        strong_early = False

        def _next_nonempty(idx: int) -> Optional[int]:
            for j in range(idx, min(idx + 40, len(lines))):
                if _safe_text(lines[j] or ""):
                    return j
            return None

        for i, ln in enumerate(lines[:120]):
            t = _safe_text(ln or "")
            if not t:
                continue
            if chapter_line_re.match(t):
                strong_early = True
                break
            if roman_line_re.match(t) or arabic_line_re.match(t):
                j = _next_nonempty(i + 1)
                if j is not None and _line_looks_like_heading_component(lines[j]):
                    strong_early = True
                    break

        if not strong_early:
            ml0 = _infer_multiline_heading_from_lines(lines, 0)
            if ml0 is not None:
                li0, lab0 = ml0
                if 0 <= li0 < len(line_offsets) and lab0:
                    headings.append((int(line_offsets[li0]), lab0))
    except Exception:
        pass

    # 0b) For malformed EPUBs with per-section notes, treat the end of each NOTES block as
    # a likely start of a new section, then infer the heading from the following lines.
    try:
        note_headers = [i for i, ln in enumerate(lines) if _is_notes_header_line(ln)]
        for hi in note_headers[:80]:
            end_idx = _find_notes_block_end_from_header(lines, hi)
            if end_idx is None or end_idx >= len(lines):
                continue
            ml = _infer_multiline_heading_from_lines(lines, end_idx)
            if ml is None:
                continue
            li, lab = ml
            if not lab or not (0 <= li < len(line_offsets)):
                continue
            # If a strong chapter heading appears immediately after, skip the synthetic heading.
            try:
                suppressed = False
                for j in range(li, min(li + 20, len(lines))):
                    t = _safe_text(lines[j] or "")
                    if not t:
                        continue
                    if _is_notes_header_line(t):
                        break
                    if chapter_line_re.match(t):
                        suppressed = True
                        break
                    if roman_line_re.match(t) or arabic_line_re.match(t):
                        k = j + 1
                        while k < len(lines) and k <= j + 30 and not _safe_text(lines[k] or ""):
                            k += 1
                        if k < len(lines) and _line_looks_like_heading_component(lines[k]):
                            suppressed = True
                            break
                if suppressed:
                    continue
            except Exception:
                pass
            if _is_strong_middoc_heading_label(lab):
                headings.append((int(line_offsets[li]), lab))
    except Exception:
        pass
    offset = 0
    def_re = _def_line_regex()

    # Suppress weak heading inference (ALLCAPS/title-card blocks, embedded inline headings)
    # when we appear to be inside a run of footnote definitions.
    #
    # Rationale: in many critical editions, a *note body* can include quoted ALLCAPS lines
    # (e.g., "... FOREWORD.") that should not split chapter grouping.
    recent_def_idxs: List[int] = []
    notes_run_active = False
    notes_run_cooldown = 0

    def _exit_notes_run_if_boundary(i0: int) -> None:
        """Exit notes suppression when prose clearly resumes.

        We require a small multi-line heading block (e.g., ALL-CAPS title + subhead)
        and that the next non-empty line is not a definition marker.
        """

        nonlocal notes_run_active, notes_run_cooldown, recent_def_idxs
        try:
            t0 = _safe_text(lines[i0] or "")
            if not t0:
                return
            t0 = _strip_trailing_footnote_marker_from_heading(t0) or t0
            if _is_notes_header_line(t0) or def_re.match(t0):
                return

            # Strong single-line boundaries that are common in critical editions.
            # Example:
            #   [PART TWO].(1)
            #   Night 62.(2) Thursday, March 6th, 1987.
            part_line = bool(re.match(r"^\s*\[?\s*PART\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\b", t0, re.IGNORECASE))
            night_line = bool(
                re.match(r"^\s*NIGHT\s+\d{1,3}\s*\.(?:\s*\(\s*\d{1,3}\s*\))?\s+\S", t0, re.IGNORECASE)
                or re.match(r"^\s*NIGHT\s+\d{1,3}\s*(?:\(\s*\d{1,3}\s*\))\s+\S", t0, re.IGNORECASE)
            )

            if not (part_line or night_line) and not _line_looks_like_heading_component(t0) and not _looks_like_chapter_heading_text(t0):
                return

            def _is_allcaps_heading_line(s: str) -> bool:
                s2 = _safe_text(s or "")
                if not s2:
                    return False
                s2 = _strip_trailing_footnote_marker_from_heading(s2) or s2
                if _is_notes_header_line(s2):
                    return False
                letters = [ch for ch in s2 if ch.isalpha()]
                if len(letters) < 6:
                    return False
                upper_count = sum(1 for ch in letters if ch.isupper())
                return (upper_count / max(1, len(letters))) >= 0.90

            def _is_strong_boundary_line(s: str) -> bool:
                s2 = _safe_text(s or "")
                if not s2:
                    return False
                s2 = _strip_trailing_footnote_marker_from_heading(s2) or s2
                if re.match(r"^\s*\[?\s*PART\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\b", s2, re.IGNORECASE):
                    return True
                if re.match(r"^\s*NIGHT\s+\d{1,3}\s*\.(?:\s*\(\s*\d{1,3}\s*\))?\s+\S", s2, re.IGNORECASE):
                    return True
                if re.match(r"^\s*NIGHT\s+\d{1,3}\s*(?:\(\s*\d{1,3}\s*\))\s+\S", s2, re.IGNORECASE):
                    return True
                if chapter_line_re.match(s2):
                    return True
                if roman_line_re.match(s2) or arabic_line_re.match(s2):
                    return True
                if _is_allcaps_heading_line(s2):
                    return True
                return False

            # Critical: do NOT end notes suppression just because we see a Title Case
            # "title-page" block inside a long footnote body. Require an ALL-CAPS or
            # explicit CHAPTER/numeral heading signal within the heading block.
            strong_seen = _is_strong_boundary_line(t0)

            # Find the next non-empty line.
            next_nonempty_idx: Optional[int] = None
            for k in range(i0 + 1, min(i0 + 12, len(lines))):
                if _safe_text(lines[k] or ""):
                    next_nonempty_idx = k
                    break
            if next_nonempty_idx is None:
                return

            t1 = _safe_text(lines[next_nonempty_idx] or "")
            t1 = _strip_trailing_footnote_marker_from_heading(t1) or t1
            if def_re.match(t1):
                return
            if _is_notes_header_line(t1):
                return

            if _is_strong_boundary_line(t1):
                strong_seen = True

            # Require a second heading-like line to avoid being fooled by quoted fragments,
            # *except* for strong single-line boundaries (PART/Night) which frequently
            # transition directly into prose.
            if not (part_line or night_line or _is_strong_boundary_line(t1)):
                if not (_line_looks_like_heading_component(t1) or _looks_like_chapter_heading_text(t1)):
                    return

            if not strong_seen:
                return

            notes_run_active = False
            notes_run_cooldown = 0
            recent_def_idxs = []
        except Exception:
            return

    for i, line in enumerate(lines):
        if i in consumed_heading_part_lines:
            offset += len(line or "") + 1
            continue
        raw = line
        t = _safe_text(raw)
        if t:
            t_clean = _strip_trailing_footnote_marker_from_heading(t) or t

            # If we see a NOTES/ENDNOTES/FOOTNOTES header line, treat the subsequent
            # region as a notes run. This suppresses weak heading inference inside
            # notes bodies (which can include quoted ALLCAPS lines).
            if _is_notes_header_line(t_clean):
                notes_run_active = True
                recent_def_idxs = []
                notes_run_cooldown = max(notes_run_cooldown, 250)
                offset += len(raw) + 1
                continue
            # Standalone note headings can act like chapters in critical editions.
            if note_heading_re.match(t_clean):
                headings.append((offset, _safe_text(t_clean)))
                offset += len(raw) + 1
                continue
            # Strong PART boundary lines can mark a new grouping inside long omnibus files,
            # e.g. "[PART TWO].(1)" before a fresh run of other entries.
            if re.match(r"^\s*\[?\s*PART\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\b", t_clean, re.IGNORECASE):
                headings.append((offset, _safe_text(t_clean)))
                notes_run_active = False
                recent_def_idxs = []
                offset += len(raw) + 1
                continue
            # Avoid misclassifying footnote definition lines like "3. Andore: ..." as chapter headings.
            # Note: the definition regex can also match single-letter markers like "V.",
            # so we must not skip lines that look like real numeral+ALLCAPS chapter headings.
            if def_re.match(t):
                # The definition regex can also match roman numeral chapter headings like "V.".
                # Detect and exempt those so we don't erroneously enter notes mode.
                looks_like_real_heading = False
                try:
                    if chapter_line_re.match(t) or roman_line_re.match(t) or paren_roman_line_re.match(t):
                        looks_like_real_heading = True
                    elif arabic_line_re.match(t) or paren_arabic_line_re.match(t):
                        looks_like_real_heading = _allow_arabic_heading(i)

                    prev_heading_like = False
                    for k in range(i - 1, max(-1, i - 8), -1):
                        prev_raw = _safe_text(lines[k] or "")
                        if not prev_raw:
                            continue
                        prev_clean = _strip_trailing_footnote_marker_from_heading(prev_raw) or prev_raw
                        prev_heading_like = bool(
                            prev_clean
                            and not _is_notes_header_line(prev_clean)
                            and (_line_looks_like_heading_component(prev_clean) or _looks_like_chapter_heading_text(prev_clean))
                        )
                        break

                    m_head = re.match(r"^\s*([IVXLC]{1,8}|\d{1,3})\.\s+(.+?)\s*$", t)
                    if m_head:
                        token = _safe_text(m_head.group(1) or "").rstrip(".")
                        rest = _safe_text(m_head.group(2) or "")
                        rest_upper = rest.upper()
                        allow_mid = _is_roman_token(token) or _allow_arabic_heading(i)
                        if rest and ":" not in rest and ";" not in rest and "$" not in rest and "SEE" not in rest_upper:
                            letters = [ch for ch in rest if ch.isalpha()]
                            if len(letters) >= 6:
                                upper_count = sum(1 for ch in letters if ch.isupper())
                                if (upper_count / max(1, len(letters))) >= 0.85:
                                    # Looks like "V. CHAPTERNAME."; allow processing,
                                    # but keep mid-document Arabic note subheads out.
                                    looks_like_real_heading = allow_mid
                                elif _is_title_case_heading(rest) and allow_mid and prev_heading_like:
                                    # Allow title-card continuations like:
                                    # Chaptername.
                                    #   1. Chapter name.
                                    looks_like_real_heading = True
                                else:
                                    offset += len(raw) + 1
                                    continue
                            else:
                                offset += len(raw) + 1
                                continue
                        else:
                            offset += len(raw) + 1
                            continue
                    else:
                        offset += len(raw) + 1
                        continue
                except Exception:
                    offset += len(raw) + 1
                    continue

                if not looks_like_real_heading:
                    # Track likely definition clusters; used to suppress weak heading inference.
                    try:
                        recent_def_idxs.append(i)
                        while recent_def_idxs and (i - recent_def_idxs[0]) > 140:
                            recent_def_idxs.pop(0)
                    except Exception:
                        pass
                    # Even a single definition marker is a strong indicator we're in notes.
                    # Keep suppression sticky; only a strong boundary can exit notes mode.
                    notes_run_active = True
                    notes_run_cooldown = max(notes_run_cooldown, 250)

            # Cooldown-based stickiness: after seeing notes headers/defs, keep suppression
            # active for a while even if definitions are sparse (note bodies can be long).
            if notes_run_cooldown > 0:
                notes_run_cooldown -= 1
                notes_run_active = True

            # If we hit a clear section-heading boundary, stop suppressing.
            if notes_run_active:
                _exit_notes_run_if_boundary(i)

            # Mid-document title-card / section heading blocks.
            # Many broken EPUB extracts put section headers on their own lines,
            # sometimes with blank lines between components.
            try:
                if notes_run_active:
                    raise RuntimeError("suppress_weak_heading_in_notes")
                t2 = _strip_trailing_footnote_marker_from_heading(t) or t
                if (
                    t2
                    and _line_looks_like_heading_component(t2)
                    and not chapter_line_re.match(t2)
                    and not roman_line_re.match(t2)
                    and not arabic_line_re.match(t2)
                    and not inline_re.match(t2)
                ):
                    # Only trigger at the *start* of a heading block.
                    prev_nonempty: Optional[int] = None
                    for k in range(i - 1, max(-1, i - 10), -1):
                        if _safe_text(lines[k] or ""):
                            prev_nonempty = k
                            break
                    if prev_nonempty is None:
                        prev_is_heading = False
                    else:
                        prev_t = _strip_trailing_footnote_marker_from_heading(_safe_text(lines[prev_nonempty] or "")) or _safe_text(
                            lines[prev_nonempty] or ""
                        )
                        prev_is_heading = bool(prev_t and _line_looks_like_heading_component(prev_t))
                        # Treat standalone numeral lines as part of a chapter heading block,
                        # so we don't inject a duplicate title-only heading right after "I.".
                        if prev_t and (
                            chapter_line_re.match(prev_t)
                            or roman_line_re.match(prev_t)
                            or arabic_line_re.match(prev_t)
                            or paren_roman_line_re.match(prev_t)
                            or paren_arabic_line_re.match(prev_t)
                        ):
                            prev_is_heading = True

                    prev_blank = (
                        i == 0
                        or not _safe_text(lines[i - 1] or "")
                        or (i >= 2 and not _safe_text(lines[i - 2] or ""))
                    )

                    if (i <= 120 or _is_near_notes_boundary_line(i) or part_line.match(t2)) and prev_blank and not prev_is_heading:
                        ml_mid = _infer_multiline_heading_from_lines(lines, i)
                        if ml_mid is not None:
                            li_mid, lab_mid = ml_mid
                            if lab_mid and 0 <= li_mid < len(line_offsets) and _label_is_plausible_chapter_label(lab_mid):
                                # Be conservative: avoid stealing simple one-line
                                # title-case headings (these frequently belong to
                                # numeral+title pairs like "IV" + "Chapter Name.").
                                tok = _extract_chapter_token(lab_mid)
                                ok = True
                                if tok is None:
                                    letters = [ch for ch in lab_mid if ch.isalpha()]
                                    upper_ratio = (
                                        sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
                                    )
                                    if upper_ratio < 0.85 and len(lab_mid) < 28:
                                        ok = False
                                if ok and _is_strong_middoc_heading_label(lab_mid):
                                    headings.append((int(line_offsets[li_mid]), lab_mid))
            except Exception:
                pass

            # Embedded inline chapter headings (mid-line).
            # Use the raw line for offsets so heading positions line up with anchors_text.
            try:
                if notes_run_active:
                    raise RuntimeError("suppress_embedded_heading_in_notes")
                for m_emb in embedded_inline_re.finditer(raw):
                    num_raw = _safe_text(m_emb.group(1)).rstrip(".")
                    head_raw = _safe_text(m_emb.group(2)).strip()
                    if not num_raw or not head_raw:
                        continue
                    # Avoid glossary-like headings.
                    if ":" in head_raw or ";" in head_raw or "$" in head_raw:
                        continue
                    if re.search(r"\bsee\b", head_raw, re.IGNORECASE):
                        continue
                    # Require at least two words in the heading.
                    if len([w for w in head_raw.split() if w]) < 2:
                        continue
                    hpos = int(offset + m_emb.start(1))
                    head_clean = head_raw.rstrip(".")
                    headings.append((hpos, f"{num_raw}. {head_clean}."))
            except Exception:
                pass
            m_ch = chapter_line_re.match(t_clean)
            if m_ch:
                chap_num = _safe_text(m_ch.group(1)).rstrip(".") + "."
                head = None
                for j in range(i + 1, min(i + 40, len(lines))):
                    cand = lines[j]
                    if not _safe_text(cand):
                        continue
                    if _is_probable_heading(cand) or _is_title_case_heading(cand):
                        head_parts = [_safe_text(cand)]
                        consumed_heading_part_lines.add(j)
                        look = j + 1
                        while look < min(j + 10, len(lines)) and len(head_parts) < 3:
                            nxt = _safe_text(lines[look] or "")
                            if not nxt:
                                look += 1
                                continue
                            nxt2 = _strip_trailing_footnote_marker_from_heading(nxt) or nxt
                            if _line_looks_like_heading_component(nxt2) and not _is_notes_header_line(nxt2):
                                head_parts.append(nxt2)
                                consumed_heading_part_lines.add(look)
                                look += 1
                                continue
                            break
                        head = " ".join([p for p in head_parts if p]).strip()
                    break
                label = f"{chap_num} {head}" if head else f"{chap_num}"
                label = _prepend_prefix_title(j, label) if head else label
                headings.append((offset, label))
                # Strong signal; exit any notes-run suppression.
                notes_run_active = False
                recent_def_idxs = []
            else:
                m_inline = inline_re.match(t_clean)
                m_tight = tight_inline_re.match(t_clean)
                m_pr = paren_roman_line_re.match(t_clean)
                m_pa = paren_arabic_line_re.match(t_clean)

                if m_inline:
                    token = _safe_text(m_inline.group(1)).rstrip(".")
                    allow_mid = _is_roman_token(token) or _allow_arabic_heading(i)
                    if allow_mid and (_is_probable_heading(m_inline.group(2)) or _is_title_case_heading(m_inline.group(2))):
                        num = token + "."
                        head = _safe_text(m_inline.group(2))
                        headings.append((offset, _prepend_prefix_title(i, f"{num} {head}")))
                        notes_run_active = False
                        recent_def_idxs = []
                elif m_tight:
                    head = _safe_text(m_tight.group(2))
                    token = _safe_text(m_tight.group(1)).rstrip(".")
                    allow_mid = _is_roman_token(token) or _allow_arabic_heading(i)
                    if head and allow_mid and (_is_probable_heading(head) or _is_title_case_heading(head)):
                        num = token + "."
                        headings.append((offset, _prepend_prefix_title(i, f"{num} {head}")))
                        notes_run_active = False
                        recent_def_idxs = []
                elif roman_line_re.match(t_clean) or arabic_line_re.match(t_clean):
                    allow_mid = _is_roman_token(t_clean) or _allow_arabic_heading(i)
                    for j in range(i + 1, min(i + 80, len(lines))):
                        cand = lines[j]
                        if not _safe_text(cand):
                            continue
                        if allow_mid and (_is_probable_heading(cand) or _is_title_case_heading(cand)):
                            num = _safe_text(t_clean).rstrip(".") + "."
                            head_parts = [_safe_text(cand)]
                            consumed_heading_part_lines.add(j)
                            look = j + 1
                            while look < min(j + 12, len(lines)) and len(head_parts) < 3:
                                nxt = _safe_text(lines[look] or "")
                                if not nxt:
                                    look += 1
                                    continue
                                nxt2 = _strip_trailing_footnote_marker_from_heading(nxt) or nxt
                                if _line_looks_like_heading_component(nxt2) and not _is_notes_header_line(nxt2):
                                    head_parts.append(nxt2)
                                    consumed_heading_part_lines.add(look)
                                    look += 1
                                    continue
                                break
                            head = " ".join([p for p in head_parts if p]).strip()
                            headings.append((offset, _prepend_prefix_title(j, f"{num} {head}".strip())))
                            notes_run_active = False
                            recent_def_idxs = []
                        break
                elif (m_pr or m_pa):
                    # Parenthesized numeral line pattern: "(IV)" then heading line.
                    num_src = _safe_text((m_pr or m_pa).group(1))
                    allow_mid = _is_roman_token(num_src) or _allow_arabic_heading(i)
                    for j in range(i + 1, min(i + 80, len(lines))):
                        cand = lines[j]
                        if not _safe_text(cand):
                            continue
                        if allow_mid and (_is_probable_heading(cand) or _is_title_case_heading(cand)):
                            num = _safe_text(num_src).rstrip(".") + "."
                            head_parts = [_safe_text(cand)]
                            consumed_heading_part_lines.add(j)
                            look = j + 1
                            while look < min(j + 12, len(lines)) and len(head_parts) < 3:
                                nxt = _safe_text(lines[look] or "")
                                if not nxt:
                                    look += 1
                                    continue
                                nxt2 = _strip_trailing_footnote_marker_from_heading(nxt) or nxt
                                if _line_looks_like_heading_component(nxt2) and not _is_notes_header_line(nxt2):
                                    head_parts.append(nxt2)
                                    consumed_heading_part_lines.add(look)
                                    look += 1
                                    continue
                                break
                            head = " ".join([p for p in head_parts if p]).strip()
                            headings.append((offset, _prepend_prefix_title(j, f"{num} {head}".strip())))
                            notes_run_active = False
                            recent_def_idxs = []
                        break

        offset += len(raw) + 1  # +1 for newline

    # Sort by position and de-dup by label; malformed text can repeat headings.
    headings.sort(key=lambda x: int(x[0]) if isinstance(x[0], int) else 0)

    # If multiple synthetic headings land at the same offset, prefer the more
    # specific label when one is just a prefix of another.
    try:
        def _same_pos_norm(label: str) -> str:
            t = _safe_text(label or "")
            t = _strip_trailing_footnote_marker_from_heading(t) or t
            t = t.upper()
            t = re.sub(r"[\[\]\(\)]", "", t)
            t = re.sub(r"\s+", " ", t).strip(" .:-_")
            return t

        collapsed: List[Tuple[int, str]] = []
        idx = 0
        while idx < len(headings):
            pos = headings[idx][0]
            same_pos: List[Tuple[int, str]] = []
            while idx < len(headings) and headings[idx][0] == pos:
                same_pos.append(headings[idx])
                idx += 1

            if len(same_pos) <= 1:
                collapsed.extend(same_pos)
                continue

            keep: List[Tuple[int, str]] = []
            norms = [_same_pos_norm(lab) for _p, lab in same_pos]
            for item, norm in zip(same_pos, norms):
                overshadowed = False
                for other in norms:
                    if not norm or not other or other == norm:
                        continue
                    if other.startswith(norm) and len(other) > len(norm):
                        overshadowed = True
                        break
                if not overshadowed:
                    keep.append(item)

            collapsed.extend(keep or [max(same_pos, key=lambda it: len(_same_pos_norm(it[1])))])

        headings = collapsed
    except Exception:
        pass

    # Drop title-only headings that duplicate a nearby numeral-bearing heading.
    # This prevents cases like:
    #   IX. CHAPTERNAME.
    #   CHAPTERNAME.
    # where the later title-only heading would otherwise override per-anchor chapter labels.
    try:
        def _norm_title(label: str) -> str:
            t = _safe_text(label or "").strip()
            if not t:
                return ""
            # Remove leading CHAPTER N or leading numeral tokens.
            t2 = re.sub(
                r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b\s*[:\.]?\s*", "", t, flags=re.IGNORECASE
            )
            t2 = re.sub(r"^\s*([IVXLC]{1,12}|\d{1,3})\.\s+", "", t2)
            t2 = re.sub(r"^\s*([IVXLC]{1,12}|\d{1,3})\s+", "", t2)
            t2 = t2.strip().strip(".:-— ")
            t2 = t2.upper()
            t2 = re.sub(r"[^A-Z0-9]+", "", t2)
            return t2

        numeric_titles: List[Tuple[int, str]] = []
        for pos, lab in headings:
            if _extract_chapter_token(lab) is None:
                continue
            nt = _norm_title(lab)
            if nt:
                numeric_titles.append((int(pos), nt))

        if numeric_titles:
            filtered: List[Tuple[int, str]] = []
            for pos, lab in headings:
                if _extract_chapter_token(lab) is not None:
                    filtered.append((pos, lab))
                    continue
                nt = _norm_title(lab)
                if not nt:
                    filtered.append((pos, lab))
                    continue
                # Drop if a matching numeral-bearing heading is very near.
                near = False
                p = int(pos)
                for np, ntt in numeric_titles:
                    if ntt != nt:
                        continue
                    if abs(np - p) <= 250:
                        near = True
                        break
                if not near:
                    filtered.append((pos, lab))
            headings = filtered
    except Exception:
        pass

    seen = set()
    out: List[Tuple[int, str]] = []
    for pos, lab in headings:
        key = _safe_text(lab).strip()
        key = re.sub(r"\s+", " ", key)
        key = key.rstrip(".").rstrip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((pos, lab))
    return out


def _looks_like_chapter_heading_text(txt: str) -> bool:
    """Return True if txt looks like an actual chapter heading/title.

    Intentionally conservative to avoid grouping by random section headings.
    """

    t = _safe_text(txt)
    if not t:
        return False

    upper = t.upper().strip(".")
    if upper in {"NOTES", "FOOTNOTES", "ENDNOTES", "CONTENTS"}:
        return False

    if len(t) < 4 or len(t) > 120:
        return False

    # Reject headings that look like sentences/notes.
    if t.count(",") >= 2:
        return False
    if t.endswith(":"):
        return False

    # Strong accept: explicit chapter marker.
    if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})(?:\b|\s*[:\.]\s*.*)$", t, re.IGNORECASE):
        return True

    # Accept: Roman numeral alone (common chapter-number heading).
    if re.match(r"^\s*[IVXLC]{1,12}\.\s*$", t) or re.match(r"^\s*[IVXLC]{1,12}\s*$", t):
        return True

    # Accept: Arabic numeral alone (some headings use plain digits).
    if re.match(r"^\s*\d{1,3}\.\s*$", t) or re.match(r"^\s*\d{1,3}\s*$", t):
        return True

    # Accept: ALL CAPS headings.
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) >= 6:
        upper_count = sum(1 for ch in letters if ch.isupper())
        if (upper_count / max(1, len(letters))) >= 0.85:
            return True

    # Accept: Title Case-ish headings.
    if len(t) <= 80 and not t.endswith("."):
        words = [w for w in re.split(r"\s+", t) if w]
        if len(words) >= 2:
            starts_caps = 0
            alpha_words = 0
            for w in words:
                w2 = re.sub(r"[^A-Za-z]", "", w)
                if not w2:
                    continue
                alpha_words += 1
                if w2[0].isupper():
                    starts_caps += 1
            if alpha_words >= 2 and (starts_caps / max(1, alpha_words)) >= 0.7:
                return True

    return False


def _chapter_group_key(label: Optional[str]) -> Optional[str]:
    """Create a stable grouping key from a chapter label."""

    t = _safe_text(label or "")
    if not t:
        return None
    t = t.upper()
    t = re.sub(r"[^A-Z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or None


def _roman_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    t = re.sub(r"[^IVXLC]", "", s.upper())
    if not t:
        return None
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    prev = 0
    for ch in reversed(t):
        v = vals.get(ch)
        if v is None:
            return None
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    if total <= 0 or total > 500:
        return None
    return total


def _extract_chapter_token(label: Optional[str]) -> Optional[int]:
    """Extract a numeric chapter token from a label like 'III. ...' or 'CHAPTER 3.'"""

    t = _safe_text(label or "")
    if not t:
        return None

    m = re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", t, re.IGNORECASE)
    if m:
        token = m.group(1)
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                return None
        return _roman_to_int(token)

    # Numeral at start, with a dot. Do NOT use a word-boundary after '.',
    # because '.' and space are both non-word chars.
    m2 = re.match(r"^\s*([IVXLC]{1,12}|\d{1,3})\.(?:\s|$)", t)
    if m2:
        token = m2.group(1)
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                return None
        return _roman_to_int(token)

    # Some extracts omit the dot: "IX THE ...".
    # Keep this conservative: require whitespace after the numeral.
    m3 = re.match(r"^\s*([IVXLC]{1,12}|\d{1,3})(?:\s+)(?=\S)", t)
    if m3:
        token = m3.group(1)
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                return None
        return _roman_to_int(token)

    return None


def _extract_notes_header_chapter_token(lines: List[str], split_main_end_index: int) -> Optional[int]:
    """If a notes header includes chapter number, extract it.

    Examples:
      NOTES TO CHAPTER VI
      NOTES ON CHAPTER 6
    """

    if not lines:
        return None
    idx = int(split_main_end_index or 0)
    if idx < 0 or idx >= len(lines):
        return None
    header = _safe_text(lines[idx] or "")
    if not header:
        return None
    m = re.match(
        r"^\s*(?:FOOTNOTES|FOOTNOTES\s+AND\s+ENDNOTES|ENDNOTES|NOTES)(?:\s+(?:TO|ON)\s+CHAPTER\s+([IVXLC]{1,12}|\d{1,3}))?\s*[:\.]?\s*$",
        header,
        re.IGNORECASE,
    )
    if not m:
        return None
    token = m.group(1)
    if not token:
        return None
    if token.isdigit():
        try:
            return int(token)
        except Exception:
            return None
    return _roman_to_int(token)


def _infer_chapter_label_from_soup(soup: BeautifulSoup) -> Optional[str]:
    """Try to infer chapter label from HTML headings before falling back to line scans."""

    if soup is None:
        return None

    def _is_part_like(txt: str) -> bool:
        return bool(re.match(r"^\s*\[?\s*(?:PART|BOOK)\b", txt or "", re.IGNORECASE))

    def _extract_num_only(txt: str) -> Optional[str]:
        t = _safe_text(txt).strip()
        if not t:
            return None
        m = re.match(r"^\(?\s*([IVXLC]{1,12}|\d{1,3})\s*\)?\.?\s*$", t, re.IGNORECASE)
        if not m:
            return None
        return _safe_text(m.group(1)).rstrip(".")

    def _collect_following_heading_parts(tags: List[Any], start_idx: int) -> str:
        parts: List[str] = []
        # Collect up to a few subsequent heading-like components. This captures:
        #   I.
        #   CHAPTER NAME
        #   IS THIS.
        for k in range(start_idx + 1, min(start_idx + 6, len(tags))):
            t = _safe_text(tags[k].get_text(" ") if tags[k] is not None else "")
            t = _strip_trailing_footnote_marker_from_heading(t) or t
            if not t:
                continue
            if _is_notes_header_line(t):
                break
            # Stop if we hit another numeric/CHAPTER marker.
            if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", t, re.IGNORECASE):
                break
            if _extract_num_only(t) is not None:
                break
            if _line_looks_like_heading_component(t) or _looks_like_chapter_heading_text(t):
                parts.append(t)
                continue
            break
        return " ".join(parts).strip()

    # Prefer numeral-bearing headings over PART/BOOK headings.
    try:
        tags = list(soup.find_all(["h1", "h2", "h3", "h4"]))
        # First pass: find a strong CHAPTER X or numeral-only heading and assemble a label.
        for idx, tag in enumerate(tags[:80]):
            txt = _safe_text(tag.get_text(" "))
            txt = _strip_trailing_footnote_marker_from_heading(txt) or txt
            if not txt:
                continue
            if _is_notes_header_line(txt):
                continue

            m = re.match(
                r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\s*[:\.]?\s*(.*)$",
                txt,
                re.IGNORECASE,
            )
            if m:
                num = _safe_text(m.group(1)).rstrip(".")
                rest = _safe_text(m.group(2))
                if not rest:
                    rest = _collect_following_heading_parts(tags, idx)
                if rest:
                    return f"{num}. {rest}".strip()
                return f"{num}."

            num_only = _extract_num_only(txt)
            if num_only:
                rest = _collect_following_heading_parts(tags, idx)
                if rest:
                    return f"{num_only}. {rest}".strip()
                return f"{num_only}."

            # Inline pattern like "II. THE TOWER ..." inside a single tag.
            m_inline = re.match(r"^\s*([IVXLC]{1,12}|\d{1,3})\.?\s+(.+?)\s*$", txt)
            if m_inline:
                head = _safe_text(m_inline.group(2))
                if head and (_line_looks_like_heading_component(head) or _looks_like_chapter_heading_text(head)):
                    num = _safe_text(m_inline.group(1)).rstrip(".")
                    return f"{num}. {head}".strip()

        # Second pass: pick a non-PART/BOOK plausible heading if that's all we have.
        part_fallback: Optional[str] = None
        for tag in tags[:80]:
            txt = _safe_text(tag.get_text(" "))
            txt = _strip_trailing_footnote_marker_from_heading(txt) or txt
            if not txt:
                continue
            if not _looks_like_chapter_heading_text(txt):
                continue
            if _is_part_like(txt):
                if part_fallback is None:
                    part_fallback = txt
                continue
            return txt

        # If we only found PART/BOOK-like headings, return None so line-scan can find
        # the real numeric chapter label in the text (I., II., ...).
        return None
    except Exception:
        pass

    # Fallback: <title> is often chapter-specific, but avoid PART/BOOK titles unless numeric.
    try:
        if soup.title and soup.title.string:
            t = _safe_text(soup.title.string)
            t = _strip_trailing_footnote_marker_from_heading(t) or t
            if not t:
                return None
            if _extract_chapter_token(t) is not None:
                return t
    except Exception:
        pass

    return None


def _infer_chapter_label_from_item_name(chapter_name: Optional[str]) -> Optional[str]:
    """Infer a chapter label from an EPUB spine item name/path.

    Many EPUBs use filenames like "chapter03.xhtml" or "ch_04.xhtml".
    This is a last-resort fallback when headings are missing.
    """

    if not chapter_name:
        return None
    base = str(chapter_name).replace("\\", "/").split("/")[-1]
    base = _safe_text(base)
    if not base:
        return None

    m = re.search(r"(?:\bchapter\b|\bch\b|\bchap\b)[\s_\-]*0*(\d{1,3})", base, re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 400:
                return f"CHAPTER {n}."
        except Exception:
            pass

    m2 = re.search(r"\b(\d{1,3})\b", base)
    if m2:
        try:
            n = int(m2.group(1))
            if 1 <= n <= 400:
                return f"CHAPTER {n}."
        except Exception:
            pass

    return None
