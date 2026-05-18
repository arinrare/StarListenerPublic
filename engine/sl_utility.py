import re

# Utility function to clean and normalize text for regex parsing and heuristics.
def _safe_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# Remove invisible formatting characters that commonly appear in EPUB/PDF text, which can break regexes that rely on start-of-line matching (e.g., NOTES headers, '17.' definitions).
def _clean_line_for_parsing(line: str) -> str:
    """Remove invisible formatting characters that commonly appear in EPUB/PDF text.

    These can break regexes that rely on start-of-line matching (e.g., NOTES headers, '17.' definitions).
    """
    if line is None:
        return ""
    # BOM, zero-width space, word-joiner, zero-width no-break.
    return (
        str(line)
        # NBSP and friends that frequently appear in EPUB text.
        .replace("\u00A0", " ")
        .replace("\u202F", " ")
        .replace("\u2009", " ")
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u2060", "")
        .replace("\uFEFF", "")
    )

# Some EPUBs embed notes inline without real line breaks, e.g. "discussed in a note at the end. NOTES. 1. ... 2. ...". Normalize those into real lines so line-based parser can work.
def _preprocess_for_notes(text: str) -> str:
        """Normalize inline notes headers into parseable line structure.

        Some EPUBs (esp. scholarly/critical editions) embed notes as:
            "... discussed in a note at the end. NOTES. 1. ... 2. ..."

        This turns those into real newlines so our line-based parser can work.
        """
        t = text or ""

        # Some EPUB conversions embed *literal* backslash escapes ("\\n", "\\r")
        # into text nodes. Treat these as real line breaks so our line-based
        # heuristics (NOTES headers, definition clusters, chapter headings) work.
        if "\\n" in t or "\\r" in t:
            t = t.replace("\\r", "\n").replace("\\n", "\n")

        # Some EPUB conversions embed notes headers without punctuation, e.g.:
        #   "... (4) NOTES 1. ..."
        # Split these into real lines so infer_notes_split can detect the header.
        # Keep this conservative: only trigger when the header word is followed by
        # something that looks like a definition marker (e.g. 1., (1), [1], A.).
        inline_notes_header_re = re.compile(
            r"(?<![A-Za-z])(FOOTNOTES|ENDNOTES|NOTES)"
            r"(?:\s+(?:TO|ON)\s+CHAPTER\s+(?:[IVXLC]{1,12}|\d{1,3}))?"
            r"\s+(?=(?:\(?\s*\d{1,3}\s*\)?|\[\s*\d{1,3}\s*\]|\(?\s*[A-Za-z]\s*\)?)[\]\)\.:\-—]\s)",
            re.IGNORECASE,
        )

        def _split_inline_notes_header(m: re.Match) -> str:
            hdr = m.group(1) or ""
            # Keep conservative: only split when the header word is uppercase in
            # the source text (typical for real NOTES/ENDNOTES section headers).
            if hdr and hdr != hdr.upper():
                return m.group(0)
            # Preserve any "ON/TO CHAPTER ..." suffix if present, because
            # heuristics.py understands that form as a notes header line.
            full = (m.group(0) or "").strip()
            # Strip the trailing whitespace that was part of the match; leave the
            # following definition marker to start on the next line.
            full = re.sub(r"\s+$", "", full)
            return "\n" + full + "\n"

        t = inline_notes_header_re.sub(_split_inline_notes_header, t)
        # Insert hard breaks around common headers even when they appear inline.
        # Some extractors emit headers like "NOTES." at end-of-line with no trailing space;
        # accept either whitespace or end-of-string after the punctuation.
        t = re.sub(r"\b(FOOTNOTES|ENDNOTES|NOTES)\s*[:\.](?:\s+|$)", r"\n\1\n", t)
        # In likely notes blocks, split numbered items onto their own lines.
        # This is conservative: only triggers after a NOTES header has been introduced.
        # Examples: "1." "2." "A." "*" etc.
        t = re.sub(r"(?m)(^\s*(?:FOOTNOTES|ENDNOTES|NOTES)\s*$)([\s\S]+)$", lambda m: m.group(1) + "\n" + re.sub(r"\s+(\d{1,3}|[A-Za-z]|\*|†|‡|§)\.\s+", r"\n\1. ", m.group(2)), t)
        return t

# Normalization function for footnote markers to a stable key format.
def _normalize_marker(marker: str) -> str:
    """Normalize common footnote marker formats to a stable key.

    Examples:
      "[12]" -> "12"
      "(a)" -> "a"
      "*" -> "*"
      "<sup>3</sup>" -> "3" (when passed already-stripped)
    """
    if marker is None:
        return ""
    m = marker.strip()
    m = re.sub(r"^\[\[", "[", m)
    m = re.sub(r"\]\]$", "]", m)
    m = m.strip()

    # Strip surrounding brackets/parens.
    m = re.sub(r"^\((.*)\)$", r"\1", m)
    m = re.sub(r"^\[(.*)\]$", r"\1", m)
    m = m.strip()

    # Superscript artifacts from HTML-to-text.
    superscript_map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
    subscript_map = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
    if any(ch in m for ch in superscript_map):
        m = "".join(superscript_map.get(ch, ch) for ch in m)
    if any(ch in m for ch in subscript_map):
        m = "".join(subscript_map.get(ch, ch) for ch in m)
    m = _safe_text(m)
    return m

# Regex for detecting candidate footnote markers in running text.
def _marker_regex() -> re.Pattern:
    # Markers treated as anchors in running text.
    # Keep this conservative to avoid pairing random numbers.
    #
    # Note: Some EPUBs use Unicode superscripts (¹²³…¹⁰) for note markers.
    # We match short runs of superscript digits only when they appear in typical
    # footnote-marker contexts (after word/punctuation and before whitespace/end/punct).
    return re.compile(
        r"(\[\s*\d{1,3}\s*\]|\(\s*\d{1,3}\s*\)|\[\s*[a-zA-Z]\s*\]|\(\s*[a-zA-Z]\s*\)|\*|†|‡|§|(?:(?<=\w)|(?<=[\)\]\}\'\"\u2019\u201D\.,;:!?]))[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}(?=(?:\s|$|[\)\]\}\'\"\u2019\u201D\.,;:!?]))|(?:(?<=\w)|(?<=[\)\]\}\'\"\u2019\u201D\.,;:!?]))[₀₁₂₃₄₅₆₇₈₉]{1,4}(?=(?:\s|$|[\)\]\}\'\"\u2019\u201D\.,;:!?])))",
        re.UNICODE,
    )

# Regex for parsing definition lines in notes blocks, e.g. "1. Note text..." or "(a) Note text..." or "† Note text..."
def _def_line_regex() -> re.Pattern:
    # Definition lines often start with a marker followed by whitespace and text.
    return re.compile(
        # Guard: do NOT treat page references like "p. 178" / "pp. 21-2" as
        # definition markers. These are common inside critical editions and can
        # otherwise be misread as a letter-marker definition with marker 'p'.
        r"^(?!\s*(?:\[|\()?\s*p{1,2}\s*\.)\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}|[₀₁₂₃₄₅₆₇₈₉]{1,4}|\*|†|‡|§)\s*(?:\]|\))?\s*(?:[\]\)\.:\-—]\s*|\s+)(?:↩|\u21A9)?\s*(.+?)\s*$",
        re.UNICODE,
    )



def _marker_category_from_raw(raw: str) -> str:
    """Classify a raw marker string into a coarse category.

    Used for filtering based on selected notes-format profile.
    """
    r = (raw or "").strip()
    if re.fullmatch(r"\(\s*\d{1,3}\s*\)", r):
        return "num_paren"
    if re.fullmatch(r"\[\s*\d{1,3}\s*\]", r):
        return "num_bracket"
    if re.fullmatch(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}", r):
        return "num_sup"
    if re.fullmatch(r"[₀₁₂₃₄₅₆₇₈₉]{1,4}", r):
        return "num_sub"
    if re.fullmatch(r"\(\s*[a-zA-Z]\s*\)", r):
        return "let_paren"
    if re.fullmatch(r"\[\s*[a-zA-Z]\s*\]", r):
        return "let_bracket"
    if r in {"*", "†", "‡", "§"}:
        return "symbol"
    if re.fullmatch(r"\d{1,3}", r):
        return "num_plain"
    return "other"
