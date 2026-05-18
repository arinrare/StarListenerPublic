import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:
    from sl_utility import (
        _safe_text,
        _def_line_regex,
        _marker_category_from_raw,
    )
    from sl_chapters import _is_notes_header_line
except ModuleNotFoundError:  # pragma: no cover
    from .sl_utility import _safe_text, _def_line_regex, _marker_category_from_raw  # type: ignore
    from .sl_chapters import _is_notes_header_line  # type: ignore


# ==========================================================
# Engine Entry Points
# ==========================================================
# engine.py should only need to import/call these:
#   - anchor_is_probable_footnote(...)
#   - infer_notes_split(...)


# ==========================================================
# Notes Continuation Heuristics
# ==========================================================


def infer_notes_continuation_harvest_start(lines: List[str]) -> Optional[int]:
    """Return a plausible start index for harvesting definitions from a continuation page.

    This is used when a spine doc looks like a short continuation of an endnotes list
    but lacks an explicit NOTES header and may contain only a few definitions.

    Returns:
      - an integer start index into `lines` if the page looks like a notes continuation
      - None otherwise
    """

    if not lines:
        return None

    # Keep this stricter than `_def_line_regex()`: continuation-page inference is a
    # fallback and must not treat ordinary prose like `I had ...` or `A peculiarity ...`
    # as note definitions.
    def_re = re.compile(
        r"^\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|\*|†|‡|§)\s*(?:\]|\))?\s*[\]\)\.:\-—]\s+(.+?)\s*$",
        re.UNICODE,
    )
    # Marker-only lines appear in some EPUB conversions:
    #   13.
    #   Eldils: ...
    marker_only_re = re.compile(
        r"^\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|\*|†|‡|§)\s*(?:\]|\))?\s*(?:[\]\)\.:\-—]\s*)?$",
        re.UNICODE,
    )

    # If this spine item already contains an explicit NOTES header, defer to the
    # normal notes-split logic instead of forcing a continuation-page guess.
    if any(_is_notes_header_line(line or "") for line in lines):
        return None

    scan_end = min(len(lines), 260)
    markerish_idxs: List[int] = []
    def_like_bodies: Dict[int, str] = {}
    for i in range(0, scan_end):
        t = _safe_text(lines[i] or "")
        if not t:
            continue
        m_def = def_re.match(t)
        if m_def:
            markerish_idxs.append(i)
            def_like_bodies[i] = _safe_text(m_def.group(2) or "").strip()
        else:
            if marker_only_re.match(t):
                markerish_idxs.append(i)
        if len(markerish_idxs) >= 10:
            break

    if len(markerish_idxs) < 2:
        return None

    start = int(markerish_idxs[0])
    if start > 180:
        return None

    # A continuation page should begin close to the start of the spine item; if we have
    # already seen a substantial amount of ordinary prose, this is much more likely to be
    # a false positive from wrapped running text or TOC/index material.
    nonempty_before = 0
    long_before = 0
    for i in range(0, start):
        t = _safe_text(lines[i] or "")
        if not t:
            continue
        nonempty_before += 1
        if len(t) >= 60:
            long_before += 1
    if nonempty_before > 8 or long_before > 2:
        return None

    window_end = min(scan_end, start + 140)
    nonempty = 0
    markerish = 0
    long_defs = 0
    for i in range(start, window_end):
        t = _safe_text(lines[i] or "")
        if not t:
            continue
        nonempty += 1
        m_def = def_re.match(t)
        if m_def:
            markerish += 1
            body = _safe_text(m_def.group(2) or "").strip()
            if len(body) >= 28:
                long_defs += 1
            continue
        if marker_only_re.match(t):
            markerish += 1
            # For marker-only lines, use the next non-empty line as a proxy for body length.
            for j in range(i + 1, min(i + 6, window_end)):
                nxt = _safe_text(lines[j] or "")
                if not nxt:
                    continue
                if len(nxt) >= 28:
                    long_defs += 1
                break

    if markerish < 2 or nonempty <= 0:
        return None
    density = float(markerish) / float(nonempty)
    if density < 0.10:
        return None
    if long_defs < 2:
        return None
    return start


def looks_like_notes_continuation_page(lines: List[str]) -> bool:
    """Compatibility wrapper: True iff infer_notes_continuation_harvest_start(...) succeeds."""

    return infer_notes_continuation_harvest_start(lines) is not None


# ==========================================================
# Notes Definition Boundary Heuristics
# ==========================================================


def looks_like_post_notes_section_enumerator_line(line: str) -> bool:
    """True if `line` looks like a chapter-section enumerator like '(ii) ...'.

    Some critical editions place a NOTES block mid-spine and then continue the
    chapter with enumerated sections '(ii)', '(iii)', etc. When parsing note
    definitions, these lines should usually *not* be appended to the previous
    numeric note definition.

    This is intentionally conservative: it only matches roman numerals in
    parentheses at the start of the line, followed by a space and an uppercase
    letter.
    """

    t = _safe_text(line or "").strip()
    if not t:
        return False
    return bool(re.match(r"^\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s+[A-Z]", t))


def looks_like_false_single_letter_definition_restart(
    line: str,
    *,
    current_marker: str,
    current_text: str,
    matched_marker_raw: str,
    matched_body: str,
) -> bool:
    """True when a def-like line is more likely a wrapped continuation than a new note.

    Critical editions with numeric notes can wrap a clause onto a new line that
    begins with a single lowercase word, e.g.:

      captured the messenger -
      a monkey. How does Michael know?

    Since `_def_line_regex()` accepts single-letter note markers, wrapped prose can
    be misread as a new lettered definition. Keep this narrow: only suppress a
    restart when we are already inside a numeric note and the candidate looks like
    a prose continuation such as `a monkey...`, `A.`, or `C. Thus ...`.
    """

    cur = _safe_text(current_marker or "").strip()
    cur_text = (current_text or "").rstrip()
    raw = (matched_marker_raw or "").strip()
    body = (matched_body or "").lstrip()
    text = (line or "").lstrip()

    if not re.fullmatch(r"\d{1,3}", cur):
        return False
    if not re.fullmatch(r"[A-Za-z]", raw):
        return False
    if not text.startswith(raw):
        return False
    next_char = text[1] if len(text) > 1 else ""
    if next_char and next_char not in {" ", "\t", ".", ":", "-", "\u2014"}:
        return False

    # Existing lowercase wrapped-prose case: "a monkey. How does ..."
    if re.fullmatch(r"[a-z]", raw) and body and re.match(r"^[a-z]", body):
        return True

    # Dialogue/prose continuation case: a wrapped line can begin with a bare
    # uppercase pronoun like "I didn't..." and be misread as a lettered note.
    # Only allow this when the current numeric note appears unfinished.
    if raw == "I" and body and re.match(r"^[a-z]", body):
        if cur_text and not re.search(r"[\.!?]['\")\]]?$", cur_text):
            return True

    # Single-letter editorial abbreviations and sigla often continue numeric notes,
    # e.g. "manuscript C. Thus ...", "do not appear in A.", or "as in F (pp. ...)".
    if next_char == ".":
        if not cur_text:
            return False
        if re.search(r"[\.!?]['\")\]]?$", cur_text):
            return False
        if not body:
            return True
        if re.match(r"^(?:\d{1,3}\.|[A-Z]\.|[A-Z][a-z]|[a-z])", body):
            return True
        if re.fullmatch(r"[IVXLC]", raw, re.IGNORECASE):
            if re.search(r"\b(?:see|cf\.?|compare|in)$", cur_text, re.IGNORECASE) and re.match(r"^\d{1,3}(?:\s*[-,]\s*(?:\d{1,3})?)?\.?$", body):
                return True

    if next_char in {" ", "\t"} and re.fullmatch(r"[A-Z]", raw):
        if not cur_text:
            return False
        if re.search(r"[\.!?]['\")\]]?$", cur_text):
            return False
        if re.search(r"\b(?:in|as in|from|cf\.?|compare|see)$", cur_text, re.IGNORECASE) and (
            not body or body.startswith("(") or re.match(r"^(?:p{1,2}\.|\d)", body, re.IGNORECASE)
        ):
            return True

    return False


def looks_like_false_numeric_crossref_restart(
    line: str,
    *,
    current_marker: str,
    current_text: str,
    next_definition_marker: str,
) -> bool:
    """True when a marker-only numeric chunk is really a wrapped cross-reference.

    Example:

      50. ... Lewis's novels (see note
      13).
      51. In A it is Dolbear ...

    Here the isolated ``13).`` is part of the current note body, not a new
    definition. Keep this narrow: only trigger inside an existing numeric note,
    and only when the current note text clearly ends with a cross-reference cue.
    """

    cur = _safe_text(current_marker or "").strip()
    txt = _safe_text(current_text or "")
    nxt = _safe_text(next_definition_marker or "").strip()
    raw = _safe_text(line or "")

    if not re.fullmatch(r"\d{1,3}", cur):
        return False
    m = re.match(r"^(?:\(|\[)?\s*(\d{1,3})(?:\)|\])?[\)\.]?\s*(.*)$", raw)
    if not m:
        return False
    probe = m.group(1)
    tail = _safe_text(m.group(2) or "")

    if not txt or not re.search(r"(?:\b(?:see|cf\.?|compare)\s+note(?:s)?|\(?note(?:s)?)$", txt, re.IGNORECASE):
        return False
    try:
        cur_i = int(cur)
        probe_i = int(probe)
    except Exception:
        return False

    nxt_i = None
    if nxt and re.fullmatch(r"\d{1,3}", nxt):
        try:
            nxt_i = int(nxt)
        except Exception:
            nxt_i = None

    if probe_i >= cur_i:
        return False
    if nxt_i is not None and nxt_i <= cur_i:
        return False

    if tail and re.fullmatch(r"\d{1,3}", tail):
        return False
    return True


def looks_like_false_numeric_date_restart(
    line: str,
    *,
    current_marker: str,
    current_text: str,
    matched_marker_raw: str,
    matched_body: str,
) -> bool:
    """True when a def-like numeric line is really a wrapped calendar date.

    Example:

      ... in his letter to Tom Brady of
      4 March 1938 (Letters no. 26), my father had said ...

    Here the leading ``4`` is a day-of-month, not a new note marker.
    Keep this narrow: only trigger inside an existing numeric note when the
    current text clearly looks unfinished and the wrapped line starts with a
    month name.
    """

    cur = _safe_text(current_marker or "").strip()
    txt = _safe_text(current_text or "")
    raw = _safe_text(matched_marker_raw or "").strip()
    body = _safe_text(matched_body or "")

    if not re.fullmatch(r"\d{1,3}", cur):
        return False
    if not re.fullmatch(r"\d{1,2}", raw):
        return False
    if not body or not re.match(
        r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        body,
        re.IGNORECASE,
    ):
        return False
    if not txt or re.search(r"[\.!?]['\")\]]?$", txt):
        return False
    if not re.search(r"\b(?:of|on|dated|date|from)$", txt, re.IGNORECASE):
        return False

    probe = _safe_text(line or "")
    if not probe.startswith(raw):
        return False
    return True


def looks_like_false_numeric_editorial_reference_restart(
    line: str,
    *,
    current_marker: str,
    current_text: str,
    matched_marker_raw: str,
    matched_body: str,
) -> bool:
    """True when a numeric def-like line is really a wrapped editorial reference.

    Example:

      ... the account of Night
      68 begins with the words ...

    Here ``68`` is part of `Night 68`, not a new note definition.
    """

    cur = _safe_text(current_marker or "").strip()
    txt = _safe_text(current_text or "")
    raw = _safe_text(matched_marker_raw or "").strip()
    body = _safe_text(matched_body or "")

    if not re.fullmatch(r"\d{1,3}", cur):
        return False
    if not re.fullmatch(r"\d{1,3}", raw):
        return False
    if not body or not re.match(r"^[a-z]", body):
        return False
    if not re.search(r"\b(?:night|note|page|pages|part|chapter|draft|text)$", txt, re.IGNORECASE):
        return False

    try:
        if int(raw) >= int(cur):
            return False
    except Exception:
        return False

    probe = _safe_text(line or "")
    if not probe.startswith(raw):
        return False
    return True


def looks_like_false_numeric_age_restart(
    line: str,
    *,
    current_marker: str,
    current_text: str,
    matched_marker_raw: str,
    matched_body: str,
) -> bool:
    """True when a numeric def-like line is really a wrapped age/range phrase.

    Example:

      ... Ted Mason was born in 1956, and was now
      48 or 49.

    Here `48` and `49` are ages in prose, not note markers.
    """

    cur = _safe_text(current_marker or "").strip()
    txt = _safe_text(current_text or "")
    raw = _safe_text(matched_marker_raw or "").strip()
    body = _safe_text(matched_body or "")

    if not re.fullmatch(r"\d{1,3}", cur):
        return False
    if not re.fullmatch(r"\d{1,3}", raw):
        return False
    if re.search(r"\b(?:was|were|aged?|age|now|then)$", txt, re.IGNORECASE):
        if not body or not re.match(r"^(?:or\b|and\b|to\b|[-\u2013\u2014])", body, re.IGNORECASE):
            return False
    elif re.search(r"\b\d{1,3}\s+(?:or|and|to|[-\u2013\u2014])$", txt, re.IGNORECASE):
        if body and not re.match(r"^[A-Z\(']", body):
            return False
    else:
        return False

    try:
        probe_i = int(raw)
        cur_i = int(cur)
    except Exception:
        return False

    if probe_i >= cur_i:
        return False

    probe = _safe_text(line or "")
    if not probe.startswith(raw):
        return False
    return True


def looks_like_false_numeric_bibliographic_restart(
    line: str,
    *,
    current_marker: str,
    current_text: str,
    matched_marker_raw: str,
    matched_body: str,
) -> bool:
    """True when a numeric def-like line is really a wrapped bibliographic continuation.

    Example:

      45. Cf. my father's letter to T. B. Brady of 7 June 1955 (Letters no.
      163):

    Here ``163`` is part of ``Letters no. 163``, not a new note marker.
    """

    cur = _safe_text(current_marker or "").strip()
    txt = _safe_text(current_text or "")
    raw = _safe_text(matched_marker_raw or "").strip()
    body = _safe_text(matched_body or "")
    probe = _safe_text(line or "").strip()

    if not re.fullmatch(r"\d{1,3}", cur):
        return False
    if not re.fullmatch(r"\d{1,4}", raw):
        return False
    if not txt:
        return False
    if not re.search(r"\b(?:letters?|letter)\s+no\.?$", txt, re.IGNORECASE):
        return False
    if body and not re.fullmatch(r"[\)\]:;.,\s-]*", body):
        return False
    if not re.fullmatch(rf"{re.escape(raw)}(?:\)|\]|\:|\.)*\s*", probe):
        return False
    return True


# ==========================================================
# Anchor Heuristics
# ==========================================================
# Order within this section is intentionally:
#   types → registry → registered rules → entrypoint


# -----------------------
# Anchor Types
# -----------------------
@dataclass(frozen=True)
class AnchorHeuristicInput:
    """Input to an anchor heuristic rule.
    Heuristics should be pure (no side effects) and must not rely on other
    heuristics being run before/after them.
    """

    marker_raw: str
    marker_norm: str
    context: str
    has_href: bool


# Master list of anchor heuristics.
#
# Rule contract:
#   - Return False to reject the anchor as a footnote marker.
#   - Return None to abstain (no opinion).
#   - (Return True is allowed but currently unused; the default is accept.)
ANCHOR_HEURISTICS: List[Tuple[str, Callable[[AnchorHeuristicInput], Optional[bool]]]] = []


def register_anchor_heuristic(name: str) -> Callable[[Callable[[AnchorHeuristicInput], Optional[bool]]], Callable[[AnchorHeuristicInput], Optional[bool]]]:
    """Register an anchor heuristic component.

    Appends the decorated function into ANCHOR_HEURISTICS at import time.
    """

    def _decorator(fn: Callable[[AnchorHeuristicInput], Optional[bool]]):
        ANCHOR_HEURISTICS.append((name, fn))
        return fn

    return _decorator

# -----------------------
# Anchor Heuristic Rules
# -----------------------
@register_anchor_heuristic("reject_punctuation_asterisk")
def _h_reject_punctuation_asterisk(inp: AnchorHeuristicInput) -> Optional[bool]:
    if inp.marker_norm != "*":
        return None
    if _is_likely_punctuation_asterisk(inp.context):
        return False
    return None


@register_anchor_heuristic("reject_bracketed_single_letter")
def _h_reject_bracketed_single_letter(inp: AnchorHeuristicInput) -> Optional[bool]:
    raw = inp.marker_raw
    norm = inp.marker_norm
    if raw.startswith("[") and raw.endswith("]") and re.fullmatch(r"[A-Za-z]", norm):
        return False
    return None


@register_anchor_heuristic("reject_page_number_bracket_anchor")
def _h_reject_page_number_bracket_anchor(inp: AnchorHeuristicInput) -> Optional[bool]:
    if _is_likely_page_number_bracket_anchor(inp.context, inp.marker_norm, inp.marker_raw):
        return False
    return None


@register_anchor_heuristic("reject_paren_numeric_false_positive")
def _h_reject_paren_numeric_false_positive(inp: AnchorHeuristicInput) -> Optional[bool]:
    raw = inp.marker_raw
    norm = inp.marker_norm
    if not (raw.startswith("(") and raw.endswith(")") and re.fullmatch(r"\d{1,3}", norm)):
        return None
    # Use a wider probe window for numeric anchors: date/chronology false-positives
    # (e.g., "March 10") can be more than 35 chars away from the marker.
    probe = _citation_probe_context(raw, norm, inp.context, radius=140)
    if _is_likely_figure_label_anchor(probe, norm):
        return False
    if _is_likely_non_footnote_numeric_anchor(probe, norm):
        return False
    return None


@register_anchor_heuristic("reject_paren_letter_false_positive")
def _h_reject_paren_letter_false_positive(inp: AnchorHeuristicInput) -> Optional[bool]:
    raw = inp.marker_raw
    norm = inp.marker_norm
    if not (raw.startswith("(") and raw.endswith(")") and re.fullmatch(r"[a-zA-Z]", norm)):
        return None
    probe = _citation_probe_context(raw, norm, inp.context, radius=90)
    if _is_likely_parenthetical_letter_suffix(probe, norm, raw):
        return False
    if _is_likely_non_footnote_letter_anchor(probe, norm, raw):
        return False
    if _is_likely_letter_enumeration_list(probe, norm, raw):
        return False
    if _is_likely_citation_context(probe):
        return False
    return None


@register_anchor_heuristic("reject_bracket_numeric_false_positive")
def _h_reject_bracket_numeric_false_positive(inp: AnchorHeuristicInput) -> Optional[bool]:
    raw = inp.marker_raw
    norm = inp.marker_norm
    if not (raw.startswith("[") and raw.endswith("]") and re.fullmatch(r"\d{1,3}", norm)):
        return None
    probe = _citation_probe_context(raw, norm, inp.context, radius=140)
    if _is_likely_figure_label_anchor(probe, norm):
        return False
    if _is_likely_non_footnote_numeric_anchor(probe, norm):
        return False
    return None


@register_anchor_heuristic("reject_bracket_letter_citation_context")
def _h_reject_bracket_letter_citation_context(inp: AnchorHeuristicInput) -> Optional[bool]:
    raw = inp.marker_raw
    norm = inp.marker_norm
    if not (raw.startswith("[") and raw.endswith("]") and re.fullmatch(r"[a-zA-Z]", norm)):
        return None
    probe = _citation_probe_context(raw, norm, inp.context, radius=35)
    if _is_likely_citation_context(probe):
        return False
    return None


# -----------------------
# Anchor Entrypoint
# -----------------------
def anchor_is_probable_footnote(marker_raw: str, marker_norm: str, context: str, *, has_href: bool = False) -> bool:
    """ENTRYPOINT: Decide whether to keep a candidate marker as a footnote anchor.

    engine.py calls this to filter candidate markers like (3), [6], *, etc.
    Internally this runs the registered anchor heuristic components in ANCHOR_HEURISTICS.
    """

    # If the EPUB explicitly links (noteref/id_link), trust it.
    if has_href:
        return True

    raw = (marker_raw or "").strip()
    norm = (marker_norm or "").strip()

    if not norm:
        return False

    inp = AnchorHeuristicInput(
        marker_raw=raw,
        marker_norm=norm,
        context=_safe_text(context)[:4000],
        has_href=bool(has_href),
    )

    for _, rule in ANCHOR_HEURISTICS:
        try:
            decision = rule(inp)
        except Exception:
            # Heuristics should never be able to crash extraction.
            decision = None
        if decision is False:
            return False
        if decision is True:
            return True

    # Default to accept if nothing rejected.
    return True


# ==========================================================
# Marker Profile Heuristics
# ==========================================================


def _infer_marker_profile_heuristic(text: str) -> str:
    """Heuristically infer numeric/symbol/letter dominance.

    Used to choose which marker families to trust when extracting and pairing.

    Real footnote definitions appear in dense clusters (e.g. "1. ...", "2. ...",
    "3. ..." in a notes section). Isolated matches like "§ 177" (section headers
    in prose) are ignored so they don't pollute the family counts.
    """

    sample = (text or "")[:200_000]
    if not sample:
        return "auto_heur"

    lines = sample.split("\n")
    def_re = _def_line_regex()

    # Collect all definition-line matches with their line indices.
    all_matches: List[Tuple[int, str]] = []
    for i, line in enumerate(lines[:6000]):
        m = def_re.match(line or "")
        if not m:
            continue
        raw_marker = (m.group(1) or "").strip()
        if not raw_marker:
            continue
        cat = _marker_category_from_raw(raw_marker)
        if cat.startswith("num_"):
            family = "numeric"
        elif cat.startswith("let_"):
            family = "letter"
        elif cat == "symbol":
            family = "symbol"
        else:
            continue
        all_matches.append((i, family))

    # Also scan the last portion — notes sections are often at the end.
    end_start = max(0, len(lines) - 3000)
    for i in range(end_start, len(lines)):
        line = lines[i] if i < len(lines) else ""
        m = def_re.match(line or "")
        if not m:
            continue
        raw_marker = (m.group(1) or "").strip()
        if not raw_marker:
            continue
        cat = _marker_category_from_raw(raw_marker)
        if cat.startswith("num_"):
            family = "numeric"
        elif cat.startswith("let_"):
            family = "letter"
        elif cat == "symbol":
            family = "symbol"
        else:
            continue
        all_matches.append((i, family))

    # Only count matches that appear in clusters — 3+ matches within a 50-line
    # window. This filters out isolated section markers ("§ 177") scattered
    # through prose while keeping genuine footnote definition clusters.
    CLUSTER_WINDOW = 50
    MIN_CLUSTER_SIZE = 4

    if not all_matches:
        return "auto_heur"

    # Sort by line index, deduplicate same-line matches.
    all_matches.sort(key=lambda x: x[0])
    unique: List[Tuple[int, str]] = []
    seen_lines: set[int] = set()
    for li, fam in all_matches:
        if li not in seen_lines:
            seen_lines.add(li)
            unique.append((li, fam))

    in_cluster: set[int] = set()
    for j in range(len(unique)):
        line_idx = unique[j][0]
        cluster_count = sum(1 for mi, _ in unique if line_idx <= mi < line_idx + CLUSTER_WINDOW)
        if cluster_count >= MIN_CLUSTER_SIZE:
            for k in range(j, len(unique)):
                if unique[k][0] < line_idx + CLUSTER_WINDOW:
                    in_cluster.add(k)
                else:
                    break

    def_counts: Dict[str, int] = defaultdict(int)
    for j in in_cluster:
        def_counts[unique[j][1]] += 1

    total = sum(def_counts.values())
    if total <= 0:
        return "auto_heur"

    top_family, top_count = max(def_counts.items(), key=lambda kv: kv[1])
    if total >= 8 and (top_count / max(1, total)) >= 0.70:
        return top_family
    return "auto_heur"


# ==========================================================
# Notes Split Heuristics
# ==========================================================
# Order within this section is intentionally:
#   types → registry → registered rules → entrypoint


# -----------------------
# Notes Split Types
# -----------------------


@dataclass(frozen=True)
class NotesSplitInput:
    """Input to notes-block split heuristics."""

    lines: List[str]


@dataclass(frozen=True)
class NotesSplitResult:
    """Represents a proposed split between main text and notes definitions."""

    main_end_index: int
    defs_start_index: int


NOTES_SPLIT_HEURISTICS: List[Tuple[str, Callable[[NotesSplitInput], Optional[NotesSplitResult]]]] = []


def register_notes_split_heuristic(name: str) -> Callable[[Callable[[NotesSplitInput], Optional[NotesSplitResult]]], Callable[[NotesSplitInput], Optional[NotesSplitResult]]]:
    """Register a notes-split heuristic component.

    Appends the decorated function into NOTES_SPLIT_HEURISTICS at import time.
    """

    def _decorator(fn: Callable[[NotesSplitInput], Optional[NotesSplitResult]]):
        NOTES_SPLIT_HEURISTICS.append((name, fn))
        return fn

    return _decorator

# ---------------------------
# Notes Split Heuristic Rules
# ---------------------------

@register_notes_split_heuristic("notes_header")
def _ns_notes_header(inp: NotesSplitInput) -> Optional[NotesSplitResult]:
    lines = inp.lines
    if not lines:
        return None
    marker_only_re = re.compile(
        r"^\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|\*|†|‡|§)\s*(?:\]|\))?\s*(?:[\]\)\.:\-—]\s*)?$",
        re.UNICODE,
    )

    # Accept common headers and chapter-scoped variants:
    #   NOTES
    #   NOTES.
    #   NOTES TO CHAPTER VI
    #   NOTES ON CHAPTER 6
    header_re = re.compile(
        r"^\s*(FOOTNOTES|FOOTNOTES\s+AND\s+ENDNOTES|ENDNOTES|NOTES)(?:\s+(?:TO|ON)\s+CHAPTER\b.*)?\s*[:\.]?\s*$",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines):
        if header_re.match(line or ""):
            # Find the first non-empty line after the header.
            defs_start = None
            for j in range(i + 1, min(i + 30, len(lines))):
                if _safe_text(lines[j] or ""):
                    defs_start = j
                    break
            if defs_start is None:
                continue

            # Must actually look like a definition line or a marker-only line.
            m0 = _def_line_regex().match(lines[defs_start] or "")
            mo0 = marker_only_re.match(lines[defs_start] or "")
            if not m0 and not mo0:
                continue

            # Require a small cluster of definition-like or marker-only lines nearby.
            look_end = min(len(lines), defs_start + 160)
            def_like = [k for k in range(defs_start, look_end) if _def_line_regex().match(lines[k] or "")]
            marker_only = [k for k in range(defs_start, look_end) if marker_only_re.match(lines[k] or "")]
            if len(def_like) < 2 and len(marker_only) < 2 and (len(def_like) + len(marker_only)) < 3:
                continue

            # Guard against Table-of-Contents/outline pages that include a NOTES header.
            # If the NOTES header appears early, require that the first marker is referenced
            # in the preceding text (e.g., "(1)" or "[1]") so we don't treat outline numbering
            # as a notes block.
            mk0 = (m0.group(1) or "").strip() if m0 else (mo0.group(1) or "").strip() if mo0 else ""
            if mk0:
                before = "\n".join(lines[:i])
                referenced = (f"({mk0})" in before) or (f"[{mk0}]" in before)
                if not referenced and i < int(len(lines) * 0.50):
                    continue

            return NotesSplitResult(main_end_index=i, defs_start_index=defs_start)
    return None


@register_notes_split_heuristic("generic_header_before_definition_cluster")
def _ns_generic_header_before_definition_cluster(inp: NotesSplitInput) -> Optional[NotesSplitResult]:
    """Detect notes split even when the header word isn't NOTES/ENDNOTES.

    Many books use headings like "ANNOTATIONS", "COMMENTARY", "EXPLANATORY NOTES",
    or language-specific variants. Structurally, the notes section is typically a
    short standalone heading line followed by a dense cluster of definition lines
    like "1. ...", "2. ...", etc.
    """

    lines = inp.lines
    if not lines:
        return None

    def _is_header_candidate(s: str) -> bool:
        t = _safe_text(s)
        if not t:
            return False

        # Keep it short and standalone-ish.
        if len(t) > 60:
            return False

        # Avoid picking a definition line itself.
        if _def_line_regex().match(t):
            return False

        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) < 4:
            return False

        # If it's mostly ALL CAPS, that's a strong header signal.
        upper = sum(1 for ch in letters if ch.isupper())
        upper_ratio = upper / max(1, len(letters))

        # Or Title-Case-ish ("Explanatory Notes").
        words = [w for w in re.split(r"\s+", t.strip(" .:\t")) if w]
        titleish = False
        if 1 <= len(words) <= 6:
            alpha_words = 0
            caps_words = 0
            for w in words:
                w2 = re.sub(r"[^A-Za-z]", "", w)
                if not w2:
                    continue
                alpha_words += 1
                if w2[0].isupper():
                    caps_words += 1
            if alpha_words >= 1 and (caps_words / max(1, alpha_words)) >= 0.7:
                titleish = True

        if not (upper_ratio >= 0.80 or titleish):
            return False

        return True

    # Find a plausible start of a definitions cluster.
    start_scan = max(0, int(len(lines) * 0.35))
    def_idxs = [i for i in range(start_scan, len(lines)) if _def_line_regex().match(lines[i] or "")]
    if len(def_idxs) < 2:
        return None

    # Pick the first index that is part of a small cluster.
    first_def = None
    for i in def_idxs:
        # Need at least 2 def-like lines in the next 80 lines.
        near = [j for j in def_idxs if j >= i and j <= i + 80]
        if len(near) >= 2:
            first_def = i
            break
    if first_def is None:
        return None

    # Search backwards from first_def for a header line.
    # Prefer the closest header candidate above the cluster.
    for i in range(first_def - 1, max(-1, first_def - 80), -1):
        if i < 0:
            break
        t = _safe_text(lines[i] or "")
        if not t:
            continue
        if not _is_header_candidate(t):
            continue

        # Require that the next non-empty line is a definition line.
        for j in range(i + 1, min(i + 10, len(lines))):
            if not _safe_text(lines[j] or ""):
                continue
            if _def_line_regex().match(lines[j] or ""):
                return NotesSplitResult(main_end_index=i, defs_start_index=j)
            break

    return None


@register_notes_split_heuristic("tail_definition_cluster")
def _ns_tail_definition_cluster(inp: NotesSplitInput) -> Optional[NotesSplitResult]:
    lines = inp.lines
    if not lines:
        return None

    # Scan a bit earlier than the extreme tail: some segmented spine items place
    # notes around ~60% into the file. We keep this conservative via a cluster
    # size requirement and an anchor-reference guard below.
    tail_start = max(0, int(len(lines) * 0.6))
    tail = lines[tail_start:]
    if not tail:
        return None

    def_like = [j for j, l in enumerate(tail) if _def_line_regex().match(l or "")]
    if len(def_like) >= 3:
        first_def = tail_start + def_like[0]

        # Guard against false positives from ordinary numbered lists: require that
        # the first marker appears earlier in the text as an anchor reference.
        try:
            m0 = _def_line_regex().match(lines[first_def] or "")
            mk0 = (m0.group(1) if m0 else "")
            mk0 = (mk0 or "").strip()
            if mk0:
                before0 = "\n".join(lines[:first_def])
                if mk0 and (f"({mk0})" not in before0 and f"[{mk0}]" not in before0):
                    return None
        except Exception:
            return None

        return NotesSplitResult(main_end_index=first_def, defs_start_index=first_def)
    return None


@register_notes_split_heuristic("notes_continuation_page")
def _ns_notes_continuation_page(inp: NotesSplitInput) -> Optional[NotesSplitResult]:
    """Detect notes blocks on continuation pages without an explicit NOTES header.

    Some EPUBs split endnotes across spine items. In those continuation pages, the
    first definition marker is often *not* referenced earlier in the same document
    (the anchors are in the previous spine item), so anchor-reference guards used
    by other split heuristics can incorrectly return None.

    We reuse `infer_notes_continuation_harvest_start(...)` (which is intentionally
    conservative) and surface it as a split result so downstream scanners can:
      - parse definitions from the start of the notes run
      - still detect bounded notes-in-the-middle cases (notes followed by prose)
    """

    lines = inp.lines
    if not lines:
        return None

    start = infer_notes_continuation_harvest_start(lines)
    if start is None:
        return None

    s = int(start)
    if s < 0 or s >= len(lines):
        return None

    return NotesSplitResult(main_end_index=s, defs_start_index=s)


@register_notes_split_heuristic("strict_numeric_cluster")
def _ns_strict_numeric_cluster(inp: NotesSplitInput) -> Optional[NotesSplitResult]:
    """More aggressive split inference when there is no explicit NOTES header."""

    lines = inp.lines
    if not lines:
        return None

    # Strict pattern for numeric definition lines; avoids matching dates like "31 January".
    strict_def = re.compile(r"^\s*(?:\[|\()?\s*(\d{1,3})\s*(?:\]|\))?\s*[\]\)\.:\-—]\s*(.+?)\s*$", re.UNICODE)
    # Start a bit earlier than mid-file: many critical editions place notes near
    # the end of a chapter but well before the 45% mark in short/segmented spine
    # items. We keep strong guards below to avoid false positives from ordinary
    # numbered lists.
    start_scan = max(0, int(len(lines) * 0.30))
    idxs = [i for i in range(start_scan, len(lines)) if strict_def.match(lines[i] or "")]
    if not idxs:
        return None

    # Guard against false positives from ordinary numbered lists in the running text.
    # If the first strict def-like marker is never referenced earlier as an anchor,
    # it's much less likely this is a notes section.
    first_i = idxs[0]
    first_m = strict_def.match(lines[first_i] or "")
    if first_m:
        mk0 = (first_m.group(1) or "").strip()
        before0 = "\n".join(lines[:first_i])
        if mk0 and (f"({mk0})" not in before0 and f"[{mk0}]" not in before0):
            return None

    # If we only found one strict def-like line, accept it iff its marker is referenced earlier.
    if len(idxs) == 1:
        i = idxs[0]
        m = strict_def.match(lines[i] or "")
        if not m:
            return None
        mk = m.group(1)
        before = "\n".join(lines[:i])
        if mk and (f"({mk})" in before or f"[{mk}]" in before):
            return NotesSplitResult(main_end_index=i, defs_start_index=i)
        return None

    # Pick the first index that is part of a small cluster (>=2 additional defs within 200 lines).
    for i in idxs:
        near = [j for j in idxs if j >= i and (j - i) <= 200]
        if len(near) >= 3:
            return NotesSplitResult(main_end_index=i, defs_start_index=i)

    return NotesSplitResult(main_end_index=idxs[0], defs_start_index=idxs[0])


# -----------------------
# Notes Split Entrypoint
# -----------------------
def infer_notes_split(lines: List[str]) -> Optional[NotesSplitResult]:
    """ENTRYPOINT: Find (main_end_index, defs_start_index) for notes parsing.

    engine.py calls this to decide where main text ends and notes begin.
    """

    inp = NotesSplitInput(lines=list(lines or []))
    for _, rule in NOTES_SPLIT_HEURISTICS:
        try:
            res = rule(inp)
        except Exception:
            res = None
        if res is not None:
            return res
    return None


# -----------------------
# Heuristic Introspection
# -----------------------
def list_registered_anchor_heuristics() -> List[str]:
    """Return the currently registered anchor heuristic component names."""

    return [name for name, _ in ANCHOR_HEURISTICS]


def list_registered_notes_split_heuristics() -> List[str]:
    """Return the currently registered notes-split heuristic component names."""

    return [name for name, _ in NOTES_SPLIT_HEURISTICS]


# -----------------------
# Loose Helper Rules
# -----------------------

# These helpers are intentionally NOT registered. They are called internally by the
# registered component rules above (which run via the two engine entrypoints).
#
# Note: they do not need to be imported by engine.py; they live in this module and
# are referenced by the registered rule functions when those rules execute.


# -----------------------
# Text Stats
# -----------------------

# Helper for letter-case analysis in citation/TOC detection.
def _letter_case_stats(text: str) -> Tuple[int, int]:
    upper = 0
    lower = 0
    for ch in text or "":
        if ch.isalpha():
            if ch.isupper():
                upper += 1
            elif ch.islower():
                lower += 1
    return upper, lower



# -----------------------
# Citation Context
# -----------------------

def _is_likely_citation_context(context: str) -> bool:
    """Heuristic: filter out scholarly/TOC references that look like footnotes but aren't.

    Critical editions commonly contain:
      - ALL-CAPS contents/outline lines with (1), (2), etc.
      - scholarly citations like 'VII.212', 'TT pp. 186-7', 'Ch. II'
    """
    ctx = _safe_text(context)
    if not ctx:
        return True

    upper, lower = _letter_case_stats(ctx)
    letters = upper + lower
    if letters >= 20:
        upper_ratio = upper / max(1, letters)
        # Strong signal for contents/outline lines.
        if upper_ratio >= 0.75 and lower <= 3:
            return True

    # Lots of section numbering like "III." "IV." etc.
    if len(re.findall(r"\b[IVX]{1,6}\.", ctx)) >= 2:
        return True

    # Scholarly citation patterns.
    if re.search(r"\bpp?\.?\s*\d", ctx, re.IGNORECASE):
        return True
    if re.search(r"\bch\.?\s*[IVX\d]", ctx, re.IGNORECASE):
        return True

    if re.search(r"\b[IVX]{1,6}\.\d{1,4}\b", ctx):
        return True
    if re.search(r"\bTT\b|\bRK\b|\bLR\b", ctx):
        # Abbreviated volume refs often coincide with citations.
        if re.search(r"\d", ctx):
            return True

    return False



def _citation_probe_context(marker_raw: str, marker_norm: str, context: str, *, radius: int = 35) -> str:
    """Return a small context window around the marker for citation heuristics."""
    ctx = _safe_text(context)
    if not ctx:
        return ctx

    raw = (marker_raw or "").strip()
    norm = (marker_norm or "").strip()

    candidates: list[str] = []
    if raw:
        candidates.append(raw)
    if norm:
        candidates.extend([f"({norm})", f"[{norm}]", norm])

    for pat in candidates:
        if not pat:
            continue
        idx = ctx.find(pat)
        if idx != -1:
            start = max(0, idx - radius)
            end = min(len(ctx), idx + len(pat) + radius)
            return ctx[start:end]

    return ctx[: max(40, radius * 2)]


# -----------------------
# Numeric Anchors
# -----------------------

def _is_likely_non_footnote_numeric_anchor(context: str, marker_norm: str) -> bool:
    """Heuristic for rejecting numeric anchors like "(8)" that are likely *not* footnotes.

    Important: scholarly prose can contain both citations ("TT p. 150") and true note markers
    ("(8)") in the same sentence. We should not reject the anchor just because a citation
    occurs nearby; instead only reject when the citation explicitly references the same number
    (e.g., "p. 8"), or when the surrounding text looks like TOC/outline material.
    """
    ctx = _safe_text(context)
    if not ctx:
        return True

    # Reuse the strongest outline/TOC signals.
    upper, lower = _letter_case_stats(ctx)
    letters = upper + lower
    if letters >= 20:
        upper_ratio = upper / max(1, letters)
        if upper_ratio >= 0.75 and lower <= 3:
            return True

    if len(re.findall(r"\b[IVX]{1,6}\.", ctx)) >= 2:
        return True

    n = (marker_norm or "").strip()
    if not re.fullmatch(r"\d{1,3}", n):
        return False

    # Footnotes numbered zero are not used in these critical-edition sources;
    # bare "(0)" markers almost always come from grammar/object-case notation.
    if n == "0":
        return True

    # Dictionary/lexicon entry sense markers, e.g.:
    #   "... English Dialect Dictionary ... Break 27 (3), ..."
    # Here "(3)" is a sense number for entry 27, not a footnote.
    # Keep this conservative by requiring nearby dictionary/editorial keywords.
    try:
        low = ctx.lower()
        dictish = (
            "dictionary" in low
            or "dialect" in low
            or "dict." in low
            or " lexicon" in low
            or "glossary" in low
            or "ed." in low
            or " ed " in low
            or " ed," in low
            or "ed," in low
        )
        if dictish:
            # lemma + entry-number + (sense-number)
            if re.search(rf"\b[A-Za-z][A-Za-z'’\-]{{1,24}}\s+\d{{1,3}}\s*\(\s*{re.escape(n)}\s*\)", ctx):
                return True
            # Also allow abbreviated lemma forms like "Br." rarely seen in notes.
            if re.search(rf"\b[A-Za-z]{{1,4}}\.?\s+\d{{1,3}}\s*\(\s*{re.escape(n)}\s*\)", ctx):
                return True
    except Exception:
        pass

    # Reject if the same number is clearly used as a page/chapter reference.
    if re.search(rf"\bpp?\.?\s*{re.escape(n)}\b", ctx, re.IGNORECASE):
        return True
    if re.search(rf"\bch\.?\s*{re.escape(n)}\b", ctx, re.IGNORECASE):
        return True

    # Chapter-title citations like:
    #   "... see especially the chapter 'Many Roads Lead Eastward (1)' ..."
    # These are not footnote anchors.
    try:
        n_esc = re.escape(n)
        if re.search(
            rf"\bchapter\b[^\n]{{0,30}}['\"\u2018\u2019\u201C\u201D][^\n]{{0,120}}\(\s*{n_esc}\s*\)",
            ctx,
            re.IGNORECASE,
        ):
            return True
    except Exception:
        pass

    # Year / chronology false-positives:
    #   "... the older date (465) for his birth ..."
    # Parenthesized numbers that are year references, not footnote markers.
    #
    # Be conservative: common words like "year", "born" also appear near
    # genuine footnotes (e.g. "New Year': see p. 46.(12)" or
    # "where I was born.(16) Anyone come with me?"). Only apply when there
    # are multiple corroborating signals.
    try:
        low = ctx.lower()
        n_int = int(n)
        if n_int >= 10 and re.search(rf"\(\s*{re.escape(n)}\s*\)", low):
            n_esc2 = re.escape(n)

            # "older date (N)" / "date of (N)" — explicit date disambiguation.
            if re.search(rf"\b(?:older\s+date|date\s+of)\b[^\n]{{0,60}}\(\s*{n_esc2}\s*\)", low):
                return True

            # "born in (N)" — explicit birth-year phrasing.
            if re.search(rf"\bborn\s+in\s*\(\s*{n_esc2}\s*\)", low):
                return True

            # "the year NNN" / "the year (NNN)" — but NOT "New Year".
            # Use a short window (20 chars) to avoid matching distant footnotes.
            if re.search(
                rf"\b(?<!\bnew\s)(?:the\s+)?year\b[^\n]{{0,20}}\(\s*{n_esc2}\s*\)",
                low,
            ):
                return True

            # Numbers >= 1000 in parens near date/chronology language
            # are almost certainly years, not footnote markers.
            if n_int >= 1000:
                chrono_cues = [
                    r"\bolder\s+date\b", r"\bdate\b", r"\byear\b",
                    r"\bborn\b", r"\bbirth\b", r"\bdied\b", r"\bdeath\b",
                    r"\bcentury\b",
                ]
                if any(re.search(cue, low) for cue in chrono_cues):
                    return True

            # Numbers >= 300 in parens near explicit date-of-birth/death
            # language (not just generic date/year words).
            if n_int >= 300:
                strong_chrono = [
                    r"\bolder\s+date\b", r"\bdate\s+of\b",
                    r"\b(?:was|were|is|be)\s+born\b",
                    r"\b(?:was|were|is|be)\s+died\b",
                ]
                if any(re.search(cue, low) for cue in strong_chrono):
                    return True

            # "born" / "birth" / "died" / "death" near paren number,
            # but only when another bare year-like number (>= 100) is present.
            life_cues = [r"\bborn\b", r"\bbirth\b", r"\bdied\b", r"\bdeath\b"]
            if any(re.search(cue, low) for cue in life_cues):
                all_nums = set(re.findall(r"\b[1-9]\d{2,3}\b", low))
                page_ref_nums = set(re.findall(r"\bpp?\.\s*([1-9]\d{2,3})\b", low))
                paren_nums3 = set(re.findall(r"\(\s*([1-9]\d{2,3})\s*\)", low))
                bare_years = all_nums - page_ref_nums - paren_nums3
                if bare_years:
                    return True
    except Exception:
        pass

    # Date-list / chronology patterns (common in critical editions):
    #   "... dates in the month of March. (9) ... on 9th ..."
    #   "... evening of March 10. ..." (no ordinal suffix)
    # Here "(9)" / "(10)" is a day number label, not a footnote.
    try:
        low = ctx.lower()
        month_re = r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
        if re.search(month_re, low):
            has_marker = bool(re.search(rf"\(\s*{re.escape(n)}\s*\)", low))
            if has_marker:
                # If the same day appears as an ordinal near month/date language.
                if ("date" in low or "dates" in low or "day" in low or "month" in low) and re.search(
                    rf"\b{re.escape(n)}(?:st|nd|rd|th)\b", low
                ):
                    return True

                # Also handle plain "March 10" / "10 March" and variants like "of March 10".
                # IMPORTANT: do not let the marker itself satisfy the day-number test.
                # We only want to match a day number that appears *as plain text*, not
                # the parenthesized footnote marker like "(2)".
                day_token = rf"(?<!\()\b{re.escape(n)}\b(?!\))"
                day_near_month = bool(
                    re.search(rf"(?:\bof\s+)?{month_re}[^\n]{{0,12}}{day_token}", low)
                    or re.search(rf"{day_token}[^\n]{{0,12}}{month_re}", low)
                )
                if day_near_month:
                    # Strengthen signal: this is prose about dates/timing.
                    if re.search(r"\b(morning|evening|dusk|dawn|night|noon|afternoon|camp(?:s|ed)?|cross(?:es|ed)?)\b", low):
                        return True

                    # Or: multiple day numbers appear in the same window (chronology lists).
                    day_re = r"\b(?:[1-9]|[12]\d|3[01])\b"
                    if len(re.findall(day_re, low)) >= 3:
                        return True
    except Exception:
        pass

    # Grammar/outline enumerations, e.g.:
    #   "There were four tenses: (1) aorist ... (2) continuative ..."
    #   "(i) as the object ... Subjective (S) or Objective (0) ..."
    # These are list labels, not footnotes.
    try:
        low = ctx.lower()
        paren_nums = re.findall(r"\(\s*(\d{1,3})\s*\)", ctx)
        grammar_cues = [
            "tenses",
            "tense",
            "aorist",
            "continuative",
            "past tense",
            "present",
            "objective",
            "subjective",
            "compound expressions",
            "object of a verb",
            "verb",
            "verbs",
            "case",
            "cases",
            "form",
            "forms",
            "language",
            "adunaic",
        ]
        if len(paren_nums) >= 2 and len(set(paren_nums)) >= 2:
            if any(cue in low for cue in grammar_cues):
                return True
            if re.search(r":\s*\(\s*\d{1,3}\s*\)", ctx):
                return True
            if re.search(r"\([ivx]{1,6}\)", low):
                return True
        if any(cue in low for cue in grammar_cues):
            if re.search(r":\s*\(\s*" + re.escape(n) + r"\s*\)\s+[a-z]", low):
                return True
            if re.search(rf"\(\s*{re.escape(n)}\s*\)\s+(?:aorist|continuative|present|past|future|plural|singular|dual)\b", low):
                return True
    except Exception:
        pass

    return False


def _is_likely_figure_label_anchor(context: str, marker_norm: str) -> bool:
    """Detect figure/illustration labels that look like footnote anchors."""
    ctx = _safe_text(context).strip()
    n = (marker_norm or "").strip()

    # Basic guard: marker must be a 1-3 digit number
    if not ctx or not re.fullmatch(r"\d{1,3}", n):
        return False

    low = ctx.lower()

    # Broader cue words, including plural and adjectival forms.
    # Keep these intentionally editorial/figure-ish to avoid rejecting normal prose.
    cue_words = [
        "picture",
        "pictures",
        "sketch",
        "sketches",
        "drawing",
        "drawings",
        "plate",
        "plates",
        "figure",
        "figures",
        "fig.",
        "illustration",
        "illustrations",
        "illustrated",
        "map",
        "maps",
        "reproduced",
        "reproduction",
        "reproductions",
    ]

    # Require that the marker itself appears as a parenthetical number in context.
    if not re.search(rf"\(\s*{re.escape(n)}\s*\)", ctx):
        return False

    cue_re = rf"(?:{'|'.join(re.escape(w) for w in cue_words)})"

    # Strong: explicit cue immediately introducing the marker.
    #   "figure (4)", "fig. (4)", "drawings (4)", etc.
    if re.search(rf"\b{cue_re}\b\s*\(\s*{re.escape(n)}\s*\)", low, re.IGNORECASE):
        return True

    # Strong: cue word occurs nearby (editorial list-of-figures style).
    # Examples:
    #   "illustrated ... drawings 'Melbourne (3)' and '(4)'"
    #   "see plate ... (4)"
    if re.search(rf"\b{cue_re}\b", low, re.IGNORECASE):
        # If multiple parenthetical numbers appear in the same snippet, it's
        # very likely a list of drawings/plates/figures, not footnote markers.
        nums = set(re.findall(r"\(\s*(\d{1,3})\s*\)", ctx))
        if len(nums) >= 2:
            return True

        # Quoted label containing a numbered figure name.
        if re.search(rf"['\"\u2018\u2019\u201c\u201d][^'\"\u2018\u2019\u201c\u201d]{{0,80}}\(\s*{re.escape(n)}\s*\)[^'\"\u2018\u2019\u201c\u201d]{{0,80}}['\"\u2018\u2019\u201c\u201d]", ctx):
            return True

        # Proper-noun label patterns ("Melbourne (3)") often accompany figure references.
        # If we see a proper-noun figure label *near* this marker and any cue word,
        # treat the marker as a figure label too.
        if re.search(r"\b[A-Z][a-z]{1,20}\s*\(\s*\d{1,3}\s*\)", ctx):
            return True

    # Legacy: single proper noun directly followed by this marker ("Melbourne (4)").
    # This suppresses a lot of non-footnote numbering in critical editions, but it
    # can also hide legitimate anchors in running prose (e.g. "Ransom (12) got...").
    #
    # Keep this rejection only for *caption/list-style* usage where the marker ends
    # the label (end of string or punctuation), not when a normal word continues.
    m = re.search(rf"\b[A-Z][a-z]{{1,20}}\s*\(\s*{re.escape(n)}\s*\)", ctx)
    if m:
        tail = (ctx[m.end() :] or "").lstrip()
        if not tail:
            return True
        # If a normal word follows, assume this is prose and keep it.
        if tail[:1].isalpha():
            return False
        # Otherwise treat as a label-like reference.
        if tail[:1] in {",", ".", ";", ":", "!", "?", ")", "]", "}"}:
            return True

    return False


def _is_likely_page_number_bracket_anchor(context: str, marker_norm: str, marker_raw: str) -> bool:
    """Detect bracketed numbers like '[6]' used as manuscript page-number references."""
    raw = (marker_raw or "").strip()
    norm = (marker_norm or "").strip()
    if not (raw.startswith("[") and raw.endswith("]") and re.fullmatch(r"\d{1,3}", norm)):
        return False

    ctx = _safe_text(context).lower()
    if not ctx:
        return False

    keywords = [
        "page-number",
        "page number",
        "duly numbered",
        "numbered",
        "recto",
        "verso",
        "manuscript",
        "folio",
        "bodleian",
        "marquette",
        "preserved",
    ]
    if any(k in ctx for k in keywords):
        return True
    if "following" in ctx and raw.strip("[]") in ctx:
        return True
    return False



# -----------------------
# Letter Anchors
# -----------------------

def _is_likely_non_footnote_letter_anchor(context: str, marker_norm: str, marker_raw: str) -> bool:
    """Detect structured appendix/scheme labels like '(B)' '(C)' '(D)' used in prose.

    In analysis-critical style texts, parenthetical uppercase letters are often *text/draft*
    labels (A/B fair copy variants, openings, drafts), not footnote markers.
    This stays conservative: it only triggers for single UPPERCASE letters in parentheses
    and requires strong nearby keywords/patterns.
    """
    ctx = _safe_text(context)
    norm = (marker_norm or "").strip()
    raw = (marker_raw or "").strip()
    if not ctx or not re.fullmatch(r"[A-Z]", norm):
        return False
    if not (raw.startswith("(") and raw.endswith(")")):
        return False

    ctx_l = ctx.lower()

    # Strong signal: explicit draft/text-version wording immediately around '(B)'.
    # Examples:
    #   "new opening (B)"
    #   "fair copy (B)"
    #   "draft (B)"
    #   "text (B)"
    version_keywords = [
        "opening",
        "draft",
        "text",
        "copy",
        "fair copy",
        "typescript",
        "ms",
        "manuscript",
        "revision",
    ]
    kw_pat = "|".join([re.escape(k) for k in version_keywords])
    if re.search(rf"\b(?:{kw_pat})\b\s*\(\s*{re.escape(norm.lower())}\s*\)", ctx_l):
        return True

    # Also common: lettered variants discussed inline, e.g. "text A ... opening (B) ...".
    # If multiple variant letters appear nearby, treat as a scheme rather than footnotes.
    if len(re.findall(r"\btext\s+[a-z]\b", ctx_l)) >= 1 and len(re.findall(r"\([a-z]\)", ctx_l)) >= 2:
        return True

    if "scheme" in ctx_l and re.search(rf"\bscheme\s+{re.escape(norm.lower())}\b", ctx_l):
        return True

    if re.search(r":\s*\(" + re.escape(norm) + r"\)(?=\s|$)", ctx):
        return True
    if re.search(
        r"\b(it continues|he wrote|she wrote|my father wrote)\s*:\s*\(" + re.escape(norm.lower()) + r"\)",
        ctx_l,
    ):
        return True

    if len(re.findall(r"\([A-Z]\)", ctx)) >= 2:
        return True
    return False


# Heuristic for rejecting letter anchors that look like enumeration labels rather than footnotes.
def _is_likely_letter_enumeration_list(context: str, marker_norm: str, marker_raw: str) -> bool:
    """Detect prose enumerations like '(a) ... (b) ...' (not footnotes)."""
    norm = (marker_norm or "").strip()
    raw = (marker_raw or "").strip()
    if not (re.fullmatch(r"[a-z]", norm) and raw.startswith("(") and raw.endswith(")")):
        return False

    ctx = _safe_text(context)
    if not ctx:
        return False
    ctx_l = ctx.lower()

    letters = re.findall(r"\(([a-z])\)", ctx_l)
    if len(letters) >= 2 and len(set(letters)) >= 2:
        return True
    if re.search(r"\([a-z]\)\s+to\b", ctx_l) and len(set(letters)) >= 2:
        return True
    return False


def _is_likely_parenthetical_letter_suffix(context: str, marker_norm: str, marker_raw: str) -> bool:
    """Detect patterns like 'Name(e)' which are not footnotes."""
    raw = (marker_raw or "").strip()
    norm = (marker_norm or "").strip()
    if not (raw.startswith("(") and raw.endswith(")") and re.fullmatch(r"[a-z]", norm)):
        return False
    ctx = _safe_text(context)
    if not ctx:
        return False
    return re.search(rf"[A-Za-z]{{3,}}\({re.escape(norm)}\)", ctx) is not None



# -----------------------
# Punctuation / Separators
# -----------------------

def _is_likely_punctuation_asterisk(context: str) -> bool:
    """Detect '*' used as punctuation/separators rather than note markers."""
    ctx = _safe_text(context)
    if not ctx:
        return True
    if re.search(r"\w\*[,.;:!?]", ctx):
        return True
    if re.search(r"[,.;:!?]\*\s", ctx):
        return True
    if re.search(r"[.!?]\s*\*\s*[A-Z]", ctx):
        return True
    return False


# -----------------------
# Legacy Wrapper Functions
# -----------------------

# Heuristic function to find the start of a notes/footnotes block in the text lines.
def _find_notes_block(lines: List[str]) -> Optional[Tuple[int, int]]:
    """Backward-compatible wrapper.

    Prefer calling infer_notes_split(lines) in new code.
    """
    res = infer_notes_split(lines)
    if res is None:
        return None
    return res.main_end_index, res.defs_start_index


# More aggressive heuristic to find the start of a notes/footnotes block when no explicit header is present.
def _infer_defs_start_index(lines: List[str]) -> Optional[int]:
    """Backward-compatible wrapper.

    Prefer calling infer_notes_split(lines) in new code.
    """
    res = infer_notes_split(lines)
    if res is None:
        return None
    return res.defs_start_index
