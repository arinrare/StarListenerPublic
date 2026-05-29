import bisect
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

try:
    from sl_heuristics import (
        anchor_is_probable_footnote,
        looks_like_false_numeric_crossref_restart,
        looks_like_false_numeric_date_restart,
        looks_like_false_numeric_age_restart,
        looks_like_false_numeric_bibliographic_restart,
        looks_like_false_numeric_editorial_reference_restart,
        looks_like_false_single_letter_definition_restart,
        looks_like_post_notes_section_enumerator_line,
    )
    from sl_ai import _ai_disambiguate_pairs
    from sl_chapters import (
        _is_notes_header_line,
        _line_looks_like_heading_component,
        _looks_like_chapter_heading_text,
        _roman_to_int,
        _strip_trailing_footnote_marker_from_heading,
    )
    from sl_utility import (
        _def_line_regex,
        _marker_category_from_raw,
        _marker_regex,
        _normalize_marker,
        _safe_text,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .sl_heuristics import (  # type: ignore
        anchor_is_probable_footnote,
        looks_like_false_numeric_crossref_restart,
        looks_like_false_numeric_date_restart,
        looks_like_false_numeric_age_restart,
        looks_like_false_numeric_bibliographic_restart,
        looks_like_false_numeric_editorial_reference_restart,
        looks_like_false_single_letter_definition_restart,
        looks_like_post_notes_section_enumerator_line,
    )
    from .sl_ai import _ai_disambiguate_pairs  # type: ignore
    from .sl_chapters import (  # type: ignore
        _is_notes_header_line,
        _line_looks_like_heading_component,
        _looks_like_chapter_heading_text,
        _roman_to_int,
        _strip_trailing_footnote_marker_from_heading,
    )
    from .sl_utility import (  # type: ignore
        _def_line_regex,
        _marker_category_from_raw,
        _marker_regex,
        _normalize_marker,
        _safe_text,
    )


def _anchor_text_excluding_definitions(lines: List[str], defs_start: int) -> str:
    """Return a text view suitable for anchor extraction.

    We want to find footnote anchors in the running prose even when the EPUB is
    poorly segmented (e.g., more prose appears after a NOTES block).

    This function returns all lines, but excludes definition blocks starting at
    defs_start (definition line + continuations), so we avoid extracting anchors
    from the footnote definitions themselves.
    """
    if not lines:
        return ""

    start = max(0, int(defs_start or 0))
    if start >= len(lines):
        return "\n".join(lines)

    def_re = _def_line_regex()
    roman_re = re.compile(r"^\s*[IVXLC]{1,8}\.?(?:\s+)?$")
    arabic_re = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")

    kept: List[str] = []

    # Always keep pre-notes region.
    kept.extend(lines[:start])

    in_def = False
    for i in range(start, len(lines)):
        line = lines[i] or ""
        stripped = line.strip()
        stripped_clean = _strip_trailing_footnote_marker_from_heading(stripped) or stripped

        # Break out of a definition block on blank lines.
        if not stripped:
            in_def = False
            kept.append(line)
            continue

        # Heuristic: headings strongly indicate we're back in prose, even if the
        # EPUB omitted blank lines between notes and chapter text.
        if roman_re.match(stripped_clean) or arabic_re.match(stripped_clean):
            in_def = False

        # Recovery: if our notes-split inference was off and we hit a real chapter
        # heading while still inside a definition block, treat that as a return to
        # prose so we don't drop the next chapter's anchors.
        if in_def:
            upper = stripped_clean.upper()
            if "NOTE" not in upper and _looks_like_chapter_heading_text(stripped_clean):
                in_def = False

        if def_re.match(line):
            in_def = True
            continue

        if in_def:
            # Continuation line of a definition.
            continue

        kept.append(line)

    return "\n".join(kept)


def _build_definition_exclusion_mask(lines: List[str], defs_start: int) -> Optional[Tuple[List[bool], List[int]]]:
    """Return (excluded_mask, line_starts) for lines inside a definitions block.

    This is used to suppress anchors/headings that occur inside inferred footnote
    definitions (where cross-references and numbering can look like anchors or
    chapter headings).

    The exclusion follows the same recovery logic as `_anchor_text_excluding_definitions`:
    once a real chapter heading is encountered after notes begin, we stop excluding
    subsequent lines (to avoid dropping anchors in prose that continues after notes).
    """

    if not lines:
        return None

    start = max(0, int(defs_start or 0))
    if start < 0 or start >= len(lines):
        return None

    def_re = _def_line_regex()
    roman_re = re.compile(r"^\s*[IVXLC]{1,8}\.?(?:\s+)?$")
    arabic_re = re.compile(r"^\s*\d{1,3}\.?(?:\s+)?$")

    def _is_strong_heading_after_notes(line: str, li: int) -> bool:
        """Heuristic: does this line look like we've returned to real prose headings?

        IMPORTANT: must be stricter than _looks_like_chapter_heading_text.
        We intentionally do NOT accept generic Title Case lines here because
        footnote bodies can quote title pages and section headers.
        """

        t = _safe_text(line or "")
        if not t:
            return False
        t = _strip_trailing_footnote_marker_from_heading(t) or t
        u = t.upper().strip()

        if _is_notes_header_line(t):
            return False

        # Strong: explicit CHAPTER marker.
        if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})(?:\b|\s*[:\.]\s*.*)$", t, re.IGNORECASE):
            return True

        # Strong: ALL-CAPS heading line (FOREWORD, INTRODUCTION, PART TWO, etc.).
        letters = [ch for ch in t if ch.isalpha()]
        if len(letters) >= 6:
            upper_count = sum(1 for ch in letters if ch.isupper())
            if (upper_count / max(1, len(letters))) >= 0.90 and "NOTE" not in u:
                return True

        # Strong: numeral line followed soon by a heading-like component.
        if roman_re.match(t) or arabic_re.match(t):
            try:
                for k in range(li + 1, min(li + 6, len(lines))):
                    tk = _safe_text(lines[k] or "")
                    if not tk:
                        continue
                    tk = _strip_trailing_footnote_marker_from_heading(tk) or tk
                    if def_re.match(tk) or _is_notes_header_line(tk):
                        return False
                    if _line_looks_like_heading_component(tk):
                        return True
                    return False
            except Exception:
                return False

        return False

    excluded: List[bool] = [False] * len(lines)
    in_def = False
    saw_defs = False
    notes_closed = False

    for li in range(start, len(lines)):
        line = lines[li] or ""
        stripped = line.strip()
        stripped_clean = _strip_trailing_footnote_marker_from_heading(stripped) or stripped

        if notes_closed:
            break

        if not stripped:
            in_def = False
            continue

        # Once we have entered a definitions block, a real chapter heading signals
        # we are back in prose and should stop excluding further lines.
        if saw_defs:
            if _is_strong_heading_after_notes(stripped_clean, li):
                notes_closed = True
                in_def = False
                continue

        if def_re.match(line):
            saw_defs = True
            in_def = True
            excluded[li] = True
            continue

        if in_def:
            excluded[li] = True

    line_starts: List[int] = []
    off = 0
    for ln in lines:
        line_starts.append(off)
        off += len(ln) + 1

    return excluded, line_starts


def _assign_positions_to_soup_anchors(anchors: List[Dict[str, Any]], lines: List[str], soup: Optional[BeautifulSoup] = None) -> None:
    """Best-effort: assign `position` to soup-derived anchors.

    Soup extraction is strong for EPUBs, but it doesn't naturally yield a character
    offset into `anchors_text`. When a single spine item contains multiple chapter
    headings, we rely on `position` to assign anchors to the nearest preceding
    heading. Without this, anchors after later headings can be incorrectly grouped
    under the spine item's inferred chapter label.

    If `soup` is provided, prefer a token-based offset mapping for href anchors.
    This avoids ambiguous matches where parenthetical numbers like “(3)” occur in
    running prose (figure refs, page refs, outlines).
    """

    if not anchors or not lines:
        return

    # 1) Prefer token-based offsets when soup is available.
    # This yields offsets directly from the HTML text stream, avoiding ambiguous
    # parenthetical numbers elsewhere in the chapter.
    if soup is not None:
        try:
            href_anchors: List[Dict[str, Any]] = []
            sup_anchors: List[Dict[str, Any]] = []
            for a in anchors:
                pos = a.get("position")
                if isinstance(pos, int) and pos >= 0:
                    continue
                if a.get("_has_href") and _safe_text(a.get("href") or ""):
                    href_anchors.append(a)
                    continue
                # Superscript-only markers (no href) still have a stable location
                # in the soup text stream; map those too.
                if not a.get("_has_href") and _safe_text(a.get("marker") or ""):
                    sup_anchors.append(a)

            if href_anchors or sup_anchors:
                # Copy soup so we don't mutate the caller's DOM.
                soup2 = BeautifulSoup(str(soup), "html.parser")

                # Fixed-length token so we can correct offsets deterministically.
                tok_tpl = "\uE000SL{idx:06d}\uE001"
                tok_len = len(tok_tpl.format(idx=0))

                token_to_anchor: Dict[str, Dict[str, Any]] = {}
                token_counter = 0

                # 1a) Map href anchors by locating their <a href="..."> tags.
                if href_anchors:
                    by_href: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for a in href_anchors:
                        by_href[_safe_text(a.get("href") or "")].append(a)

                    for href, group in by_href.items():
                        if not href:
                            continue
                        tags = soup2.find_all("a", href=href)
                        if not tags:
                            continue

                        # Some EPUBs reuse the same href for cross-references
                        # ("see note 8") *and* true footnote markers ("(8)").
                        # Never assume the first <a href> is the right one.
                        unused = list(tags)

                        def _tag_marker_text(t) -> str:
                            try:
                                return _safe_text(t.get_text(" ")).strip()
                            except Exception:
                                return ""

                        def _tag_container_text(t) -> str:
                            try:
                                container = t.find_parent(["p", "li", "dd"]) if getattr(t, "find_parent", None) else None
                                if container is not None and getattr(container, "get_text", None):
                                    return _safe_text(container.get_text(" "))
                            except Exception:
                                pass
                            try:
                                p = getattr(t, "parent", None)
                                if p is not None and getattr(p, "get_text", None):
                                    return _safe_text(p.get_text(" "))
                            except Exception:
                                pass
                            return ""

                        stop = {
                            "the",
                            "and",
                            "of",
                            "to",
                            "in",
                            "a",
                            "an",
                            "on",
                            "for",
                            "as",
                            "at",
                            "by",
                            "with",
                            "from",
                            "that",
                            "this",
                            "is",
                            "was",
                            "were",
                            "be",
                            "it",
                        }

                        def _context_keywords(ctx: str) -> List[str]:
                            try:
                                words = re.findall(r"[A-Za-z]{4,}", ctx or "")
                            except Exception:
                                words = []
                            out: List[str] = []
                            for w in words:
                                wl = w.lower()
                                if wl in stop:
                                    continue
                                if wl not in out:
                                    out.append(wl)
                                if len(out) >= 6:
                                    break
                            return out

                        for anchor_dict in group:
                            if not unused:
                                break

                            mk_norm = _safe_text(anchor_dict.get("marker") or "")
                            mk_raw = _safe_text(anchor_dict.get("marker_raw") or "")
                            ctx = _safe_text(anchor_dict.get("context") or "")
                            kws = _context_keywords(ctx)

                            best = None
                            best_score = -999

                            for t in unused:
                                t_txt = _tag_marker_text(t)
                                t_container = _tag_container_text(t).lower()

                                score = 0
                                if mk_raw and t_txt and t_txt == mk_raw.strip():
                                    score += 6
                                if mk_raw and mk_raw.strip() and mk_raw.strip() in t_txt:
                                    score += 3
                                if mk_norm and mk_norm in t_txt:
                                    score += 1
                                try:
                                    if t.find_parent("sup") is not None:
                                        score += 2
                                except Exception:
                                    pass

                                # Prefer tags whose container matches the anchor context.
                                if kws and t_container:
                                    score += sum(1 for kw in kws if kw in t_container)

                                # Penalize explicit cross-reference phrasing.
                                if mk_norm:
                                    try:
                                        mkx = re.escape(mk_norm.lower())
                                        if re.search(rf"\bsee\s+note(?:s)?\s*\(?\s*{mkx}\s*\)?\b", t_container):
                                            score -= 4
                                    except Exception:
                                        pass

                                if score > best_score:
                                    best_score = score
                                    best = t

                            chosen = best or unused[0]
                            try:
                                unused.remove(chosen)
                            except Exception:
                                pass

                            tok = tok_tpl.format(idx=token_counter)
                            token_counter += 1
                            try:
                                chosen.insert_before(tok)
                                token_to_anchor[tok] = anchor_dict
                            except Exception:
                                continue

                # 1b) Map superscript-only anchors by locating matching <sup> tags.
                # We mirror the extraction constraints to avoid grabbing arbitrary
                # superscripts or definition-leading markers.
                if sup_anchors:
                    sup_allowed_re = re.compile(r"^(?:\d{1,3}|\*+|ΓÇá+|ΓÇí+|┬º+|[a-zA-Z])$")

                    def _sup_is_definition_marker(sup_tag, marker_norm: str) -> bool:
                        try:
                            container = sup_tag.find_parent(["p", "li", "dd"]) if getattr(sup_tag, "find_parent", None) else None
                            if container is None or not getattr(container, "get_text", None):
                                return False
                            container_text = _safe_text(container.get_text(" "))
                            dm = _def_line_regex().match(container_text)
                            if dm and _normalize_marker(dm.group(1)) == marker_norm:
                                return True
                        except Exception:
                            return False
                        return False

                    sup_candidates_by_marker: Dict[str, List[Any]] = defaultdict(list)
                    for sup in soup2.find_all("sup"):
                        try:
                            txt = _safe_text(sup.get_text(" "))
                        except Exception:
                            txt = ""
                        if not txt:
                            continue
                        if not sup_allowed_re.fullmatch(txt):
                            continue
                        mk_norm = _normalize_marker(txt)
                        if not mk_norm:
                            continue
                        if _sup_is_definition_marker(sup, mk_norm):
                            continue
                        sup_candidates_by_marker[mk_norm].append(sup)

                    by_marker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for a in sup_anchors:
                        by_marker[_safe_text(a.get("marker") or "")].append(a)

                    for mk, group in by_marker.items():
                        if not mk:
                            continue
                        tags = sup_candidates_by_marker.get(mk) or []
                        if not tags:
                            continue
                        for i, anchor_dict in enumerate(group):
                            if i >= len(tags):
                                break
                            tok = tok_tpl.format(idx=token_counter)
                            token_counter += 1
                            try:
                                tags[i].insert_before(tok)
                                token_to_anchor[tok] = anchor_dict
                            except Exception:
                                continue

                if token_to_anchor:
                    # Produce the same coordinate space as `lines` (preprocess + clean + join).
                    try:
                        from sl_utility import _preprocess_for_notes, _clean_line_for_parsing  # type: ignore
                    except ModuleNotFoundError:  # pragma: no cover
                        from .sl_utility import _preprocess_for_notes, _clean_line_for_parsing  # type: ignore

                    chapter_text_tok = _preprocess_for_notes(soup2.get_text("\n"))
                    tok_lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in chapter_text_tok.split("\n")]
                    anchors_text_tok = "\n".join(tok_lines)

                    hits: List[Tuple[int, str, Dict[str, Any]]] = []
                    for tok, anchor_dict in token_to_anchor.items():
                        j = anchors_text_tok.find(tok)
                        if j != -1:
                            hits.append((int(j), tok, anchor_dict))

                    hits.sort(key=lambda x: x[0])
                    seen = 0
                    for j, _tok, anchor_dict in hits:
                        corrected = int(j - (seen * tok_len))
                        if corrected < 0:
                            corrected = 0
                        anchor_dict["position"] = corrected
                        seen += 1
        except Exception:
            # Fall back to heuristic matching below.
            pass

    # 2) Fallback heuristic for any anchors still missing a position.
    # Precompute line starts in the same coordinate space as anchors_text = "\n".join(lines)
    line_starts: List[int] = []
    off = 0
    for ln in lines:
        line_starts.append(off)
        off += len(ln) + 1

    stopwords = {
        "the",
        "and",
        "of",
        "to",
        "in",
        "a",
        "an",
        "on",
        "for",
        "as",
        "at",
        "by",
        "with",
        "from",
        "that",
        "this",
        "is",
        "was",
        "were",
        "be",
        "it",
    }

    def _keywords(ctx: str, marker_norm: str) -> List[str]:
        # Extract keywords from a window around the marker occurrence in the
        # context, not from the start of the paragraph. In long paragraphs,
        # the early words can be unrelated to the marker line (and EPUB text
        # extraction can split the marker onto its own line).
        c = ctx or ""
        mk2 = marker_norm or ""
        focus = c
        try:
            if mk2:
                pats = [f"({mk2})", f"[{mk2}]", mk2]
                hit = -1
                for p in pats:
                    j = c.find(p)
                    if j != -1 and (hit == -1 or j < hit):
                        hit = j
                if hit != -1:
                    start = max(0, hit - 180)
                    end = min(len(c), hit + 180)
                    focus = c[start:end]
        except Exception:
            focus = c

        words = re.findall(r"[A-Za-z]{4,}", focus)
        out: List[str] = []
        for w in words:
            wl = w.lower()
            if wl in stopwords:
                continue
            if wl not in out:
                out.append(wl)
            if len(out) >= 6:
                break
        return out

    for a in anchors:
        try:
            pos = a.get("position")
            if isinstance(pos, int) and pos >= 0:
                continue

            mk = _safe_text(a.get("marker") or "")
            if not mk:
                continue

            # Try to match how the marker appears in running text.
            # NOTE: we treat parenthetical/bracketed occurrences as "strong" matches.
            # Bare-number matches ("5") are ambiguous in scholarly editions and can
            # appear in dates, page refs, definitions, etc.
            strong_patterns = [f"({mk})", f"[{mk}]"]
            marker_patterns = [*strong_patterns, mk]

            ctx = _safe_text(a.get("context") or "")
            kws = _keywords(ctx, mk)

            chosen: Optional[Tuple[int, int]] = None
            fallback: Optional[Tuple[int, int]] = None

            # Scan from the beginning and choose the earliest plausible match.
            # Keyword overlap is a confirmation signal to avoid matching unrelated
            # parenthetical numbers elsewhere in the text.
            for li, line in enumerate(lines[:30000]):
                if not line:
                    continue
                hit_idx = None
                hit_pat = None
                for pat in marker_patterns:
                    if not pat:
                        continue

                    # For the bare marker case, avoid matching inside larger tokens
                    # (e.g., marker "14" must not match the "14" in "149").
                    if pat == mk:
                        try:
                            if re.fullmatch(r"\d{1,3}", mk):
                                m_bare = re.search(rf"(?<!\d){re.escape(mk)}(?!\d)", line)
                                if m_bare:
                                    hit_idx = int(m_bare.start())
                                    hit_pat = pat
                                    break
                                continue
                            if re.fullmatch(r"[A-Za-z]", mk):
                                m_bare = re.search(rf"(?<![A-Za-z]){re.escape(mk)}(?![A-Za-z])", line)
                                if m_bare:
                                    hit_idx = int(m_bare.start())
                                    hit_pat = pat
                                    break
                                continue
                        except Exception:
                            # Fall back to substring search below.
                            pass

                    idx = line.find(pat)
                    if idx != -1:
                        hit_idx = idx
                        hit_pat = pat
                        break
                if hit_idx is None:
                    continue

                if fallback is None:
                    fallback = (li, hit_idx)

                if not kws:
                    chosen = (li, hit_idx)
                    break

                # Confirmation step: prefer occurrences that align with the
                # anchor's surrounding context. This prevents accidentally
                # matching unrelated parenthetical numbers like "(14)" that
                # can appear in scholarly prose.
                try:
                    prev_line = lines[li - 1] if li - 1 >= 0 else ""
                    next_line = lines[li + 1] if (li + 1) < len(lines) else ""
                except Exception:
                    prev_line = ""
                    next_line = ""

                # If the marker is isolated on its own line (common in EPUB
                # text extraction), look at neighbors for context confirmation.
                try:
                    w = (prev_line or "") + " " + (line or "") + " " + (next_line or "")
                except Exception:
                    w = line

                wl = w.lower()
                score = sum(1 for kw in kws if kw and kw in wl)

                # For strong patterns like "(5)", require at least one context
                # keyword in the window.
                if hit_pat in strong_patterns:
                    if score >= 1:
                        chosen = (li, hit_idx)
                        break
                    continue

                if score >= 1:
                    chosen = (li, hit_idx)
                    break

            if chosen is None:
                chosen = fallback

            if chosen is not None:
                (best_li, best_idx) = chosen
                a["position"] = int(line_starts[best_li] + best_idx)
        except Exception:
            continue


def _extract_anchors_from_soup(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Core function to extract candidate footnote anchors from HTML soup."""

    anchors: List[Dict[str, Any]] = []

    # Common fragment id naming schemes for note targets.
    # Keep this conservative; this is only a *candidate* signal (combined with
    # other signals like epub:type=noteref or <sup> containment).
    id_re = re.compile(
        r"^(?:fn|fnref|fnt|footnote|footnoteref|note|noteref|endnote|en|ref|n)[-_]?\d{1,4}[a-z]?$",
        re.IGNORECASE,
    )
    _pt4en_id_re = re.compile(r"^r?pt4en\d{1,4}[a-z]?$", re.IGNORECASE)

    def is_noteref(a_tag) -> bool:
        # EPUBs often mark footnote refs via epub:type="noteref" or classes like "noteref".
        epub_type = (a_tag.get("epub:type") or a_tag.get("type") or "").lower()
        if "noteref" in epub_type:
            return True
        role = (a_tag.get("role") or "").lower()
        if "doc-noteref" in role:
            return True
        cls = " ".join(a_tag.get("class") or []).lower()
        if (
            "noteref" in cls
            or "fnref" in cls
            or "footnote-ref" in cls
            or "note-ref" in cls
            or "footnote" in cls
            or "endnote" in cls
        ):
            return True
        return False

    def href_fragment(href: str) -> str:
        if not href or "#" not in href:
            return ""
        return href.split("#", 1)[1].strip()

    def is_superscript_ref(a_tag) -> bool:
        # Many App Specific-produced EPUBs don't mark footnote refs with epub:type.
        # A numeric link inside <sup> is a strong footnote-ref signal.
        try:
            return a_tag.find_parent("sup") is not None
        except Exception:
            return False

    def _is_superscript_related(a_tag) -> bool:
        """Check whether an <a> tag is related to a superscript footnote marker.

        Handles both normal nesting (<sup><a>N</a></sup>) and inverted nesting
        (<a><sup>N</sup></a>) which occurs in some calibre-produced EPUBs.
        """
        try:
            if is_superscript_ref(a_tag):
                return True
            # Inverted nesting: <a> wraps <sup> (e.g., <a id="ref395a"><sup>31</sup></a>)
            if a_tag.find("sup") is not None:
                return True
        except Exception:
            pass
        return False

    def _context_text_for_tag(tag, fallback: str) -> str:
        """Return a stable surrounding context for an anchor-ish tag.

        Important: if a marker is represented as <sup><a>12</a></sup>, using
        a.parent.get_text() yields only "12" while the <sup> path yields the
        whole paragraph, causing duplicate anchors. Prefer container-level text.
        """

        try:
            container = tag.find_parent(["p", "li", "dd", "td", "th"]) if getattr(tag, "find_parent", None) else None
            if container is not None:
                t = _safe_text(container.get_text(" "))
                if t:
                    return t
        except Exception:
            pass
        try:
            parent = getattr(tag, "parent", None)
            if parent is not None and getattr(parent, "get_text", None):
                t = _safe_text(parent.get_text(" "))
                if t:
                    return t
        except Exception:
            pass
        return fallback

    def _context_window(full_text: str, marker_norm: str) -> str:
        """Return a short context window centered on the marker occurrence.

        If we keep the whole paragraph and then truncate from the start, long
        paragraphs can lose the actual marker occurrence. That breaks later
        href-vs-regex de-duplication and makes the UI context misleading.
        """

        t = _safe_text(full_text)
        mk = _safe_text(marker_norm)
        if not t:
            return ""
        if not mk:
            return t[:400]

        idx = -1
        hit_len = 0

        try:
            if re.fullmatch(r"\d{1,3}", mk):
                pats = [
                    rf"\(\s*{re.escape(mk)}\s*\)",
                    rf"\[\s*{re.escape(mk)}\s*\]",
                    rf"(?<!\d){re.escape(mk)}(?!\d)",
                ]
                for pat in pats:
                    m = re.search(pat, t)
                    if m:
                        idx = int(m.start())
                        hit_len = int(m.end() - m.start())
                        break
            elif re.fullmatch(r"[A-Za-z]", mk):
                pats = [
                    rf"\(\s*{re.escape(mk)}\s*\)",
                    rf"\[\s*{re.escape(mk)}\s*\]",
                    rf"(?<![A-Za-z]){re.escape(mk)}(?![A-Za-z])",
                ]
                for pat in pats:
                    m = re.search(pat, t)
                    if m:
                        idx = int(m.start())
                        hit_len = int(m.end() - m.start())
                        break
            else:
                for pat in (f"({mk})", f"[{mk}]", mk):
                    j = t.find(pat)
                    if j != -1:
                        idx = int(j)
                        hit_len = len(pat)
                        break
        except Exception:
            idx = -1

        if idx < 0:
            return t[:400]

        radius = 220
        start = max(0, idx - radius)
        end = min(len(t), idx + max(1, hit_len) + radius)
        return t[start:end]

    # 1) Explicit links to footnotes (common in EPUBs)
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        # Allow intra-file (#fn1) and cross-file (notes.xhtml#fn1) fragments.
        if "#" not in href:
            continue
        if href.lower().startswith("http://") or href.lower().startswith("https://"):
            continue

        frag = href_fragment(href)
        # Also recognise the st{N}/rst{N} footnote convention where anchors
        # in prose link to rst-prefixed or st-prefixed definition ids.
        # Direction 1 (Book 11): anchor=st{N}, fragment=rst{N}
        # Direction 2 (Book 12): anchor=rst{N}, fragment=st{N}
        _is_rst_frag = bool(frag and re.match(r"^rst\d+[a-z]?$", frag, re.IGNORECASE))
        _is_st_frag = bool(frag and re.match(r"^st\d+[a-z]?$", frag, re.IGNORECASE))
        # Only consider anchors that look like they reference a footnote target.
        # This prevents TOC/section navigation links (e.g., "Foreword", chapter titles) from being treated as footnotes.
        if not (frag and (id_re.match(frag) or _is_rst_frag or _is_st_frag or _pt4en_id_re.match(frag))) and not is_noteref(a) and not _is_superscript_related(a):
            continue

        txt = _safe_text(a.get_text(" "))
        marker_txt = txt
        if not marker_txt:
            # Try extracting a numeric marker from the href like #fn12
            m = re.search(r"(\d{1,3})", href)
            marker_txt = m.group(1) if m else ""

        # Normalize common variants like "(10)" or "10.".
        if marker_txt and not re.fullmatch(r"\d{1,3}|\*+|ΓÇá+|ΓÇí+|┬º+|[a-zA-Z]", marker_txt):
            m = re.fullmatch(r"\s*[\(\[]?\s*(\d{1,3}|\*+|ΓÇá+|ΓÇí+|┬º+|[a-zA-Z])\s*[\)\]]?\s*\.?\s*", marker_txt)
            if m:
                marker_txt = m.group(1)

        # Only accept small marker-like texts.
        if not re.fullmatch(r"\d{1,3}|\*+|ΓÇá+|ΓÇí+|┬º+|[a-zA-Z]", marker_txt or ""):
            continue

        marker_norm = _normalize_marker(marker_txt)
        if not marker_norm:
            continue

        parent_text_full = _context_text_for_tag(a, txt)
        parent_text = _context_window(parent_text_full, marker_norm)

        # Cross-reference suppression: scholarly editions often contain links like
        # "(see note 8)" where the linked token is just "8". These are not
        # footnote *markers* in running prose; treating them as anchors can steal
        # the earliest id_link for a marker and disrupt reading-order display.
        #
        # Only apply to non-superscript anchors: real footnotes live in <sup> tags
        # and should never be suppressed by cross-reference patterns.
        if not _is_superscript_related(a):
            try:
                # Search only the ±220-char window around the anchor (parent_text),
                # not the full paragraph. This prevents false suppression when a
                # page reference appears far from the footnote marker in the same
                # paragraph.
                tlow = (parent_text or "").lower()
                mk = re.escape(str(marker_norm).lower())
                cross_pats = [
                    rf"\bsee\s+note(?:s)?\s*\(?\s*{mk}\s*\)?\b",
                    rf"\bcf\.?\s+note(?:s)?\s*\(?\s*{mk}\s*\)?\b",
                    rf"\bcompare\s+note(?:s)?\s*\(?\s*{mk}\s*\)?\b",
                    rf"\bsee\s+n\.?\s*{mk}\b",
                    rf"\bin\s+note\s+{mk}\b",
                    rf"\bcf\.?\s*{mk}\b",
                    rf"\bp\.?\s*{mk}\b",
                    rf"\bpp\.?\s*{mk}\b",
                    rf"\bpage(?:s)?\s*{mk}\b",
                    rf"\bp\.?\s*\d{{1,3}}[-\u2013\u2014]\s*{mk}\b",
                    rf"\bpp\.?\s*\d{{1,3}}[-\u2013\u2014]\s*{mk}\b",
                    rf"\bp\.?\s*\d{{1,3}}[-\u2013\u2014]\d{{1,3}}\s*[,;&]\s*{mk}\b",
                    rf"\bpp\.?\s*\d{{1,3}}[-\u2013\u2014]\d{{1,3}}\s*[,;&]\s*{mk}\b",
                    rf"\bpp?\.?\s*\d{{1,3}}\s*[,\uFF0C;&]\s*{mk}\b",
                    rf"\bpp?\.?\s*\d{{1,3}}(?:\s*[,;&]\s*\d{{1,3}})*\s*[,;&]\s*{mk}\b",
                    rf"\bpp?\.?\s*\d{{1,3}}\s*[,;&]\s*\d{{1,3}}\s*[,;&]\s*{mk}\b",
                ]
                if any(re.search(p, tlow) for p in cross_pats):
                    continue
            except Exception:
                pass

        # If this link is actually the leading marker of a *definition* paragraph
        # (i.e., we're inside the notes list itself), do not treat it as an anchor.
        # Important: the immediate parent of the <a> is often <sup>, whose text is
        # just "12"; so we must check a wider container like the nearest <p>/<li>.
        try:
            container = a.find_parent(["p", "li", "dd"]) if getattr(a, "find_parent", None) else None
            container_text = _safe_text(container.get_text(" ")) if container is not None else parent_text
            dm = _def_line_regex().match(container_text)
            if dm and _normalize_marker(dm.group(1)) == marker_norm:
                continue
            # Also skip anchors whose container has a note-like class — these
            # are back-links within definition paragraphs, not prose anchors.
            if container is not None:
                c_cls = " ".join(container.get("class") or []).lower()
                if any(tok in c_cls for tok in ("note", "endnote", "footnote")):
                    continue
        except Exception:
            pass

        anchors.append(
            {
                "marker_raw": txt or marker_txt,
                "marker": marker_norm,
                "position": -1,
                "context": parent_text[:400],
                "href": href,
                "_has_href": True,
            }
        )

    # 2) Superscripts (often just the marker)
    for sup in soup.find_all("sup"):
        txt = _safe_text(sup.get_text(" "))
        if not txt:
            continue
        # Keep only small digit markers or known symbols.
        if not re.fullmatch(r"\d{1,3}|\*+|ΓÇá+|ΓÇí+|┬º+|[a-zA-Z]", txt):
            continue
        marker_norm = _normalize_marker(txt)
        if not marker_norm:
            continue
        parent_text_full = _context_text_for_tag(sup, txt)
        parent_text = _context_window(parent_text_full, marker_norm)

        # Same guard as above: ignore superscripts that are actually the marker
        # starting a notes definition paragraph.
        try:
            container = sup.find_parent(["p", "li", "dd"]) if getattr(sup, "find_parent", None) else None
            container_text = _safe_text(container.get_text(" ")) if container is not None else parent_text
            dm = _def_line_regex().match(container_text)
            if dm and _normalize_marker(dm.group(1)) == marker_norm:
                continue
        except Exception:
            pass

        anchors.append(
            {
                "marker_raw": txt,
                "marker": marker_norm,
                "position": -1,
                "context": parent_text[:400],
                "_has_href": False,
            }
        )

    # De-dup by (marker, context)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for a in anchors:
        key = (a.get("marker"), a.get("context"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq


def _harvest_specific_filepos_note_targets(soup: BeautifulSoup) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Extract Specific App-style popup footnote definitions.

    Pattern observed in some EPUBs:
      - A self-closing or empty tag with an id like "filepos1458711"
      - Followed by a paragraph that starts with a superscripted link marker (e.g., "1")
        and then the note text.

    Returns:
      - id_map: filepos-id -> definition text
      - marker_defs: list of {marker, text}
    """

    id_map: Dict[str, str] = {}
    marker_defs: List[Dict[str, str]] = []

    if soup is None:
        return id_map, marker_defs

    # Walk paragraphs; only treat them as note definitions when we can bind them
    # to a filepos id anchor (either on the paragraph itself or on an immediately
    # preceding empty sibling tag). This avoids misclassifying normal running
    # text paragraphs that happen to start with a superscript footnote ref.
    for p in soup.find_all("p"):
        sup = p.find("sup")
        if not sup:
            continue
        a = sup.find("a")
        if not a:
            continue
        mk_raw = _safe_text(a.get_text(" "))
        if not re.fullmatch(r"\d{1,3}|\*+|ΓÇá+|ΓÇí+|┬º+|[a-zA-Z]", mk_raw or ""):
            continue

        # Get full paragraph text and strip the marker prefix.
        p_text = _safe_text(p.get_text(" "))
        if not p_text or len(p_text) < 20:
            continue

        m = _def_line_regex().match(p_text)
        if not m:
            continue
        marker_norm = _normalize_marker(m.group(1))
        def_text = _safe_text(m.group(2))
        if not marker_norm or not def_text:
            continue

        bound_id: Optional[str] = None

        # Case 1: paragraph itself is the anchor.
        pid_self = p.get("id") if getattr(p, "get", None) else None
        if pid_self and str(pid_self).lower().startswith("filepos"):
            bound_id = str(pid_self)
        else:
            # Case 2: immediately preceding (often empty) filepos anchor tag.
            prev = p.find_previous_sibling()
            while prev is not None and getattr(prev, "name", None) is None:
                prev = prev.find_previous_sibling()
            if prev is not None:
                pid = prev.get("id") if getattr(prev, "get", None) else None
                if pid and str(pid).lower().startswith("filepos"):
                    prev_txt = _safe_text(prev.get_text(" ")) if getattr(prev, "get_text", None) else ""
                    if not prev_txt:
                        bound_id = str(pid)

        if bound_id:
            id_map[bound_id] = def_text
            marker_defs.append({"marker": marker_norm, "text": def_text})

    return id_map, marker_defs


def _harvest_structured_notes_section_targets(soup: BeautifulSoup) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Extract note definitions from an in-document NOTES section.

    Some EPUBs keep note definitions inline in the same chapter XHTML and format
    them as linked marker paragraphs under a NOTES/ENDNOTES heading. In those
    cases, flattened text parsing can corrupt markers (for example, `406. 2`).
    Prefer the HTML structure when available.
    """

    id_map: Dict[str, str] = {}
    marker_defs: List[Dict[str, Any]] = []

    if soup is None:
        return id_map, marker_defs

    marker_text_re = re.compile(r"\d{1,3}|\*+|[A-Za-z]", re.UNICODE)
    boundary_tag_names = {"h1", "h2", "h3", "h4", "h5", "h6", "title"}
    scanned_ids: set[int] = set()

    def _block_text(tag: Any) -> str:
        try:
            return _safe_text(tag.get_text(" "))
        except Exception:
            return ""

    def _extract_structured_note_from_block(tag: Any) -> Optional[Dict[str, str]]:
        text = _block_text(tag)
        if not text:
            return None

        tag_name = str(getattr(tag, "name", "") or "").lower()
        if not tag_name or tag_name in {"a", "span", "em", "i", "strong", "b", "sup", "sub"}:
            return None

        first_anchor = None
        try:
            for a in tag.find_all("a"):
                try:
                    if a.find_parent("sup") is not None:
                        continue
                except Exception:
                    pass
                first_anchor = a
                break
        except Exception:
            first_anchor = None

        if first_anchor is None:
            return None

        href = _safe_text(first_anchor.get("href") or "").strip()
        if href.lower().startswith("http://") or href.lower().startswith("https://"):
            return None

        marker_raw = _safe_text(first_anchor.get_text(" ")).strip()
        marker_norm = _normalize_marker(marker_raw)
        # Strip trailing punctuation from definition markers (e.g., "1." → "1")
        # so they match the marker_text_re pattern.
        if marker_norm and len(marker_norm) > 1 and marker_norm[-1] in ".:)]":
            marker_norm = marker_norm[:-1]
        if not marker_norm or not marker_text_re.fullmatch(marker_norm):
            return None

        lead_re = re.compile(
            rf"^\s*(?:\(|\[)?\s*{re.escape(marker_norm)}\s*(?:\)|\])?\s*(?:[\]\)\.:\-—]\s*)?"
        )
        body = lead_re.sub("", text, count=1).strip()
        if not body or body == text.strip():
            return None

        tag_id = _safe_text(getattr(tag, "get", lambda _k, _d=None: None)("id") or "").strip()
        return {"id": tag_id, "marker": marker_norm, "text": body, "tag_name": tag_name}

    def _candidate_blocks_from_node(node: Any) -> List[Any]:
        candidates: List[Any] = []
        node_name = str(getattr(node, "name", "") or "").lower()
        if node_name:
            direct = _extract_structured_note_from_block(node)
            if direct is not None:
                candidates.append(node)
                return candidates

        try:
            for desc in getattr(node, "find_all", lambda *args, **kwargs: [])(True):
                desc_id = id(desc)
                if desc_id in scanned_ids:
                    continue
                note_entry = _extract_structured_note_from_block(desc)
                if note_entry is None:
                    continue
                candidates.append(desc)
        except Exception:
            return candidates
        return candidates

    for tag in soup.find_all(True):
        tag_id = _safe_text(getattr(tag, "get", lambda _k, _d=None: None)("id") or "").strip()
        if "_rfn" not in tag_id:
            continue
        note_entry = _extract_structured_note_from_block(tag)
        if note_entry is None:
            continue
        marker_defs.append(
            {
                "id": note_entry["id"],
                "marker": note_entry["marker"],
                "text": note_entry["text"],
                "line_index": len(marker_defs) + 1,
            }
        )
        if note_entry["id"]:
            id_map[note_entry["id"]] = note_entry["text"]
        scanned_ids.add(id(tag))

    if marker_defs:
        return id_map, marker_defs

    # Second branch: detect well-formed EPUBs that use class="footnote" or "footnotet"
    # CSS classes for definition blocks (common in calibre-produced scholarly editions).
    # This handles books where each footnote definition is in its own <p> element with
    # an explicit footnote class, even when _rfn ids are absent.
    footnote_class_elts = soup.select(".footnote, .footnotet, .noindent-x1")
    if footnote_class_elts:
        for elt in footnote_class_elts:
            note_entry = _extract_structured_note_from_block(elt)
            if note_entry is None:
                continue
            marker_defs.append(
                {
                    "id": note_entry["id"],
                    "marker": note_entry["marker"],
                    "text": note_entry["text"],
                    "line_index": len(marker_defs) + 1,
                }
            )
            if note_entry["id"]:
                id_map[note_entry["id"]] = note_entry["text"]
            scanned_ids.add(id(elt))
        if marker_defs:
            return id_map, marker_defs

    header_tag_blacklist = {"a", "span", "em", "i", "strong", "b", "sup", "sub"}

    for header in soup.find_all(True):
        header_name = str(getattr(header, "name", "") or "").lower()
        if not header_name or header_name in header_tag_blacklist:
            continue
        if not _is_notes_header_line(_block_text(header)):
            continue

        started = False
        matched_tag_name: Optional[str] = None
        matched_parent_id: Optional[int] = None
        for sibling in header.find_next_siblings():
            sibling_name = str(getattr(sibling, "name", "") or "").lower()
            if not sibling_name:
                continue

            sibling_text = _block_text(sibling)
            if not sibling_text:
                continue

            if _is_notes_header_line(sibling_text):
                if started:
                    break
                continue

            candidates = _candidate_blocks_from_node(sibling)
            accepted_here = 0
            for candidate in candidates:
                cand_id = id(candidate)
                if cand_id in scanned_ids:
                    continue

                note_entry = _extract_structured_note_from_block(candidate)
                if note_entry is None:
                    continue

                candidate_parent = getattr(candidate, "parent", None)
                candidate_parent_id = id(candidate_parent) if candidate_parent is not None else None
                candidate_tag_name = str(note_entry.get("tag_name") or "")

                if started:
                    same_parent = matched_parent_id is not None and candidate_parent_id == matched_parent_id
                    same_tag = bool(matched_tag_name) and candidate_tag_name == matched_tag_name
                    cls = " ".join(getattr(candidate, "get", lambda _k, _d=None: [])("class") or []).lower()
                    noteish_class = any(tok in cls for tok in ["note", "footnote", "endnote"])
                    if not (same_parent or same_tag or noteish_class):
                        continue

                started = True
                if matched_tag_name is None:
                    matched_tag_name = candidate_tag_name
                if matched_parent_id is None and candidate_parent_id is not None:
                    matched_parent_id = candidate_parent_id

                marker_defs.append(
                    {
                        "id": note_entry["id"],
                        "marker": note_entry["marker"],
                        "text": note_entry["text"],
                        "line_index": len(marker_defs) + 1,
                    }
                )
                if note_entry["id"]:
                    id_map[note_entry["id"]] = note_entry["text"]
                scanned_ids.add(cand_id)
                accepted_here += 1

            if accepted_here:
                continue

            if started:
                cls = " ".join(getattr(sibling, "get", lambda _k, _d=None: [])("class") or []).lower()
                if sibling_name == "div" and any(tok.startswith("top") for tok in cls.split()):
                    continue
                if sibling_name in boundary_tag_names:
                    break
                if _line_looks_like_heading_component(sibling_text):
                    continue
                break

    return id_map, marker_defs


_st_rst_id_re = re.compile(r"^st(\d+)([a-z]?)$", re.IGNORECASE)


def _harvest_st_rst_footnotes(soup: BeautifulSoup) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Harvest footnote anchors and definitions that use the st{N}/rst{N} bidirectional link convention.

    Supports both naming directions:
      Direction 1 (Book 11): anchor <a id="st3" href="...#rst3">, def <p class="footnote" id="rst3">
      Direction 2 (Book 12): anchor <a id="rst1" href="...#st1">, def <p id="st1">

    Returns:
      - id_map: definition-id -> definition text
      - anchors: list of anchor dicts {marker, id, href_target_id, href_file, ...}
      - marker_defs: list of {id, marker, text}
    """

    id_map: Dict[str, str] = {}
    anchors: List[Dict[str, Any]] = []
    marker_defs: List[Dict[str, Any]] = []

    if soup is None:
        return id_map, anchors, marker_defs

    def _body_after_lead_marker(text: str, lead_marker: str) -> str:
        m = re.match(rf"^\s*(?:\(|\[)?\s*{re.escape(lead_marker)}\s*(?:\)|\])?\s*(?:[\]\)\.:\-—]\s*)?", text)
        if m:
            body = text[m.end():].strip()
            if body:
                return body
        stripped = text.strip()
        if stripped.lower().startswith(lead_marker.lower()):
            body = stripped[len(lead_marker):].strip()
            body = re.sub(r"^[\]\)\.:\-—]\s*", "", body)
            if body:
                return body
        return ""

    def _is_footnote_def_block(p: Any) -> bool:
        cls = " ".join(p.get("class") or []).lower()
        return any(tok in cls for tok in ["footnote", "footnotet", "noindent-x1"])

    # Phase 1a: Book 11 direction — st{N} anchors linking to rst{N} definitions.
    for a in soup.find_all("a"):
        tag_id = _safe_text(a.get("id") or "").strip()
        m = _st_rst_id_re.match(tag_id)
        if not m:
            continue
        href = _safe_text(a.get("href") or "").strip()
        if not href or "#" not in href:
            continue
        href_file, href_frag = href.rsplit("#", 1)
        href_file = href_file.strip()
        href_frag = href_frag.strip()
        rst_m = re.match(r"^rst(\d+)([a-z]?)$", href_frag, re.IGNORECASE)
        if not rst_m:
            continue
        marker_raw = _safe_text(a.get_text(" ")).strip()
        if not marker_raw:
            continue
        marker_norm = _normalize_marker(marker_raw)
        if not marker_norm:
            continue
        anchors.append({
            "marker_raw": marker_raw,
            "marker": marker_norm,
            "id": tag_id,
            "href_target_id": href_frag,
            "href_file": href_file,
            "href": href,
            "_has_href": True,
            "match_method_hint": "st_rst",
        })

    # Phase 1b: Book 12 direction — rst{N} anchors linking to st{N} definitions.
    for a in soup.find_all("a"):
        tag_id = _safe_text(a.get("id") or "").strip()
        rst_m = re.match(r"^rst(\d+)([a-z]?)$", tag_id, re.IGNORECASE)
        if not rst_m:
            continue
        href = _safe_text(a.get("href") or "").strip()
        if not href or "#" not in href:
            continue
        href_file, href_frag = href.rsplit("#", 1)
        href_file = href_file.strip()
        href_frag = href_frag.strip()
        st_m = _st_rst_id_re.match(href_frag)
        if not st_m:
            continue
        marker_raw = _safe_text(a.get_text(" ")).strip()
        if not marker_raw:
            continue
        marker_norm = _normalize_marker(marker_raw)
        if not marker_norm:
            continue
        anchors.append({
            "marker_raw": marker_raw,
            "marker": marker_norm,
            "id": tag_id,
            "href_target_id": href_frag,
            "href_file": href_file,
            "href": href,
            "_has_href": True,
            "match_method_hint": "st_rst",
        })

    # Phase 2a: Book 11 direction — rst{N} footnote definitions.
    for p in soup.find_all(["p", "div", "li", "blockquote"]):
        tag_id = _safe_text(p.get("id") or "").strip()
        rst_m = re.match(r"^rst(\d+)([a-z]?)$", tag_id, re.IGNORECASE)
        if not rst_m:
            continue
        is_footnote_block = _is_footnote_def_block(p)
        text = _safe_text(p.get_text(" ")).strip()
        if not text:
            continue
        back_links = []
        for a in p.find_all("a"):
            ahref = _safe_text(a.get("href") or "").strip()
            if "#" in ahref:
                a_frag = ahref.rsplit("#", 1)[1].strip()
                if re.match(r"^st\d+[a-z]?$", a_frag, re.IGNORECASE):
                    back_links.append((a, a_frag))
        if not back_links:
            if not is_footnote_block:
                continue
            a = p.find("a")
            if a is not None:
                back_links.append((a, ""))
        for back_a, _back_frag in back_links[:1]:
            marker_raw = _safe_text(back_a.get_text(" ")).strip()
            if not marker_raw:
                marker_raw = rst_m.group(1)
            marker_norm = _normalize_marker(marker_raw)
            if not marker_norm:
                continue
            body = _body_after_lead_marker(text, marker_norm)
            if not body:
                body = text
            id_map[tag_id] = body
            marker_defs.append({"id": tag_id, "marker": marker_norm, "text": body, "tag_name": p.name or "p"})
            break

    # Phase 2b: Book 12 direction — st{N} definitions (in separate footnote files).
    for p in soup.find_all(["p", "div", "li", "blockquote"]):
        tag_id = _safe_text(p.get("id") or "").strip()
        st_m = _st_rst_id_re.match(tag_id)
        if not st_m:
            continue
        text = _safe_text(p.get_text(" ")).strip()
        if not text:
            continue
        back_links = []
        for a in p.find_all("a"):
            ahref = _safe_text(a.get("href") or "").strip()
            if "#" in ahref:
                a_frag = ahref.rsplit("#", 1)[1].strip()
                if re.match(r"^rst\d+[a-z]?$", a_frag, re.IGNORECASE):
                    back_links.append((a, a_frag))
        if not back_links:
            continue
        for back_a, _back_frag in back_links[:1]:
            marker_raw = _safe_text(back_a.get_text(" ")).strip()
            if not marker_raw:
                marker_raw = st_m.group(1)
            marker_norm = _normalize_marker(marker_raw)
            if not marker_norm:
                continue
            body = _body_after_lead_marker(text, marker_norm)
            if not body:
                body = text
            id_map[tag_id] = body
            marker_defs.append({"id": tag_id, "marker": marker_norm, "text": body, "tag_name": p.name or "p"})
            break

    return id_map, anchors, marker_defs


def _harvest_footnote_class_definitions(soup: BeautifulSoup) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Harvest footnote definitions from elements with class='footnote' or class='footnotet'.

    This handles well-formed EPUBs where definitions are explicitly marked with CSS
    footnote classes (common in calibre-produced scholarly editions). Works even when
    the _rfn id convention is not used.
    """

    id_map: Dict[str, str] = {}
    marker_defs: List[Dict[str, Any]] = []

    if soup is None:
        return id_map, marker_defs

    for p in soup.select(".footnote, .footnotet, .noindent-x1"):
        tag_id = _safe_text(p.get("id") or "").strip()

        # Find the marker from the back-link <a> element.
        a = p.find("a")
        marker_raw = _safe_text(a.get_text(" ")).strip() if a is not None else ""
        if not marker_raw and a is not None:
            marker_raw = _safe_text(a.get("href") or "").strip()
            if "#" in marker_raw:
                marker_raw = marker_raw.rsplit("#", 1)[-1]

        text = _safe_text(p.get_text(" ")).strip()
        if not text:
            continue

        m = _def_line_regex().match(text)
        if m:
            marker_norm = _normalize_marker(_safe_text(m.group(1)))
            def_text = _safe_text(m.group(2))
            if marker_norm and def_text:
                marker_defs.append({
                    "id": tag_id,
                    "marker": marker_norm,
                    "text": def_text,
                    "tag_name": p.name or "p",
                })
                if tag_id:
                    id_map[tag_id] = def_text
        else:
            marker_norm = _normalize_marker(marker_raw) if marker_raw else ""
            if marker_norm and len(text) > len(marker_raw) + 2:
                body = text
                stripped = text.strip()
                if stripped.lower().startswith(marker_raw.lower()):
                    body = stripped[len(marker_raw):].strip()
                    body = re.sub(r"^[\]\)\.:\-—]\s*", "", body)
                if body:
                    marker_defs.append({
                        "id": tag_id,
                        "marker": marker_norm,
                        "text": body,
                        "tag_name": p.name or "p",
                    })
                    if tag_id:
                        id_map[tag_id] = body

    return id_map, marker_defs


def _extract_bare_digit_anchors_from_text(text: str, allowed_markers: List[str]) -> List[Dict[str, Any]]:
    """Find superscript-like bare digits in running text.

    Only searches for digits that appear as definition markers to reduce false positives.
    """

    anchors: List[Dict[str, Any]] = []
    if not text or not allowed_markers:
        return anchors

    allowed_nums = [m for m in allowed_markers if re.fullmatch(r"\d{1,3}", m or "")]
    # Be conservative: single-digit bare suffixes are extremely ambiguous in scholarly
    # texts (e.g., linguistic bases like "PA3"). Only attempt bare-digit anchors for
    # multi-digit markers.
    allowed_nums = [n for n in allowed_nums if 10 <= int(n) <= 200]
    if not allowed_nums:
        return anchors

    # Look for word+number patterns like "word12".
    # Important: avoid matching inside larger numbers (e.g. matching "3" inside "13").
    for n in allowed_nums:
        rx = re.compile(rf"(?<=[^\W\d_]){re.escape(n)}(?!\d)")
        for m in rx.finditer(text):
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            context = _safe_text(text[start:end])
            if not anchor_is_probable_footnote(n, n, context, has_href=False):
                continue
            anchors.append(
                {
                    "marker_raw": n,
                    "marker": n,
                    "position": m.start(),
                    "context": context,
                    "_has_href": False,
                }
            )
    return anchors


# Split points for inline numeric definitions inside a single extracted line.
#
# Some extractors output compact runs like:
#   "11. ... 12.The next ... 13. ..."
# (note missing whitespace after the dot). We accept optional whitespace and
# keep additional conservative guards in _split_inline_numeric_definitions.
_INLINE_DEF_SPLIT_RE = re.compile(r"(?<!\d)(\d{1,3})\.\s*", re.UNICODE)


def _split_inline_numeric_definitions(line: str) -> List[str]:
    """Split a single extracted line that contains multiple numeric definitions."""

    s = line or ""
    if not s:
        return [s]

    matches = list(_INLINE_DEF_SPLIT_RE.finditer(s))
    if not matches:
        return [s]

    split_positions: List[int] = []
    for m in matches:
        start = m.start(1)
        if start <= 0:
            continue

        left = s[:start].rstrip()
        if not left:
            continue

        # Avoid splitting page refs like "p. 93." / "pp. 93.".
        # IMPORTANT: only treat p./pp. as a standalone token. Words like "map."
        # must NOT trigger this guard.
        left_l = left.lower()
        # Also avoid splitting on page-range abbreviations like "pp. 21-2." or
        # "pp. 233 - 4, 281 - 2." where the trailing "2." is a continuation of
        # a page span, not a footnote marker.
        if (
            re.search(r"(?:^|\s)pp?\.", left_l)
            and re.search(r"\b\d{1,4}\s*[-–—]\s*$", left_l)
        ):
            continue
        if re.search(r"(?:^|\s)pp?\.$", left_l):
            continue

        # Avoid splitting common critical-edition citations like "VI.335." / "VII.448.".
        # These are volume.page references, not footnote definition markers.
        # If the token immediately preceding the digits looks like a Roman numeral + '.',
        # treat it as a citation boundary and keep it within the current definition.
        if re.search(r"\b[IVXLC]{1,6}\.$", left):
            continue

        # Require a non-alphanumeric boundary before the marker (e.g., ")", ".", ":").
        prev = left[-1]
        if prev.isalnum():
            continue

        split_positions.append(start)

    if not split_positions:
        return [s]

    out: List[str] = []
    last = 0
    for pos in split_positions:
        chunk = s[last:pos].strip()
        if chunk:
            out.append(chunk)
        last = pos
    tail = s[last:].strip()
    if tail:
        out.append(tail)
    return out


def _extract_definitions_from_lines(lines: List[str], start_index: int) -> List[Dict[str, Any]]:
    """Parse definition blocks from lines[start_index:].

    Handles multi-line definitions by accumulating lines until next marker line.
    """

    # Pre-compute character offsets for each line so definitions carry a
    # position that proximity-based pairing can use.
    _line_offsets: List[int] = []
    _off = 0
    for _ln in lines:
        _line_offsets.append(_off)
        _off += len(_ln) + 1  # +1 for newline

    defs: list[dict] = []
    current = None
    double_numeric_marker_only_re = re.compile(r"^\s*(\d{1,3})\.\s*(\d{1,3})\.\s*$", re.UNICODE)
    page_ref_re = re.compile(r"^\s*p{1,2}\.\s*\d", re.IGNORECASE)
    marker_only_re = re.compile(
        r"^\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|\*+|ΓÇá+|ΓÇí+|┬º+)\s*(?:\]|\))?\s*(?:[\]\)\.\:\-ΓÇö]\s*)?(?:Γå⌐|\u21A9)?\s*$",
        re.UNICODE,
    )
    def_like_re = _def_line_regex()

    def _looks_like_notes_to_prose_boundary(line0: str, idx0: int, *, allow_strong_single: bool) -> bool:
        """Return True if line0 looks like a real section heading that ends a notes block."""

        t0 = _safe_text(line0 or "")
        if not t0:
            return False
        if _is_notes_header_line(t0):
            return False
        if def_like_re.match(t0) or marker_only_re.match(t0) or page_ref_re.match(t0):
            return False

        # Normalize: strip trailing footnote-like markers that often appear on headings.
        t0c = _strip_trailing_footnote_marker_from_heading(t0) or t0
        u0 = t0c.upper().strip()

        # Strong single-line boundaries that frequently occur in critical editions.
        # Example:
        #   [PART TWO].(1)
        #   Night 62.(2) Thursday, March 6th, 1987.
        part_line = allow_strong_single and bool(re.match(r"^\s*\[?\s*PART\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\b", u0))
        night_line = allow_strong_single and bool(
            re.match(r"^\s*NIGHT\s+\d{1,3}\s*\.(?:\s*\(\s*\d{1,3}\s*\))?\s+\S", t0, re.IGNORECASE)
            or re.match(r"^\s*NIGHT\s+\d{1,3}\s*(?:\(\s*\d{1,3}\s*\))\s+\S", t0, re.IGNORECASE)
        )

        looks_like_heading = _looks_like_chapter_heading_text(t0c) or _line_looks_like_heading_component(t0c)
        if not (looks_like_heading or part_line or night_line):
            return False

        # Look ahead: next non-empty line must not be a definition marker.
        # For generic headings, require a second heading-ish line (multiline title card).
        # For strong boundaries (PART/Night), allow a single heading line.
        saw_second_heading = False
        for k in range(idx0 + 1, min(idx0 + 12, len(lines))):
            tk = _safe_text(lines[k] or "")
            if not tk:
                continue
            tkc = _strip_trailing_footnote_marker_from_heading(tk) or tk
            if def_like_re.match(tk) or marker_only_re.match(tk):
                return False
            if _is_notes_header_line(tk):
                return False
            if part_line or night_line:
                # No extra requirement; this is already a strong boundary.
                break
            if _looks_like_chapter_heading_text(tkc) or _line_looks_like_heading_component(tkc):
                saw_second_heading = True
            break

        if not (part_line or night_line) and not saw_second_heading:
            return False

        # Guard: notes often *quote* multi-line title-page blocks inside a single
        # definition (e.g. "Leaves from the Club Papers / II / ...").
        # If a new definition marker starts immediately after such a quoted block,
        # we are still in notes and must NOT stop parsing.
        # Wider guard: notes often quote heading/title-card blocks inside a definition.
        # If we see another definition marker within a reasonable lookahead, we're
        # almost certainly still in notes and must NOT stop parsing.
        non_empty_seen = 0
        for k in range(idx0 + 1, min(idx0 + 220, len(lines))):
            tk = _safe_text(lines[k] or "")
            if not tk:
                continue
            non_empty_seen += 1
            if def_like_re.match(tk) or marker_only_re.match(tk):
                return False
            if non_empty_seen >= 40:
                break
        return True

    defs_seen = 0
    blank_run = 0
    for idx in range(start_index, len(lines)):
        raw_line = lines[idx].rstrip("\n")
        if not _safe_text(raw_line).strip():
            blank_run += 1
        else:
            # We'll reset this after processing the non-empty line.
            pass

        # Recovery: if notes-split inference was too aggressive and prose resumes,
        # stop parsing definitions so we don't swallow whole chapters.
        if current is not None and _looks_like_notes_to_prose_boundary(raw_line, idx, allow_strong_single=(defs_seen >= 12)):
            current["text"] = _safe_text(current["text"])
            defs.append(current)
            return [d for d in defs if _safe_text(d.get("text") or "").strip()]

        chunks = _split_inline_numeric_definitions(raw_line)
        for ci, line in enumerate(chunks):
            next_line = chunks[ci + 1] if (ci + 1) < len(chunks) else ""

            next_nonempty_line = ""
            if next_line and _safe_text(next_line or ""):
                next_nonempty_line = next_line
            else:
                for probe_idx in range(idx + 1, min(idx + 7, len(lines))):
                    probe_line = _safe_text(lines[probe_idx] or "")
                    if probe_line:
                        next_nonempty_line = probe_line
                        break

            # If we just had a blank-line break and the chapter resumes with
            # enumerated prose sections like "(ii) ...", do not append it to the
            # prior numeric note definition.
            try:
                if current is not None and blank_run >= 1:
                    cur_mk = str(current.get("marker") or "").strip()
                    if re.fullmatch(r"\d{1,3}", cur_mk) and looks_like_post_notes_section_enumerator_line(line):
                        current["text"] = _safe_text(current["text"])
                        defs.append(current)
                        current = None
                        continue
            except Exception:
                pass
            # Critical editions sometimes interleave page references inside notes blocks.
            # Example: "p. 93. ..." which should be part of the current note, not a new marker.
            if page_ref_re.match(line):
                if current and line.strip():
                    current["text"] += " " + line.strip()
                continue

            # Some apparatus formats encode note markers as "<page>. <note>." on its own line.
            # Example: "23. 3." or "22. 17." followed by the note text on the next line.
            dm = double_numeric_marker_only_re.match(line)
            if dm:
                if current:
                    current["text"] = _safe_text(current["text"])
                    defs.append(current)
                marker_norm = _normalize_marker(dm.group(2))
                defs_seen += 1
                current = {
                    "marker": marker_norm,
                    "text": "",
                    "line_index": idx,
                }
                continue

            # Handle the common two-line format:
            #   1.
            #    Definition text...
            mo = marker_only_re.match(line)
            if mo and not _def_line_regex().match(line):
                # Guard: inline-splitting can produce a stray marker-only chunk at the
                # start of a wrapped line like "... notes 3 and 4. 2. The name ...".
                # In that case, the marker-only token ("4.") is part of the previous
                # definition's prose, not a new note.
                if current and next_line and def_like_re.match(next_line):
                    cur_mk = str(current.get("marker") or "").strip()
                    marker_norm_peek = _normalize_marker(mo.group(1))
                    next_m = def_like_re.match(next_line)
                    next_mk = _normalize_marker(next_m.group(1)) if next_m else ""
                    if re.fullmatch(r"\d{1,3}", str(marker_norm_peek or "")):
                        current["text"] += " " + line.strip()
                        continue
                    if (
                        re.fullmatch(r"\d{1,3}", cur_mk)
                        and re.fullmatch(r"[A-Za-z]", str(marker_norm_peek or ""))
                        and re.fullmatch(r"\d{1,3}", str(next_mk or ""))
                    ):
                        current["text"] += " " + line.strip()
                        continue
                if current and looks_like_false_numeric_crossref_restart(
                    line,
                    current_marker=str(current.get("marker") or ""),
                    current_text=str(current.get("text") or ""),
                    next_definition_marker=(
                        _normalize_marker(def_like_re.match(next_nonempty_line).group(1))
                        if next_nonempty_line and def_like_re.match(next_nonempty_line)
                        else ""
                    ),
                ):
                    current["text"] += " " + line.strip()
                    continue
                if current:
                    current["text"] = _safe_text(current["text"])
                    defs.append(current)
                marker_norm = _normalize_marker(mo.group(1))
                defs_seen += 1
                current = {
                    "marker": marker_norm,
                    "text": "",
                    "line_index": idx,
                }
                continue

            m = def_like_re.match(line)
            if m:
                # Avoid splitting a definition on nested parenthesized numbering.
                line_stripped = line.lstrip()
                if current:
                    cur_mk = str(current.get("marker") or "").strip()
                    new_mk = _normalize_marker(m.group(1))
                    if looks_like_false_numeric_crossref_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        next_definition_marker=(
                            _normalize_marker(def_like_re.match(next_nonempty_line).group(1))
                            if next_nonempty_line and def_like_re.match(next_nonempty_line)
                            else ""
                        ),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_date_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_editorial_reference_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_age_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_bibliographic_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_single_letter_definition_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if (
                        line_stripped.startswith("(")
                        and re.fullmatch(r"\d{1,3}", cur_mk)
                        and re.fullmatch(r"\d{1,3}", new_mk)
                    ):
                        try:
                            if int(new_mk) <= int(cur_mk):
                                if line.strip():
                                    current["text"] += " " + line.strip()
                                continue
                        except Exception:
                            pass

                if current:
                    current["text"] = _safe_text(current["text"])
                    defs.append(current)
                marker_norm = _normalize_marker(m.group(1))
                defs_seen += 1
                current = {
                    "marker": marker_norm,
                    "text": m.group(2).strip(),
                    "line_index": idx,
                }
                continue

            if current and line.strip():
                current["text"] += " " + line.strip()

        if _safe_text(raw_line).strip():
            blank_run = 0

    if current:
        current["text"] = _safe_text(current["text"])
        defs.append(current)

    # Drop empty-text definitions (can occur with stray marker-only lines
    # at page/spine boundaries or when marker-only lines are separated from
    # their body by non-text artifacts).
    kept = [d for d in defs if _safe_text(d.get("text") or "").strip()]
    # Populate char_position from line_index so proximity-based pairing can use it.
    for d in kept:
        li = d.get("line_index")
        if isinstance(li, int) and 0 <= li < len(_line_offsets):
            d["char_position"] = _line_offsets[li]
    return kept


def _extract_definitions_from_lines_scoped(
    lines: List[str],
    start_index: int,
    *,
    initial_chapter_token: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Like _extract_definitions_from_lines, but tags each definition with a chapter token."""

    # Pre-compute character offsets for each line.
    _line_offsets_scoped: List[int] = []
    _off = 0
    for _ln in lines:
        _line_offsets_scoped.append(_off)
        _off += len(_ln) + 1  # +1 for newline

    def _token_from_heading_line(line: str) -> Optional[int]:
        t = _safe_text(line)
        if not t:
            return None
        # Tolerate extra punctuation and short trailing text.
        m = re.match(
            r"^\s*(?:FOOTNOTES|FOOTNOTES\s+AND\s+ENDNOTES|ENDNOTES|NOTES)\b(?:\s+(?:TO|ON)\s+CHAPTER\s+([IVXLC]{1,12}|\d{1,3}))\b.*$",
            t,
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

    defs: list[dict] = []
    current = None
    current_token = initial_chapter_token

    double_numeric_marker_only_re = re.compile(r"^\s*(\d{1,3})\.\s*(\d{1,3})\.\s*$", re.UNICODE)
    page_ref_re = re.compile(r"^\s*p{1,2}\.\s*\d", re.IGNORECASE)
    marker_only_re = re.compile(
        r"^\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|\*+|ΓÇá+|ΓÇí+|┬º+)\s*(?:\]|\))?\s*(?:[\]\)\.\:\-ΓÇö]\s*)?(?:Γå⌐|\u21A9)?\s*$",
        re.UNICODE,
    )
    def_like_re = _def_line_regex()

    def _looks_like_notes_to_prose_boundary(line0: str, idx0: int, *, allow_strong_single: bool) -> bool:
        """Return True if line0 looks like a real section heading that ends a notes block."""

        t0 = _safe_text(line0 or "")
        if not t0:
            return False
        if _is_notes_header_line(t0):
            return False
        if def_like_re.match(t0) or marker_only_re.match(t0) or page_ref_re.match(t0):
            return False

        t0c = _strip_trailing_footnote_marker_from_heading(t0) or t0
        u0 = t0c.upper().strip()

        part_line = allow_strong_single and bool(re.match(r"^\s*\[?\s*PART\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\b", u0))
        night_line = allow_strong_single and bool(
            re.match(r"^\s*NIGHT\s+\d{1,3}\s*\.(?:\s*\(\s*\d{1,3}\s*\))?\s+\S", t0, re.IGNORECASE)
            or re.match(r"^\s*NIGHT\s+\d{1,3}\s*(?:\(\s*\d{1,3}\s*\))\s+\S", t0, re.IGNORECASE)
        )

        looks_like_heading = _looks_like_chapter_heading_text(t0c) or _line_looks_like_heading_component(t0c)
        if not (looks_like_heading or part_line or night_line):
            return False

        saw_second_heading = False
        for k in range(idx0 + 1, min(idx0 + 12, len(lines))):
            tk = _safe_text(lines[k] or "")
            if not tk:
                continue
            tkc = _strip_trailing_footnote_marker_from_heading(tk) or tk
            if def_like_re.match(tk) or marker_only_re.match(tk):
                return False
            if _is_notes_header_line(tk):
                return False
            if part_line or night_line:
                break
            if _looks_like_chapter_heading_text(tkc) or _line_looks_like_heading_component(tkc):
                saw_second_heading = True
            break

        if not (part_line or night_line) and not saw_second_heading:
            return False

        non_empty_seen = 0
        for k in range(idx0 + 1, min(idx0 + 220, len(lines))):
            tk = _safe_text(lines[k] or "")
            if not tk:
                continue
            non_empty_seen += 1
            if def_like_re.match(tk) or marker_only_re.match(tk):
                return False
            if non_empty_seen >= 40:
                break
        return True

    defs_seen = 0
    blank_run = 0
    for idx in range(start_index, len(lines)):
        raw_line = lines[idx].rstrip("\n")

        if not _safe_text(raw_line).strip():
            blank_run += 1
        else:
            # We'll reset this after processing the non-empty line.
            pass

        # Recovery: if notes-split inference was too aggressive and prose resumes,
        # stop parsing definitions so we don't swallow whole chapters.
        if current is not None and _looks_like_notes_to_prose_boundary(raw_line, idx, allow_strong_single=(defs_seen >= 12)):
            current["text"] = _safe_text(current["text"])
            defs.append(current)
            return [d for d in defs if _safe_text(d.get("text") or "").strip()]

        chunks = _split_inline_numeric_definitions(raw_line)
        for ci, line in enumerate(chunks):
            next_line = chunks[ci + 1] if (ci + 1) < len(chunks) else ""

            next_nonempty_line = ""
            if next_line and _safe_text(next_line or ""):
                next_nonempty_line = next_line
            else:
                for probe_idx in range(idx + 1, min(idx + 7, len(lines))):
                    probe_line = _safe_text(lines[probe_idx] or "")
                    if probe_line:
                        next_nonempty_line = probe_line
                        break

            # If we just had a blank-line break and the chapter resumes with
            # enumerated prose sections like "(ii) ...", do not append it to the
            # prior numeric note definition.
            try:
                if current is not None and blank_run >= 1:
                    cur_mk = str(current.get("marker") or "").strip()
                    if re.fullmatch(r"\d{1,3}", cur_mk) and looks_like_post_notes_section_enumerator_line(line):
                        current["text"] = _safe_text(current["text"])
                        defs.append(current)
                        current = None
                        continue
            except Exception:
                pass
            # Update current chapter token when we see a notes-to-chapter header.
            tok = _token_from_heading_line(line)
            if tok is not None:
                current_token = tok
                continue

            if page_ref_re.match(line):
                if current and line.strip():
                    current["text"] += " " + line.strip()
                continue

            dm = double_numeric_marker_only_re.match(line)
            if dm:
                if current:
                    current["text"] = _safe_text(current["text"])
                    defs.append(current)
                marker_norm = _normalize_marker(dm.group(2))
                defs_seen += 1
                current = {
                    "marker": marker_norm,
                    "text": "",
                    "line_index": idx,
                    "chapter_token": current_token,
                }
                continue

            mo = marker_only_re.match(line)
            if mo and not _def_line_regex().match(line):
                if current and next_line and def_like_re.match(next_line):
                    cur_mk = str(current.get("marker") or "").strip()
                    marker_norm_peek = _normalize_marker(mo.group(1))
                    next_m = def_like_re.match(next_line)
                    next_mk = _normalize_marker(next_m.group(1)) if next_m else ""
                    if re.fullmatch(r"\d{1,3}", str(marker_norm_peek or "")):
                        current["text"] += " " + line.strip()
                        continue
                    if (
                        re.fullmatch(r"\d{1,3}", cur_mk)
                        and re.fullmatch(r"[A-Za-z]", str(marker_norm_peek or ""))
                        and re.fullmatch(r"\d{1,3}", str(next_mk or ""))
                    ):
                        current["text"] += " " + line.strip()
                        continue
                if current and looks_like_false_numeric_crossref_restart(
                    line,
                    current_marker=str(current.get("marker") or ""),
                    current_text=str(current.get("text") or ""),
                    next_definition_marker=(
                        _normalize_marker(def_like_re.match(next_nonempty_line).group(1))
                        if next_nonempty_line and def_like_re.match(next_nonempty_line)
                        else ""
                    ),
                ):
                    current["text"] += " " + line.strip()
                    continue
                if current:
                    current["text"] = _safe_text(current["text"])
                    defs.append(current)
                marker_norm = _normalize_marker(mo.group(1))
                defs_seen += 1
                current = {
                    "marker": marker_norm,
                    "text": "",
                    "line_index": idx,
                    "chapter_token": current_token,
                }
                continue

            m = def_like_re.match(line)
            if m:
                line_stripped = line.lstrip()
                if current:
                    cur_mk = str(current.get("marker") or "").strip()
                    new_mk = _normalize_marker(m.group(1))
                    if looks_like_false_numeric_crossref_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        next_definition_marker=(
                            _normalize_marker(def_like_re.match(next_nonempty_line).group(1))
                            if next_nonempty_line and def_like_re.match(next_nonempty_line)
                            else ""
                        ),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_date_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_editorial_reference_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_age_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_numeric_bibliographic_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if looks_like_false_single_letter_definition_restart(
                        line,
                        current_marker=cur_mk,
                        current_text=str(current.get("text") or ""),
                        matched_marker_raw=str(m.group(1) or ""),
                        matched_body=str(m.group(2) or ""),
                    ):
                        if line.strip():
                            current["text"] += " " + line.strip()
                        continue
                    if (
                        line_stripped.startswith("(")
                        and re.fullmatch(r"\d{1,3}", cur_mk)
                        and re.fullmatch(r"\d{1,3}", new_mk)
                    ):
                        try:
                            if int(new_mk) <= int(cur_mk):
                                if line.strip():
                                    current["text"] += " " + line.strip()
                                continue
                        except Exception:
                            pass

                if current:
                    current["text"] = _safe_text(current["text"])
                    defs.append(current)
                marker_norm = _normalize_marker(m.group(1))
                defs_seen += 1
                current = {
                    "marker": marker_norm,
                    "text": m.group(2).strip(),
                    "line_index": idx,
                    "chapter_token": current_token,
                }
                continue

            if current and line.strip():
                current["text"] += " " + line.strip()

        if _safe_text(raw_line).strip():
            blank_run = 0

    if current:
        current["text"] = _safe_text(current["text"])
        defs.append(current)

    # Populate char_position from line_index.
    for d in defs:
        li = d.get("line_index")
        if isinstance(li, int) and 0 <= li < len(_line_offsets_scoped):
            d["char_position"] = _line_offsets_scoped[li]
    return defs


def _extract_anchors_from_text(text: str) -> List[Dict[str, Any]]:
    """Fallback: extract candidate anchors via regex."""

    anchors: list[dict] = []
    rx = _marker_regex()
    text = text or ""

    matches = list(rx.finditer(text))

    def _marker_category(raw: str) -> str:
        if re.fullmatch(r"\(\s*\d{1,3}\s*\)", raw):
            return "num_paren"
        if re.fullmatch(r"\[\s*\d{1,3}\s*\]", raw):
            return "num_bracket"
        if re.fullmatch(r"[Γü░┬╣┬▓┬│Γü┤Γü╡Γü╢Γü╖Γü╕Γü╣]{1,4}", raw):
            return "num_sup"
        if re.fullmatch(r"\(\s*[a-zA-Z]\s*\)", raw):
            return "let_paren"
        if re.fullmatch(r"\[\s*[a-zA-Z]\s*\]", raw):
            return "let_bracket"
        if re.fullmatch(r"[\*ΓÇáΓÇí┬º]+", raw):
            return "symbol"
        return "other"

    # If there is a strong established pattern in this chapter/text, suppress unlikely categories.
    counts: Dict[str, int] = defaultdict(int)
    for m in matches:
        counts[_marker_category(m.group(0))] += 1

    total = sum(counts.values())
    dominant = None
    if total >= 6:
        top_cat, top_count = max(counts.items(), key=lambda kv: kv[1])
        if top_count >= 5 and (top_count / max(1, total)) >= 0.8:
            dominant = top_cat

    suppress_letter_markers = dominant in {"num_paren", "num_bracket"}

    # If numeric markers are clearly using one delimiter style, suppress the other.
    paren_num_count = int(counts.get("num_paren", 0) or 0)
    bracket_num_count = int(counts.get("num_bracket", 0) or 0)

    suppress_bracket_numeric = dominant == "num_paren" or (paren_num_count >= 3 and bracket_num_count <= 1)
    suppress_paren_numeric = dominant == "num_bracket" or (bracket_num_count >= 3 and paren_num_count <= 1)

    for m in matches:
        raw = m.group(0)
        cat = _marker_category(raw)
        if suppress_letter_markers and cat in {"let_paren", "let_bracket"}:
            continue
        if suppress_bracket_numeric and cat == "num_bracket":
            continue
        if suppress_paren_numeric and cat == "num_paren":
            continue

        # Conservative guard: avoid treating likely mathematical exponents like "10┬▓" as footnotes.
        if cat == "num_sup":
            prev_ch = text[m.start() - 1] if (m.start() - 1) >= 0 else ""
            if prev_ch.isdigit():
                continue

        marker_norm = _normalize_marker(raw)
        if not marker_norm:
            continue

        # Ignore editorial insertions like "Sydn[e]y" / "Name[s]".
        if raw.startswith("[") and raw.endswith("]") and re.fullmatch(r"[A-Za-z]", marker_norm):
            prev_ch = text[m.start() - 1] if m.start() - 1 >= 0 else ""
            next_ch = text[m.end()] if m.end() < len(text) else ""
            if prev_ch.isalpha() and next_ch.isalpha():
                continue

        # Context for AI and UI.
        #
        # IMPORTANT: for multi-digit numeric anchors (>=10), allow a wider probe
        # window for plausibility checks. Some EPUB extracts insert extreme
        # whitespace/newlines, so date cues like "8 March" can be hundreds of
        # characters away from the marker and would otherwise be missed.
        display_radius = 80
        probe_radius = display_radius
        try:
            if re.fullmatch(r"\d{1,3}", marker_norm) and int(marker_norm) >= 10:
                probe_radius = 650
        except Exception:
            probe_radius = display_radius

        start = max(0, m.start() - display_radius)
        end = min(len(text), m.end() + display_radius)
        context = _safe_text(text[start:end])

        pstart = max(0, m.start() - probe_radius)
        pend = min(len(text), m.end() + probe_radius)
        probe_context = _safe_text(text[pstart:pend])

        if len(context) < 8:
            continue
        if not anchor_is_probable_footnote(raw, marker_norm, probe_context, has_href=False):
            continue
        anchors.append(
            {
                "marker_raw": raw,
                "marker": marker_norm,
                "position": m.start(),
                "context": context,
                "_has_href": False,
            }
        )
    return anchors


def _filter_anchors_by_profile(anchors: List[Dict[str, Any]], allowed_categories: Optional[set[str]]) -> List[Dict[str, Any]]:
    if not anchors or not allowed_categories:
        return anchors
    out: List[Dict[str, Any]] = []
    for a in anchors:
        raw = (a.get("marker_raw") or a.get("marker") or "").strip()
        cat = _marker_category_from_raw(raw)
        if cat in allowed_categories:
            out.append(a)
    return out


def _filter_definitions_by_profile(definitions: List[Dict[str, Any]], allowed_categories: Optional[set[str]]) -> List[Dict[str, Any]]:
    if not definitions or not allowed_categories:
        return definitions

    out: List[Dict[str, Any]] = []
    for d in definitions:
        mk = (d.get("marker") or "").strip()
        if not mk:
            continue
        cat = _marker_category_from_raw(mk)
        # Normalized digits will be "num_plain".
        if cat in allowed_categories:
            out.append(d)
    return out


def _pair_anchors_to_definitions(
    anchors: List[Dict[str, Any]],
    definitions: List[Dict[str, Any]],
    source_meta: Dict[str, Any],
    definitions_by_id: Optional[Dict[str, str]] = None,
    id_start: int = 0,
    *,
    forward_looking: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    """Pair anchors to definitions using marker matching + AI fallback.

    Returns (results, next_id)
    """

    defs_by_marker: dict[str, list[dict]] = defaultdict(list)
    for d in definitions:
        defs_by_marker[d["marker"]].append(d)

    def _anchor_group(a: Dict[str, Any]) -> Optional[str]:
        g = a.get("_chapter_group")
        if g is None:
            g = source_meta.get("chapter_group")
        return str(g) if g is not None else None

    anchor_total_by_marker_group: Dict[tuple[str, Optional[str]], int] = defaultdict(int)
    for a in anchors:
        mk = a.get("marker")
        if mk:
            anchor_total_by_marker_group[(mk, _anchor_group(a))] += 1

    numeric_anchor_in_sequence: Dict[int, bool] = {}

    def _lnds_positions(values: List[int]) -> set[int]:
        """Return positions (0..n-1) that form one Longest Non-Decreasing Subsequence."""
        if not values:
            return set()
        tails: List[int] = []
        tails_idx: List[int] = []
        prev: List[int] = [-1] * len(values)

        for i, v in enumerate(values):
            pos = bisect.bisect_right(tails, v)
            if pos == len(tails):
                tails.append(v)
                tails_idx.append(i)
            else:
                tails[pos] = v
                tails_idx[pos] = i
            prev[i] = tails_idx[pos - 1] if pos > 0 else -1

        seq: set[int] = set()
        k = tails_idx[-1] if tails_idx else -1
        while k != -1:
            seq.add(k)
            k = prev[k]
        return seq

    numeric_by_group: Dict[Optional[str], List[Tuple[int, int]]] = defaultdict(list)
    for anchor_index, a in enumerate(anchors):
        mk = (a.get("marker") or "").strip()
        if not re.fullmatch(r"\d{1,3}", mk):
            continue
        try:
            v = int(mk)
        except Exception:
            continue
        numeric_by_group[_anchor_group(a)].append((anchor_index, v))

    for gkey, items in numeric_by_group.items():
        if len(items) <= 2:
            for anchor_index, _ in items:
                numeric_anchor_in_sequence[anchor_index] = True
            continue

        values = [v for _, v in items]
        keep_positions = _lnds_positions(values)
        for pos, (anchor_index, _) in enumerate(items):
            numeric_anchor_in_sequence[anchor_index] = pos in keep_positions

    results: list[dict] = []
    next_id = id_start

    marker_seen_count: dict[tuple[str, Optional[str]], int] = defaultdict(int)
    ai_batch: list[dict] = []

    for anchor_index, a in enumerate(anchors):
        marker = a.get("marker") or ""
        if not marker:
            continue

        group_key = _anchor_group(a)
        marker_seen_count[(marker, group_key)] += 1
        occurrence_index = marker_seen_count[(marker, group_key)] - 1

        candidates_all = defs_by_marker.get(marker, [])

        local_all = [c for c in candidates_all if c.get("line_index", -1) != -1]
        global_all = [c for c in candidates_all if c.get("line_index", -1) == -1]

        global_filtered = global_all
        if global_all and group_key is not None:
            tagged = [c for c in global_all if c.get("chapter_group") is not None]
            if tagged:
                group_match = [c for c in global_all if str(c.get("chapter_group")) == str(group_key)]
                if group_match:
                    global_filtered = group_match
                else:
                    ambiguous = [c for c in global_all if c.get("chapter_group") is None]
                    global_filtered = ambiguous if ambiguous else global_all

        local_candidates = local_all
        global_candidates = global_filtered

        # Prefer local definitions when any exist.
        # Rationale: mixing local + global candidates can make numeric markers
        # appear ambiguous even when the local (same-spine or spine-bridged)
        # definition is uniquely correct.
        if local_candidates:
            candidates = local_candidates
            global_candidates = []
        else:
            candidates = global_candidates
        suggested_def = None
        confidence_score = 0.0
        confidence = "Manual Review Required"
        match_method = "none"

        # 0) If the EPUB provides a fragment id, use it (highest confidence)
        href = a.get("href")
        if definitions_by_id and href and "#" in href:
            frag = href.split("#", 1)[1]
            frag = (frag or "").strip()
            if frag:
                def_text = definitions_by_id.get(frag)
                if def_text:
                    suggested_def = def_text
                    confidence_score = 0.98
                    confidence = "High (ID Link)"
                    match_method = "id_link"

        if match_method != "id_link":
            anchor_total = anchor_total_by_marker_group.get((marker, group_key), 1)

            if len(local_candidates) == 0 and len(global_candidates) > 1 and not re.fullmatch(r"\d{1,3}", marker.strip()):
                chap_idx = source_meta.get("chapter_index")
                if isinstance(chap_idx, int):
                    with_idx = [c for c in global_candidates if isinstance(c.get("origin_index"), int)]
                    if with_idx:
                        with_idx.sort(key=lambda c: abs(int(c.get("origin_index")) - chap_idx))
                        best = with_idx[0]
                        best_dist = abs(int(best.get("origin_index")) - chap_idx)
                        second_dist = None
                        if len(with_idx) >= 2:
                            second_dist = abs(int(with_idx[1].get("origin_index")) - chap_idx)

                        if best_dist <= 2 or (second_dist is not None and (best_dist * 2) < second_dist):
                            suggested_def = best.get("text")
                            confidence_score = 0.65
                            confidence = "Medium (Nearest Notes)"
                            match_method = "global_nearest"

            if len(candidates) == 0:
                match_method = "none"

            elif len(candidates) == 1:
                if anchor_total == 1:
                    suggested_def = candidates[0]["text"]
                    if len(local_candidates) == 0 and len(global_candidates) == 1 and re.fullmatch(r"\d{1,3}", marker.strip()):
                        confidence_score = 0.65
                        confidence = "Medium (Global Marker)"
                        match_method = "global_unique"
                    else:
                        confidence_score = 0.9
                        confidence = "High (Marker Match)"
                        match_method = "marker_unique"
                else:
                    if anchor_total <= 3:
                        suggested_def = candidates[0]["text"]
                        in_seq = numeric_anchor_in_sequence.get(anchor_index, True)
                        if in_seq:
                            confidence_score = 0.85
                            confidence = "High (Repeated Marker)"
                        else:
                            confidence_score = 0.55
                            confidence = "Low (Repeated Marker / Out of Order)"
                        match_method = "marker_repeat_first" if occurrence_index == 0 else "marker_repeat_reuse"
                    else:
                        confidence_score = 0.50
                        confidence = "Manual Review Required"
                        match_method = "marker_repeat_unpaired"

            elif len(candidates) > 1 and match_method == "none":
                if len(local_candidates) == 0 and len(global_candidates) > 1:
                    if re.fullmatch(r"\d{1,3}", marker.strip()):
                        try:
                            chap_idx = source_meta.get("chapter_index")
                            if (
                                anchor_total == 1
                                and isinstance(chap_idx, int)
                                and group_key is not None
                                and re.fullmatch(r"PART_[A-Z0-9]+", str(group_key))
                            ):
                                with_idx = [c for c in global_candidates if isinstance(c.get("origin_index"), int)]
                                if with_idx:
                                    dist_pairs = [
                                        (abs(int(c.get("origin_index")) - int(chap_idx)), c)
                                        for c in with_idx
                                    ]
                                    best_dist = min(dist for dist, _c in dist_pairs)
                                    distinct_dists = sorted({dist for dist, _c in dist_pairs})
                                    second_dist = distinct_dists[1] if len(distinct_dists) >= 2 else None
                                    nearest = [c for dist, c in dist_pairs if dist == best_dist]
                                    if best_dist <= 2 and (second_dist is None or best_dist < second_dist):
                                        suggested_def = nearest[0].get("text")
                                        confidence_score = 0.66
                                        confidence = "Medium (Nearest Notes Order)"
                                        match_method = "global_nearest_order"
                        except Exception:
                            pass

                    if re.fullmatch(r"\d{1,3}", marker.strip()) and match_method == "none":
                        # Narrow fallback for malformed EPUBs that emit multiple NOTES
                        # blocks in the same source item for one structural chapter.
                        # If all global candidates come from the same nearby source item
                        # and same chapter-group, preserve the harvested order instead of
                        # leaving the anchor unpaired.
                        try:
                            if anchor_total == 1 and group_key is not None:
                                tagged_globals = [c for c in global_candidates if c.get("chapter_group") is not None]
                                origin_indexes = {
                                    int(c.get("origin_index"))
                                    for c in tagged_globals
                                    if isinstance(c.get("origin_index"), int)
                                }
                                group_keys = {str(c.get("chapter_group")) for c in tagged_globals}
                                if tagged_globals and len(origin_indexes) == 1 and group_keys == {str(group_key)}:
                                    suggested_def = tagged_globals[0].get("text")
                                    confidence_score = 0.60
                                    confidence = "Medium (Same-Source Order)"
                                    match_method = "global_same_source_order"
                        except Exception:
                            pass

                    if re.fullmatch(r"\d{1,3}", marker.strip()) and match_method == "none":
                        # Conservative disambiguation for numeric markers using anchor context.
                        #
                        # Many critical editions restart numeric notes, so we normally refuse
                        # to guess among multiple global candidates. However, when the anchor
                        # is unique within its chapter-group and one candidate has a clearly
                        # stronger lexical match to the anchor context, we can safely pick it.
                        try:
                            if anchor_total == 1:
                                stop = {
                                    "the",
                                    "and",
                                    "that",
                                    "with",
                                    "from",
                                    "this",
                                    "have",
                                    "will",
                                    "were",
                                    "been",
                                    "they",
                                    "them",
                                    "into",
                                    "about",
                                    "under",
                                    "over",
                                    "your",
                                    "their",
                                    "there",
                                    "which",
                                    "when",
                                    "what",
                                    "would",
                                    "could",
                                    "should",
                                    "then",
                                    "than",
                                    "also",
                                    "only",
                                    "more",
                                }

                                stop_caps = {
                                    "the",
                                    "and",
                                    "for",
                                    "with",
                                    "from",
                                    "this",
                                    "that",
                                    "but",
                                    "not",
                                    "see",
                                    "said",
                                }

                                def _kw_set(s: str) -> set[str]:
                                    words = re.findall(r"[A-Za-z]{5,}", (s or "").lower())
                                    return {w for w in words if w not in stop}

                                def _caps_set(s: str) -> set[str]:
                                    # Capture likely proper nouns / titled phrases.
                                    words = re.findall(r"\b[A-Z][A-Za-z'’\-]{2,}\b", (s or ""))
                                    return {w.lower() for w in words if w.lower() not in stop_caps}

                                ctx = _safe_text(a.get("context") or "")
                                ctx_kw = _kw_set(ctx)
                                ctx_caps = _caps_set(ctx)
                                if ctx_kw or ctx_caps:
                                    scored: List[Tuple[int, Dict[str, Any], int, int]] = []
                                    for c in global_candidates:
                                        txt = _safe_text(c.get("text") or "")
                                        if not txt:
                                            continue
                                        txt_kw = _kw_set(txt)
                                        txt_caps = _caps_set(txt)
                                        kw_overlap = len(ctx_kw.intersection(txt_kw)) if ctx_kw and txt_kw else 0
                                        caps_overlap = len(ctx_caps.intersection(txt_caps)) if ctx_caps and txt_caps else 0
                                        score = (caps_overlap * 3) + kw_overlap
                                        scored.append((score, c, caps_overlap, kw_overlap))
                                    if scored:
                                        scored.sort(key=lambda t: t[0], reverse=True)
                                        best_score, best_c, best_caps, best_kw = scored[0]
                                        second_score = scored[1][0] if len(scored) >= 2 else -1

                                        # Require a clear margin to avoid false certainty.
                                        #
                                        # Accept if:
                                        #  - There is at least one proper-noun token overlap AND
                                        #    the best beats the runner-up by a wide margin; OR
                                        #  - The best has some signal and the runner-up has none.
                                        accept = False
                                        # Minimum-signal requirement: weighted score >= 3
                                        # (e.g., at least one proper-noun overlap) with a clear margin.
                                        if best_score >= 3 and best_score >= (second_score + 3):
                                            accept = True
                                        # Even stronger signal: multiple proper-noun overlaps.
                                        if best_caps >= 2 and best_score >= (second_score + 2):
                                            accept = True

                                        if accept:
                                            suggested_def = best_c.get("text")
                                            confidence_score = 0.62
                                            confidence = "Medium (Context Match)"
                                            match_method = "global_context"
                                        else:
                                            match_method = "none"
                                            confidence_score = 0.50
                                            confidence = "Manual Review Required"
                                    else:
                                        match_method = "none"
                                        confidence_score = 0.50
                                        confidence = "Manual Review Required"
                                else:
                                    match_method = "none"
                                    confidence_score = 0.50
                                    confidence = "Manual Review Required"
                            else:
                                match_method = "none"
                                confidence_score = 0.50
                                confidence = "Manual Review Required"
                        except Exception:
                            match_method = "none"
                            confidence_score = 0.50
                            confidence = "Manual Review Required"
                    else:
                        match_method = "ai"
                else:
                    if forward_looking and candidates:
                        # Sort candidates by line_index so we find the closest
                        # definition AFTER the anchor, not just the first in
                        # insertion order.
                        al = a.get("line_index")
                        if isinstance(al, (int, float)):
                            sorted_cands = sorted(
                                [c for c in candidates if isinstance(c.get("line_index"), (int, float))],
                                key=lambda c: c.get("line_index", 0),
                            )
                            best, best_l = None, None
                            for c in sorted_cands:
                                cl = c.get("line_index")
                                # Use a tolerance of 1 to compensate for ratio
                                # rounding errors that cause definitions to
                                # appear fractionally before their anchor.
                                if isinstance(cl, (int, float)) and cl >= int(al) - 1:
                                    if best is None or cl < best_l:
                                        best, best_l = c, cl
                            if best is not None:
                                suggested_def = best.get("text")
                                confidence_score = 0.75
                                confidence = "Medium (Forward Match)"
                                match_method = "forward_looking"
                    if match_method == "none":
                        if occurrence_index < len(candidates) and anchor_total <= len(candidates):
                            suggested_def = candidates[occurrence_index]["text"]
                            confidence_score = 0.75
                            confidence = "Medium (Order Match)"
                            match_method = "marker_order"
                        elif occurrence_index < len(candidates):
                            suggested_def = candidates[occurrence_index]["text"]
                            confidence_score = 0.55
                            confidence = "Low (Marker/Order)"
                            match_method = "marker_order_low"
                        else:
                            # Too many anchors for strict order matching.
                            # Fall back to proximity: pair with the candidate whose
                            # char_position is nearest to (and ideally after) the anchor.
                            anchor_pos = a.get("position")
                            if isinstance(anchor_pos, (int, float)) and candidates:
                                best = None
                                best_dist = None
                                for c in candidates:
                                    cpos = c.get("char_position")
                                    if isinstance(cpos, (int, float)):
                                        dist = abs(cpos - int(anchor_pos))
                                        if best is None or dist < best_dist:
                                            best = c
                                            best_dist = dist
                                if best is not None and best_dist is not None and best_dist < 50000:
                                    suggested_def = best.get("text")
                                    confidence_score = 0.55
                                    confidence = "Low (Proximity Match)"
                                    match_method = "proximity_local"
                                else:
                                    match_method = "ai"
                            else:
                                match_method = "ai"

        item = {
            "type": "footnote",
            "marker": marker,
            "marker_raw": a.get("marker_raw"),
            "context": a.get("context"),
            "href": a.get("href"),
            "position": a.get("position"),
            "suggested_definition": suggested_def,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "match_method": match_method,
            "id": next_id,
        }
        item.update(source_meta)

        try:
            base = 0
            if item.get("source") == "epub" and isinstance(item.get("chapter_index"), int):
                base = int(item.get("chapter_index"))
            elif item.get("source") == "pdf" and isinstance(item.get("page_index"), int):
                base = int(item.get("page_index"))
            pos = item.get("position")
            pos_i = int(pos) if isinstance(pos, int) and pos >= 0 else 999_999_999
            item["order_key"] = base * 1_000_000_000 + min(pos_i, 999_999_999)
        except Exception:
            pass
        next_id += 1

        if match_method == "ai" and candidates:
            ordered = local_candidates + global_candidates
            item["candidates"] = ordered[:8]
            item["confidence_score"] = 0.55
            item["ai_status"] = "pending"
            ai_batch.append(item)

        results.append(item)

        if len(ai_batch) >= 5:
            _ai_disambiguate_pairs(ai_batch)
            ai_batch = []
            # Brief pause between AI calls to avoid hammering the server
            # and to give it time to clear prompt cache / finish cleanup.
            import time
            time.sleep(0.5)

    if ai_batch:
        _ai_disambiguate_pairs(ai_batch)

    for r in results:
        r.pop("candidates", None)

    return results, next_id
