import sys
import ebooklib
import re
import bisect
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
try:
    from sl_types import ScanOptions
    from sl_debug import _stderr_log, _debug_markers_set, _debug_markers_verbose_enabled
    from sl_ai import _infer_marker_profile_ai, _ai_infer_notes_split
    from sl_extract import (
        _anchor_text_excluding_definitions,
        _build_definition_exclusion_mask,
        _assign_positions_to_soup_anchors,
        _extract_anchors_from_soup,
        _harvest_structured_notes_section_targets,
        _harvest_specific_filepos_note_targets,
        _harvest_st_rst_footnotes,
        _harvest_footnote_class_definitions,
        _extract_bare_digit_anchors_from_text,
        _extract_definitions_from_lines,
        _extract_definitions_from_lines_scoped,
        _extract_anchors_from_text,
        _filter_anchors_by_profile,
        _filter_definitions_by_profile,
        _pair_anchors_to_definitions,
        _split_inline_numeric_definitions,
    )
    from sl_chapters import (
        _is_notes_header_line,
        _find_notes_block_end_from_header,
        _strip_trailing_footnote_marker_from_heading,
        _infer_logical_chapter_label,
        _line_looks_like_heading_component,
        _label_is_plausible_chapter_label,
        _find_chapter_headings_in_text,
        _chapter_group_key,
        _extract_chapter_token,
        _extract_notes_header_chapter_token,
        _infer_chapter_label_from_soup,
        _infer_chapter_label_from_item_name,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .sl_types import ScanOptions  # type: ignore
    from .sl_debug import _stderr_log, _debug_markers_set, _debug_markers_verbose_enabled  # type: ignore
    from .sl_ai import _infer_marker_profile_ai, _ai_infer_notes_split  # type: ignore
    from .sl_extract import (  # type: ignore
        _anchor_text_excluding_definitions,
        _build_definition_exclusion_mask,
        _assign_positions_to_soup_anchors,
        _extract_anchors_from_soup,
        _harvest_structured_notes_section_targets,
        _harvest_specific_filepos_note_targets,
        _harvest_st_rst_footnotes,
        _harvest_footnote_class_definitions,
        _extract_bare_digit_anchors_from_text,
        _extract_definitions_from_lines,
        _extract_definitions_from_lines_scoped,
        _extract_anchors_from_text,
        _filter_anchors_by_profile,
        _filter_definitions_by_profile,
        _pair_anchors_to_definitions,
        _split_inline_numeric_definitions,
    )
    from .sl_chapters import (  # type: ignore
        _is_notes_header_line,
        _find_notes_block_end_from_header,
        _strip_trailing_footnote_marker_from_heading,
        _infer_logical_chapter_label,
        _line_looks_like_heading_component,
        _label_is_plausible_chapter_label,
        _find_chapter_headings_in_text,
        _chapter_group_key,
        _extract_chapter_token,
        _extract_notes_header_chapter_token,
        _infer_chapter_label_from_soup,
        _infer_chapter_label_from_item_name,
    )

try:
    from sl_utility import (
        _safe_text,
        _clean_line_for_parsing,
        _preprocess_for_notes,
        _def_line_regex,
        _marker_category_from_raw,
        _normalize_marker,
    )
    from sl_heuristics import (
        anchor_is_probable_footnote,
        infer_notes_split,
        _infer_marker_profile_heuristic,
        looks_like_notes_continuation_page,
        infer_notes_continuation_harvest_start,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .sl_utility import (  # type: ignore
        _safe_text,
        _clean_line_for_parsing,
        _preprocess_for_notes,
        _def_line_regex,
        _marker_category_from_raw,
        _normalize_marker,
    )
    from .sl_heuristics import (  # type: ignore
        anchor_is_probable_footnote,
        infer_notes_split,
        _infer_marker_profile_heuristic,
        looks_like_notes_continuation_page,
        infer_notes_continuation_harvest_start,
    )


def _allowed_categories_for_profile(profile: str) -> Optional[set[str]]:
    p = (profile or "").strip().lower()
    if p in {"", "auto", "auto_heur", "auto_ai"}:
        return None
    if p == "numeric":
        return {"num_paren", "num_bracket", "num_sup", "num_sub", "num_plain"}
    if p == "symbol":
        return {"symbol"}
    if p == "letter":
        return {"let_paren", "let_bracket"}
    return None


def _marker_family(marker_raw: Any, marker_norm: Any) -> str:
    """Classify marker into a broad family for outlier detection.

    This is intentionally separate from `_marker_category_from_raw`, which is
    used for marker-profile filtering.
    """

    raw = _safe_text(marker_raw or marker_norm or "").strip()
    if not raw:
        return "unknown"

    # Strip simple wrappers.
    raw2 = raw.strip()
    if (raw2.startswith("(") and raw2.endswith(")")) or (raw2.startswith("[") and raw2.endswith("]")):
        raw2 = raw2[1:-1].strip()

    # Numeric families (including Unicode superscripts/subscripts).
    if re.fullmatch(r"\d{1,3}", raw2):
        return "numeric"
    if re.fullmatch(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}", raw2) or re.fullmatch(r"[₀₁₂₃₄₅₆₇₈₉]{1,4}", raw2):
        return "numeric"

    # Common note symbols.
    if re.fullmatch(r"[\*†‡§]+", raw2):
        return "symbol"

    # Roman numerals (common in some books for notes/chapters).
    u = raw2.upper().rstrip(".")
    if re.fullmatch(r"[IVXLC]{1,12}", u):
        return "roman"

    # Single letters.
    if re.fullmatch(r"[A-Za-z]", raw2):
        return "letter"

    return "other"


def _apply_marker_family_outlier_penalty(results: List[Dict[str, Any]]) -> None:
    """Downgrade confidence for marker-family outliers.

    This runs on the final result list (entire book/file) so we can use a
    *global* dominant marker family as a baseline. Per-chapter/page dominance is
    still preferred when a group is internally consistent.

    Goal: suspicious oddballs like a lone roman numeral ("I") among numeric
    footnotes should be downgraded even if their local chapter has only 1–2
    notes.
    """

    if not results:
        return

    def _group_key(r: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        src = _safe_text(r.get("source") or "") or "unknown"
        g: Optional[str] = None
        if src == "epub":
            cg = r.get("chapter_group")
            if cg is not None:
                g = str(cg)
            elif r.get("chapter_label") is not None:
                g = str(r.get("chapter_label"))
            elif r.get("chapter_index") is not None:
                g = str(r.get("chapter_index"))
        elif src == "pdf":
            if r.get("page_index") is not None:
                g = str(r.get("page_index"))
        else:
            if r.get("file_name") is not None:
                g = str(r.get("file_name"))
        return (src, g)

    def _dominant_family(fam_map: Dict[str, int], total: int) -> Optional[Tuple[str, float]]:
        if not fam_map or total <= 0:
            return None
        fam, cnt = max(fam_map.items(), key=lambda kv: kv[1])
        return fam, (cnt / max(1, total))

    # 1) Count families globally per source, and per group (chapter/page/file).
    global_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    global_totals: Dict[str, int] = defaultdict(int)
    group_counts: Dict[Tuple[str, Optional[str]], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_totals: Dict[Tuple[str, Optional[str]], int] = defaultdict(int)

    for r in results:
        if not isinstance(r, dict):
            continue
        mk = _safe_text(r.get("marker") or "")
        if not mk:
            continue
        src = _safe_text(r.get("source") or "") or "unknown"
        fam = _marker_family(r.get("marker_raw"), mk)

        global_counts[src][fam] += 1
        global_totals[src] += 1

        gk = _group_key(r)
        group_counts[gk][fam] += 1
        group_totals[gk] += 1

    # 2) Determine a global dominant family per source (if strong).
    global_dom: Dict[str, Tuple[str, float, int]] = {}
    for src, fam_map in global_counts.items():
        total = int(global_totals.get(src, 0))
        dom = _dominant_family(fam_map, total)
        if not dom:
            continue
        fam, ratio = dom
        # Require strong dominance so we don't punish books that legitimately mix schemes.
        if total >= 10 and ratio >= 0.75:
            global_dom[src] = (fam, ratio, total)

    # 3) Apply penalty.
    for r in results:
        if not isinstance(r, dict):
            continue

        # Never penalize explicit EPUB id links.
        if r.get("match_method") == "id_link":
            continue

        mk = _safe_text(r.get("marker") or "")
        if not mk:
            continue
        src = _safe_text(r.get("source") or "") or "unknown"
        fam = _marker_family(r.get("marker_raw"), mk)

        gk = _group_key(r)
        g_total = int(group_totals.get(gk, 0))
        g_map = group_counts.get(gk) or {}
        g_dom = _dominant_family(g_map, g_total)

        # Prefer a local dominant family when the group is big enough and clear.
        ref_family: Optional[str] = None
        ref_total: int = 0
        ref_map: Dict[str, int] = {}
        if g_dom and g_total >= 5 and g_dom[1] >= 0.70:
            ref_family = g_dom[0]
            ref_total = g_total
            ref_map = g_map
        else:
            gd = global_dom.get(src)
            if gd:
                ref_family = gd[0]
                ref_total = int(global_totals.get(src, 0))
                ref_map = global_counts.get(src) or {}

        if not ref_family:
            continue
        if fam == ref_family:
            continue

        # Only penalize truly rare outliers in the chosen reference scope.
        fam_count = int(ref_map.get(fam, 0))
        fam_ratio = fam_count / max(1, ref_total)
        if fam_count > 2 and fam_ratio > 0.20:
            continue

        old = r.get("confidence_score")
        try:
            old_f = float(old) if old is not None else 0.0
        except Exception:
            old_f = 0.0

        if "confidence_score_base" not in r:
            r["confidence_score_base"] = old_f

        new_f = min(old_f, 0.55)
        if new_f < old_f:
            r["confidence_score"] = new_f
            cur = _safe_text(r.get("confidence") or "")
            if not cur or cur == "Manual Review Required" or cur.lower().startswith("manual"):
                r["confidence"] = "Manual Review Required"
            else:
                r["confidence"] = "Low (Marker Outlier)"

            flags = r.get("confidence_flags")
            if not isinstance(flags, list):
                flags = []
            if "marker_family_outlier" not in flags:
                flags.append("marker_family_outlier")
            r["confidence_flags"] = flags


def _dedupe_numeric_results_prefer_id_link(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Conservative cleanup: when an ID-linked entry exists for a numeric marker in a chapter,
    drop any additional entries for that same (chapter_group, marker).

    This targets a common artifact where regex-derived anchors duplicate real EPUB noteref links.
    """

    if not results:
        return results

    groups: Dict[Tuple[Optional[str], str], List[int]] = defaultdict(list)
    for i, r in enumerate(results):
        mk = (r.get("marker") or "").strip()
        if not re.fullmatch(r"\d{1,3}", mk):
            continue
        g = r.get("chapter_group")
        gkey = str(g) if g is not None else None
        groups[(gkey, mk)].append(i)

    drop: set[int] = set()
    for _, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        id_link_idxs = [i for i in idxs if results[i].get("match_method") == "id_link"]
        if not id_link_idxs:
            continue

        # Distinct structured note refs can legitimately reuse the same numeric marker
        # within one logical chapter group (for example c2_rfn1, c2a_rfn1, c2b_rfn1).
        # Keep all explicit id-links and only drop non-id-link duplicates.
        for i in idxs:
            if results[i].get("match_method") == "id_link":
                continue
            drop.add(i)
        continue

        def _sort_key(i: int) -> Tuple[int, int, int, int]:
            r = results[i]
            is_id = 0 if r.get("match_method") == "id_link" else 1
            ok = r.get("order_key")
            pos = r.get("position")
            okv = int(ok) if isinstance(ok, int) else 1_000_000_000_000_000_000
            posv = int(pos) if isinstance(pos, int) else 1_000_000_000_000_000_000
            return (is_id, okv, posv, i)

        keep = min(idxs, key=_sort_key)
        for i in idxs:
            if i != keep:
                drop.add(i)

    if not drop:
        return results
    return [r for i, r in enumerate(results) if i not in drop]


def _looks_like_structural_part_or_book_heading(label: Any) -> bool:
    text = _safe_text(label or "")
    if not text:
        return False
    probe = re.sub(r"^[\s\[]+", "", text).strip()
    m = re.match(r"^(PART|BOOK)\s+([^\s\].,:;!?]+)", probe, re.IGNORECASE)
    if not m:
        return False
    token = _safe_text(m.group(2) or "").strip().strip(".:")
    if not token:
        return False
    if re.fullmatch(r"\d{1,3}", token):
        return True
    if re.fullmatch(r"[IVXLC]{1,12}", token, re.IGNORECASE):
        return True
    if token in {"ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"}:
        return True
    return token.isupper() and len(token) >= 2


def _parent_structural_chapter_group(label: Any) -> Optional[str]:
    text = _safe_text(label or "")
    if not text:
        return None
    m = re.search(r"\((PART|BOOK)\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\)\.?$", text, re.IGNORECASE)
    if not m:
        return None
    kind = _safe_text(m.group(1) or "").upper()
    token = _safe_text(m.group(2) or "").upper()
    if not kind or not token:
        return None
    return _chapter_group_key(f"[{kind} {token}].")


def _add_orphan_numeric_definitions(
    results: List[Dict[str, Any]],
    definitions: List[Dict[str, Any]],
    source_meta: Dict[str, Any],
    next_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Add synthetic entries for numeric definitions unreferenced in running text.

    This is intentionally conservative:
      - Prefers *local* definitions (line_index >= 0)
      - May use imported/global definitions only for single-digit markers (1-9)
      - Only fills *mid-sequence* gaps (has neighbor markers on both sides)
      - Only when the chapter already has numeric anchors
    """

    if not results or not definitions:
        return results, next_id

    # Pick the dominant chapter_group/label among existing results.
    group_counts: Dict[Optional[str], int] = defaultdict(int)
    label_counts: Dict[Tuple[Optional[str], Optional[str]], int] = defaultdict(int)
    for r in results:
        g = r.get("chapter_group")
        gkey = str(g) if g is not None else None
        group_counts[gkey] += 1
        lbl = r.get("chapter_label")
        lbl_key = str(lbl) if lbl is not None else None
        label_counts[(gkey, lbl_key)] += 1

    dominant_group: Optional[str] = None
    if group_counts:
        dominant_group = max(group_counts.items(), key=lambda kv: kv[1])[0]

    dominant_label: Optional[str] = None
    if dominant_group is not None:
        candidates = [(k, c) for k, c in label_counts.items() if k[0] == dominant_group and k[1] is not None]
        if candidates:
            dominant_label = max(candidates, key=lambda kv: kv[1])[0][1]

    if dominant_group is None:
        g = source_meta.get("chapter_group")
        dominant_group = str(g) if g is not None else None
    if dominant_label is None:
        lbl = source_meta.get("chapter_label")
        dominant_label = str(lbl) if lbl is not None else None

    # Existing numeric markers in this dominant group.
    present_nums: set[int] = set()
    for r in results:
        if (str(r.get("chapter_group")) if r.get("chapter_group") is not None else None) != dominant_group:
            continue
        mk = (r.get("marker") or "").strip()
        if not re.fullmatch(r"\d{1,3}", mk):
            continue
        try:
            present_nums.add(int(mk))
        except Exception:
            continue

    if len(present_nums) < 2:
        return results, next_id

    min_present = min(present_nums)
    max_present = max(present_nums)

    # Prefer local definitions. As a fallback, allow imported/global definitions only
    # for single-digit markers so recovered_anchor can still run.
    local_defs: Dict[int, str] = {}
    imported_defs: Dict[int, str] = {}
    for d in definitions:
        mk = (d.get("marker") or "").strip()
        if not re.fullmatch(r"\d{1,3}", mk):
            continue
        try:
            num = int(mk)
        except Exception:
            continue

        txt = d.get("text")
        if not txt:
            continue

        li = d.get("line_index")
        if isinstance(li, int) and li >= 0:
            if num not in local_defs:
                local_defs[num] = str(txt)
        else:
            if 1 <= num <= 9 and num not in imported_defs:
                imported_defs[num] = str(txt)

    if not local_defs and not imported_defs:
        return results, next_id

    combined_defs: Dict[int, str] = dict(imported_defs)
    combined_defs.update(local_defs)
    def_nums = set(combined_defs.keys())

    added: List[Dict[str, Any]] = []
    for num, def_text in sorted(combined_defs.items(), key=lambda kv: kv[0]):
        if num in present_nums:
            continue
        if not (min_present <= num <= max_present):
            continue
        if not ((num - 1) in present_nums or (num - 1) in def_nums):
            continue
        if not ((num + 1) in present_nums or (num + 1) in def_nums):
            continue

        item: Dict[str, Any] = {
            "type": "footnote",
            "marker": str(num),
            "marker_raw": str(num),
            "context": "",
            "href": None,
            "position": 999_999_999,
            "suggested_definition": def_text,
            "confidence": "Low (Orphan Definition)",
            "confidence_score": 0.40,
            "match_method": "orphan_definition",
            "id": next_id,
        }
        item.update(source_meta)
        if dominant_label is not None:
            item["chapter_label"] = dominant_label
        if dominant_group is not None:
            item["chapter_group"] = dominant_group

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
        added.append(item)

    if not added:
        return results, next_id

    results = list(results)
    results.extend(added)
    return results, next_id


def _repair_structural_parent_note_swaps(results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    parent_rows_by_label: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    child_rows_by_parent: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    parent_label_re = re.compile(r"^\[(PART|BOOK)\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\]\.?")

    def _parent_label_for_row(row: Dict[str, Any]) -> Optional[str]:
        label = _safe_text(row.get("chapter_label") or "")
        if not label or row.get("source") != "epub":
            return None
        m = re.search(r"\((PART|BOOK)\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\)\.?$", label, re.IGNORECASE)
        if not m:
            return None
        return f"[{str(m.group(1) or '').upper()} {str(m.group(2) or '').upper()}]."

    for row in results:
        label = _safe_text(row.get("chapter_label") or "")
        marker = _safe_text(row.get("marker") or "")
        if row.get("source") != "epub" or not label or not re.fullmatch(r"\d{1,3}", marker):
            continue
        if parent_label_re.fullmatch(label):
            parent_rows_by_label[label][marker] = row
            continue
        parent_label = _parent_label_for_row(row)
        if parent_label:
            child_rows_by_parent[parent_label][marker] = row

    preferred_child_methods = {
        "marker_order",
        "marker_order_low",
        "marker_unique",
        "global_context",
        "global_unique",
    }

    for parent_label, parent_rows in parent_rows_by_label.items():
        child_rows = child_rows_by_parent.get(parent_label)
        if not child_rows:
            continue

        swap_markers: List[str] = []
        for marker, parent_row in parent_rows.items():
            child_row = child_rows.get(marker)
            if child_row is None:
                continue
            if parent_row.get("match_method") != "ai":
                continue
            if child_row.get("match_method") not in preferred_child_methods:
                continue
            if not parent_row.get("suggested_definition") or not child_row.get("suggested_definition"):
                continue
            parent_order = parent_row.get("order_key")
            child_order = child_row.get("order_key")
            if isinstance(parent_order, int) and isinstance(child_order, int) and not (parent_order < child_order):
                continue
            swap_markers.append(marker)

        if len(swap_markers) < 5:
            continue

        for marker in swap_markers:
            parent_row = parent_rows[marker]
            child_row = child_rows[marker]

            parent_def = parent_row.get("suggested_definition")
            parent_score = parent_row.get("confidence_score")

            parent_row["suggested_definition"] = child_row.get("suggested_definition")
            parent_row["confidence"] = "Medium (Structural Notes Swap)"
            parent_row["confidence_score"] = max(float(child_row.get("confidence_score") or 0.0), 0.78)
            parent_row["match_method"] = "structural_parent_swap"

            child_row["suggested_definition"] = parent_def
            child_row["confidence"] = "Medium (Structural Notes Swap)"
            child_row["confidence_score"] = max(float(parent_score or 0.0), 0.78)
            child_row["match_method"] = "structural_child_swap"


# High-level function to scan a text blob for footnotes, supporting both inline and end-of-chapter/notes styles.
def _scan_text_blob_for_footnotes(
    text: str,
    source_meta: Dict[str, Any],
    id_start: int = 0,
    *,
    options: Optional[ScanOptions] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Scan an arbitrary text blob, supporting end-of-chapter/notes style footnotes."""
    options = options or ScanOptions()
    text = _preprocess_for_notes(text or "")

    effective_profile = (options.marker_profile or "auto_heur").strip().lower()
    if effective_profile in {"auto", "auto_heur", "auto_ai"}:
        inferred = _infer_marker_profile_heuristic(text)
        if effective_profile == "auto_ai" and inferred == "auto_heur":
            # Only call AI when heuristics are uncertain — avoid wasting
            # an AI call when the heuristic already has a confident answer.
            ai_prof = _infer_marker_profile_ai(text)
            if ai_prof:
                inferred = ai_prof
        effective_profile = inferred if inferred != "auto_heur" else "auto_heur"
    allowed_categories = _allowed_categories_for_profile(effective_profile)

    # Preserve newlines for definition parsing.
    lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in text.split("\n")]

    split = infer_notes_split(lines)
    if split is None:
        # Fallback when we can't confidently detect a notes block:
        # - Prefer scanning the entire text for anchors (prevents missing markers when
        #   the EPUB/text is poorly segmented or contains very few line breaks).
        # - Only carve off a notes block if we see a definition-like cluster near the tail.
        if not lines:
            main_end = 0
            defs_start = 0
        else:
            tail_start = max(0, int(len(lines) * 0.6))
            tail_def_like = [i for i in range(tail_start, len(lines)) if _def_line_regex().match(lines[i] or "")]
            if len(tail_def_like) >= 3:
                defs_start = tail_def_like[0]
                main_end = defs_start
            else:
                main_end = len(lines)
                defs_start = len(lines)
    else:
        main_end = split.main_end_index
        defs_start = split.defs_start_index

    definitions = _extract_definitions_from_lines(lines, defs_start)
    definitions = _filter_definitions_by_profile(definitions, allowed_categories)
    anchor_text = _anchor_text_excluding_definitions(lines, defs_start)
    anchors = _extract_anchors_from_text(anchor_text)
    anchors = _filter_anchors_by_profile(anchors, allowed_categories)

    # Targeted rescue for single-digit markers: if a marker exists in definitions but
    # produced zero anchors (often due to ProperNoun(n) suppression), attempt to
    # add back the first ProperNoun (n) occurrence.
    try:
        def_single = {
            str(d.get("marker") or "").strip()
            for d in definitions
            if re.fullmatch(r"[1-9]", str(d.get("marker") or "").strip())
        }
        present = {str(a.get("marker") or "").strip() for a in anchors}
        missing = sorted([m for m in def_single if m not in present], key=lambda s: int(s))
        if missing and anchor_text:
            for mk in missing:
                rx = re.compile(rf"\b[A-Z][A-Za-z'’\-]{{1,28}}\s*\(\s*{re.escape(mk)}\s*\)")
                m = rx.search(anchor_text)
                if not m:
                    continue
                s = m.group(0)
                rel = s.find("(")
                if rel < 0:
                    continue
                pos = m.start() + rel
                start = max(0, pos - 80)
                end = min(len(anchor_text), pos + 80)
                ctx = _safe_text(anchor_text[start:end])
                anchors.append(
                    {
                        "marker_raw": f"({mk})",
                        "marker": mk,
                        "position": int(pos),
                        "context": ctx,
                        "_has_href": False,
                    }
                )
    except Exception:
        pass

    # If there are numeric definitions but no detected anchors (common for PDF text extraction),
    # look for bare digits that correspond to known definition markers.
    def_markers = [d.get("marker") for d in definitions if d.get("marker")]
    if def_markers:
        anchors.extend(_extract_bare_digit_anchors_from_text(anchor_text, def_markers))

    results, next_id = _pair_anchors_to_definitions(anchors, definitions, source_meta, id_start=id_start)
    for r in results:
        r.setdefault("marker_profile", effective_profile)
    return results, next_id

# Main function to scan an EPUB file for footnotes, handling both inline and end-of-chapter/notes styles, and building a global id and marker map for cross-referencing.
def scan_epub_for_footnotes(epub_path: str, *, options: Optional[ScanOptions] = None) -> str:
    options = options or ScanOptions()
    book = epub.read_epub(epub_path)
    all_doc_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    def _def_line_looks_like_combined_chapter_heading(line: str) -> bool:
        t = _safe_text(line or "").strip()
        if not t:
            return False
        if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", t, re.IGNORECASE):
            return True

        m = re.match(r"^\s*([IVXLC]{1,12}|\d{1,3})\.\s+(.+?)\s*$", t)
        if not m:
            return False

        title = _safe_text(m.group(2) or "")
        if not title or _is_notes_header_line(title):
            return False

        letters = [ch for ch in title if ch.isalpha()]
        if len(letters) < 6:
            return False

        upper = sum(1 for ch in letters if ch.isupper())
        if (upper / max(1, len(letters))) >= 0.85:
            return True

        words = [w for w in re.split(r"\s+", title) if w]
        if len(words) < 2:
            return False
        if all(re.match(r"^[A-Z][A-Za-z'’\-]*\.?$", w) for w in words):
            return True

        return False

    def _looks_like_navigation_or_index_doc(name: str, lines: List[str]) -> bool:
        name_l = _safe_text(name or "").strip().lower()
        base = os.path.basename(name_l)
        if base == "nav.xhtml":
            return True

        nonempty: List[str] = []
        for line in lines:
            t = _safe_text(line or "").strip()
            if not t:
                continue
            nonempty.append(t)
            if len(nonempty) >= 6:
                break

        if not nonempty:
            return False

        first = nonempty[0].strip(" .:\t").lower()
        if first in {"contents", "table of contents", "index"}:
            return True

        return False

    def _parse_toc_chapter_map() -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, Optional[str]]]], bool]:
        """Parse toc.ncx to extract chapter label information from the navMap.

        Returns:
          (file_to_label, toc_file_entries, is_high_quality)

          file_to_label: dict mapping file basename → first chapter label (existing contract)
          toc_file_entries: dict mapping file basename → [(label, anchor_fragment), ...]
            for ALL TOC entries per file. Anchor fragment may be None when the TOC src
            has no #anchor. This is used for per-anchor position-based chapter splitting
            when a single file contains multiple TOC entries.
          is_high_quality: bool
        """

        file_to_label: Dict[str, str] = {}
        toc_file_entries: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
        try:
            toc_item = book.get_item_with_href("toc.ncx") or book.get_item_with_id("ncx")
            if toc_item is None:
                for it in book.get_items():
                    name = (getattr(it, "get_name", lambda: "")() if hasattr(it, "get_name") else "").lower()
                    if name.endswith(".ncx") or name == "toc.ncx":
                        toc_item = it
                        break
            if toc_item is None:
                return file_to_label, toc_file_entries, False

            toc_xml = toc_item.get_content()
            toc_text = toc_xml.decode("utf-8", errors="ignore") if isinstance(toc_xml, bytes) else str(toc_xml)
            toc_soup = BeautifulSoup(toc_text, "xml")

            valid_labels = 0
            total_points = 0
            frontback_keywords = {"title page", "copyright", "contents", "cover", "foreword",
                                  "about the publisher", "other books", "searchable terms",
                                  "note on accessibility"}

            for np in toc_soup.find_all("navPoint"):
                total_points += 1
                label_el = np.find("navLabel")
                text_el = label_el.find("text") if label_el else None
                content_el = np.find("content")
                if text_el is None or content_el is None:
                    continue

                label = _safe_text(text_el.get_text(" ")).strip()
                src = _safe_text(content_el.get("src") or "").strip()
                if not label or not src:
                    continue

                # Extract the file basename and anchor fragment from src.
                file_part = src.split("#")[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
                anchor_frag: Optional[str] = None
                if "#" in src:
                    anchor_frag = src.split("#", 1)[1].strip() or None

                # Count non-trivial, non-frontmatter labels as high-quality.
                label_lower = label.lower().strip()
                if (
                    len(label) >= 4
                    and label_lower not in frontback_keywords
                    and not re.fullmatch(r"^\s*[IVXLC]{1,8}\.?\s*$", label)
                    and not re.fullmatch(r"^\s*\d{1,3}\.?\s*$", label)
                ):
                    valid_labels += 1

                # Map file → first label (existing flat mapping).
                if file_part not in file_to_label:
                    file_to_label[file_part] = label
                # Map file → all (label, anchor) entries (new; for multi-entry splitting).
                toc_file_entries[file_part].append((label, anchor_frag))

            is_high_quality = valid_labels >= 5 and (total_points < 1 or valid_labels / max(1, total_points) >= 0.25)
            return file_to_label, toc_file_entries, is_high_quality

        except Exception:
            return file_to_label, toc_file_entries, False

    def _items_in_spine_order() -> List[Any]:
        """Return EPUB document items in spine (reading) order.

        ebooklib's ITEM_DOCUMENT iteration order is not guaranteed to match the
        spine. Scanning in manifest order can surface appendices/notes before
        the actual Chapter I/II/... reading order.
        """

        docs_by_id: Dict[str, Any] = {}
        for it in all_doc_items:
            try:
                docs_by_id[str(it.get_id())] = it
            except Exception:
                continue

        ordered: List[Any] = []
        try:
            for ent in getattr(book, "spine", []) or []:
                idref = ent[0] if isinstance(ent, tuple) else ent
                if not idref:
                    continue
                if str(idref).lower() == "nav":
                    continue
                it = None
                try:
                    it = book.get_item_with_id(idref)
                except Exception:
                    it = None
                if it is None:
                    it = docs_by_id.get(str(idref))
                if it is None:
                    continue
                try:
                    if getattr(it, "get_type", None) and it.get_type() != ebooklib.ITEM_DOCUMENT:
                        continue
                except Exception:
                    continue
                ordered.append(it)
        except Exception:
            ordered = []

        # De-dup while preserving order.
        seen_ids = set()
        uniq: List[Any] = []
        for it in ordered:
            try:
                iid = str(it.get_id())
            except Exception:
                iid = None
            if not iid or iid in seen_ids:
                continue
            seen_ids.add(iid)
            uniq.append(it)

        # Fall back to manifest/doc iteration order if spine is missing.
        return uniq if uniq else all_doc_items

    items = _items_in_spine_order()
    all_results: list[dict] = []
    next_id = 0

    # Detect well-structured EPUBs that use explicit bidirectional HTML footnote links.
    # When an EPUB consistently has <a> tags in prose linking to definition elements via
    # explicit ID fragments (any naming scheme), we can skip AI calls and trust the
    # HTML DOM for pairing. This is pattern-agnostic — it catches st/rst, c_rfn, fn{N},
    # pt4en{N}, filepos{N}, and any future convention automatically.
    structured_footnote_epub = False
    _st_rst_convention = False  # kept narrow for st/rst-specific harvest code
    st_rst_rst_id_map: Dict[str, str] = {}  # rst-id -> definition text
    st_rst_st_anchors: Dict[str, Dict[str, Any]] = {}  # st-id -> anchor info
    st_rst_global_defs: Dict[str, Dict[str, Any]] = {}  # rst-id -> full definition info
    _st_id_re = re.compile(r"^st(\d+)([a-z]?)$", re.IGNORECASE)
    _rst_id_re = re.compile(r"^rst(\d+)([a-z]?)$", re.IGNORECASE)
    try:
        # --- Phase 1: Collect all element IDs across the entire EPUB ---
        all_elem_ids: set[str] = set()
        for it in items:
            try:
                html_content = it.get_content()
                html_text = html_content.decode("utf-8", errors="ignore") if isinstance(html_content, bytes) else str(html_content)
                probe_soup = BeautifulSoup(html_text, "html.parser")
                for tag in probe_soup.find_all(True):
                    tid = _safe_text(tag.get("id") or "").strip()
                    if tid:
                        all_elem_ids.add(tid)
            except Exception:
                continue

        # --- Phase 2: Count bidirectional footnote links ---
        # A "bidirectional footnote link" is an <a> tag whose href fragment points to
        # an element ID we found elsewhere in the EPUB, AND whose visible text is small
        # (a typical footnote marker like "1", "*", "a" — not a TOC/chapter title).
        total_bidi_anchors = 0
        total_bidi_defs = 0
        referenced_def_ids: set[str] = set()
        anchor_files_with_bidi: set[int] = set()
        _small_marker_re = re.compile(r"^\s*[\(\[]?\s*(?:\d{1,3}|\*+|\u2020+|\u2021+|\u00a7+|[a-zA-Z])\s*[\)\]]?\s*$", re.UNICODE)

        for idx, it in enumerate(items):
            try:
                html_content = it.get_content()
                html_text = html_content.decode("utf-8", errors="ignore") if isinstance(html_content, bytes) else str(html_content)
                probe_soup = BeautifulSoup(html_text, "html.parser")

                # Count bidirectional anchors.
                for a in probe_soup.find_all("a"):
                    href = _safe_text(a.get("href") or "").strip()
                    if "#" not in href:
                        continue
                    if href.lower().startswith("http://") or href.lower().startswith("https://"):
                        continue
                    frag = href.rsplit("#", 1)[1].strip()
                    if frag not in all_elem_ids:
                        continue
                    txt = _safe_text(a.get_text(" ")).strip()
                    if not _small_marker_re.match(txt):
                        continue
                    total_bidi_anchors += 1
                    referenced_def_ids.add(frag)
                    anchor_files_with_bidi.add(int(idx))

                # Count st{N}/rst{N} anchors (for the narrow st/rst harvest path).
                for a in probe_soup.find_all("a"):
                    aid = _safe_text(a.get("id") or "").strip()
                    if _st_id_re.match(aid):
                        href = _safe_text(a.get("href") or "").strip()
                        if "#" in href:
                            frag = href.rsplit("#", 1)[1].strip()
                            if _rst_id_re.match(frag):
                                _st_rst_convention = True
                    if _rst_id_re.match(aid):
                        href = _safe_text(a.get("href") or "").strip()
                        if "#" in href:
                            frag = href.rsplit("#", 1)[1].strip()
                            if _st_id_re.match(frag):
                                _st_rst_convention = True

                # Count footnote-class definition elements with IDs.
                for p in probe_soup.select(".footnote, .footnotet, .noindent-x1"):
                    pid = _safe_text(p.get("id") or "").strip()
                    if pid and pid in referenced_def_ids:
                        total_bidi_defs += 1

                # Also count any element whose ID was referenced by an anchor above.
                for tag in probe_soup.find_all(True):
                    tid = _safe_text(tag.get("id") or "").strip()
                    if tid and tid in referenced_def_ids:
                        total_bidi_defs += 1
            except Exception:
                continue

        # --- Phase 3: Threshold check ---
        # Convention is active when we see at least 10 bidirectional footnote links
        # spread across at least 3 different spine items.
        structured_footnote_epub = (
            total_bidi_anchors >= 10
            and total_bidi_defs >= 5
            and len(anchor_files_with_bidi) >= 3
        )
    except Exception:
        structured_footnote_epub = False
        _st_rst_convention = False

    # When a structured footnote convention is detected, the EPUB is well-structured —
    # all real footnotes have explicit HTML links. Disable ALL AI calls globally
    # for the duration of this scan. Using the env var blocks at the lowest level
    # (_call_ai), catching even unguarded paths like _ai_disambiguate_pairs.
    _saved_ai_disabled = os.environ.get("STARLISTENER_AI_DISABLED")
    _ai_was_disabled = False
    if structured_footnote_epub and _saved_ai_disabled != "1":
        os.environ["STARLISTENER_AI_DISABLED"] = "1"
        _ai_was_disabled = True

    # Parse the TOC for chapter structure. A high-quality TOC provides chapter labels
    # and boundaries that are more reliable than text-based heuristics.
    toc_file_to_label, toc_file_entries, toc_is_high_quality = _parse_toc_chapter_map()

    def _resolve_toc_anchor_positions(
        entries: List[Tuple[str, Optional[str]]],
        soup: BeautifulSoup,
        lines: List[str],
    ) -> List[Tuple[str, int]]:
        """Resolve TOC anchor fragments to character positions in the text.

        For each (label, anchor_fragment) in `entries`, finds the DOM element
        with that ID, inserts a token at that position, then finds the token in
        a fresh text derived from the modified soup. This avoids the stale-text
        problem where anchors_text was computed before token insertion.

        Returns a sorted list of (label, char_position) boundaries.

        Entries whose anchor_fragment is None or whose element is not found
        are silently skipped.
        """
        boundaries: List[Tuple[str, int]] = []
        if not entries or not soup:
            return boundaries

        token_tpl = "\uE002TOC{idx:06d}\uE003"
        token_len = len(token_tpl.format(idx=0))
        token_counter = 0
        label_map: Dict[str, str] = {}  # token → label

        # Copy soup so tokens don't affect downstream parsing.
        soup2 = BeautifulSoup(str(soup), "html.parser")

        for label, anchor_frag in entries:
            if not anchor_frag:
                continue
            tag = soup2.find(id=anchor_frag)
            if not tag:
                continue
            tok = token_tpl.format(idx=token_counter)
            token_counter += 1
            try:
                tag.insert_before(tok)
            except Exception:
                continue
            label_map[tok] = label

        if not label_map:
            return boundaries

        # Recompute text from the token-instrumented soup.
        try:
            from sl_utility import _preprocess_for_notes, _clean_line_for_parsing  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover
            from .sl_utility import _preprocess_for_notes, _clean_line_for_parsing  # type: ignore

        chapter_text_instr = _preprocess_for_notes(soup2.get_text("\n"))
        tok_lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in chapter_text_instr.split("\n")]
        anchors_text_instr = "\n".join(tok_lines)

        for tok, label in label_map.items():
            j = anchors_text_instr.find(tok)
            if j == -1:
                continue
            corrected = int(j - (boundaries.__len__() * token_len))
            if corrected < 0:
                corrected = 0
            boundaries.append((label, corrected))

        boundaries.sort(key=lambda kv: kv[1])
        return boundaries

    def _label_by_toc_position(
        boundaries: List[Tuple[str, int]],
        pos: Any,
    ) -> Optional[str]:
        """Return the TOC label that applies at the given character position.

        Boundaries are sorted by character position. The label at or after `pos`
        is the label for that position. If `pos` is before the first boundary,
        returns None (caller should use the inherited/carried context).
        """
        if not boundaries or not isinstance(pos, (int, float)):
            return None
        ipos = int(pos)
        if ipos < 0:
            return None
        # Find the last boundary whose position is <= ipos.
        label = None
        for lbl, bpos in boundaries:
            if int(bpos) <= ipos:
                label = lbl
            else:
                break
        return _strip_trailing_footnote_marker_from_heading(label) or label if label else None

    # Decide marker profile once per book (used to suppress obviously-wrong marker families).
    effective_profile = (options.marker_profile or "auto_heur").strip().lower()
    if effective_profile in {"auto", "auto_heur", "auto_ai"}:
        if structured_footnote_epub:
            # Well-structured EPUB with explicit footnote links — no AI needed.
            effective_profile = "auto_heur"
        else:
            sample_text_parts: List[str] = []
            for it in items[: min(6, len(items))]:
                try:
                    html_content = it.get_content()
                    html_text = html_content.decode("utf-8", errors="ignore") if isinstance(html_content, bytes) else str(html_content)
                    soup = BeautifulSoup(html_text, "html.parser")
                    sample_text_parts.append(soup.get_text("\n"))
                except Exception:
                    continue
            sample_text = _preprocess_for_notes("\n".join(sample_text_parts))
            inferred = _infer_marker_profile_heuristic(sample_text)
            if effective_profile == "auto_ai" and inferred == "auto_heur":
                # Only call AI when heuristics are uncertain.
                ai_prof = _infer_marker_profile_ai(sample_text)
                if ai_prof:
                    inferred = ai_prof
            effective_profile = inferred if inferred != "auto_heur" else "auto_heur"
    allowed_categories = _allowed_categories_for_profile(effective_profile)

    # Build a global id->definition map (common in EPUBs where notes live in separate files).
    global_defs_by_id: Dict[str, str] = {}
    global_defs_by_marker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # Common fragment/id naming schemes for note targets.
    # This is used for harvesting *definitions* and for identifying note-target pages.
    id_re = re.compile(
        r"^(?:fn|fnref|fnt|footnote|footnoteref|note|noteref|endnote|en|ref|n)[-_]?\d{1,4}[a-z]?$",
        re.IGNORECASE,
    )
    _pt4en_id_re = re.compile(r"^r?pt4en\d{1,4}[a-z]?$", re.IGNORECASE)
    _rref_id_re = re.compile(r"^rref\d{1,4}[a-z]?$", re.IGNORECASE)
    _rss_id_re = re.compile(r"^rss\d{1,4}[a-z]?$", re.IGNORECASE)
    _note_class_re = re.compile(r"\b(?:note|endnote|footnote)", re.IGNORECASE)

    def _harvest_id_note_block(tag: Any) -> Optional[str]:
        start = None
        try:
            if hasattr(tag, "find_parent"):
                start = tag.find_parent(["p", "blockquote", "li", "div"])
        except Exception:
            start = None
        if start is None:
            return None

        def _normalize_block_text(node: Any) -> str:
            try:
                raw = node.get_text(" ")
            except Exception:
                raw = ""
            return _safe_text(raw)

        def _note_block_style(text: str) -> str:
            t = (text or "").lstrip()
            if re.match(r"^\d{1,3}\s*[\]\)\.:\-—]\s+", t):
                return "num_plain"
            if re.match(r"^[\(\[]\s*\d{1,3}\s*[\)\]]\s*[\]\)\.:\-—]?\s+", t):
                return "num_wrapped"
            if re.match(r"^[A-Za-z]\s*[\]\)\.:\-—]\s+", t):
                return "letter_plain"
            if re.match(r"^[\(\[]\s*[A-Za-z]\s*[\)\]]\s*[\]\)\.:\-—]?\s+", t):
                return "letter_wrapped"
            if re.match(r"^(\*+|†+|‡+|§+)\s+", t):
                return "symbol"
            return "other"

        blocks: List[str] = []
        cur = start
        first = start
        first_style = _note_block_style(_normalize_block_text(first))
        while cur is not None:
            txt = _normalize_block_text(cur)

            if cur is not first:
                try:
                    has_note_id = False
                    for sub in cur.find_all(True):
                        sid = (sub.get("id") or "").strip()
                        if sid and (id_re.match(sid) or _pt4en_id_re.match(sid)):
                            has_note_id = True
                            break
                    if has_note_id:
                        break
                except Exception:
                    pass

                if txt and _line_looks_like_heading_component(txt):
                    break

                if txt and _def_line_regex().match(txt):
                    if _note_block_style(txt) == first_style and first_style != "other":
                        break

            if txt:
                blocks.append(txt)

            nxt = None
            try:
                nxt = cur.find_next_sibling()
            except Exception:
                nxt = None
            if nxt is None or getattr(nxt, "name", None) not in {"p", "blockquote", "li", "div"}:
                break
            cur = nxt

        if not blocks:
            return None

        first_txt = blocks[0]
        m = _def_line_regex().match(first_txt)
        if m:
            body = _safe_text(m.group(2))
            if body:
                blocks[0] = body

        joined = "\n\n".join([b for b in blocks if b])
        return joined if joined else None

    # Track the last known logical chapter group while walking spine items.
    # This lets us associate standalone notes/endnotes files with the nearest
    # chapter, and later prevent definitions from bleeding across chapters when
    # numbering restarts.
    context_chapter_label: Optional[str] = None
    context_chapter_group: Optional[str] = None
    for origin_index, item in enumerate(items):
        html_content = item.get_content()
        html_text = html_content.decode("utf-8", errors="ignore") if isinstance(html_content, bytes) else str(html_content)
        soup = BeautifulSoup(html_text, "html.parser")
        name = (getattr(item, "get_name", lambda: "")() if hasattr(item, "get_name") else "")
        text_all = _preprocess_for_notes(soup.get_text("\n"))
        lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in text_all.split("\n")]

        if _looks_like_navigation_or_index_doc(name, lines):
            continue

        # Per-item context used when tagging harvested definitions.
        item_context_group = context_chapter_group
        item_context_group_ambiguous = False
        carried_context_label = context_chapter_label
        carried_context_group = item_context_group

        # Update current chapter context if this item contains a real chapter heading.
        # Avoid letting notes-like pages overwrite chapter context.
        try:
            tmp_label = _infer_chapter_label_from_soup(soup)
            if not tmp_label:
                tmp_text = _preprocess_for_notes(soup.get_text("\n"))
                # Prefer the *last* heading in the file, since malformed EPUBs can
                # contain multiple chapters within one spine item.
                tmp_heads = _find_chapter_headings_in_text(tmp_text)
                if tmp_heads:
                    tmp_label = tmp_heads[-1][1]
                    item_context_group = _chapter_group_key(tmp_label)
                    item_context_group_ambiguous = len(tmp_heads) >= 2
                else:
                    tmp_lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in tmp_text.split("\n")]
                    tmp_label = _infer_logical_chapter_label(tmp_lines)
            if tmp_label and _label_is_plausible_chapter_label(tmp_label):
                context_chapter_label = tmp_label
                context_chapter_group = _chapter_group_key(tmp_label)
                item_context_group = context_chapter_group
        except Exception:
            pass

        # Specific App-style popup note targets keyed by filepos ids.
        fp_id_map, fp_marker_defs = _harvest_specific_filepos_note_targets(soup)
        for fid, txt in fp_id_map.items():
            if fid and txt:
                # Avoid overwriting if already harvested via a stronger signal.
                global_defs_by_id.setdefault(fid, txt)
        for d in fp_marker_defs:
            mk = d.get("marker")
            txt = d.get("text")
            if mk and txt:
                if allowed_categories and _marker_category_from_raw(mk) not in allowed_categories:
                    continue
                existing = global_defs_by_marker.get(mk, [])
                if not any(x.get("text") == txt for x in existing):
                    global_defs_by_marker[mk].append(
                        {
                            "text": txt,
                            "origin": name,
                            "origin_index": origin_index,
                            "chapter_group": (None if item_context_group_ambiguous else item_context_group),
                        }
                    )

        structured_id_map, structured_marker_defs = _harvest_structured_notes_section_targets(soup)
        for fid, txt in structured_id_map.items():
            if fid and txt:
                global_defs_by_id.setdefault(fid, txt)
        for d in structured_marker_defs:
            mk = d.get("marker")
            txt = d.get("text")
            if mk and txt:
                if allowed_categories and _marker_category_from_raw(mk) not in allowed_categories:
                    continue
                existing = global_defs_by_marker.get(mk, [])
                if not any(x.get("text") == txt for x in existing):
                    global_defs_by_marker[mk].append(
                        {
                            "text": txt,
                            "origin": name,
                            "origin_index": origin_index,
                            "chapter_group": (None if item_context_group_ambiguous else item_context_group),
                        }
                    )

        # st/rst convention: harvest explicit st{N} anchors and rst{N} footnote definitions.
        # For well-formed EPUBs this gives us direct anchor-definition pairs with no heuristics.
        if _st_rst_convention:
            rst_id_map, st_anchors_from_item, rst_defs_from_item = _harvest_st_rst_footnotes(soup)
            for rst_id, txt in rst_id_map.items():
                if rst_id and txt:
                    global_defs_by_id.setdefault(rst_id, txt)
                    st_rst_rst_id_map.setdefault(rst_id, txt)
            for d in rst_defs_from_item:
                def_id = _safe_text(d.get("id") or "")
                if def_id:
                    st_rst_global_defs.setdefault(def_id, d)
            for a in st_anchors_from_item:
                anchor_id = _safe_text(a.get("id") or "")
                if anchor_id:
                    a["origin_index"] = origin_index
                    a["chapter_name"] = name
                    st_rst_st_anchors.setdefault(anchor_id, a)

        # 1) Harvest id-based definitions
        for tag in soup.find_all(True):
            tag_id = tag.get("id")
            if not tag_id:
                continue
            matches_id = id_re.match(tag_id) or _pt4en_id_re.match(tag_id)
            # Extended check: rref/rss prefixed ids inside note-like containers
            # (e.g., <p class="note" id="rref552"> or <a id="rss559b"> inside <p class="note">)
            if not matches_id and (_rref_id_re.match(tag_id) or _rss_id_re.match(tag_id)):
                try:
                    cls = " ".join(tag.get("class") or []).lower()
                    parent = tag.find_parent(["p", "div", "li"])
                    parent_cls = " ".join(parent.get("class") or []).lower() if parent is not None else ""
                    if _note_class_re.search(cls) or _note_class_re.search(parent_cls):
                        matches_id = True
                except Exception:
                    pass
            if not matches_id:
                continue

            harvested_block = _harvest_id_note_block(tag)
            if harvested_block:
                global_defs_by_id[tag_id] = harvested_block
                # For <a> tags inside note-like parents, also store the clean
                # parent-paragraph text as a fallback. The harvested block can
                # concatenate multiple definitions, losing the individual note.
                if tag.name == "a":
                    try:
                        p = tag.find_parent("p")
                        if p is not None:
                            p_cls = " ".join(p.get("class") or []).lower()
                            if _note_class_re.search(p_cls):
                                p_text = _safe_text(p.get_text(" "))
                                if p_text:
                                    m = _def_line_regex().match(p_text)
                                    if m:
                                        single = _safe_text(m.group(2))
                                        if single and len(single) > 10:
                                            global_defs_by_id[tag_id] = single
                                    elif len(p_text) > 20:
                                        global_defs_by_id[tag_id] = p_text
                    except Exception:
                        pass
                continue

            text = _safe_text(tag.get_text(" "))
            if not text:
                continue

            # If the id tag only contains a marker like "10.", harvest the surrounding paragraph.
            if re.fullmatch(r"\d{1,3}[\.:\)]?", text):
                p = tag.find_parent("p")
                if p is not None:
                    p_text = _safe_text(p.get_text(" "))
                    m = _def_line_regex().match(p_text)
                    if m:
                        harvested = _safe_text(m.group(2))
                        if harvested:
                            global_defs_by_id[tag_id] = harvested
                            continue
                    if p_text and len(p_text) > len(text) + 8:
                        global_defs_by_id[tag_id] = p_text
                        continue

            # Some EPUBs include the marker itself at the start; keep it as-is.
            global_defs_by_id[tag_id] = text

        # 2) Harvest marker-based definitions from notes-like documents (covers books without per-note ids)
        name_l = (name or "").lower()
        looks_like_notes = any(k in name_l for k in ["footnote", "endnote", "notes", "note"])
        cont_start = infer_notes_continuation_harvest_start(lines)
        looks_like_notes_cont = cont_start is not None

        # If this spine doc contains multiple chapter headings, we need to tag
        # harvested definitions to the nearest preceding heading so downstream
        # imports can be scoped correctly.
        joined = "\n".join(lines)
        heads_for_item = _find_chapter_headings_in_text(joined) if joined else []
        multiple_heads = bool(heads_for_item and len(heads_for_item) >= 2)
        line_starts: List[int] = []
        try:
            off = 0
            for ln in lines:
                line_starts.append(off)
                off += len(ln) + 1
        except Exception:
            line_starts = []

        def _label_for_def_line_index(li: Any) -> Optional[str]:
            if not isinstance(li, int) or li < 0 or li >= len(line_starts):
                return carried_context_label
            pos = int(line_starts[li])
            chosen = None
            for hpos, hlabel in heads_for_item:
                if not isinstance(hpos, int):
                    continue
                if hpos <= pos:
                    chosen = hlabel
                else:
                    break
            if chosen:
                return _strip_trailing_footnote_marker_from_heading(chosen) or chosen
            # If the def occurs before the first heading in this spine doc,
            # inherit the carried chapter context from the previous spine item.
            return carried_context_label

        def _group_for_def_line_index(li: Any) -> Optional[str]:
            chosen = _label_for_def_line_index(li)
            g = _chapter_group_key(chosen)
            if g:
                return g
            return carried_context_group

        defs: List[Dict[str, Any]] = []
        split = infer_notes_split(lines)
        notes_chapter_token: Optional[int] = None
        if split is not None:
            notes_chapter_token = _extract_notes_header_chapter_token(lines, split.main_end_index)
            defs = _extract_definitions_from_lines_scoped(lines, split.defs_start_index, initial_chapter_token=notes_chapter_token)
        elif looks_like_notes:
            # Auto-AI: if we couldn't split, ask AI to locate the notes header.
            if (options.marker_profile or "").strip().lower() == "auto_ai" and not structured_footnote_epub:
                ai_split = _ai_infer_notes_split(lines)
                if ai_split is not None:
                    _, ai_defs_start = ai_split
                    defs = _extract_definitions_from_lines_scoped(lines, ai_defs_start, initial_chapter_token=None)
                else:
                    # If the file name screams notes but we didn't detect a block, still attempt tail parsing.
                    tail_start = max(0, int(len(lines) * 0.6))
                    defs = _extract_definitions_from_lines_scoped(lines, tail_start, initial_chapter_token=None)
            else:
                # If the file name screams notes but we didn't detect a block, still attempt tail parsing.
                tail_start = max(0, int(len(lines) * 0.6))
                defs = _extract_definitions_from_lines_scoped(lines, tail_start, initial_chapter_token=None)
        else:
            # No identifiable notes section was found.  Instead of parsing the
            # entire text body as a single notes block, use the EPUB's paragraph
            # (&lt;p&gt;) structure to identify definitions.  A paragraph whose text
            # starts with a symbol definition marker is a definition paragraph;
            # everything else is prose.  This handles inline-definition EPUBs
            # where each definition occupies its own &lt;p&gt; intermixed with prose.
            def_re = _def_line_regex()
            def_paragraphs: List[BeautifulSoup] = []
            for p in soup.find_all("p"):
                pt = _safe_text(p.get_text(" ") or "").strip()
                if pt and def_re.match(pt):
                    def_paragraphs.append(p)

            if len(def_paragraphs) >= 5:
                # Pattern for inline secondary definition markers within a shared &lt;p&gt;
                # (e.g. "* Def1 ** Def2" — we want both as separate definitions).
                _inline_next_marker_re = re.compile(
                    r"(?:\s+|^)(\*+|†+|‡+|§+)\s+",
                    re.UNICODE,
                )

                soup_defs: List[Dict[str, Any]] = []
                for p in def_paragraphs:
                    pt = _safe_text(p.get_text(" ") or "").strip()
                    if not pt:
                        continue

                    # Extract primary definition from the paragraph start.
                    m = def_re.match(pt)
                    if not m:
                        continue
                    marker_norm = _normalize_marker(m.group(1))
                    if re.fullmatch(r"[A-Za-z]", marker_norm):
                        cat = _marker_category_from_raw(m.group(1))
                        if cat not in {"let_paren", "let_bracket"}:
                            continue
                    body = _safe_text(m.group(2) or "").strip()

                    # If the paragraph contains an inline secondary definition
                    # (e.g. "** Slang: ..." after the first definition), extract
                    # that too instead of absorbing it into the first definition.
                    inline_split = _inline_next_marker_re.search(body)
                    if inline_split:
                        remainder = body[inline_split.end() :].strip()
                        body = _safe_text(body[: inline_split.start()]).strip()
                        # Parse the secondary definition from the remainder.
                        m2 = def_re.match(inline_split.group(1) + " " + remainder)
                        if m2:
                            mk2 = _normalize_marker(m2.group(1))
                            bd2 = _safe_text(m2.group(2) or "").strip()
                            if bd2:
                                soup_defs.append({
                                    "marker": mk2,
                                    "text": bd2,
                                })

                    soup_defs.append({
                        "marker": marker_norm,
                        "text": body,
                    })
                defs = soup_defs
            elif cont_start is not None:
                defs = _extract_definitions_from_lines(lines, int(cont_start))

        # Some spine docs contain multiple independent NOTES blocks.
        # `infer_notes_split` returns only one, so also scan for additional
        # notes-header occurrences and parse definitions after each.
        try:
            marker_only_re = re.compile(
                r"^\s*(?:\[|\()?\s*(\d{1,3}|[a-zA-Z]|\*+|†+|‡+|§+)\s*(?:\]|\))?\s*(?:[\]\)\.:\-—]\s*)?\s*$",
                re.UNICODE,
            )
            def_like_re = _def_line_regex()
            header_idxs = [i for i, l in enumerate(lines) if _is_notes_header_line(l)]
            strong_heads_for_item: List[Tuple[int, str]] = []
            for hpos, hlabel in heads_for_item:
                clean_hlabel = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                if (
                    _looks_like_structural_part_or_book_heading(clean_hlabel)
                    or re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", clean_hlabel, re.IGNORECASE)
                    or _extract_chapter_token(clean_hlabel) is not None
                ):
                    strong_heads_for_item.append((hpos, clean_hlabel))
            extra_defs: List[Dict[str, Any]] = []
            for hi in header_idxs:
                # Find the next non-empty line.
                start = None
                for j in range(hi + 1, min(hi + 30, len(lines))):
                    if _safe_text(lines[j] or ""):
                        start = j
                        break
                if start is None:
                    continue

                # Must actually look like definitions nearby; otherwise we match
                # incidental phrases like "notes on distances".
                look_end = min(len(lines), start + 200)
                def_cnt = 0
                mk_cnt = 0
                for k in range(start, look_end):
                    t = lines[k] or ""
                    if def_like_re.match(t):
                        def_cnt += 1
                    elif marker_only_re.match(t):
                        mk_cnt += 1
                    if def_cnt >= 2:
                        break
                if def_cnt < 2 and mk_cnt < 2:
                    continue

                tok = _extract_notes_header_chapter_token(lines, hi)
                block_defs = _extract_definitions_from_lines_scoped(lines, start, initial_chapter_token=tok)

                # If the carried context is a PART_* group (e.g. PART_TWO) and this NOTES
                # header sits under a short subheading (e.g. TITLE), treat the notes
                # definitions as belonging to the carried PART group. This prevents
                # global imports from failing when anchors are grouped at the PART level.
                try:
                    if tok is None and carried_context_group is not None and block_defs:
                        carried_g = str(carried_context_group)
                        if re.fullmatch(r"PART_[A-Z0-9]+", carried_g):
                            hi_pos = int(line_starts[hi]) if isinstance(hi, int) and 0 <= hi < len(line_starts) else None
                            has_prior_notes_header = any(prev_hi < hi for prev_hi in header_idxs)
                            if not has_prior_notes_header:
                                for d in block_defs:
                                    d["chapter_group"] = carried_g
                            elif hi_pos is not None:
                                later_heads = [(hpos, hlabel) for hpos, hlabel in strong_heads_for_item if int(hpos) > hi_pos]
                                if later_heads:
                                    next_group = _chapter_group_key(later_heads[0][1])
                                    if next_group is not None:
                                        for d in block_defs:
                                            d["chapter_group"] = next_group
                            else:
                                hi_g = _group_for_def_line_index(int(hi))
                                if hi_g is not None:
                                    hi_gs = str(hi_g)
                                    # Only override when the local heading is a short label
                                    # (subsection) rather than a full chapter title.
                                    if (
                                        hi_gs != carried_g
                                        and not hi_gs.startswith("PART_")
                                        and len(hi_gs) <= 18
                                    ):
                                        for d in block_defs:
                                            d["chapter_group"] = carried_g
                except Exception:
                    pass

                extra_defs.extend(block_defs)

            if extra_defs:
                defs.extend(extra_defs)
        except Exception:
            pass

        # If we detected an explicit notes block, it's a strong signal even for short lists.
        # Also allow "continuation" pages with only a couple of defs.
        if split is not None or looks_like_notes or looks_like_notes_cont or len(defs) >= 5:
            for d in defs:
                mk = d.get("marker")
                txt = d.get("text")
                if mk and txt:
                    if allowed_categories and _marker_category_from_raw(mk) not in allowed_categories:
                        continue
                    existing = global_defs_by_marker.get(mk, [])
                    if not any(x.get("text") == txt for x in existing):
                        ent_token = d.get("chapter_token")
                        if ent_token is None:
                            ent_token = notes_chapter_token

                        ent_group = d.get("chapter_group")
                        if ent_group is None:
                            ent_group = _group_for_def_line_index(d.get("line_index"))

                        ent_label = _label_for_def_line_index(d.get("line_index"))
                        global_defs_by_marker[mk].append(
                            {
                                "text": txt,
                                "origin": name,
                                "origin_index": origin_index,
                                "chapter_token": ent_token,
                                "chapter_group": ent_group,
                                "parent_chapter_group": _parent_structural_chapter_group(ent_label),
                            }
                        )

    current_chapter_label: Optional[str] = None
    current_chapter_group: Optional[str] = None
    skip_spine_items: set[int] = set()
    debug_markers = _debug_markers_set()
    for chapter_index, item in enumerate(items):
        if chapter_index in skip_spine_items:
            continue
        prev_chapter_label = current_chapter_label
        prev_chapter_group = current_chapter_group

        html_content = item.get_content()
        html_text = html_content.decode("utf-8", errors="ignore") if isinstance(html_content, bytes) else str(html_content)
        soup = BeautifulSoup(html_text, "html.parser")
        chapter_text = _preprocess_for_notes(soup.get_text("\n"))
        source_meta = {
            "source": "epub",
            "chapter_index": chapter_index,
            "chapter_name": getattr(item, "get_name", lambda: None)() if hasattr(item, "get_name") else None,
        }

        # If this document is a footnote/endnote *target* page, don't emit anchor results from it.
        # Already harvested its definitions into global_defs_by_id.
        chapter_name = (source_meta.get("chapter_name") or "").lower()
        looks_like_notes_file = any(k in chapter_name for k in ["footnote", "endnote", "notes", "note"])
        has_note_targets = False

        # Also treat specific App-style filepos note lists as note-target pages.
        fp_id_map, fp_marker_defs = _harvest_specific_filepos_note_targets(soup)
        if len(fp_marker_defs) >= 1:
            has_note_targets = True
        for tag in soup.find_all(True):
            tid = tag.get("id")
            if tid and (id_re.match(tid) or _pt4en_id_re.match(tid)):
                has_note_targets = True
                break
            epub_type = (tag.get("epub:type") or "").lower()
            if "footnote" in epub_type or "endnote" in epub_type or "note" in epub_type:
                has_note_targets = True
                break
        if has_note_targets and (looks_like_notes_file or len(fp_marker_defs) >= 1):
            continue

        # Prefer explicit HTML anchors when present; then augment with regex anchors from main text.
        # Still rely on line-based notes parsing for definitions.
        lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in chapter_text.split("\n")]

        if _looks_like_navigation_or_index_doc(chapter_name, lines):
            continue

        structured_note_id_map, structured_note_defs = _harvest_structured_notes_section_targets(soup)
        structured_note_defs = _filter_definitions_by_profile(structured_note_defs, allowed_categories)

        # When a structured footnote convention is active, skip AI inference and use
        # HTML structure directly. If this spine item has class="footnote" definitions
        # but no prose anchor links (small-marker <a> tags with href fragments),
        # it's a pure footnote-definition page -- skip it (defs already harvested in Pass 1).
        if structured_footnote_epub:
            has_footnote_defs = bool(soup.select(".footnote, .footnotet, .noindent-x1"))
            _small_marker_skip_re = re.compile(r"^\s*(?:\d{1,3}|\*+|\u2020+|\u2021+|\u00a7+|[a-zA-Z])\s*$", re.UNICODE)
            has_prose_anchors = any(
                _small_marker_skip_re.match(_safe_text(a.get_text(" ")).strip())
                and "#" in _safe_text(a.get("href") or "").strip()
                for a in soup.find_all("a")
            )
            if has_footnote_defs and not has_prose_anchors and len(_safe_text(chapter_text)) < 400:
                continue

        split = infer_notes_split(lines)
        if split is None:
            if (options.marker_profile or "").strip().lower() == "auto_ai":
                # Skip AI notes split when structured convention is active.
                if structured_footnote_epub:
                    ai_split = None
                else:
                    ai_split = _ai_infer_notes_split(lines)
                if ai_split is not None:
                    ai_main_end, ai_defs_start = ai_split
                    main_end = ai_main_end
                    defs_start = ai_defs_start
                    split = None
                else:
                    split = None
            if not lines:
                main_end = 0
                defs_start = 0
            else:
                # If a document is predominantly a notes/definitions list and starts
                # with definition-like lines, treat it as a note-target page even if
                # it lacks explicit NOTE headers or id-based targets.
                head_scan_end = min(len(lines), 240)
                head_def_like = [
                    i
                    for i in range(0, head_scan_end)
                    if _def_line_regex().match(lines[i] or "")
                    and not _def_line_looks_like_combined_chapter_heading(lines[i] or "")
                ]
                if len(head_def_like) >= 6 and (head_def_like[0] <= 8):
                    # Only treat as a note-target page when no chapter headings
                    # appear before the first definition-like line.
                    early_text = "\n".join(lines[: min(head_def_like[0] + 40, len(lines))])
                    early_headings = _find_chapter_headings_in_text(early_text)
                    if not early_headings:
                        defs_start = head_def_like[0]
                        main_end = defs_start
                else:
                    tail_start = max(0, int(len(lines) * 0.6))
                    tail_def_like = [
                        i
                        for i in range(tail_start, len(lines))
                        if _def_line_regex().match(lines[i] or "")
                        and not _def_line_looks_like_combined_chapter_heading(lines[i] or "")
                    ]
                    if len(tail_def_like) >= 3:
                        defs_start = tail_def_like[0]
                        main_end = defs_start
                    else:
                        main_end = len(lines)
                        defs_start = len(lines)
        else:
            main_end = split.main_end_index
            defs_start = split.defs_start_index

        # NOTES-block XHTML handling: some EPUBs place NOTES blocks mid-file,
        # with real prose and subsequent chapters after the notes.
        #
        # If we can detect NOTES header(s), treat each detected block as bounded
        # (header..end), and build an exclusion mask for just those spans so:
        #  - anchors/headings in intervening/trailing prose are not filtered out
        #  - definition parsing cannot leak into prose/outlines after the notes
        multi_notes_blocks: Optional[List[Tuple[int, int]]] = None
        try:
            # IMPORTANT: do not rely on a single inferred `defs_start` here.
            # Some XHTML files contain multiple NOTES blocks *and* intervening prose;
            # in those cases `infer_notes_split()` can be ambiguous and return None.
            header_idxs = [i for i in range(0, len(lines)) if _is_notes_header_line(lines[i] or "")]
            header_idxs = sorted({int(i) for i in header_idxs if 0 <= int(i) < len(lines)})
            if header_idxs:
                blocks: List[Tuple[int, int]] = []
                for hi in header_idxs:
                    end = _find_notes_block_end_from_header(lines, hi)
                    if not isinstance(end, int):
                        continue
                    end = max(hi + 1, min(int(end), len(lines)))
                    if end <= (hi + 1):
                        continue
                    blocks.append((hi, end))
                blocks = sorted(set(blocks), key=lambda t: t[0])

                if len(blocks) >= 2:
                    multi_notes_blocks = blocks

            # Fallback: notes-in-the-middle without an explicit NOTES header.
            # If infer_notes_split() triggered and we can see a chapter heading
            # after defs_start, bound the notes block to the first such heading
            # *only when* definition-like density after the heading is low.
            if multi_notes_blocks is None and split is not None and isinstance(defs_start, int) and 0 <= defs_start < len(lines):
                anchors_text0 = "\n".join(lines)

                tmp_heads = _find_chapter_headings_in_text(anchors_text0)
                if tmp_heads:
                    # Map char offsets -> line indices.
                    line_starts0: List[int] = []
                    run0 = 0
                    for ln in lines:
                        line_starts0.append(run0)
                        run0 += len(ln) + 1

                    defs_pos = int(line_starts0[defs_start])
                    next_heading_pos: Optional[int] = None
                    for hpos, _hlab in tmp_heads:
                        if isinstance(hpos, int) and hpos > defs_pos:
                            next_heading_pos = int(hpos)
                            break
                    if next_heading_pos is not None:
                        # Map char offset -> containing line index.
                        # NOTE: headings can appear mid-line (e.g. "... chamber). [PART TWO].(1)")
                        # so bisect_left() would incorrectly point to the *next* line.
                        next_heading_line = bisect.bisect_right(line_starts0, next_heading_pos) - 1
                        next_heading_line = max(0, min(int(next_heading_line), len(lines)))

                        # Require at least a small definitions cluster before the heading.
                        def_re0 = _def_line_regex()
                        before_def_like = 0
                        for i in range(defs_start, min(next_heading_line, len(lines))):
                            if def_re0.match(lines[i] or ""):
                                before_def_like += 1

                        # If there are lots of definition-like lines after the heading,
                        # this is likely a multi-chapter notes file; do NOT bound it.
                        after_scan_end = min(len(lines), next_heading_line + 160)
                        after_def_like = 0
                        for i in range(next_heading_line, after_scan_end):
                            if def_re0.match(lines[i] or ""):
                                after_def_like += 1

                        # Only treat this as notes-in-the-middle if the inferred
                        # notes start well before EOF; otherwise normal endnotes
                        # at the end of a chapter would be mis-bounded.
                        notes_start_ratio = float(defs_start) / max(1.0, float(len(lines)))
                        if before_def_like >= 1 and after_def_like <= 1 and next_heading_line > defs_start and notes_start_ratio <= 0.70:
                            multi_notes_blocks = [(int(defs_start), int(next_heading_line))]
        except Exception:
            multi_notes_blocks = None

        if multi_notes_blocks:
            seen_defs: set[tuple[str, str, int]] = set()
            definitions = []
            for hi, end in multi_notes_blocks:
                sidx = min(len(lines), int(hi))
                # If this block starts with an explicit NOTES header line, skip it.
                if 0 <= sidx < len(lines) and _is_notes_header_line(lines[sidx] or ""):
                    sidx = min(len(lines), sidx + 1)
                # Skip blank padding immediately after header.
                for j in range(sidx, min(sidx + 30, int(end))):
                    if _safe_text(lines[j] or ""):
                        sidx = j
                        break

                block_lines = lines[sidx:end]
                for d in _extract_definitions_from_lines(block_lines, 0):
                    mk = str(d.get("marker") or "")
                    txt = str(d.get("text") or "")
                    li = -1
                    try:
                        raw_li = d.get("line_index")
                        if raw_li is not None:
                            li = int(raw_li)
                    except Exception:
                        li = -1
                    if li >= 0:
                        d["line_index"] = int(sidx) + int(li)
                        try:
                            li = int(d.get("line_index"))
                        except Exception:
                            li = -1
                    key = (mk, txt, li)
                    if key in seen_defs:
                        continue
                    seen_defs.add(key)
                    definitions.append(d)
            definitions = _filter_definitions_by_profile(definitions, allowed_categories)
        else:
            definitions = _extract_definitions_from_lines(lines, defs_start)

            # When the primary notes-split heuristic found no notes section
            # at all (split is None), the text-based extraction produces
            # poor results for inline-definition EPUBs where definitions are
            # scattered throughout the prose.  Use soup-level paragraph
            # detection instead: a paragraph whose text starts with a symbol
            # marker is a definition paragraph.
            if split is None:
                def_re2 = _def_line_regex()
                _inline_next_marker_re = re.compile(
                    r"(?:\s+|^)(\*+|†+|‡+|§+)\s+",
                    re.UNICODE,
                )
                all_ps = list(soup.find_all("p"))
                soup_defs2: List[Dict[str, Any]] = []

                # ---- Pass 1: extract definitions from <p> tags that start
                #              with a symbol marker. ----
                # Each entry: (paragraph_index, marker, body)
                raw_defs: List[Tuple[int, str, str]] = []
                for pi, p in enumerate(all_ps):
                    pt = _safe_text(p.get_text(" ") or "").strip()
                    if not pt:
                        continue
                    # Fix 2: convert typographic markers like "**\u2022" to "***"
                    # so _def_line_regex sees a valid separator after the marker.
                    if pt.startswith("**\u2022"):
                        pt = "*** " + pt[3:].strip()
                    elif pt.startswith("**") and len(pt) > 2 and not pt[2].isalnum() and not pt[2].isspace():
                        pt = "*** " + pt[3:].strip()
                    m_start = def_re2.match(pt)
                    if not m_start:
                        continue
                    marker_norm = _normalize_marker(m_start.group(1))
                    if re.fullmatch(r"[A-Za-z]", marker_norm):
                        cat = _marker_category_from_raw(m_start.group(1))
                        if cat not in {"let_paren", "let_bracket"}:
                            continue
                    body = _safe_text(m_start.group(2) or "").strip()

                    # Handle inline secondary definition inside same <p>.
                    inline = _inline_next_marker_re.search(body)
                    if inline:
                        remainder = body[inline.end() :].strip()
                        body = _safe_text(body[: inline.start()]).strip()
                        m2 = def_re2.match(inline.group(1) + " " + remainder)
                        if m2:
                            mk2 = _normalize_marker(m2.group(1))
                            bd2 = _safe_text(m2.group(2) or "").strip()
                            if bd2:
                                raw_defs.append((pi, mk2, bd2))
                    raw_defs.append((pi, marker_norm, body))

                if len(raw_defs) >= 5:
                    # ---- Pass 2: merge continuation <p> tags.  Some EPUBs
                    #     split a definition across consecutive <p> tags;
                    #     the first carries the marker, the rest is plain
                    #     text.  A continuation is a short (<80 chars) <p>
                    #     that does not start with a definition marker.
                    # ---------------------------------------------------------
                    for di, (pi, mk, body) in enumerate(raw_defs):
                        # Find where the NEXT definition starts.
                        next_pi = None
                        if di + 1 < len(raw_defs):
                            next_pi = raw_defs[di + 1][0]

                        # Scan forward for continuation <p> tags.
                        for ci in range(pi + 1, min(pi + 3, len(all_ps))):
                            if next_pi is not None and ci >= next_pi:
                                break
                            ct = _safe_text(all_ps[ci].get_text(" ") or "").strip()
                            if not ct or def_re2.match(ct):
                                break
                            if len(ct) > 80:
                                break
                            if ct[0].isupper() and len(ct) > 40:
                                break
                            body += " " + ct

                        soup_defs2.append({
                            "marker": mk,
                            "text": body,
                            "line_index": pi + 1,
                            "origin_index": chapter_index,
                        })

                if len(soup_defs2) >= 5:
                    definitions = soup_defs2
            definitions = _filter_definitions_by_profile(definitions, allowed_categories)

        # Build a set of definition body prefixes for the definition-header
        # anchor filter (used later in the deduped filtering step).
        _def_starts: set[str] = set()
        if split is None:
            for d in definitions:
                txt = _safe_text((d.get("text") or "")).strip()
                if len(txt) >= 15:
                    _def_starts.add(txt[:40].lower())

        # Build section labels when multiple NOTES blocks exist in one spine item.
        # Use globally-assigned numbered labels: "NOTES Section 1", "NOTES Section 2", etc.
        # Only active for books where a structured convention is detected.
        multi_section_labels: Optional[List[str]] = None
        if structured_footnote_epub and multi_notes_blocks:
            multi_section_labels = [
                f"NOTES Section {block_idx + 1}"
                for block_idx in range(len(multi_notes_blocks))
            ]

        # NOTES can spill across spine items (split mid-definition or split mid-notes-list).
        # If we have a NOTES block that reaches EOF in this spine item and the *next*
        # spine item starts with continuation/definition text (before any new chapter),
        # pull that continuation + subsequent defs forward so anchors in this item can
        # still be paired correctly.
        spill_start_line: Optional[int] = None
        spill_base: Optional[int] = None
        try:
            spill_ok = False
            if multi_notes_blocks:
                # Only consider the *last* detected notes block. Earlier blocks ending
                # before EOF are common in multi-chapter spine docs and should not
                # disable spillover handling for the final notes block.
                try:
                    last_hi, last_end = sorted(multi_notes_blocks, key=lambda t: int(t[0]))[-1]
                    if isinstance(last_end, int) and int(last_end) >= len(lines):
                        spill_ok = True
                        spill_start_line = int(last_hi)
                except Exception:
                    spill_ok = False
                    spill_start_line = None
            elif split is not None and isinstance(defs_start, int) and defs_start < len(lines):
                spill_ok = True
                spill_start_line = int(defs_start)

            if spill_ok and definitions and (chapter_index + 1) < len(items):
                next_item = items[chapter_index + 1]
                next_html = next_item.get_content()
                next_text = next_html.decode("utf-8", errors="ignore") if isinstance(next_html, bytes) else str(next_html)
                next_soup = BeautifulSoup(next_text, "html.parser")
                next_chapter_text = _preprocess_for_notes(next_soup.get_text("\n"))
                next_lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in next_chapter_text.split("\n")]

                # Parse definitions from the start of the next item. The extractor
                # can split inline markers like "... workus). 16. I do not know ...".
                extra_defs = _extract_definitions_from_lines(next_lines, 0)

                # Ensure the prefix doesn't look like the start of a new chapter.
                def _prefix_looks_like_new_chapter(prefix: List[str]) -> bool:
                    for ln in prefix:
                        t = _safe_text(ln or "").strip()
                        if not t:
                            continue
                        u = t.upper()
                        if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", t, re.IGNORECASE):
                            return True
                        if re.match(r"^\s*[IVXLC]{1,12}\s*\.?\s*$", t):
                            return True
                        if _def_line_looks_like_combined_chapter_heading(t):
                            return True
                        if "BOOK" in u and len(u) <= 18:
                            return True
                    return False

                first_def_idx = extra_defs[0].get("line_index") if extra_defs else None

                try:
                    if isinstance(first_def_idx, int) and 0 <= first_def_idx < len(next_lines):
                        first_def_line = next_lines[int(first_def_idx)] or ""
                        if _def_line_looks_like_combined_chapter_heading(first_def_line):
                            first_def_idx = None
                            extra_defs = []
                except Exception:
                    pass

                if isinstance(first_def_idx, int) and first_def_idx <= 90:
                    prefix = list(next_lines[:first_def_idx])

                    # Continuation text can share a line with the next definition marker,
                    # e.g. "... Sally of Nottingham (see next note). 13. ...". Preserve
                    # the pre-marker fragment as spillover continuation for the prior note.
                    try:
                        if 0 <= int(first_def_idx) < len(next_lines):
                            first_def_line = next_lines[int(first_def_idx)] or ""
                            for chunk in _split_inline_numeric_definitions(first_def_line):
                                if _def_line_regex().match(chunk or ""):
                                    break
                                if _safe_text(chunk or ""):
                                    prefix.append(chunk)
                    except Exception:
                        pass

                    if not _prefix_looks_like_new_chapter(prefix):
                        if debug_markers and _debug_markers_verbose_enabled():
                            try:
                                _stderr_log(
                                    f"[DBG] spillover: chap_index={chapter_index} next='{next_item.get_name() if hasattr(next_item,'get_name') else ''}' extra_defs={len(extra_defs)} first_def_idx={first_def_idx}"
                                )
                            except Exception:
                                pass
                        cont = " ".join([p.strip() for p in prefix if (p or "").strip()])
                        if cont:
                            last = definitions[-1]
                            last["text"] = _safe_text(f"{_safe_text(last.get('text') or '')} {cont}")

                        if extra_defs:
                            # Mark these as local candidates for pairing purposes.
                            base = len(lines) + 100_000
                            spill_base = base

                            # If this spine item contains multiple chapter headings,
                            # pairing is scoped by `chapter_group`. Spilled definitions
                            # from the next spine item must be tagged to the relevant
                            # group (usually the last chapter in this spine doc).
                            spill_group = None
                            try:
                                spill_heads = _find_chapter_headings_in_text("\n".join(lines))

                                # Mirror the later heading filtering that excludes headings
                                # occurring inside notes/definitions spans. This keeps
                                # spillover tagging consistent with anchor grouping.
                                try:
                                    if spill_heads and multi_notes_blocks:
                                        excluded_total: List[bool] = [False] * len(lines)
                                        for hi2, end2 in multi_notes_blocks:
                                            for i2 in range(int(hi2), int(end2)):
                                                if 0 <= i2 < len(excluded_total):
                                                    excluded_total[i2] = True

                                        line_starts_spill_h: List[int] = []
                                        run_spill_h = 0
                                        for ln in lines:
                                            line_starts_spill_h.append(run_spill_h)
                                            run_spill_h += len(ln) + 1

                                        filtered: List[Tuple[int, str]] = []
                                        for hpos, hlabel in spill_heads:
                                            if not isinstance(hpos, int) or hpos < 0:
                                                continue
                                            li2 = bisect.bisect_right(line_starts_spill_h, hpos) - 1
                                            if 0 <= li2 < len(excluded_total) and excluded_total[li2]:
                                                continue
                                            filtered.append((hpos, hlabel))
                                        spill_heads = filtered
                                except Exception:
                                    pass

                                if spill_heads:
                                    # Prefer the heading that precedes the NOTES block
                                    # that is spilling across this spine boundary.
                                    spill_label = None
                                    if spill_start_line is not None:
                                        try:
                                            line_starts_spill: List[int] = []
                                            run_spill = 0
                                            for ln in lines:
                                                line_starts_spill.append(run_spill)
                                                run_spill += len(ln) + 1
                                            if 0 <= int(spill_start_line) < len(line_starts_spill):
                                                spill_pos = int(line_starts_spill[int(spill_start_line)])
                                                chosen = None
                                                for hpos, hlabel in spill_heads:
                                                    if isinstance(hpos, int) and hpos <= spill_pos:
                                                        chosen = hlabel
                                                    else:
                                                        break
                                                spill_label = chosen
                                        except Exception:
                                            spill_label = None
                                    if not spill_label:
                                        spill_label = spill_heads[-1][1]
                                    spill_label = _strip_trailing_footnote_marker_from_heading(spill_label) or spill_label
                                    spill_group = _chapter_group_key(spill_label)
                            except Exception:
                                spill_group = None

                            for d in extra_defs:
                                try:
                                    d["line_index"] = base + int(d.get("line_index") or 0)
                                except Exception:
                                    d["line_index"] = base
                                if spill_group and d.get("chapter_group") is None:
                                    d["chapter_group"] = spill_group
                            extra_defs = _filter_definitions_by_profile(extra_defs, allowed_categories)
                            definitions.extend(extra_defs)

                            if debug_markers and _debug_markers_verbose_enabled():
                                try:
                                    c13 = sum(1 for d in extra_defs if str(d.get('marker') or '') in {'13','14'})
                                    _stderr_log(
                                        f"[DBG] spillover: imported_defs={len(extra_defs)} imported_13_14={c13} spill_group={spill_group}"
                                    )
                                except Exception:
                                    pass

                        # If the next item looks like it's *only* notes/definitions,
                        # skip scanning it to avoid emitting duplicates.
                        if len(extra_defs) >= 6 and not _prefix_looks_like_new_chapter(next_lines[: min(60, len(next_lines))]):
                            # IMPORTANT: some spine items contain a NOTES/definitions run
                            # and then continue with the next chapter/prose. In those
                            # cases, skipping the entire next item would drop anchors
                            # and chapter transitions (e.g., "[PART TWO].(1)").
                            skip_ok = True
                            try:
                                # If the next spine item still contains multiple anchor-like
                                # markers in running text, do not skip it: it's not just a
                                # pure continuation-notes page.
                                next_anchor_probe = _filter_anchors_by_profile(
                                    _extract_anchors_from_text("\n".join(next_lines)),
                                    allowed_categories,
                                )
                                if len(next_anchor_probe) >= 3:
                                    skip_ok = False

                                heads_next = _find_chapter_headings_in_text("\n".join(next_lines))
                                if heads_next:
                                    # Compute an approximate position for the first definition.
                                    line_starts_next: List[int] = []
                                    run_next = 0
                                    for ln in next_lines:
                                        line_starts_next.append(run_next)
                                        run_next += len(ln) + 1
                                    def_pos_next = 0
                                    if isinstance(first_def_idx, int) and 0 <= first_def_idx < len(line_starts_next):
                                        def_pos_next = int(line_starts_next[first_def_idx])

                                    # If we see a heading *after* the definitions begin,
                                    # treat this as notes-in-the-middle; do NOT skip.
                                    for hpos, _hlab in heads_next:
                                        if isinstance(hpos, int) and hpos > (def_pos_next + 40):
                                            skip_ok = False
                                            break
                            except Exception:
                                skip_ok = True

                            if skip_ok:
                                skip_spine_items.add(chapter_index + 1)
        except Exception:
            pass
        main_text = "\n".join(lines[:main_end])

        # Two text views:
        #  - anchor_text: definition-excluding view (good for inferring chapter labels)
        #  - anchors_text: full text view used for regex anchor positions + heading offsets
        #
        # Rationale: some EPUB conversions can cause our definition-exclusion heuristic
        # to skip large spans of prose (e.g., when notes split is misdetected). Regex
        # anchor extraction must not silently lose legitimate markers.
        if multi_notes_blocks:
            excluded_total: List[bool] = [False] * len(lines)
            for hi, end in multi_notes_blocks:
                for i in range(int(hi), int(end)):
                    if 0 <= i < len(excluded_total):
                        excluded_total[i] = True

            line_starts: List[int] = []
            run = 0
            for ln in lines:
                line_starts.append(run)
                run += len(ln) + 1
            mask = (excluded_total, line_starts)
            anchor_text = "\n".join([ln for i, ln in enumerate(lines) if not excluded_total[i]])
        else:
            anchor_text = _anchor_text_excluding_definitions(lines, defs_start)
        anchors_text = "\n".join(lines)

        # Build a line-level exclusion mask for the inferred definitions region.
        # We use this to filter regex anchors (cross-references inside defs) and
        # also to drop soup-derived anchors that occur inside defs (often backlinks).
        if not multi_notes_blocks:
            mask = _build_definition_exclusion_mask(lines, defs_start)

        if debug_markers and _debug_markers_verbose_enabled():
            try:
                chap_lab = _safe_text(source_meta.get("chapter_label") or "")
                chap_name = _safe_text(source_meta.get("chapter_name") or "")
                at_upper = anchor_text.upper()
                has_pp = ("PP." in at_upper and "21" in at_upper and "-2" in at_upper) or ("PP" in at_upper and "21-2" in at_upper)
                _stderr_log(
                    f"[DBG] chap_index={chapter_index} chap='{chap_lab}' file='{chap_name}' main_end={main_end} defs_start={defs_start} lines={len(lines)} anchor_text_len={len(anchor_text)} has_pp21_2={has_pp}"
                )
            except Exception:
                pass

        # Attempt to infer a stable logical chapter label so split chapters can be grouped.
        # Some EPUBs contain a NOTES block mid-file and then continue with the next chapter;
        # infer from the prose-only view so we can still see headings after the notes.
        #
        # When the TOC is high-quality and the convention is active, use it to
        # help identify chapter boundaries.
        label_lines = anchor_text.split("\n")
        toc_label: Optional[str] = None
        toc_boundaries: List[Tuple[str, int]] = []  # [(label, char_position), ...] sorted by position
        chapter_file_key = (source_meta.get("chapter_name") or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        if structured_footnote_epub and toc_is_high_quality:
            if chapter_file_key in toc_file_to_label:
                toc_label = toc_file_to_label[chapter_file_key]
            # Build TOC boundary positions when the current file has multiple TOC entries.
            toc_entries_for_file = toc_file_entries.get(chapter_file_key)
            if toc_entries_for_file and len(toc_entries_for_file) >= 2:
                toc_boundaries = _resolve_toc_anchor_positions(
                    toc_entries_for_file, soup, lines
                )
        if toc_label:
            inferred_label = toc_label
            soup_label = toc_label
            logical_label = toc_label
        else:
            #
            # IMPORTANT: A soup-derived heading can be title-only (e.g. <h2>THE SCRUBBING...</h2>)
            # while the Roman/Arabic numeral appears in a separate tag/line (e.g. "IX.").
            # In that case, prefer the logical (line-scan) label when it yields a chapter token.
            inferred_label = _infer_chapter_label_from_soup(soup)
            soup_label = inferred_label
            logical_label: Optional[str] = None
            if inferred_label:
                try:
                    if _extract_chapter_token(inferred_label) is None:
                        logical_label = _infer_logical_chapter_label(label_lines)
                        if logical_label and _extract_chapter_token(logical_label) is not None:
                            inferred_label = logical_label
                except Exception:
                    logical_label = None
            if not inferred_label:
                logical_label = _infer_logical_chapter_label(label_lines)
                inferred_label = logical_label

            # In omnibus spine items, the first real heading near the start of the file is often
            # the stable chapter label, while later numeric headings are internal sub-sections.
            # If the next detected heading is far downstream, prefer that early dominant heading.
            try:
                early_heads = _find_chapter_headings_in_text(anchor_text)
                if early_heads:
                    first_hpos, first_hlabel = early_heads[0]
                    second_hpos = early_heads[1][0] if len(early_heads) >= 2 else None
                    if (
                        isinstance(first_hpos, int)
                        and first_hpos <= 80
                        and first_hlabel
                        and _label_is_plausible_chapter_label(first_hlabel)
                        and (
                            second_hpos is None
                            or (isinstance(second_hpos, int) and int(second_hpos) - int(first_hpos) >= 20000)
                        )
                    ):
                        inferred_label = _strip_trailing_footnote_marker_from_heading(first_hlabel) or first_hlabel
            except Exception:
                pass

            if not inferred_label:
                # Weak fallback: do not override an already-known chapter label.
                if current_chapter_label is None:
                    inferred_label = _infer_chapter_label_from_item_name(source_meta.get("chapter_name"))

        # Marker-driven diagnostics to make chapter-label issues debuggable in one run.
        # We only dump details for chapters that actually contain one of the debug markers
        # in their parsed definitions (so output stays small).
        if debug_markers and _debug_markers_verbose_enabled():
            try:
                hit = False
                for d in definitions:
                    mk = _safe_text(d.get("marker") or "")
                    if mk and mk in debug_markers:
                        hit = True
                        break
                if hit:
                    chap_name = _safe_text(source_meta.get("chapter_name") or "")
                    s_lab = _safe_text(soup_label or "")
                    l_lab = _safe_text(logical_label or "")
                    f_lab = _safe_text(inferred_label or "")
                    s_tok = _extract_chapter_token(s_lab)
                    l_tok = _extract_chapter_token(l_lab)
                    f_tok = _extract_chapter_token(f_lab)
                    # Quick signal: do we even see a standalone Roman numeral line anywhere?
                    roman_line_hits = []
                    roman_line_re_dbg = re.compile(r"^\s*[IVXLC]{1,12}\.?(?:\s+)?$")
                    for ln in label_lines[: min(len(label_lines), 600)]:
                        tt = _safe_text(ln or "")
                        tt = _strip_trailing_footnote_marker_from_heading(tt) or tt
                        if tt and roman_line_re_dbg.match(tt):
                            roman_line_hits.append(tt.strip())
                            if len(roman_line_hits) >= 6:
                                break
                    _stderr_log(
                        f"[DBG] chap_index={chapter_index} file='{chap_name}' soup_label='{s_lab}' soup_tok={s_tok} logical_label='{l_lab}' logical_tok={l_tok} final_label='{f_lab}' final_tok={f_tok} roman_lines={roman_line_hits}"
                    )
            except Exception:
                pass

        if inferred_label:
            inferred_label = _strip_trailing_footnote_marker_from_heading(inferred_label)

        # Determine if this item provides a chapter boundary we should trust.
        # When structured_footnote_epub is active with a high-quality TOC:
        #   - Files with a TOC entry use the TOC label.
        #   - Files without a TOC entry normally inherit the previous label, BUT
        #     when they contain strong internal headings (PART/BOOK/CHAPTER/roman),
        #     allow those headings to set the chapter context. This handles spine
        #     items that fall between TOC entries but still have clear boundaries.
        heuristic_label_is_plausible = bool(inferred_label and _label_is_plausible_chapter_label(inferred_label))
        heuristic_blocked_by_toc = False
        if structured_footnote_epub and toc_is_high_quality and not toc_label:
            heuristic_blocked_by_toc = True
            # Unblock when the file has strong structural headings.
            if heuristic_label_is_plausible and inferred_label:
                has_structural_heading = (
                    _looks_like_structural_part_or_book_heading(inferred_label)
                    or bool(re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", str(inferred_label), re.IGNORECASE))
                    or (_extract_chapter_token(str(inferred_label)) is not None)
                )
                if has_structural_heading:
                    heuristic_blocked_by_toc = False
        use_heuristic_label = heuristic_label_is_plausible and not heuristic_blocked_by_toc

        if use_heuristic_label:
            current_chapter_label = inferred_label

        # Stable grouping key: prevents minor label variations from splitting groups.
        if use_heuristic_label:
            current_chapter_group = _chapter_group_key(inferred_label)
        chosen_label = inferred_label if use_heuristic_label else current_chapter_label
        source_meta["chapter_label"] = chosen_label
        source_meta["chapter_group"] = current_chapter_group
        chapter_label_local = bool(use_heuristic_label)
        supplemental_inherited_defs: Dict[str, str] = {}
        supplemental_inherited_label: Optional[str] = None
        supplemental_inherited_group: Optional[str] = None

        # In inherited-label items with no reliable local notes split, recover the early
        # numeric definitions before anchor extraction so later bare-digit anchors can be
        # recognized and paired normally.
        try:
            numeric_defs_now = sum(
                1 for d in definitions if re.fullmatch(r"\d{1,3}", _safe_text(d.get("marker") or ""))
            )
            if not chapter_label_local and split is None and numeric_defs_now < 12:
                strong_heads_for_recovery = []
                for hpos, hlabel in _find_chapter_headings_in_text(anchors_text):
                    clean_hlabel = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                    if (
                        _looks_like_structural_part_or_book_heading(clean_hlabel)
                        or re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", clean_hlabel, re.IGNORECASE)
                        or _extract_chapter_token(clean_hlabel) is not None
                    ):
                        strong_heads_for_recovery.append((hpos, hlabel))

                early_numeric_def_lines = []
                for i_probe, ln_probe in enumerate(lines[: min(len(lines), 400)]):
                    m_probe = _def_line_regex().match(ln_probe or "")
                    if not m_probe:
                        continue
                    mk_probe = _safe_text(m_probe.group(1) or "")
                    if re.fullmatch(r"\d{1,3}", mk_probe):
                        early_numeric_def_lines.append(i_probe)

                if strong_heads_for_recovery and len(early_numeric_def_lines) >= 5:
                    first_recovery_label = _strip_trailing_footnote_marker_from_heading(strong_heads_for_recovery[0][1]) or strong_heads_for_recovery[0][1]
                    supplemental_inherited_label = first_recovery_label
                    supplemental_inherited_group = _chapter_group_key(first_recovery_label)
                    first_def_line = int(early_numeric_def_lines[0])
                    supplemental_defs = _extract_definitions_from_lines(lines, first_def_line)
                    supplemental_defs = _filter_definitions_by_profile(supplemental_defs, allowed_categories)
                    if supplemental_defs:
                        seen_local: set[tuple[str, str, int]] = set()
                        merged_defs: List[Dict[str, Any]] = []
                        for d in list(definitions) + list(supplemental_defs):
                            mk = _safe_text(d.get("marker") or "")
                            txt = _safe_text(d.get("text") or "")
                            try:
                                li = int(d.get("line_index")) if d.get("line_index") is not None else -1
                            except Exception:
                                li = -1
                            key = (mk, txt, li)
                            if key in seen_local:
                                continue
                            seen_local.add(key)
                            merged_defs.append(d)
                        definitions = merged_defs
                        marker_counts: Dict[str, int] = defaultdict(int)
                        marker_text: Dict[str, str] = {}
                        for d in supplemental_defs:
                            mk = _safe_text(d.get("marker") or "")
                            txt = _safe_text(d.get("text") or "")
                            if not re.fullmatch(r"\d{1,3}", mk) or not txt:
                                continue
                            marker_counts[mk] += 1
                            marker_text.setdefault(mk, txt)
                        supplemental_inherited_defs = {
                            mk: marker_text[mk]
                            for mk, count in marker_counts.items()
                            if count == 1 and mk in marker_text
                        }
        except Exception:
            pass

        # Definition-heavy appendix tails can look like a continued chapter while actually
        # being an editorial note list with noisy mixed markers. Do not emit anchor rows
        # from those inherited-label tails.
        try:
            if not chapter_label_local and split is not None and len(definitions) >= 80:
                numeric_markers = []
                non_numeric_markers = 0
                for d in definitions:
                    mk = _safe_text(d.get("marker") or "")
                    if re.fullmatch(r"\d{1,3}", mk):
                        try:
                            numeric_markers.append(int(mk))
                        except Exception:
                            pass
                    elif mk:
                        non_numeric_markers += 1
                if numeric_markers and len(numeric_markers) >= 20 and non_numeric_markers >= 20 and max(numeric_markers) >= 100:
                    continue
        except Exception:
            pass

        # If there's effectively no running text (e.g., a standalone footnotes.xhtml),
        # skip anchor extraction to avoid generating noise from backlinks.
        soup_anchors: List[Dict[str, Any]]
        if len(_safe_text(main_text)) < 200:
            soup_anchors = []
        else:
            soup_anchors = _extract_anchors_from_soup(soup)

        structured_notes_authoritative = False
        if soup_anchors and structured_note_id_map:
            structured_ids = set(structured_note_id_map.keys())
            structured_soup_anchors: List[Dict[str, Any]] = []
            local_href_anchor_count = 0
            chapter_file_name = _safe_text(chapter_name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
            for a in soup_anchors:
                href = _safe_text(a.get("href") or "")
                if "#" not in href:
                    continue
                href_file, frag = href.split("#", 1)
                href_file = _safe_text(href_file).replace("\\", "/").rsplit("/", 1)[-1].lower()
                is_local_href = not href_file or href_file == chapter_file_name
                if is_local_href:
                    local_href_anchor_count += 1
                if frag in structured_ids:
                    structured_soup_anchors.append(a)

            if structured_soup_anchors and local_href_anchor_count > 0:
                coverage_ratio = float(len(structured_soup_anchors)) / float(local_href_anchor_count)
                if coverage_ratio >= 0.80:
                    structured_notes_authoritative = True

            if structured_notes_authoritative:
                # Preserve anchors whose href fragments point to globally-known
                # definition IDs (e.g., multi-file chapters where anchors live in
                # one spine item but definitions in another). Without this, anchors
                # like c12b_rfn1 (in part0024_split_000.html) pointing to definitions
                # in part0024_split_002.html are silently dropped.
                for a in soup_anchors:
                    if a in structured_soup_anchors:
                        continue
                    href = _safe_text(a.get("href") or "")
                    if "#" not in href:
                        continue
                    frag = href.rsplit("#", 1)[1].strip()
                    if frag and frag in global_defs_by_id:
                        structured_soup_anchors.append(a)
                soup_anchors = structured_soup_anchors
                definitions = [dict(d) for d in structured_note_defs]

        # Give soup anchors approximate positions so multi-heading grouping works.
        if soup_anchors:
            _assign_positions_to_soup_anchors(soup_anchors, lines, soup)

            # Filter soup anchors that occur inside the inferred definitions region.
            # These are usually backlinks/crosslinks from within notes lists, not
            # anchors in the running prose.
            if mask is not None:
                try:
                    excluded, line_starts = mask
                    filtered_soup: List[Dict[str, Any]] = []
                    for a in soup_anchors:
                        pos = a.get("position")
                        if not isinstance(pos, int) or pos < 0:
                            filtered_soup.append(a)
                            continue
                        li = bisect.bisect_right(line_starts, pos) - 1
                        if 0 <= li < len(excluded) and excluded[li]:
                            continue
                        filtered_soup.append(a)
                    soup_anchors = filtered_soup
                except Exception:
                    pass

        regex_anchors = [] if (structured_notes_authoritative and soup_anchors) or structured_footnote_epub else _extract_anchors_from_text(anchors_text)

        # If we inferred a notes definitions region, filter regex anchors that occur inside
        # that region. Definitions often contain cross-references like "pp. 21-2.(1)" which
        # should not be treated as *anchors* in the running prose.
        if regex_anchors and mask is not None:
            try:
                excluded, line_starts = mask
                filtered_rx: List[Dict[str, Any]] = []
                for a in regex_anchors:
                    pos = a.get("position")
                    if not isinstance(pos, int) or pos < 0:
                        filtered_rx.append(a)
                        continue
                    li = bisect.bisect_right(line_starts, pos) - 1
                    if 0 <= li < len(excluded) and excluded[li]:
                        continue
                    filtered_rx.append(a)
                regex_anchors = filtered_rx
            except Exception:
                pass

        # Apply marker-family filtering.
        soup_anchors = _filter_anchors_by_profile(soup_anchors, allowed_categories)
        regex_anchors = _filter_anchors_by_profile(regex_anchors, allowed_categories)

        # Apply anchor plausibility heuristics to regex-derived anchors.
        # We intentionally do NOT apply this to explicit href anchors (id_link/noteref).
        if regex_anchors:
            try:
                filtered_rx2: List[Dict[str, Any]] = []
                for a in regex_anchors:
                    raw = _safe_text(a.get("marker_raw") or a.get("marker") or "")
                    norm = _safe_text(a.get("marker") or "")
                    ctx = _safe_text(a.get("context") or "")
                    if not norm:
                        continue
                    if not anchor_is_probable_footnote(raw, norm, ctx, has_href=False):
                        continue
                    filtered_rx2.append(a)
                regex_anchors = filtered_rx2
            except Exception:
                pass

        if debug_markers and _debug_markers_verbose_enabled():
            try:
                chap_lab = _safe_text(source_meta.get("chapter_label") or "")
                chap_name = _safe_text(source_meta.get("chapter_name") or "")
                for mk in sorted(debug_markers):
                    rx_hits = [a for a in regex_anchors if _safe_text(a.get("marker") or "") == mk]
                    if not rx_hits:
                        continue
                    _stderr_log(
                        f"[DBG] chap_index={chapter_index} chap='{chap_lab}' file='{chap_name}' marker={mk} regex_anchors={len(rx_hits)}"
                    )
                    for a in rx_hits[:6]:
                        pos = a.get("position")
                        raw = _safe_text(a.get("marker_raw") or "")
                        ctx = _safe_text(a.get("context") or "")
                        _stderr_log(f"[DBG]   regex marker={mk} pos={pos} raw='{raw}' ctx='{ctx[:140]}'")
            except Exception:
                pass

        anchors = list(soup_anchors)
        anchors.extend(regex_anchors)
        # Add bare-digit anchors only for markers that exist in defs.
        # When structured convention is active, skip bare-digit anchors — they are
        # never real footnote markers in well-structured EPUBs.
        def_markers = [d.get("marker") for d in definitions if d.get("marker")]
        bare_digit_anchors = [] if structured_footnote_epub else _extract_bare_digit_anchors_from_text(anchors_text, def_markers)
        if bare_digit_anchors and mask is not None:
            try:
                excluded, line_starts = mask
                filtered_bd: List[Dict[str, Any]] = []
                for a in bare_digit_anchors:
                    pos = a.get("position")
                    if not isinstance(pos, int) or pos < 0:
                        filtered_bd.append(a)
                        continue
                    li = bisect.bisect_right(line_starts, pos) - 1
                    if 0 <= li < len(excluded) and excluded[li]:
                        continue
                    filtered_bd.append(a)
                bare_digit_anchors = filtered_bd
            except Exception:
                pass

        anchors.extend(bare_digit_anchors)

        # De-dup anchors (prefer explicit href/noteref anchors over regex-only anchors).
        href_anchors = [a for a in anchors if a.get("href")]
        non_href_anchors = [a for a in anchors if not a.get("href")]

        # Drop non-href anchors that look like duplicates of an href anchor (same marker, overlapping context).
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

        def _kw_set(s: str) -> set[str]:
            words = re.findall(r"[A-Za-z]{4,}", (s or "").lower())
            out: List[str] = []
            for w in words:
                if w in stopwords:
                    continue
                if w not in out:
                    out.append(w)
                if len(out) >= 12:
                    break
            return set(out)

        filtered_non_href: List[Dict[str, Any]] = []
        for a in non_href_anchors:
            marker = a.get("marker")
            ctx = (a.get("context") or "").strip()
            if not marker or not ctx:
                continue
            is_dup = False
            for h in href_anchors:
                if h.get("marker") != marker:
                    continue
                hctx = (h.get("context") or "").strip()
                if not hctx:
                    continue
                # 1) Simple overlap check; avoids heavy NLP.
                if ctx in hctx or hctx in ctx:
                    is_dup = True
                    break

                # 2) If both anchors have approximate character offsets, treat near-coincident
                # markers as duplicates even if the captured context windows differ.
                ap = a.get("position")
                hp = h.get("position")
                if isinstance(ap, int) and isinstance(hp, int) and ap >= 0 and hp >= 0:
                    if abs(int(ap) - int(hp)) <= 25:
                        is_dup = True
                        break

                # 3) Keyword overlap fallback (for cases where positions couldn't be assigned).
                # Only apply this when both contexts are reasonably sized.
                if len(ctx) >= 40 and len(hctx) >= 40:
                    s1 = _kw_set(ctx)
                    s2 = _kw_set(hctx)
                    if s1 and s2:
                        inter = len(s1 & s2)
                        union = len(s1 | s2)
                        if inter >= 4 and (inter / max(1, union)) >= 0.55:
                            is_dup = True
                            break
            if not is_dup:
                filtered_non_href.append(a)

        anchors = href_anchors + filtered_non_href

        seen = set()
        deduped: List[Dict[str, Any]] = []
        for a in anchors:
            key = (a.get("marker"), a.get("context"), a.get("href"), a.get("position"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)

        # When definitions were extracted from soup-level paragraphs (split is None),
        # filter out anchors whose context text RIGHT AFTER the marker matches a
        # definition body start — this means the anchor is actually a definition header
        # (e.g. the "*" at the start of "* Royal Flying Corps ..."), not a prose marker.
        if _def_starts:
            filtered: List[Dict[str, Any]] = []
            for a in deduped:
                ctx = (a.get("context") or "").strip()
                mk_raw = (a.get("marker_raw") or "").strip()
                if not mk_raw:
                    filtered.append(a)
                    continue
                # Check every occurrence of the marker in the context.
                # If any occurrence's following text looks like a definition
                # body, this anchor is a definition header — skip it.
                is_def_header = False
                mk_len = len(mk_raw)
                pos = 0
                while True:
                    pos = ctx.find(mk_raw, pos)
                    if pos < 0:
                        break
                    after = ctx[pos + mk_len:].strip()[:40].lower()
                    if after and any(after.startswith(ds) for ds in _def_starts if len(ds) >= 15):
                        is_def_header = True
                        break
                    pos += max(1, mk_len)
                if is_def_header:
                    continue  # Skip — this is a definition header
                filtered.append(a)
                deduped = filtered

        # Scale both to a common line-index space.  Definitions get
        # _p_counter * ratio, anchors get bisect_line / ratio.
        if split is None:
            _all_ps3 = list(soup.find_all("p"))
            _ratio = max(1.0, len(lines) / max(1.0, len(_all_ps3)))
            for d in definitions:
                pidx = d.get("line_index")
                if isinstance(pidx, (int, float)) and pidx > 0:
                    d["line_index"] = int(pidx * _ratio)
            _offs: List[int] = []
            _o = 0
            for _ln in lines:
                _offs.append(_o)
                _o += len(_ln) + 1
            import bisect as _bi
            for a in deduped:
                ap = a.get("position")
                if isinstance(ap, (int, float)) and ap >= 0 and _offs:
                    li = _bi.bisect_right(_offs, int(ap)) - 1
                    if li >= 0:
                        a["line_index"] = int(li / _ratio)

        if debug_markers:
            try:
                chap_lab = _safe_text(source_meta.get("chapter_label") or "")
                chap_name = _safe_text(source_meta.get("chapter_name") or "")
                a_counts: Dict[str, int] = defaultdict(int)
                for a in deduped:
                    mk = _safe_text(a.get("marker") or "")
                    if mk:
                        a_counts[mk] += 1
                d_counts: Dict[str, int] = defaultdict(int)
                for d in definitions:
                    mk = _safe_text(d.get("marker") or "")
                    if mk:
                        d_counts[mk] += 1

                for mk in sorted(debug_markers):
                    if mk in a_counts or mk in d_counts:
                        _stderr_log(
                            f"[DBG] chap_index={chapter_index} chap='{chap_lab}' file='{chap_name}' marker={mk} anchors={a_counts.get(mk,0)} defs={d_counts.get(mk,0)}"
                        )

                        if _debug_markers_verbose_enabled() and a_counts.get(mk, 0) > 0:
                            shown = 0
                            for a in deduped:
                                if _safe_text(a.get("marker") or "") != mk:
                                    continue
                                pos = a.get("position")
                                raw = _safe_text(a.get("marker_raw") or "")
                                ctx = _safe_text(a.get("context") or "")
                                _stderr_log(f"[DBG]   anchor marker={mk} pos={pos} raw='{raw}' ctx='{ctx[:140]}'")
                                shown += 1
                                if shown >= 6:
                                    break
            except Exception:
                pass

        # If this spine item contains multiple chapter headings, tag each anchor with the
        # nearest preceding heading (or the previous spine's chapter for anchors before the
        # first heading). This is used during pairing to avoid cross-chapter collisions.
        headings = _find_chapter_headings_in_text(anchors_text)
        # If we have an inferred definitions region, drop headings that occur within
        # it. Notes/glossary definition lines can look like headings (e.g. "3. Andore: ...")
        # and corrupt chapter grouping.
        # Only filter when the definitions start in the latter half of the file,
        # preventing early false-positive defs_start from removing real chapter headings.
        if headings and mask is not None:
            try:
                excluded, line_starts = mask
                filtered_heads: List[Tuple[int, str]] = []
                for hpos, hlabel in headings:
                    if not isinstance(hpos, int) or hpos < 0:
                        continue
                    li = bisect.bisect_right(line_starts, hpos) - 1
                    if 0 <= li < len(excluded) and excluded[li]:
                        continue
                    filtered_heads.append((hpos, hlabel))
                headings = filtered_heads
            except Exception:
                pass

        # Definition-led spine items can carry a large local notes block before a later
        # strong chapter heading. When the initial split heuristics miss that early block,
        # add those numeric definitions back now that we have the final heading map.
        try:
            numeric_defs_after_split = sum(
                1 for d in definitions if re.fullmatch(r"\d{1,3}", _safe_text(d.get("marker") or ""))
            )
            if split is None and not chapter_label_local and numeric_defs_after_split < 12 and headings:
                strong_headings_late: List[Tuple[int, str]] = []
                for hpos, hlabel in headings:
                    clean_hlabel = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                    if (
                        _looks_like_structural_part_or_book_heading(clean_hlabel)
                        or re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", clean_hlabel, re.IGNORECASE)
                        or _extract_chapter_token(clean_hlabel) is not None
                    ):
                        strong_headings_late.append((hpos, hlabel))

                early_numeric_def_lines = []
                for i_probe, ln_probe in enumerate(lines[: min(len(lines), 400)]):
                    m_probe = _def_line_regex().match(ln_probe or "")
                    if not m_probe:
                        continue
                    mk_probe = _safe_text(m_probe.group(1) or "")
                    if re.fullmatch(r"\d{1,3}", mk_probe):
                        early_numeric_def_lines.append(i_probe)

                if strong_headings_late and len(early_numeric_def_lines) >= 5:
                    first_def_line = int(early_numeric_def_lines[0])
                    first_def_pos = sum(len(ln) + 1 for ln in lines[:first_def_line])
                    first_strong_heading_pos = int(strong_headings_late[0][0])
                    if first_strong_heading_pos > (first_def_pos + 4000):
                        supplemental_defs = _extract_definitions_from_lines(lines, first_def_line)
                        supplemental_defs = _filter_definitions_by_profile(supplemental_defs, allowed_categories)
                        if supplemental_defs:
                            seen_local: set[tuple[str, str, int]] = set()
                            merged_defs: List[Dict[str, Any]] = []
                            for d in list(definitions) + list(supplemental_defs):
                                mk = _safe_text(d.get("marker") or "")
                                txt = _safe_text(d.get("text") or "")
                                try:
                                    li = int(d.get("line_index")) if d.get("line_index") is not None else -1
                                except Exception:
                                    li = -1
                                key = (mk, txt, li)
                                if key in seen_local:
                                    continue
                                seen_local.add(key)
                                merged_defs.append(d)
                            definitions = merged_defs
        except Exception:
            pass

        # Endnotes-at-EOF files can contain editorial subsection headings inside the
        # notes tail (for example, linguistic note headings). If all parsed anchors sit
        # before the inferred notes split, those late headings must not participate in
        # chapter grouping for this spine item.
        if headings and split is not None and multi_notes_blocks is None and isinstance(defs_start, int):
            try:
                if 0 <= defs_start < len(lines):
                    if mask is not None:
                        _excluded2, line_starts2 = mask
                    else:
                        line_starts2 = []
                        run2 = 0
                        for ln in lines:
                            line_starts2.append(run2)
                            run2 += len(ln) + 1

                    anchors_after_defs = False
                    for a in deduped:
                        pos = a.get("position")
                        if not isinstance(pos, int) or pos < 0:
                            continue
                        li = bisect.bisect_right(line_starts2, pos) - 1
                        if li >= defs_start:
                            anchors_after_defs = True
                            break

                    if not anchors_after_defs:
                        filtered_heads2: List[Tuple[int, str]] = []
                        for hpos, hlabel in headings:
                            if not isinstance(hpos, int) or hpos < 0:
                                continue
                            li = bisect.bisect_right(line_starts2, hpos) - 1
                            if li >= defs_start:
                                continue
                            filtered_heads2.append((hpos, hlabel))
                        headings = filtered_heads2
            except Exception:
                pass

        # Book-series safeguard (critical editions): many spine items contain
        # section subheads or outline/TOC-like numbered lines inside notes bodies.
        # Those must NOT be treated as chapter boundaries.
        #
        # Strategy:
        #   - If we detect any roman-numeral chapter headings in this spine item,
        #     keep only roman-token headings (drop arabic numbered outline lines).
        #   - If we detect any tokened headings at all, drop tokenless headings.
        if headings:
            try:
                def _clean_heading_label(lab: str) -> str:
                    t = _safe_text(lab or "")
                    t = _strip_trailing_footnote_marker_from_heading(t) or t
                    return t.strip()

                cleaned: List[Tuple[int, str, str, Optional[int]]] = []
                for hpos, hlabel in headings:
                    t = _clean_heading_label(hlabel)
                    tok = _extract_chapter_token(t)
                    cleaned.append((int(hpos), hlabel, t, tok))

                has_tokened = any(tok is not None for _, _, _, tok in cleaned)

                roman_headings: List[Tuple[int, str]] = []
                tokened_headings: List[Tuple[int, str]] = []
                for hpos, raw_lab, t, tok in cleaned:
                    if tok is None:
                        continue
                    tokened_headings.append((hpos, raw_lab))
                    # Roman chapter heading forms like:
                    #   I. THE ...
                    #   IX. ...
                    #   CHAPTER IV
                    if re.match(r"^\s*[IVXLC]{1,12}\s*\.?\s*\b", t):
                        roman_headings.append((hpos, raw_lab))
                    elif re.match(r"^\s*CHAPTER\s+[IVXLC]{1,12}\b", t, re.IGNORECASE):
                        roman_headings.append((hpos, raw_lab))

                if roman_headings:
                    # Keep all headings from the first one up through the
                    # last Roman heading, preserving the chapter structure.
                    first_roman_hpos = roman_headings[0][0]
                    last_roman_hpos = roman_headings[-1][0]
                    result_heads: List[Tuple[int, str]] = [
                        (hpos, hlabel) for hpos, hlabel in headings
                        if isinstance(hpos, int) and hpos <= last_roman_hpos
                    ]
                    if result_heads:
                        headings = result_heads
                elif has_tokened:
                    headings = tokened_headings
            except Exception:
                pass

        # Hard guard: do NOT allow weak/inferred headings to split a sequential
        # run of numeric footnote markers (1,2,3,4,5,...). Title-page-ish text can
        # appear inside long notes, and we never want it to "interject" a chapter.
        if headings:
            try:
                def _is_hard_chapter_boundary_label(lab: str) -> bool:
                    t = _safe_text(lab or "")
                    if not t:
                        return False
                    t = _strip_trailing_footnote_marker_from_heading(t) or t
                    if re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})(?:\b|\s*[:\.]\s*.*)$", t, re.IGNORECASE):
                        return True
                    # Standalone chapter numerals.
                    if re.match(r"^\s*[IVXLC]{1,12}\.\?\s*$", t) or re.match(r"^\s*\d{1,3}\.\?\s*$", t):
                        return True
                    # Common chapter heading forms: "III. THE LAND OF SHADOW" or "3. ...".
                    if re.match(r"^\s*[IVXLC]{1,12}\s*[\.)]\s+\S", t):
                        return True
                    if re.match(r"^\s*\d{1,3}\s*[\.)]\s+\S", t):
                        return True
                    return False

                numeric_anchors: List[Tuple[int, int]] = []
                for a in deduped:
                    pos = a.get("position")
                    mk = (a.get("marker") or "").strip()
                    if not isinstance(pos, int) or pos < 0:
                        continue
                    if not re.fullmatch(r"\d{1,3}", mk):
                        continue
                    try:
                        numeric_anchors.append((pos, int(mk)))
                    except Exception:
                        continue
                numeric_anchors.sort(key=lambda x: x[0])
                na_pos = [p for p, _ in numeric_anchors]

                if len(numeric_anchors) >= 2:
                    filtered_heads2: List[Tuple[int, str]] = []
                    for hpos, hlabel in headings:
                        # Always keep hard chapter boundaries.
                        if _is_hard_chapter_boundary_label(hlabel):
                            filtered_heads2.append((hpos, hlabel))
                            continue

                        j = bisect.bisect_left(na_pos, hpos) - 1
                        if j < 0 or (j + 1) >= len(numeric_anchors):
                            filtered_heads2.append((hpos, hlabel))
                            continue
                        (p0, v0) = numeric_anchors[j]
                        (p1, v1) = numeric_anchors[j + 1]

                        # Heading is between two successive numeric anchors.
                        # If numbering is continuing (non-decreasing), it's a numeric
                        # progression run; do NOT allow weak/inferred headings to split it.
                        # (Gaps are allowed: 1,3,7,9,11...)
                        if p0 <= hpos <= p1:
                            if (v1 >= v0) and v0 <= 300 and v1 <= 300:
                                continue

                        filtered_heads2.append((hpos, hlabel))
                    headings = filtered_heads2
            except Exception:
                pass

        if debug_markers and _debug_markers_verbose_enabled():
            try:
                hit = False
                for d in definitions:
                    mk = _safe_text(d.get("marker") or "")
                    if mk and mk in debug_markers:
                        hit = True
                        break
                if hit:
                    chap_name = _safe_text(source_meta.get("chapter_name") or "")
                    shown = [(_safe_text(lab), int(pos)) for pos, lab in headings[:12]]
                    _stderr_log(f"[DBG] chap_index={chapter_index} file='{chap_name}' headings_sample={shown}")
            except Exception:
                pass

        # Tag local definitions to their nearest preceding heading early so
        # global marker-based imports can be scoped correctly per chapter-group.
        try:
            if headings and definitions:
                if mask is not None:
                    _excluded, line_starts = mask
                else:
                    line_starts = []
                    run = 0
                    for ln in lines:
                        line_starts.append(run)
                        run += len(ln) + 1

                # Tag local definitions with nearest preceding heading.
                for d in definitions:
                    if d.get("chapter_group") is not None:
                        continue
                    li = d.get("line_index")
                    if not isinstance(li, int) or li < 0 or li >= len(line_starts):
                        continue
                    dpos = int(line_starts[li])
                    chosen = None
                    for hpos, hlabel in headings:
                        if hpos <= dpos:
                            chosen = hlabel
                        else:
                            break
                    if chosen:
                        chosen = _strip_trailing_footnote_marker_from_heading(chosen) or chosen
                        d["chapter_group"] = _chapter_group_key(chosen)
        except Exception:
            pass

        # Drop numeric-only anchors that occur on the same line as a detected chapter heading.
        # In some critical editions, headings contain parenthetical numbers like "(1)" that
        # are NOT footnote markers, and allowing them creates bogus repeated-marker anchors.
        if headings:
            try:
                if mask is not None:
                    _, line_starts = mask
                else:
                    line_starts = []
                    run = 0
                    for ln in lines:
                        line_starts.append(run)
                        run += len(ln) + 1

                heading_line_idxs: set[int] = set()
                for hpos, _hlab in headings:
                    if not isinstance(hpos, int) or hpos < 0:
                        continue
                    li = bisect.bisect_right(line_starts, hpos) - 1
                    if li >= 0:
                        heading_line_idxs.add(li)

                filtered_deduped: List[Dict[str, Any]] = []
                for a in deduped:
                    if a.get("href"):
                        filtered_deduped.append(a)
                        continue
                    pos = a.get("position")
                    if not isinstance(pos, int) or pos < 0:
                        filtered_deduped.append(a)
                        continue
                    li = bisect.bisect_right(line_starts, pos) - 1
                    if li in heading_line_idxs:
                        mk = (a.get("marker") or "").strip()
                        raw = (a.get("marker_raw") or "").strip()
                        if re.fullmatch(r"\d{1,3}", mk) and (
                            (raw.startswith("(") and raw.endswith(")")) or (raw.startswith("[") and raw.endswith("]"))
                        ):
                            # Exception: strong critical-edition boundaries like
                            #   [PART TWO].(1)
                            # can carry real footnote anchors on the heading line.
                            try:
                                lt = _safe_text(lines[li] or "")
                                lt = _strip_trailing_footnote_marker_from_heading(lt) or lt
                                if re.match(r"^\s*\[?\s*PART\s+([A-Z]+|\d{1,3}|[IVXLC]{1,12})\b", lt, re.IGNORECASE):
                                    filtered_deduped.append(a)
                                    continue
                                if re.match(r"^\s*NIGHT\s+\d{1,3}\s*\.", lt, re.IGNORECASE):
                                    filtered_deduped.append(a)
                                    continue
                            except Exception:
                                pass
                            continue
                    filtered_deduped.append(a)
                deduped = filtered_deduped
            except Exception:
                pass
        if headings:
            first_heading_pos = headings[0][0]
            first_heading_label = headings[0][1]

            # Detect chapter numbering restart after the first heading.
            # If the post-heading region starts with small numeric markers (e.g., 2, 3),
            # then a stray small marker before the heading (most commonly 1) is likely
            # part of the same chapter intro rather than the previous chapter.
            min_numeric_after_heading: Optional[int] = None
            try:
                for a in deduped:
                    pos = a.get("position")
                    if not isinstance(pos, int) or pos < first_heading_pos:
                        continue
                    mk = (a.get("marker") or "").strip()
                    if not re.fullmatch(r"\d{1,3}", mk):
                        continue
                    v = int(mk)
                    min_numeric_after_heading = v if min_numeric_after_heading is None else min(min_numeric_after_heading, v)
            except Exception:
                min_numeric_after_heading = None

            for a in deduped:
                pos = a.get("position")
                if not isinstance(pos, int) or pos < 0:
                    continue
                chosen = None
                if pos < first_heading_pos and prev_chapter_label:
                    chosen = prev_chapter_label
                    # Override for likely restart markers.
                    if min_numeric_after_heading is not None and min_numeric_after_heading <= 3:
                        mk = (a.get("marker") or "").strip()
                        if re.fullmatch(r"\d{1,3}", mk):
                            try:
                                if int(mk) <= 3 and first_heading_label:
                                    chosen = first_heading_label
                            except Exception:
                                pass
                else:
                    for hpos, hlabel in headings:
                        if hpos <= pos:
                            chosen = hlabel
                        else:
                            break
                if chosen:
                    chosen = _strip_trailing_footnote_marker_from_heading(chosen) or chosen
                    a["_chapter_label"] = chosen
                    a["_chapter_group"] = _chapter_group_key(chosen)
        else:
            # Single-heading (or no-heading) spine items: use the spine's inferred group.
            if source_meta.get("chapter_label"):
                for a in deduped:
                    a["_chapter_label"] = source_meta.get("chapter_label")
                    a["_chapter_group"] = source_meta.get("chapter_group")

        # If this spine item contains a detected definitions region and then later
        # starts a new chapter heading, suppress non-href anchors that occur in the
        # interstitial prose between the *end* of definitions and that next heading.
        # This is a common critical-edition artifact where editorial/outline text
        # appears after a notes block (but before the next chapter), and can contain
        # parenthetical numbers that are not footnotes.
        try:
            if headings and len(headings) >= 2 and mask is not None and deduped:
                excluded, line_starts = mask
                # Find the last excluded line (end of definitions span).
                last_excl = None
                for i in range(len(excluded) - 1, -1, -1):
                    if excluded[i]:
                        last_excl = i
                        break
                if last_excl is not None:
                    end_line = min(len(line_starts) - 1, int(last_excl) + 1)
                    defs_end_pos = int(line_starts[end_line]) if end_line >= 0 else 0

                    next_heading_pos = None
                    for hpos, _hlabel in headings:
                        if isinstance(hpos, int) and hpos > defs_end_pos:
                            next_heading_pos = int(hpos)
                            break

                    # Only apply when there is meaningful text between the end of
                    # definitions and the next heading.
                    if next_heading_pos is not None and (next_heading_pos - defs_end_pos) >= 40:
                        filtered_deduped: List[Dict[str, Any]] = []
                        for a in deduped:
                            if a.get("href"):
                                filtered_deduped.append(a)
                                continue
                            pos = a.get("position")
                            if not isinstance(pos, int) or pos < 0:
                                filtered_deduped.append(a)
                                continue
                            if defs_end_pos <= pos < next_heading_pos:
                                continue
                            filtered_deduped.append(a)
                        deduped = filtered_deduped
        except Exception:
            pass

        # Targeted rescue (per chapter region): if a numeric marker exists in definitions
        # within a given chapter region but there are zero anchors for that marker in
        # the same region, try to add back the first suitable "(n)" occurrence from
        # the running text within that region.
        #
        # This is specifically designed to keep the broad anchor plausibility
        # heuristics (which help avoid junk numbering) while preventing cases where
        # genuine footnote anchors are rejected by context rules (common in critical
        # editions with lots of parenthetical numbers).
        try:
            if anchors_text and definitions and headings:
                # Build line_starts for mapping definition line_index -> char offset.
                if mask is not None:
                    excluded, line_starts = mask
                else:
                    excluded = None
                    line_starts = []
                    run = 0
                    for ln in lines:
                        line_starts.append(run)
                        run += len(ln) + 1

                # Build chapter regions by heading positions.
                regions = []  # (start, end, label, group)
                for i, (hpos, hlabel) in enumerate(headings):
                    if not isinstance(hpos, int) or hpos < 0:
                        continue
                    start = int(hpos)
                    end = int(headings[i + 1][0]) if (i + 1) < len(headings) else len(anchors_text)
                    lab = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                    grp = _chapter_group_key(lab)
                    regions.append((start, end, lab, grp))
                if regions:
                    region_starts = [r[0] for r in regions]

                    # Markers needed per region from definitions.
                    needed_by_group: Dict[str, set[str]] = {}
                    fallback_grp: Optional[str] = None
                    try:
                        fallback_grp = str(regions[-1][3]) if regions else None
                    except Exception:
                        fallback_grp = None
                    for d in definitions:
                        mk = str(d.get("marker") or "").strip()
                        if not re.fullmatch(r"\d{1,3}", mk):
                            continue
                        li = d.get("line_index")
                        if not (isinstance(li, int) and 0 <= li < len(line_starts)):
                            # Spine-bridge imported definitions use a synthetic line_index that
                            # is intentionally out of range. Treat them as belonging to the last
                            # chapter region in this spine item (continuation notes).
                            if fallback_grp is not None:
                                needed_by_group.setdefault(fallback_grp, set()).add(mk)
                            continue
                        dpos = int(line_starts[li])
                        # Assign definition to nearest preceding heading region.
                        j = bisect.bisect_right(region_starts, dpos) - 1
                        if j < 0:
                            continue
                        (_rs, _re, _lab, grp) = regions[j]
                        needed_by_group.setdefault(str(grp), set()).add(mk)

                    for mk, entries in global_defs_by_marker.items():
                        mk_s = str(mk or "").strip()
                        if not re.fullmatch(r"\d{1,3}", mk_s):
                            continue
                        for entry in entries:
                            parent_grp = entry.get("parent_chapter_group")
                            if parent_grp is None:
                                continue
                            oi = entry.get("origin_index")
                            if not (isinstance(oi, int) and abs(int(oi) - int(chapter_index)) <= 8):
                                continue
                            needed_by_group.setdefault(str(parent_grp), set()).add(mk_s)

                    # Present anchors per group.
                    present_by_group: Dict[str, set[str]] = {}
                    for a in deduped:
                        grp = str(a.get("_chapter_group") or "")
                        mk = str(a.get("marker") or "").strip()
                        if not grp or not re.fullmatch(r"\d{1,3}", mk):
                            continue
                        present_by_group.setdefault(grp, set()).add(mk)

                    # Rescue within each region.
                    for (rs, re_end, lab, grp) in regions:
                        grp_s = str(grp)
                        needed = needed_by_group.get(grp_s) or set()
                        if not needed:
                            continue
                        present = present_by_group.get(grp_s) or set()
                        missing = sorted([m for m in needed if m not in present], key=lambda s: int(s))
                        if not missing:
                            continue

                        region_text = anchors_text[rs:re_end]
                        for mk in missing:
                            pos = None

                            # First try: ProperNoun (n) (keeps prior behavior for many books).
                            rx = re.compile(rf"\b[A-Z][A-Za-z'’\-]{{1,28}}\s*\(\s*{re.escape(mk)}\s*\)")
                            pos = None
                            for m in rx.finditer(region_text):
                                s = m.group(0)
                                rel = s.find("(")
                                if rel < 0:
                                    continue
                                cand = rs + m.start() + rel
                                if excluded is not None:
                                    li2 = bisect.bisect_right(line_starts, cand) - 1
                                    if 0 <= li2 < len(excluded) and excluded[li2]:
                                        continue
                                pos = cand
                                break
                            if pos is None:
                                # Fallback: any "(n)" occurrence in the region.
                                rx2 = re.compile(rf"\(\s*{re.escape(mk)}\s*\)")
                                for m2 in rx2.finditer(region_text):
                                    cand = rs + m2.start()
                                    # Avoid cases like "10(12)" (no boundary).
                                    prev_ch = region_text[m2.start() - 1] if (m2.start() - 1) >= 0 else ""
                                    if prev_ch and prev_ch.isdigit():
                                        continue
                                    if excluded is not None:
                                        li2 = bisect.bisect_right(line_starts, cand) - 1
                                        if 0 <= li2 < len(excluded) and excluded[li2]:
                                            continue
                                    pos = cand
                                    break

                            if pos is None:
                                continue

                            start = max(0, pos - 80)
                            end = min(len(anchors_text), pos + 80)
                            ctx = _safe_text(anchors_text[start:end])
                            deduped.append(
                                {
                                    "marker_raw": f"({mk})",
                                    "marker": mk,
                                    "position": int(pos),
                                    "context": ctx,
                                    "_has_href": False,
                                    "_chapter_label": lab,
                                    "_chapter_group": grp_s,
                                }
                            )
                            present_by_group.setdefault(grp_s, set()).add(mk)
        except Exception:
            pass

        # If this chapter has anchors but lacks local definitions for a marker,
        # pull in global marker-based definitions (notes pages without ids).
        #
        # IMPORTANT: when a spine item contains multiple chapter headings, scope
        # imports per-anchor chapter_group to avoid importing definitions for the
        # wrong chapter and pairing them by accident.
        if global_defs_by_marker and deduped:
            anchors_by_group: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
            for a in deduped:
                g = a.get("_chapter_group")
                anchors_by_group[str(g) if g is not None else None].append(a)

            for gkey, group_anchors in anchors_by_group.items():
                # Determine the current chapter label/token for this anchor group.
                group_label = None
                for a in group_anchors:
                    lab = a.get("_chapter_label")
                    if lab:
                        group_label = lab
                        break
                cur_token = _extract_chapter_token(group_label) if group_label else _extract_chapter_token(source_meta.get("chapter_label"))
                cur_group = gkey if gkey is not None else source_meta.get("chapter_group")

                # Local markers for this group.
                local_markers = {
                    d.get("marker")
                    for d in definitions
                    if d.get("marker")
                    and (
                        (cur_group is None)
                        or (d.get("chapter_group") is None)
                        or (str(d.get("chapter_group")) == str(cur_group))
                    )
                }

                anchor_markers = {
                    a.get("marker")
                    for a in group_anchors
                    if a.get("marker")
                    and anchor_is_probable_footnote(
                        a.get("marker_raw"),
                        a.get("marker"),
                        a.get("context") or "",
                        has_href=bool(a.get("href")),
                    )
                }
                missing = [m for m in anchor_markers if m and m not in local_markers]
                if not missing:
                    continue

                try:
                    if not chapter_label_local and split is not None:
                        local_numeric = sorted(
                            {
                                int(str(m))
                                for m in local_markers
                                if re.fullmatch(r"\d{1,3}", str(m or ""))
                            }
                        )
                        missing_numeric = sorted(
                            {
                                int(str(m))
                                for m in missing
                                if re.fullmatch(r"\d{1,3}", str(m or ""))
                            }
                        )
                        if local_numeric and missing_numeric and max(missing_numeric) < min(local_numeric):
                            continue
                except Exception:
                    pass

                for mk in missing:
                    eligible: List[Dict[str, Any]] = []
                    for entry in global_defs_by_marker.get(mk, []):
                        txt = entry.get("text")
                        if not txt:
                            continue

                        # Prefer chapter_group containment; if we know the current group,
                        # do not import definitions from a different group.
                        ent_group = entry.get("chapter_group")
                        parent_group = entry.get("parent_chapter_group")
                        same_parent_group = (
                            cur_group is not None
                            and parent_group is not None
                            and str(parent_group) == str(cur_group)
                        )
                        if cur_group is not None:
                            if ent_group is not None and str(ent_group) != str(cur_group) and not same_parent_group:
                                continue
                            # If the entry isn't tagged to a group (ambiguous), only accept it
                            # when it's very close in spine order; otherwise it causes bleed.
                            if ent_group is None:
                                # If this spine item contains multiple headings, never import
                                # untagged definitions into a specific chapter-group. This is
                                # the classic bleed case: Chapter I notes get reused for Chapter II
                                # anchors within the same spine item.
                                if headings and len(headings) >= 2:
                                    continue
                                oi = entry.get("origin_index")
                                if not (isinstance(oi, int) and abs(int(oi) - int(chapter_index)) <= 1):
                                    continue

                        # If notes file says which chapter it belongs to, respect it.
                        ent_token = entry.get("chapter_token")
                        if cur_token is not None and ent_token is not None and int(ent_token) != int(cur_token):
                            continue

                        # If no token info, only import notes from nearby spine items.
                        # Apply this regardless of whether we have a current chapter token;
                        # otherwise tokenless sections (e.g., FOREWORD/INTRO) import junk.
                        if ent_token is None:
                            oi = entry.get("origin_index")
                            max_dist = 2
                            # If the harvested definition is explicitly tagged to the same
                            # chapter_group, allow a wider window to handle chapters that
                            # span multiple spine items.
                            if (
                                cur_group is not None
                                and (
                                    (ent_group is not None and str(ent_group) == str(cur_group))
                                    or same_parent_group
                                )
                            ):
                                max_dist = 8
                            if not (isinstance(oi, int) and abs(int(oi) - int(chapter_index)) <= max_dist):
                                continue

                        # If we have *neither* token nor group, be extra conservative.
                        if cur_group is None and cur_token is None:
                            oi = entry.get("origin_index")
                            if not (isinstance(oi, int) and abs(int(oi) - int(chapter_index)) <= 1):
                                continue

                        eligible.append(entry)

                    # If multiple eligible global definitions exist for the same marker,
                    # prefer the nearest by spine distance to avoid numeric ambiguity.
                    if len(eligible) > 1:
                        with_idx = [e for e in eligible if isinstance(e.get("origin_index"), int)]
                        if with_idx:
                            with_idx.sort(key=lambda e: abs(int(e.get("origin_index")) - int(chapter_index)))
                            best_dist = abs(int(with_idx[0].get("origin_index")) - int(chapter_index))
                            eligible = [
                                e
                                for e in eligible
                                if isinstance(e.get("origin_index"), int)
                                and abs(int(e.get("origin_index")) - int(chapter_index)) == best_dist
                            ]

                    for entry in eligible:
                        txt = entry.get("text")
                        if not txt:
                            continue
                        definitions.append(
                            {
                                "marker": mk,
                                "text": txt,
                                "line_index": -1,
                                "origin": entry.get("origin"),
                                "origin_index": entry.get("origin_index"),
                                "chapter_token": entry.get("chapter_token"),
                                # Tag imported defs to this anchor group to avoid bleed.
                                "chapter_group": (cur_group if cur_group is not None else entry.get("chapter_group")),
                            }
                        )

        # If a spine item contains multiple chapter headings, it can contain
        # editorial/prose sections with parenthetical numbers that are NOT
        # footnotes (e.g., outlines, citations, variant labels). If a chapter
        # region has *no* definitions available (local or imported), suppress
        # non-href anchors from that region to avoid generating bogus notes.
        try:
            if headings and len(headings) >= 2 and deduped and definitions:
                groups_with_defs = {
                    str(d.get("chapter_group"))
                    for d in definitions
                    if d.get("chapter_group") is not None and str(d.get("chapter_group")).strip()
                }
                if groups_with_defs:
                    filtered: List[Dict[str, Any]] = []
                    for a in deduped:
                        g = a.get("_chapter_group")
                        gk = str(g) if g is not None else None
                        if gk is None or gk in groups_with_defs:
                            filtered.append(a)
                            continue
                        # Keep explicit href anchors (true EPUB noterefs), but
                        # drop regex/bare-digit anchors when there are no defs.
                        if a.get("href"):
                            filtered.append(a)
                    deduped = filtered
        except Exception:
            pass

        # Remove internal-only keys from anchors before pairing output is generated.
        for a in deduped:
            a.pop("_has_href", None)
            # Keep _chapter_group/_chapter_label for pairing; they are not emitted in results.

        # If the same numeric markers restart across headings within one spine item,
        # pair anchors per chapter-group using only definitions from that same
        # chapter region (when available). This prevents cross-chapter collisions
        # when a spine item contains multiple chapters/sections.
        if headings and definitions:
            try:
                if mask is not None:
                    _excluded, line_starts = mask
                else:
                    line_starts = []
                    run = 0
                    for ln in lines:
                        line_starts.append(run)
                        run += len(ln) + 1

                # Tag local definitions with the nearest preceding strong heading.
                headings_for_defs = list(headings)
                try:
                    strong_headings_for_defs: List[Tuple[int, str]] = []
                    for hpos, hlabel in headings:
                        clean_hlabel = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                        if (
                            _looks_like_structural_part_or_book_heading(clean_hlabel)
                            or re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", clean_hlabel, re.IGNORECASE)
                            or _extract_chapter_token(clean_hlabel) is not None
                        ):
                            strong_headings_for_defs.append((hpos, hlabel))
                    if strong_headings_for_defs:
                        headings_for_defs = strong_headings_for_defs
                except Exception:
                    headings_for_defs = list(headings)

                first_heading_pos_for_defs = None
                prefer_current_group_for_pre_heading_defs = None
                try:
                    if headings_for_defs:
                        first_heading_pos_for_defs = int(headings_for_defs[0][0])
                        first_heading_label_for_defs = _strip_trailing_footnote_marker_from_heading(headings_for_defs[0][1]) or headings_for_defs[0][1]
                        prefer_current_group_for_pre_heading_defs = _chapter_group_key(first_heading_label_for_defs)
                    if (
                        first_heading_pos_for_defs is not None
                        and not chapter_label_local
                        and isinstance(defs_start, int)
                        and defs_start <= 120
                        and prefer_current_group_for_pre_heading_defs is not None
                    ):
                        prefer_current_group_for_pre_heading_defs = prefer_current_group_for_pre_heading_defs
                except Exception:
                    first_heading_pos_for_defs = None
                    prefer_current_group_for_pre_heading_defs = None

                for d in definitions:
                    if d.get("chapter_group") is not None:
                        continue
                    li = d.get("line_index")
                    if not isinstance(li, int) or li < 0 or li >= len(line_starts):
                        continue
                    dpos = int(line_starts[li])
                    chosen = None
                    for hpos, hlabel in headings_for_defs:
                        if hpos <= dpos:
                            chosen = hlabel
                        else:
                            break
                    if chosen:
                        chosen = _strip_trailing_footnote_marker_from_heading(chosen) or chosen
                        d["chapter_group"] = _chapter_group_key(chosen)
                    elif (
                        prefer_current_group_for_pre_heading_defs is not None
                        and first_heading_pos_for_defs is not None
                        and dpos < first_heading_pos_for_defs
                    ):
                        d["chapter_group"] = prefer_current_group_for_pre_heading_defs

                # Re-tag any spine-bridge spilled definitions to the same chapter_group as
                # the NOTES block they are continuing. Spillover tagging earlier in the
                # pipeline intentionally runs before all heading safeguards are applied,
                # so we normalize here using the final filtered `headings` list.
                try:
                    if spill_base is not None and spill_start_line is not None and headings:
                        if 0 <= int(spill_start_line) < len(line_starts):
                            spill_pos = int(line_starts[int(spill_start_line)])
                            spill_label = None
                            for hpos, hlabel in headings:
                                if isinstance(hpos, int) and hpos <= spill_pos:
                                    spill_label = hlabel
                                else:
                                    break
                            if not spill_label and headings:
                                spill_label = headings[-1][1]
                            if spill_label:
                                spill_label = _strip_trailing_footnote_marker_from_heading(spill_label) or spill_label
                                spill_group2 = _chapter_group_key(spill_label)
                                if spill_group2:
                                    for d in definitions:
                                        li = d.get("line_index")
                                        if isinstance(li, int) and li >= int(spill_base):
                                            d["chapter_group"] = spill_group2
                except Exception:
                    pass

                definitions_for_orphans = [dict(d) for d in definitions]

                anchors_by_group: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
                for a in deduped:
                    g = a.get("_chapter_group")
                    anchors_by_group[str(g) if g is not None else None].append(a)

                results = []
                for gkey, group_anchors in anchors_by_group.items():
                    defs_for_group = definitions
                    if gkey is not None:
                        tagged = [d for d in definitions if d.get("chapter_group") is not None]
                        if tagged:
                            group_match = [d for d in definitions if str(d.get("chapter_group")) == str(gkey)]
                            if group_match:
                                # When we have multiple headings, do not allow untagged
                                # definitions to bleed into a different chapter group.
                                if len(headings) >= 2:
                                    defs_for_group = group_match
                                else:
                                    defs_for_group = group_match + [d for d in definitions if d.get("chapter_group") is None]

                    # In a well-labeled multi-heading spine item, do not let a chapter
                    # see local definitions that occur before that chapter's own heading.
                    # This keeps earlier chapter note blocks from hijacking marker-order
                    # pairing for later chapters in the same spine item.
                    try:
                        if chapter_label_local and gkey is not None and len(headings) >= 2:
                            group_start_pos = None
                            for hpos, hlabel in headings:
                                clean_hlabel = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                                if _chapter_group_key(clean_hlabel) == str(gkey):
                                    group_start_pos = int(hpos)
                                    break
                            if group_start_pos is not None:
                                filtered_group_defs: List[Dict[str, Any]] = []
                                for d in defs_for_group:
                                    li = d.get("line_index")
                                    if not isinstance(li, int) or li < 0 or li >= len(line_starts):
                                        filtered_group_defs.append(d)
                                        continue
                                    dpos = int(line_starts[li])
                                    if dpos < group_start_pos:
                                        continue
                                    filtered_group_defs.append(d)
                                defs_for_group = filtered_group_defs
                    except Exception:
                        pass

                    try:
                        if chapter_label_local and split is not None and isinstance(defs_start, int):
                            def _defs_for_group_sort_key(d: Dict[str, Any]) -> Tuple[int, int]:
                                li = d.get("line_index")
                                if isinstance(li, int) and li >= 0:
                                    preferred_bucket = 0 if li >= defs_start else 1
                                    return (preferred_bucket, int(li))
                                return (2, 1_000_000_000)

                            defs_for_group = sorted(defs_for_group, key=_defs_for_group_sort_key)
                    except Exception:
                        pass

                    r, next_id = _pair_anchors_to_definitions(
                        group_anchors,
                        defs_for_group,
                        source_meta,
                        definitions_by_id=global_defs_by_id,
                        id_start=next_id,
                        forward_looking=(split is None),
                    )
                    results.extend(r)
            except Exception:
                results, next_id = _pair_anchors_to_definitions(
                    deduped,
                    definitions,
                    source_meta,
                    definitions_by_id=global_defs_by_id,
                    id_start=next_id,
                    forward_looking=(split is None),
                )
        else:
            definitions_for_orphans = [dict(d) for d in definitions]
            results, next_id = _pair_anchors_to_definitions(
                deduped,
                definitions,
                source_meta,
                definitions_by_id=global_defs_by_id,
                id_start=next_id,
                forward_looking=(split is None),
            )

        if 'definitions_for_orphans' not in locals():
            definitions_for_orphans = [dict(d) for d in definitions]

        # Fallback rescue: some endnotes-at-EOF spine items can confuse the chapter
        # grouping path and end up with zero results even though the local item has
        # both anchors and definitions. In that case, retry a simple local pairing
        # pass scoped to this spine item and its inferred chapter label.
        if (
            not results
            and definitions
            and anchors_text
            and source_meta.get("chapter_label")
            and split is not None
            and isinstance(defs_start, int)
            and defs_start < len(lines)
        ):
            try:
                skip_local_rescue = False
                try:
                    first_nonempty_idx = None
                    early_def_like = 0
                    def_re_rescue = _def_line_regex()
                    for i_probe, ln_probe in enumerate(lines[:80]):
                        t_probe = _safe_text(ln_probe or "")
                        if not t_probe:
                            continue
                        if first_nonempty_idx is None:
                            first_nonempty_idx = i_probe
                        if def_re_rescue.match(t_probe):
                            early_def_like += 1

                    first_heading_pos = None
                    if headings:
                        try:
                            first_heading_pos = int(headings[0][0])
                        except Exception:
                            first_heading_pos = None

                    if (
                        first_nonempty_idx is not None
                        and first_nonempty_idx <= 20
                        and early_def_like >= 3
                        and (
                            first_heading_pos is None
                            or first_heading_pos >= int(len(anchors_text) * 0.50)
                        )
                    ):
                        skip_local_rescue = True
                except Exception:
                    skip_local_rescue = False

                if skip_local_rescue:
                    raise RuntimeError("skip_local_rescue_for_definition_led_tail")

                rescue_anchors = _extract_anchors_from_text(anchors_text)
                rescue_mask = _build_definition_exclusion_mask(lines, defs_start)
                if rescue_mask is not None:
                    excluded_r, line_starts_r = rescue_mask
                    filtered_rescue: List[Dict[str, Any]] = []
                    for a in rescue_anchors:
                        pos = a.get("position")
                        if not isinstance(pos, int) or pos < 0:
                            filtered_rescue.append(a)
                            continue
                        li = bisect.bisect_right(line_starts_r, pos) - 1
                        if 0 <= li < len(excluded_r) and excluded_r[li]:
                            continue
                        filtered_rescue.append(a)
                    rescue_anchors = filtered_rescue

                rescue_anchors = _filter_anchors_by_profile(rescue_anchors, allowed_categories)

                filtered_rescue2: List[Dict[str, Any]] = []
                for a in rescue_anchors:
                    raw = _safe_text(a.get("marker_raw") or a.get("marker") or "")
                    norm = _safe_text(a.get("marker") or "")
                    ctx = _safe_text(a.get("context") or "")
                    if not norm:
                        continue
                    if not anchor_is_probable_footnote(raw, norm, ctx, has_href=bool(a.get("href"))):
                        continue
                    a["_chapter_label"] = source_meta.get("chapter_label")
                    a["_chapter_group"] = source_meta.get("chapter_group")
                    filtered_rescue2.append(a)

                rescue_seen: set[tuple[Any, Any, Any, Any]] = set()
                rescue_deduped: List[Dict[str, Any]] = []
                for a in filtered_rescue2:
                    key = (a.get("marker"), a.get("context"), a.get("href"), a.get("position"))
                    if key in rescue_seen:
                        continue
                    rescue_seen.add(key)
                    rescue_deduped.append(a)

                if rescue_deduped:
                    results, next_id = _pair_anchors_to_definitions(
                        rescue_deduped,
                        definitions,
                        source_meta,
                        definitions_by_id=global_defs_by_id,
                        id_start=next_id,
                    )
            except Exception:
                pass

        # If this spine item contains multiple chapter headings, assign each result
        # to the nearest preceding heading based on its anchor position.
        # For malformed EPUBs, a spine item can continue a chapter and then start
        # the next chapter later. Anchors before the first heading belong to the
        # previous chapter.
        # Re-run (or re-use) heading assignment for results.
        # This mutates the emitted items but does not affect pairing anymore.
        if headings:
            first_heading_pos = headings[0][0]
            first_heading_label = headings[0][1]

            min_numeric_after_heading: Optional[int] = None
            try:
                for r in results:
                    pos = r.get("position")
                    if not isinstance(pos, int) or pos < first_heading_pos:
                        continue
                    mk = (r.get("marker") or "").strip()
                    if not re.fullmatch(r"\d{1,3}", mk):
                        continue
                    v = int(mk)
                    min_numeric_after_heading = v if min_numeric_after_heading is None else min(min_numeric_after_heading, v)
            except Exception:
                min_numeric_after_heading = None

            for r in results:
                pos = r.get("position")
                if not isinstance(pos, int) or pos < 0:
                    continue
                chosen = None
                if pos < first_heading_pos and prev_chapter_label:
                    chosen = prev_chapter_label
                    # Override: when the first heading differs meaningfully from
                    # the previous chapter label AND early notes restart numbering,
                    # the rows before the heading belong to this chapter.
                    if min_numeric_after_heading is not None and min_numeric_after_heading <= 3:
                        mk = (r.get("marker") or "").strip()
                        if re.fullmatch(r"\d{1,3}", mk):
                            try:
                                if int(mk) <= 3 and first_heading_label:
                                    chosen = first_heading_label
                            except Exception:
                                pass
                    # Also override when the first heading is a clearly different
                    # chapter label from the inherited context.
                    elif first_heading_label and prev_chapter_label:
                        fhl = _safe_text(first_heading_label).strip().upper()
                        pcl = _safe_text(prev_chapter_label).strip().upper()
                        if fhl and pcl and fhl != pcl:
                            chosen = first_heading_label
                else:
                    for hpos, hlabel in headings:
                        if hpos <= pos:
                            chosen = hlabel
                        else:
                            break
                if chosen:
                    chosen = _strip_trailing_footnote_marker_from_heading(chosen) or chosen
                    # When the TOC is high-quality, only reassign chapter_label when
                    # this file has multiple TOC entries (use position-based lookup)
                    # or when it has no TOC label at all (use heading-based lookup).
                    # Single-TOC-entry files keep one label for the entire file.
                    if toc_boundaries and len(toc_boundaries) >= 2:
                        toc_pos = r.get("position")
                        chosen_toc = _label_by_toc_position(toc_boundaries, toc_pos)
                        if chosen_toc:
                            r["chapter_label"] = chosen_toc
                            r["chapter_group"] = _chapter_group_key(chosen_toc)
                    elif not (structured_footnote_epub and toc_is_high_quality) or toc_label is None:
                        r["chapter_label"] = chosen
                        r["chapter_group"] = _chapter_group_key(chosen)

            # Advance the carried chapter state.
            # - Multi-TOC-entry files: advance to the LAST TOC label so the next
            #   spine item inherits the correct chapter (e.g., Part Two, not Part One).
            # - Single-TOC-entry files: don't advance (label already set at file level).
            # - Non-TOC files with high-quality TOC: don't advance (inherited context).
            # - No TOC / low-quality TOC: advance via heading-based carry (existing behavior).
            if toc_boundaries and len(toc_boundaries) >= 2:
                last_toc_label = toc_boundaries[-1][0]
                if last_toc_label:
                    last_toc_label = _strip_trailing_footnote_marker_from_heading(last_toc_label) or last_toc_label
                    current_chapter_label = last_toc_label
                    current_chapter_group = _chapter_group_key(last_toc_label)
            elif not (structured_footnote_epub and toc_is_high_quality) or toc_label is None:
                carry_label = None
                try:
                    for _hpos, hlabel in headings:
                        cleaned = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                        if _extract_chapter_token(cleaned) is not None:
                            carry_label = cleaned
                    if not carry_label:
                        last_label = headings[-1][1]
                        carry_label = _strip_trailing_footnote_marker_from_heading(last_label) or last_label
                except Exception:
                    carry_label = None

                if carry_label:
                    current_chapter_label = carry_label
                    current_chapter_group = _chapter_group_key(carry_label)

        # Post-processing cleanup:
        #  - Prefer the explicit EPUB id_link when present (drops regex duplicates)
        #  - Surface conservative orphan definitions that sit in the middle of a sequence
        results = _dedupe_numeric_results_prefer_id_link(results)
        results, next_id = _add_orphan_numeric_definitions(results, definitions_for_orphans, source_meta, next_id)

        # Final recovery for inherited-label numeric-note sections: when we recovered a
        # unique local definition set for the item, promote unresolved rows that match
        # those markers, and upgrade orphan-only rows if a unique anchor occurrence exists.
        try:
            if not supplemental_inherited_defs and not chapter_label_local and split is None:
                early_numeric_def_lines = []
                for i_probe, ln_probe in enumerate(lines[: min(len(lines), 400)]):
                    m_probe = _def_line_regex().match(ln_probe or "")
                    if not m_probe:
                        continue
                    mk_probe = _safe_text(m_probe.group(1) or "")
                    if re.fullmatch(r"\d{1,3}", mk_probe):
                        early_numeric_def_lines.append(i_probe)
                if len(early_numeric_def_lines) >= 5:
                    strong_heads_for_recovery = []
                    for hpos, hlabel in _find_chapter_headings_in_text(anchors_text):
                        clean_hlabel = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                        if (
                            _looks_like_structural_part_or_book_heading(clean_hlabel)
                            or re.match(r"^\s*CHAPTER\s+([IVXLC]{1,12}|\d{1,3})\b", clean_hlabel, re.IGNORECASE)
                            or _extract_chapter_token(clean_hlabel) is not None
                        ):
                            strong_heads_for_recovery.append((hpos, hlabel))
                    if strong_heads_for_recovery:
                        first_recovery_label = _strip_trailing_footnote_marker_from_heading(strong_heads_for_recovery[0][1]) or strong_heads_for_recovery[0][1]
                        supplemental_inherited_label = first_recovery_label
                        supplemental_inherited_group = _chapter_group_key(first_recovery_label)
                        supplemental_defs = _extract_definitions_from_lines(lines, int(early_numeric_def_lines[0]))
                        supplemental_defs = _filter_definitions_by_profile(supplemental_defs, allowed_categories)
                        marker_counts: Dict[str, int] = defaultdict(int)
                        marker_text: Dict[str, str] = {}
                        for d in supplemental_defs:
                            mk = _safe_text(d.get("marker") or "")
                            txt = _safe_text(d.get("text") or "")
                            if not re.fullmatch(r"\d{1,3}", mk) or not txt:
                                continue
                            marker_counts[mk] += 1
                            marker_text.setdefault(mk, txt)
                        supplemental_inherited_defs = {
                            mk: marker_text[mk]
                            for mk, count in marker_counts.items()
                            if count == 1 and mk in marker_text
                        }

            if supplemental_inherited_defs:
                for r in results:
                    mk = _safe_text(r.get("marker") or "")
                    if mk not in supplemental_inherited_defs:
                        continue
                    if supplemental_inherited_group is not None and r.get("chapter_group") != supplemental_inherited_group:
                        continue
                    if r.get("match_method") == "none":
                        r["suggested_definition"] = supplemental_inherited_defs[mk]
                        r["confidence"] = "High (Marker Match)"
                        r["confidence_score"] = 0.9
                        r["match_method"] = "marker_unique"

                orphan_rows = {
                    _safe_text(r.get("marker") or ""): r
                    for r in results
                    if r.get("match_method") == "orphan_definition"
                    and (supplemental_inherited_group is None or r.get("chapter_group") == supplemental_inherited_group)
                }
                candidate_markers = [mk for mk in supplemental_inherited_defs if mk in orphan_rows]
                if candidate_markers:
                    candidate_anchors = _extract_anchors_from_text(anchors_text)
                    candidate_anchors.extend(_extract_bare_digit_anchors_from_text(anchors_text, candidate_markers))
                    candidate_hits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for a in candidate_anchors:
                        mk = _safe_text(a.get("marker") or "")
                        if mk not in orphan_rows:
                            continue
                        if not anchor_is_probable_footnote(a.get("marker_raw"), mk, a.get("context") or "", has_href=bool(a.get("href"))):
                            continue
                        candidate_hits[mk].append(a)
                    for mk, hits in candidate_hits.items():
                        if len(hits) != 1:
                            continue
                        hit = hits[0]
                        row = orphan_rows[mk]
                        row["marker_raw"] = hit.get("marker_raw") or row.get("marker_raw")
                        row["context"] = hit.get("context") or row.get("context")
                        row["href"] = hit.get("href")
                        row["position"] = hit.get("position")
                        row["suggested_definition"] = supplemental_inherited_defs[mk]
                        row["confidence"] = "High (Marker Match)"
                        row["confidence_score"] = 0.9
                        row["match_method"] = "marker_unique"
                        if supplemental_inherited_label is not None:
                            row["chapter_label"] = supplemental_inherited_label
                        if supplemental_inherited_group is not None:
                            row["chapter_group"] = supplemental_inherited_group
                        try:
                            base = 0
                            if row.get("source") == "epub" and isinstance(row.get("chapter_index"), int):
                                base = int(row.get("chapter_index"))
                            pos_i = int(row.get("position")) if isinstance(row.get("position"), int) and row.get("position") >= 0 else 999_999_999
                            row["order_key"] = base * 1_000_000_000 + min(pos_i, 999_999_999)
                        except Exception:
                            pass
        except Exception:
            pass

        # Upgrade certain orphan_definitions by finding a likely anchor occurrence.
        #
        # These orphans are created *above* by `_add_orphan_numeric_definitions`.
        # Keep broad ProperNoun(n) suppression (prevents junk), but when a single-digit
        # marker is only present as an orphan_definition, try to locate a plausible
        # in-prose anchor `ProperNoun (n)` in the same chapter region and populate
        # position/context/order_key.
        try:
            if results and anchors_text:
                local_heads = _find_chapter_headings_in_text(anchors_text)

                # Drop headings that occur inside the inferred definitions region.
                if local_heads and mask is not None:
                    try:
                        excluded, line_starts = mask
                        filtered_heads: List[Tuple[int, str]] = []
                        for hpos, hlabel in local_heads:
                            if not isinstance(hpos, int) or hpos < 0:
                                continue
                            li = bisect.bisect_right(line_starts, hpos) - 1
                            if 0 <= li < len(excluded) and excluded[li]:
                                continue
                            filtered_heads.append((hpos, hlabel))
                        local_heads = filtered_heads
                    except Exception:
                        pass

                # Keep only tokened/roman headings when present.
                if local_heads:
                    try:
                        def _clean_heading_label(lab: str) -> str:
                            t = _safe_text(lab or "")
                            t = _strip_trailing_footnote_marker_from_heading(t) or t
                            return t.strip()

                        cleaned: List[Tuple[int, str, str, Optional[int]]] = []
                        for hpos, hlabel in local_heads:
                            t = _clean_heading_label(hlabel)
                            tok = _extract_chapter_token(t)
                            cleaned.append((int(hpos), hlabel, t, tok))

                        has_tokened = any(tok is not None for _, _, _, tok in cleaned)
                        roman_headings: List[Tuple[int, str]] = []
                        tokened_headings: List[Tuple[int, str]] = []
                        for hpos, raw_lab, t, tok in cleaned:
                            if tok is None:
                                continue
                            tokened_headings.append((hpos, raw_lab))
                            if re.match(r"^\s*[IVXLC]{1,12}\s*\.?\s*\b", t) or re.match(
                                r"^\s*CHAPTER\s+[IVXLC]{1,12}\b", t, re.IGNORECASE
                            ):
                                roman_headings.append((hpos, raw_lab))
                        if roman_headings:
                            local_heads = roman_headings
                        elif has_tokened:
                            local_heads = tokened_headings
                    except Exception:
                        pass

                # Build regions from headings.
                regions: List[Tuple[str, int, int]] = []
                for i, (hpos, hlabel) in enumerate(local_heads):
                    if not isinstance(hpos, int) or hpos < 0:
                        continue
                    start = int(hpos)
                    end = int(local_heads[i + 1][0]) if (i + 1) < len(local_heads) else len(anchors_text)
                    lab = _strip_trailing_footnote_marker_from_heading(hlabel) or hlabel
                    regions.append((lab, start, end))
                region_by_label = {lab: (start, end) for (lab, start, end) in regions}

                # For robust recovery when chapter_label is wrong, keep an ordered list
                # of regions we can fall back to searching.
                ordered_regions: List[Tuple[str, int, int]] = list(regions)

                # Precompute mask for exclusion checks.
                if mask is not None:
                    excluded, line_starts = mask
                else:
                    excluded = None
                    line_starts = []
                    run = 0
                    for ln in lines:
                        line_starts.append(run)
                        run += len(ln) + 1

                dbg = _debug_markers_set()
                dbg_verbose = _debug_markers_verbose_enabled()

                # Seed per-region single-digit orphans when a definition exists but the
                # chapter region contains a mid-sequence gap (e.g., has 2 and 4 but not 3).
                # This avoids dependence on dominant-group inference in multi-heading spine
                # items and enables recovered_anchor for ProperNoun(n) cases like
                # "Celeborn (3)".
                try:
                    def_text_by_marker: Dict[str, str] = {}
                    for d in definitions:
                        mk_d = str(d.get("marker") or "").strip()
                        if not re.fullmatch(r"[1-9]", mk_d):
                            continue
                        txt = d.get("text")
                        if not txt:
                            continue
                        # Prefer local definition text when available.
                        li = d.get("line_index")
                        if mk_d not in def_text_by_marker:
                            def_text_by_marker[mk_d] = str(txt)
                        else:
                            if isinstance(li, int) and li >= 0:
                                def_text_by_marker[mk_d] = str(txt)

                    if def_text_by_marker:
                        present_by_group_num: Dict[str, set[int]] = defaultdict(set)
                        for rr in results:
                            grp_rr = str(rr.get("chapter_group") or "")
                            mk_rr = str(rr.get("marker") or "").strip()
                            if not grp_rr or not mk_rr.isdigit():
                                continue
                            try:
                                v = int(mk_rr)
                            except Exception:
                                continue
                            if 1 <= v <= 9:
                                present_by_group_num[grp_rr].add(v)

                        existing_keys = {
                            (str(rr.get("chapter_group") or ""), str(rr.get("marker") or "").strip())
                            for rr in results
                            if str(rr.get("chapter_group") or "") and str(rr.get("marker") or "").strip()
                        }

                        for (lab2, rs2, re2) in ordered_regions:
                            grp2 = _chapter_group_key(lab2)
                            present_nums2 = present_by_group_num.get(grp2) or set()
                            if len(present_nums2) < 2:
                                continue
                            for mk_s, txt in def_text_by_marker.items():
                                try:
                                    n = int(mk_s)
                                except Exception:
                                    continue
                                if n in present_nums2:
                                    continue
                                if not ((n - 1) in present_nums2 and (n + 1) in present_nums2):
                                    continue
                                if (grp2, mk_s) in existing_keys:
                                    continue

                                item: Dict[str, Any] = {
                                    "type": "footnote",
                                    "marker": mk_s,
                                    "marker_raw": mk_s,
                                    "context": "",
                                    "href": None,
                                    "position": 999_999_999,
                                    "suggested_definition": txt,
                                    "confidence": "Low (Orphan Definition)",
                                    "confidence_score": 0.40,
                                    "match_method": "orphan_definition",
                                    "id": next_id,
                                }
                                item.update(source_meta)
                                item["chapter_label"] = lab2
                                item["chapter_group"] = grp2
                                try:
                                    base = 0
                                    if item.get("source") == "epub" and isinstance(item.get("chapter_index"), int):
                                        base = int(item.get("chapter_index"))
                                    pos = item.get("position")
                                    pos_i = int(pos) if isinstance(pos, int) and pos >= 0 else 999_999_999
                                    item["order_key"] = base * 1_000_000_000 + min(pos_i, 999_999_999)
                                except Exception:
                                    pass

                                next_id += 1
                                results.append(item)
                                existing_keys.add((grp2, mk_s))
                                present_by_group_num.setdefault(grp2, set()).add(n)
                except Exception:
                    pass

                for r in results:
                    if r.get("match_method") != "orphan_definition":
                        continue
                    mk = str(r.get("marker") or "").strip()
                    if not re.fullmatch(r"\d{1,3}", mk):
                        continue

                    lab = _safe_text(r.get("chapter_label") or "")
                    if not lab:
                        continue
                    lab = _strip_trailing_footnote_marker_from_heading(lab) or lab
                    region = region_by_label.get(lab)
                    rs: int
                    re_end: int
                    region_text: str

                    rx = re.compile(rf"\b[A-Z][A-Za-z'’\-]{{1,28}}\s*\(\s*{re.escape(mk)}\s*\)")

                    def _find_pos_in_slice(slice_start: int, slice_text: str) -> Optional[int]:
                        for m in rx.finditer(slice_text):
                            rel = m.group(0).find("(")
                            if rel < 0:
                                continue
                            cand = slice_start + m.start() + rel
                            if excluded is not None:
                                li2 = bisect.bisect_right(line_starts, cand) - 1
                                if 0 <= li2 < len(excluded) and excluded[li2]:
                                    continue
                            return int(cand)

                        # Fallback: accept a unique bare "(n)" occurrence in the slice.
                        fallback_hits: List[int] = []
                        rx2 = re.compile(rf"\(\s*{re.escape(mk)}\s*\)")
                        for m2 in rx2.finditer(slice_text):
                            cand = slice_start + m2.start()
                            prev_ch = slice_text[m2.start() - 1] if (m2.start() - 1) >= 0 else ""
                            if prev_ch and prev_ch.isdigit():
                                continue
                            if excluded is not None:
                                li2 = bisect.bisect_right(line_starts, cand) - 1
                                if 0 <= li2 < len(excluded) and excluded[li2]:
                                    continue
                            fallback_hits.append(int(cand))
                            if len(fallback_hits) > 1:
                                break
                        if len(fallback_hits) == 1:
                            return fallback_hits[0]
                        return None

                    pos: Optional[int] = None
                    if region is not None:
                        rs, re_end = region
                        region_text = anchors_text[rs:re_end]
                        pos = _find_pos_in_slice(rs, region_text)

                    # Fallback: chapter_label can be wrong in multi-heading spine items.
                    # If we can't find a match in the labeled region, search all regions.
                    if pos is None:
                        for (_lab2, rs2, re2) in ordered_regions:
                            if region is not None and rs2 == rs and re2 == re_end:
                                continue
                            p2 = _find_pos_in_slice(rs2, anchors_text[rs2:re2])
                            if p2 is not None:
                                pos = p2
                                break

                    if pos is None:
                        if dbg_verbose and mk in dbg:
                            _stderr_log(
                                f"[DBG] orphan_recover mk={mk} chap_index={chapter_index} lab='{lab}' no_match"
                            )
                        continue

                    start = max(0, pos - 80)
                    end = min(len(anchors_text), pos + 80)
                    ctx = _safe_text(anchors_text[start:end])
                    if ctx:
                        r["context"] = ctx
                    r["position"] = int(pos)

                    # Re-assign chapter_label/group based on recovered position.
                    # When the TOC is high-quality:
                    #   - Multi-TOC-entry files: use position-based TOC lookup.
                    #   - Single-TOC-entry files: keep the one label (no reassignment).
                    #   - Non-TOC files: use heading-based lookup.
                    try:
                        if toc_boundaries and len(toc_boundaries) >= 2:
                            chosen_toc = _label_by_toc_position(toc_boundaries, pos)
                            if chosen_toc:
                                r["chapter_label"] = chosen_toc
                                r["chapter_group"] = _chapter_group_key(chosen_toc)
                        elif not (structured_footnote_epub and toc_is_high_quality) or toc_label is None:
                            chosen = None
                            for hpos, hlabel in local_heads:
                                if isinstance(hpos, int) and hpos <= pos:
                                    chosen = hlabel
                                else:
                                    break
                            if chosen:
                                chosen = _strip_trailing_footnote_marker_from_heading(chosen) or chosen
                                r["chapter_label"] = chosen
                                r["chapter_group"] = _chapter_group_key(chosen)
                    except Exception:
                        pass

                    # Recompute order_key.
                    try:
                        base = 0
                        if r.get("source") == "epub" and isinstance(r.get("chapter_index"), int):
                            base = int(r.get("chapter_index"))
                        pos_i = int(pos) if isinstance(pos, int) and pos >= 0 else 999_999_999
                        r["order_key"] = base * 1_000_000_000 + min(pos_i, 999_999_999)
                    except Exception:
                        pass

                    r["confidence"] = "Medium (Recovered Anchor)"
                    r["confidence_score"] = max(float(r.get("confidence_score") or 0.0), 0.65)
                    r["match_method"] = "recovered_anchor"

                    if dbg_verbose and mk in dbg:
                        _stderr_log(
                            f"[DBG] orphan_recover mk={mk} chap_index={chapter_index} lab='{lab}' recovered_pos={pos}"
                        )
        except Exception:
            pass

        # Final cleanup: if an orphan_definition exists for a marker that also has a
        # non-orphan entry in the same chapter_group, drop the orphan. This prevents
        # duplicated marker entries in the UI.
        try:
            if results:
                non_orphan_keys: set[Tuple[Optional[str], str]] = set()
                for r in results:
                    if r.get("match_method") == "orphan_definition":
                        continue
                    mk = str(r.get("marker") or "").strip()
                    if not mk:
                        continue
                    g = r.get("chapter_group")
                    gkey = str(g) if g is not None else None
                    non_orphan_keys.add((gkey, mk))

                if non_orphan_keys:
                    filtered: List[Dict[str, Any]] = []
                    for r in results:
                        if r.get("match_method") != "orphan_definition":
                            filtered.append(r)
                            continue
                        mk = str(r.get("marker") or "").strip()
                        g = r.get("chapter_group")
                        gkey = str(g) if g is not None else None
                        if (gkey, mk) in non_orphan_keys:
                            continue
                        filtered.append(r)
                    results = filtered
        except Exception:
            pass

        if (
            not results
            and definitions
            and anchors_text
            and source_meta.get("chapter_label")
            and split is not None
            and isinstance(defs_start, int)
            and defs_start < len(lines)
        ):
            try:
                skip_local_rescue = False
                try:
                    first_nonempty_idx = None
                    early_def_like = 0
                    def_re_rescue = _def_line_regex()
                    for i_probe, ln_probe in enumerate(lines[:80]):
                        t_probe = _safe_text(ln_probe or "")
                        if not t_probe:
                            continue
                        if first_nonempty_idx is None:
                            first_nonempty_idx = i_probe
                        if def_re_rescue.match(t_probe):
                            early_def_like += 1

                    first_heading_pos = None
                    if headings:
                        try:
                            first_heading_pos = int(headings[0][0])
                        except Exception:
                            first_heading_pos = None

                    if (
                        first_nonempty_idx is not None
                        and first_nonempty_idx <= 20
                        and early_def_like >= 3
                        and (
                            first_heading_pos is None
                            or first_heading_pos >= int(len(anchors_text) * 0.50)
                        )
                    ):
                        skip_local_rescue = True
                except Exception:
                    skip_local_rescue = False

                if skip_local_rescue:
                    raise RuntimeError("skip_local_rescue_for_definition_led_tail")

                rescue_anchors = _extract_anchors_from_text(anchors_text)
                rescue_mask = _build_definition_exclusion_mask(lines, defs_start)
                if rescue_mask is not None:
                    excluded_r, line_starts_r = rescue_mask
                    filtered_rescue: List[Dict[str, Any]] = []
                    for a in rescue_anchors:
                        pos = a.get("position")
                        if not isinstance(pos, int) or pos < 0:
                            filtered_rescue.append(a)
                            continue
                        li = bisect.bisect_right(line_starts_r, pos) - 1
                        if 0 <= li < len(excluded_r) and excluded_r[li]:
                            continue
                        filtered_rescue.append(a)
                    rescue_anchors = filtered_rescue

                rescue_anchors = _filter_anchors_by_profile(rescue_anchors, allowed_categories)

                filtered_rescue2: List[Dict[str, Any]] = []
                for a in rescue_anchors:
                    raw = _safe_text(a.get("marker_raw") or a.get("marker") or "")
                    norm = _safe_text(a.get("marker") or "")
                    ctx = _safe_text(a.get("context") or "")
                    if not norm:
                        continue
                    if not anchor_is_probable_footnote(raw, norm, ctx, has_href=bool(a.get("href"))):
                        continue
                    a["_chapter_label"] = source_meta.get("chapter_label")
                    a["_chapter_group"] = source_meta.get("chapter_group")
                    filtered_rescue2.append(a)

                rescue_seen: set[tuple[Any, Any, Any, Any]] = set()
                rescue_deduped: List[Dict[str, Any]] = []
                for a in filtered_rescue2:
                    key = (a.get("marker"), a.get("context"), a.get("href"), a.get("position"))
                    if key in rescue_seen:
                        continue
                    rescue_seen.add(key)
                    rescue_deduped.append(a)

                if rescue_deduped:
                    results, next_id = _pair_anchors_to_definitions(
                        rescue_deduped,
                        definitions,
                        source_meta,
                        definitions_by_id=global_defs_by_id,
                        id_start=next_id,
                    )
            except Exception:
                pass

        for r in results:
            r.setdefault("marker_profile", effective_profile)

        # When a spine item has multiple NOTES blocks, tag each result with
        # the section it belongs to by appending a numbered suffix.
        if structured_footnote_epub and multi_section_labels and multi_notes_blocks and len(multi_section_labels) >= 2:
            # Build char-position boundaries for each block.
            _sec_lbl_line_starts: List[int] = []
            _sec_lbl_run = 0
            for ln in lines:
                _sec_lbl_line_starts.append(_sec_lbl_run)
                _sec_lbl_run += len(ln) + 1
            block_boundaries: List[Tuple[int, int, str]] = []
            for idx, (hi, end) in enumerate(multi_notes_blocks):
                if idx < len(multi_section_labels):
                    bs = _sec_lbl_line_starts[min(int(hi), len(_sec_lbl_line_starts) - 1)]
                    be = _sec_lbl_line_starts[min(int(end), len(_sec_lbl_line_starts) - 1)]
                    block_boundaries.append((int(bs), int(be), multi_section_labels[idx]))

            for r in results:
                pos = r.get("position")
                if isinstance(pos, int) and pos >= 0:
                    matched = False
                    for bs, be, label in block_boundaries:
                        if bs <= pos < be:
                            cur = _safe_text(r.get("chapter_label") or "")
                            new_label = f"{cur} \u00b7 {label}"
                            r["chapter_label"] = new_label
                            r["chapter_group"] = _chapter_group_key(new_label)
                            matched = True
                            break
                    if not matched:
                        for idx in range(len(block_boundaries)):
                            bs, be, label = block_boundaries[idx]
                            if pos < bs:
                                cur = _safe_text(r.get("chapter_label") or "")
                                new_label = f"{cur} \u00b7 {label}"
                                r["chapter_label"] = new_label
                                r["chapter_group"] = _chapter_group_key(new_label)
                                matched = True
                                break
                            if idx + 1 < len(block_boundaries) and pos >= be and pos < block_boundaries[idx + 1][0]:
                                cur = _safe_text(r.get("chapter_label") or "")
                                new_label = f"{cur} \u00b7 {block_boundaries[idx + 1][2]}"
                                r["chapter_label"] = new_label
                                r["chapter_group"] = _chapter_group_key(new_label)
                                matched = True
                                break
                        if not matched and block_boundaries:
                            cur = _safe_text(r.get("chapter_label") or "")
                            new_label = f"{cur} \u00b7 {block_boundaries[-1][2]}"
                            r["chapter_label"] = new_label
                            r["chapter_group"] = _chapter_group_key(new_label)

        all_results.extend(results)

    try:
        chapter_rows_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in all_results:
            chapter_name = _safe_text(row.get("chapter_name") or "")
            if chapter_name:
                chapter_rows_by_name[chapter_name].append(row)

        structured_replacements: Dict[str, List[Dict[str, Any]]] = {}
        replacement_next_id = 0
        for row in all_results:
            try:
                rid = int(row.get("id")) if row.get("id") is not None else -1
            except Exception:
                rid = -1
            replacement_next_id = max(replacement_next_id, rid + 1)

        for chapter_index, item in enumerate(items):
            chapter_name = _safe_text(getattr(item, "get_name", lambda: None)() if hasattr(item, "get_name") else "")
            if not chapter_name:
                continue

            existing_rows = chapter_rows_by_name.get(chapter_name) or []
            if not existing_rows:
                continue
            html_content = item.get_content()
            html_text = html_content.decode("utf-8", errors="ignore") if isinstance(html_content, bytes) else str(html_content)
            soup = BeautifulSoup(html_text, "html.parser")
            chapter_text = _preprocess_for_notes(soup.get_text("\n"))
            lines = [_clean_line_for_parsing(l.rstrip("\r")) for l in chapter_text.split("\n")]

            id_map, marker_defs = _harvest_structured_notes_section_targets(soup)
            marker_defs = _filter_definitions_by_profile(marker_defs, allowed_categories)
            if not id_map or not marker_defs:
                continue

            soup_anchors = _extract_anchors_from_soup(soup)
            structured_ids = set(id_map.keys())
            soup_anchors = [
                a
                for a in soup_anchors
                if "#" in _safe_text(a.get("href") or "")
                and _safe_text(a.get("href") or "").split("#", 1)[1].strip() in structured_ids
            ]
            if not soup_anchors:
                continue

            _assign_positions_to_soup_anchors(soup_anchors, lines, soup)
            split_override = infer_notes_split(lines)
            mask_override = _build_definition_exclusion_mask(lines, split_override.defs_start_index) if split_override is not None else None
            if mask_override is not None:
                excluded, line_starts = mask_override
                filtered_soup: List[Dict[str, Any]] = []
                for a in soup_anchors:
                    pos = a.get("position")
                    if not isinstance(pos, int) or pos < 0:
                        filtered_soup.append(a)
                        continue
                    li = bisect.bisect_right(line_starts, pos) - 1
                    if 0 <= li < len(excluded) and excluded[li]:
                        continue
                    filtered_soup.append(a)
                soup_anchors = filtered_soup
            if not soup_anchors:
                continue

            template = existing_rows[0]
            template_chapter_label = template.get("chapter_label")
            template_chapter_group = template.get("chapter_group")

            local_heads = _find_chapter_headings_in_text(chapter_text)
            if local_heads and mask_override is not None:
                try:
                    excluded, line_starts = mask_override
                    filtered_heads: List[Tuple[int, str]] = []
                    for hpos, hlabel in local_heads:
                        if not isinstance(hpos, int) or hpos < 0:
                            continue
                        li = bisect.bisect_right(line_starts, hpos) - 1
                        if 0 <= li < len(excluded) and excluded[li]:
                            continue
                        filtered_heads.append((hpos, hlabel))
                    if filtered_heads:
                        local_heads = filtered_heads
                except Exception:
                    pass

            def _chapter_meta_for_rebuilt_pos(pos: Any) -> Tuple[Any, Any]:
                label = template_chapter_label
                group = template_chapter_group
                try:
                    if not isinstance(pos, int) or pos < 0 or not local_heads:
                        return label, group
                    chosen = None
                    for hpos, hlabel in local_heads:
                        if isinstance(hpos, int) and hpos <= pos:
                            chosen = hlabel
                        else:
                            break
                    if chosen:
                        chosen = _strip_trailing_footnote_marker_from_heading(chosen) or chosen
                        label = chosen
                        group = _chapter_group_key(chosen)
                except Exception:
                    pass
                return label, group

            pending_replacement_rows: List[Dict[str, Any]] = []
            seen_anchor_keys: set[Tuple[str, Any, Any]] = set()
            seen_frag_ids: set[str] = set()
            for a in soup_anchors:
                href = _safe_text(a.get("href") or "")
                frag = href.split("#", 1)[1].strip() if "#" in href else ""
                if not frag or frag not in id_map:
                    continue
                key = (frag, a.get("position"), a.get("context"))
                if key in seen_anchor_keys:
                    continue
                seen_anchor_keys.add(key)
                seen_frag_ids.add(frag)
                pos = a.get("position")
                pos_i = int(pos) if isinstance(pos, int) and pos >= 0 else 999_999_999
                chapter_label, chapter_group = _chapter_meta_for_rebuilt_pos(pos)
                pending_replacement_rows.append(
                    {
                        "type": "footnote",
                        "marker": a.get("marker"),
                        "marker_raw": a.get("marker_raw") or a.get("marker"),
                        "context": a.get("context") or "",
                        "href": a.get("href"),
                        "position": a.get("position"),
                        "suggested_definition": id_map.get(frag) or "",
                        "confidence": "High (ID Link)",
                        "confidence_score": 0.98,
                        "match_method": "id_link",
                        "id": replacement_next_id,
                        "source": "epub",
                        "chapter_index": chapter_index,
                        "chapter_name": chapter_name,
                        "chapter_label": chapter_label,
                        "chapter_group": chapter_group,
                        "order_key": int(chapter_index) * 1_000_000_000 + min(pos_i, 999_999_999),
                        "marker_profile": effective_profile,
                        "_series_prefix": frag.split("_", 1)[0] if "_" in frag else frag,
                    }
                )
                replacement_next_id += 1

            for d in marker_defs:
                def_id = _safe_text(d.get("id") or "")
                if not def_id or def_id in seen_frag_ids:
                    continue
                chapter_label, chapter_group = template_chapter_label, template_chapter_group
                pending_replacement_rows.append(
                    {
                        "type": "footnote",
                        "marker": d.get("marker"),
                        "marker_raw": d.get("marker"),
                        "context": "",
                        "href": None,
                        "position": None,
                        "suggested_definition": d.get("text") or "",
                        "confidence": "Low (Definition Only)",
                        "confidence_score": 0.40,
                        "match_method": "orphan_definition",
                        "id": replacement_next_id,
                        "source": "epub",
                        "chapter_index": chapter_index,
                        "chapter_name": chapter_name,
                        "chapter_label": chapter_label,
                        "chapter_group": chapter_group,
                        "order_key": int(chapter_index) * 1_000_000_000 + 999_999_999,
                        "marker_profile": effective_profile,
                        "_series_prefix": def_id.split("_", 1)[0] if "_" in def_id else def_id,
                    }
                )
                replacement_next_id += 1

            replacement_rows: List[Dict[str, Any]] = []
            if pending_replacement_rows:
                prefix_label_counts: Dict[str, Counter[Tuple[Any, Any]]] = defaultdict(Counter)
                for row in pending_replacement_rows:
                    prefix = _safe_text(row.get("_series_prefix") or "")
                    if not prefix:
                        continue
                    prefix_label_counts[prefix][(row.get("chapter_label"), row.get("chapter_group"))] += 1

                dominant_meta_by_prefix: Dict[str, Tuple[Any, Any]] = {}
                for prefix, counts in prefix_label_counts.items():
                    dominant_meta_by_prefix[prefix] = counts.most_common(1)[0][0]

                prefixes_by_meta: Dict[Tuple[Any, Any], List[str]] = defaultdict(list)
                for prefix, meta in dominant_meta_by_prefix.items():
                    prefixes_by_meta[meta].append(prefix)

                for row in pending_replacement_rows:
                    prefix = _safe_text(row.pop("_series_prefix", "") or "")
                    if prefix and prefix in dominant_meta_by_prefix:
                        label, group = dominant_meta_by_prefix[prefix]
                        colliding_prefixes = prefixes_by_meta.get((label, group), [])
                        if len(colliding_prefixes) >= 2:
                            label = f"{label} [{prefix}]" if label else prefix
                            group = _chapter_group_key(label)
                        row["chapter_label"] = label
                        row["chapter_group"] = group
                    replacement_rows.append(row)

            if replacement_rows:
                current_id_links = sum(1 for r in existing_rows if r.get("match_method") == "id_link")
                replacement_id_links = sum(1 for r in replacement_rows if r.get("match_method") == "id_link")
                has_orphan_defs = any(r.get("match_method") == "orphan_definition" for r in existing_rows)
                large_structured_rebuild = replacement_id_links >= 20 and replacement_id_links >= (current_id_links + 10)
                if replacement_id_links > current_id_links and (has_orphan_defs or large_structured_rebuild):
                    structured_replacements[chapter_name] = replacement_rows

        if structured_replacements:
            replaced_names: set[str] = set()
            rebuilt_results: List[Dict[str, Any]] = []
            for row in all_results:
                chapter_name = _safe_text(row.get("chapter_name") or "")
                if chapter_name in structured_replacements:
                    if chapter_name in replaced_names:
                        continue
                    rebuilt_results.extend(structured_replacements[chapter_name])
                    replaced_names.add(chapter_name)
                    continue
                rebuilt_results.append(row)
            all_results = rebuilt_results
    except Exception:
        pass

    # When a structured footnote convention is active, the EPUB's footnote structure
    # is explicit in the HTML DOM. Only id_link results (from explicit href/fragment
    # pairs) are real footnotes. Drop all heuristic matches to prevent AI disambiguation.
    if structured_footnote_epub:
        filtered_results: list[dict] = []
        for r in all_results:
            if not isinstance(r, dict):
                continue
            method = r.get("match_method")
            if method in ("id_link", "orphan_definition"):
                filtered_results.append(r)
        all_results = filtered_results

    _repair_structural_parent_note_swaps(all_results)
    _apply_marker_family_outlier_penalty(all_results)
    result_json = json.dumps(all_results, indent=2)

    # Restore the AI env var if we set it for this scan.
    if _ai_was_disabled:
        if _saved_ai_disabled is not None:
            os.environ["STARLISTENER_AI_DISABLED"] = _saved_ai_disabled
        else:
            os.environ.pop("STARLISTENER_AI_DISABLED", None)

    return result_json

# Main function to scan a PDF file for footnotes, using only text extraction and heuristics since PDFs lack structured markup. Results are generally lower confidence than EPUBs.
def scan_pdf_for_footnotes(pdf_path: str, *, options: Optional[ScanOptions] = None) -> str:
    options = options or ScanOptions()
    try:
        import fitz  # type: ignore  # PyMuPDF
    except Exception:
        return json.dumps(
            [
                {
                    "type": "error",
                    "marker": "!",
                    "context": "PDF support requires PyMuPDF. Install with: pip install pymupdf",
                    "id": 0,
                }
            ],
            indent=2,
        )

    doc = fitz.open(pdf_path)
    all_results: list[dict] = []
    next_id = 0
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text")
        source_meta = {
            "source": "pdf",
            "page_index": page_index,
        }
        results, next_id = _scan_text_blob_for_footnotes(text, source_meta, id_start=next_id, options=options)
        # For PDF boost confidence when anchor and def likely share a page (we only scan per-page).
        for r in results:
            if r.get("suggested_definition") and (r.get("confidence_score") or 0) >= 0.75:
                r["confidence_score"] = min(1.0, (r.get("confidence_score") or 0) + 0.05)
                if r["confidence"].startswith("High"):
                    r["confidence"] = "High (Same Page)"
        all_results.extend(results)

    _apply_marker_family_outlier_penalty(all_results)
    return json.dumps(all_results, indent=2)


def scan_file_for_footnotes(file_path: str, *, options: Optional[ScanOptions] = None) -> str:
    options = options or ScanOptions()
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".epub":
        return scan_epub_for_footnotes(str(path), options=options)
    if ext == ".pdf":
        return scan_pdf_for_footnotes(str(path), options=options)

    # Treat everything else as a text-like file.
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return json.dumps([{"type": "error", "marker": "!", "context": str(e), "id": 0}], indent=2)

    # If it's HTML-ish, strip tags.
    if ext in {".html", ".htm", ".xhtml"} or ("<html" in raw.lower() or "<body" in raw.lower()):
        soup = BeautifulSoup(raw, "html.parser")
        raw = soup.get_text("\n")

    results, _ = _scan_text_blob_for_footnotes(raw, {"source": "text", "file_name": path.name}, id_start=0, options=options)
    _apply_marker_family_outlier_penalty(results)
    return json.dumps(results, indent=2)

def scan_for_footnotes(epub_path, *, options: Optional[ScanOptions] = None):
    try:
        return scan_file_for_footnotes(epub_path, options=options)
    except Exception as e:
        return json.dumps([{"type": "error", "marker": "!", "context": str(e), "id": 0}], indent=2)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_arg = sys.argv[1]
        file_path = raw_arg
        options = ScanOptions()

        # Accept either a raw file path or a JSON payload like:
        #   {"path": "...", "options": {"marker_profile": "numeric", "allow_ai_pairing": true, "ai_debug": false}}
        if isinstance(raw_arg, str) and raw_arg.strip().startswith("{"):
            try:
                payload = json.loads(raw_arg)
                if isinstance(payload, dict):
                    file_path = payload.get("path") or payload.get("file_path") or file_path
                    opt = payload.get("options") or {}
                    if isinstance(opt, dict):
                        options = ScanOptions(
                            marker_profile=str(opt.get("marker_profile") or opt.get("markerProfile") or options.marker_profile),
                            allow_ai_pairing=bool(opt.get("allow_ai_pairing") if opt.get("allow_ai_pairing") is not None else opt.get("allowAIPairing") if opt.get("allowAIPairing") is not None else options.allow_ai_pairing),
                            ai_debug=bool(opt.get("ai_debug") if opt.get("ai_debug") is not None else opt.get("aiDebug") if opt.get("aiDebug") is not None else options.ai_debug),
                        )
            except Exception:
                file_path = raw_arg

        if not options.allow_ai_pairing:
            os.environ["STARLISTENER_AI_DISABLED"] = "1"
        if options.ai_debug:
            os.environ["STARLISTENER_AI_DEBUG"] = "1"

        # Use sys.stdout.write for clean pipe to Electron
        sys.stdout.write(scan_for_footnotes(str(file_path), options=options))
        sys.stdout.flush()
    else:
        print(json.dumps([{"error": "No file path provided"}]))
