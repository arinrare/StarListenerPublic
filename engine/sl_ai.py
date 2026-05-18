import os
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

import requests

try:
    from sl_debug import _stderr_log
    from sl_utility import _safe_text, _def_line_regex
except ModuleNotFoundError:  # pragma: no cover
    from .sl_debug import _stderr_log  # type: ignore
    from .sl_utility import _safe_text, _def_line_regex  # type: ignore


def _normalize_ai_endpoint(url: Optional[str]) -> str:
    """Normalize a user-provided AI endpoint into an OpenAI-compatible chat completions URL.

    Accepts either:
      - base API root like http://localhost:1234/v1
      - full endpoint like http://localhost:1234/v1/chat/completions
    """

    default_full = "http://localhost:8080/v1/chat/completions"
    u = (url or "").strip()
    u = u.strip('"').strip("'").strip()
    if not u:
        return default_full

    # Strip trailing slash for consistency.
    u = u.rstrip("/")

    # If already points at chat completions, keep it.
    if u.lower().endswith("/chat/completions"):
        return u

    # If it's exactly .../v1, append chat/completions.
    if u.lower().endswith("/v1"):
        return u + "/chat/completions"

    # If user passed bare host (no path), assume /v1/chat/completions.
    try:
        parsed = urlparse(u)
        if parsed.scheme in {"http", "https"} and (parsed.path == "" or parsed.path == "/"):
            return u + "/v1/chat/completions"
    except Exception:
        pass

    # Otherwise, leave as-is (user may be pointing at a proxy path).
    return u


AI_ENDPOINT = _normalize_ai_endpoint(os.environ.get("STARLISTENER_AI_ENDPOINT"))
AI_MODEL = "qwen3-coder-next"


def _call_ai(prompt: str, max_tokens: int = 700, timeout: int = 25) -> Optional[str]:
    """Call the configured OpenAI-compatible endpoint (if enabled)."""

    if os.environ.get("STARLISTENER_AI_DISABLED") == "1":
        return None
    try:
        if os.environ.get("STARLISTENER_AI_DEBUG") == "1":
            _stderr_log(f"[AI] POST {AI_ENDPOINT} model={AI_MODEL} max_tokens={max_tokens}")
        response = requests.post(
            AI_ENDPOINT,
            json={
                "model": AI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        if os.environ.get("STARLISTENER_AI_DEBUG") == "1":
            _stderr_log("[AI] request failed")
        return None


def _infer_marker_profile_ai(text: str) -> Optional[str]:
    """Ask the local OpenAI-compatible server to classify marker profile.

    Returns: numeric | symbol | letter | None
    """

    sample = (text or "")[:5_000]
    if not sample:
        return None

    prompt = (
        "Classify how footnotes are marked in this book excerpt. "
        "Return ONLY one token: numeric, symbol, letter, unknown.\n\n"
        "EXCERPT:\n" + sample
    )
    ai = _call_ai(prompt, max_tokens=10, timeout=120)
    if not ai:
        return None
    token = _safe_text(ai).split(" ")[0].strip().lower()
    if token in {"numeric", "symbol", "letter"}:
        return token
    return None


def _ai_infer_notes_split(lines: List[str]) -> Optional[Tuple[int, int]]:
    """Ask AI to find the notes header line when heuristics fail.

    Returns (main_end_index, defs_start_index) or None.
    This is intentionally conservative and only uses a small window around the
    first dense cluster of definition-like lines.
    """

    if not lines:
        return None

    # Find a dense cluster of definition-like lines in the latter portion.
    start_scan = max(0, int(len(lines) * 0.35))
    def_idxs = [i for i in range(start_scan, len(lines)) if _def_line_regex().match(lines[i] or "")]
    if len(def_idxs) < 2:
        return None

    first_def = None
    for i in def_idxs:
        near = [j for j in def_idxs if j >= i and j <= i + 80]
        if len(near) >= 2:
            first_def = i
            break
    if first_def is None:
        return None

    win_start = max(0, first_def - 80)
    win_end = min(len(lines), first_def + 40)
    snippet_lines = []
    for idx in range(win_start, win_end):
        t = _safe_text(lines[idx] or "")
        if not t:
            continue
        snippet_lines.append(f"{idx}: {t}")

    if not snippet_lines:
        return None

    prompt = (
        "We are parsing a book file that likely contains a NOTES/ENDNOTES section. "
        "Below are numbered lines (index: text). Identify the single line index that is the "
        "section heading immediately introducing the notes definitions (e.g., 'NOTES.', 'ANNOTATIONS', 'ENDNOTES'). "
        "Return ONLY the integer line index. If none, return -1.\n\n"
        + "\n".join(snippet_lines)
    )

    ai = _call_ai(prompt, max_tokens=6, timeout=120)
    if not ai:
        return None
    m = re.search(r"-?\d+", ai)
    if not m:
        return None
    try:
        header_idx = int(m.group(0))
    except Exception:
        return None
    if header_idx < 0 or header_idx >= len(lines):
        return None

    # defs_start is the next non-empty line after the header.
    defs_start = None
    for j in range(header_idx + 1, min(header_idx + 25, len(lines))):
        if not _safe_text(lines[j] or ""):
            continue
        defs_start = j
        break
    if defs_start is None:
        return None

    # Must actually look like definitions.
    if not _def_line_regex().match(lines[defs_start] or ""):
        return None

    return (header_idx, defs_start)


def _ai_disambiguate_pairs(batch: List[Dict[str, Any]]) -> None:
    """Given a batch of items each with anchor + candidate definitions, pick best candidate.

    Mutates items in-place setting suggested_definition + confidence_score.
    """

    if not batch:
        return

    payload = []
    for item in batch:
        candidates = item.get("candidates") or []
        if not candidates:
            continue

        def _cand_line(idx: int, c: Dict[str, Any]) -> str:
            origin = c.get("origin")
            origin_part = f" ({origin})" if origin else ""
            return f"  - C{idx}{origin_part}: {c['text'][:200]}"

        c_text = "\n".join([_cand_line(idx, c) for idx, c in enumerate(candidates)])
        payload.append(
            f"ID: {item['id']}\nMarker: {item['marker']}\nAnchor context: {item.get('context','')[:250]}\nCandidates:\n{c_text}\n"
        )

    if not payload:
        return

    prompt = (
        "You are pairing footnote anchors to footnote definitions. "
        "For each ID, choose the single best candidate definition (C0..Cn) or NONE. "
        "Return ONLY lines in this exact format: ID:<id> BEST:<Ck|NONE> CONF:<0.00-1.00>.\n\n"
        + "\n---\n".join(payload)
        + "\nAnswer:"
    )

    ai = _call_ai(prompt, timeout=300)
    if not ai:
        for item in batch:
            item["ai_status"] = "unavailable"
        return

    for line in ai.splitlines():
        m = re.match(
            r"\s*ID\s*:\s*(\d+)\s+BEST\s*:\s*(C\d+|NONE)\s+CONF\s*:\s*([01](?:\.\d+)?)\s*",
            line.strip(),
            re.IGNORECASE,
        )
        if not m:
            continue
        item_id = int(m.group(1))
        best = m.group(2).upper()
        conf = float(m.group(3))
        # Find the item
        for item in batch:
            if item.get("id") != item_id:
                continue
            if best == "NONE":
                item["confidence"] = "Manual Review Required"
                item["confidence_score"] = min(item.get("confidence_score") or 0.0, 0.5)
                item["ai_status"] = "none"
                continue
            idx = int(best[1:])
            candidates = item.get("candidates") or []
            if 0 <= idx < len(candidates):
                item["suggested_definition"] = candidates[idx]["text"]
                item["confidence_score"] = max(item.get("confidence_score") or 0.0, conf)
                item["confidence"] = f"AI Match ({item['confidence_score']:.2f})"
                item["ai_status"] = "matched"

    for item in batch:
        if item.get("ai_status") == "pending":
            item["ai_status"] = "unparsed"
