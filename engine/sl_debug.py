import os
import sys

try:
    from sl_utility import _safe_text
except ModuleNotFoundError:  # pragma: no cover
    from .sl_utility import _safe_text  # type: ignore


def _stderr_log(msg: str) -> None:
    try:
        line = str(msg).rstrip() + "\n"

        # Optional: write debug to a file to avoid flooding the Electron terminal.
        # Useful when STARLISTENER_DEBUG_MARKERS_VERBOSE=1.
        dbg_path = (os.environ.get("STARLISTENER_DEBUG_LOG_FILE") or "").strip()
        if dbg_path:
            try:
                with open(dbg_path, "a", encoding="utf-8", errors="ignore") as f:
                    f.write(line)
                return
            except Exception:
                # Fall back to stderr.
                pass

        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass


def _debug_markers_set() -> set[str]:
    """Return a set of marker strings to debug (from env var).

    Set `STARLISTENER_DEBUG_MARKERS` to a comma-separated list like:
      1,12,13
    to emit per-chapter diagnostics to stderr.
    """

    raw = (os.environ.get("STARLISTENER_DEBUG_MARKERS") or "").strip()
    if not raw:
        return set()
    out: set[str] = set()
    for part in raw.split(","):
        t = _safe_text(part)
        if t:
            out.add(t)
    return out


def _debug_markers_verbose_enabled() -> bool:
    return (os.environ.get("STARLISTENER_DEBUG_MARKERS_VERBOSE") or "").strip() == "1"
