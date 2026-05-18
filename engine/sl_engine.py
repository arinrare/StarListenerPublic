import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from sl_scan import scan_file_for_footnotes
    from sl_types import ScanOptions
except ModuleNotFoundError:  # pragma: no cover
    from .sl_scan import scan_file_for_footnotes  # type: ignore
    from .sl_types import ScanOptions  # type: ignore


def scan_for_footnotes(file_path: str, *, options: Optional[ScanOptions] = None) -> str:
    """Backwards-compatible wrapper used by the Electron app."""
    try:
        return scan_file_for_footnotes(file_path, options=options)
    except Exception as e:
        return json.dumps([{"type": "error", "marker": "!", "context": str(e), "id": 0}], indent=2)


def _env_flag_enabled(name: str) -> bool:
    value = str(os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_book_label(file_path: str) -> str:
    stem = Path(file_path).stem.strip() or "scan"
    cleaned = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_", " ", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    label = "".join(cleaned).strip().replace(" ", "_")
    return label or "scan"


def _scan_options_payload(options: ScanOptions) -> Dict[str, Any]:
    return {
        "marker_profile": options.marker_profile,
        "allow_ai_pairing": options.allow_ai_pairing,
        "ai_debug": options.ai_debug,
    }


def _regression_json_payload(raw_json: str, options: ScanOptions) -> str:
    rows = json.loads(raw_json)
    payload = {
        "scan_options": _scan_options_payload(options),
        "rows": rows,
    }
    return json.dumps(payload, indent=2)


def _maybe_write_regression_json(file_path: str, raw_json: str, options: ScanOptions) -> None:
    if not _env_flag_enabled("REGRESSION_TESTING"):
        return

    try:
        root = Path(__file__).resolve().parents[1]
        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{_safe_book_label(file_path)}_{timestamp}.json"
        (tests_dir / out_name).write_text(_regression_json_payload(raw_json, options), encoding="utf-8")
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if len(argv) < 1:
        sys.stdout.write(json.dumps([{"error": "No file path provided"}]))
        sys.stdout.flush()
        return 0

    raw_arg = argv[0]
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
                        allow_ai_pairing=bool(
                            opt.get("allow_ai_pairing")
                            if opt.get("allow_ai_pairing") is not None
                            else opt.get("allowAIPairing")
                            if opt.get("allowAIPairing") is not None
                            else options.allow_ai_pairing
                        ),
                        ai_debug=bool(
                            opt.get("ai_debug")
                            if opt.get("ai_debug") is not None
                            else opt.get("aiDebug")
                            if opt.get("aiDebug") is not None
                            else options.ai_debug
                        ),
                    )
        except Exception:
            file_path = raw_arg

    if not options.allow_ai_pairing:
        os.environ["STARLISTENER_AI_DISABLED"] = "1"
    if options.ai_debug:
        os.environ["STARLISTENER_AI_DEBUG"] = "1"

    result_json = scan_for_footnotes(str(file_path), options=options)
    _maybe_write_regression_json(str(file_path), result_json, options)

    # Use sys.stdout.write for clean pipe to Electron
    sys.stdout.write(result_json)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
