import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env so the scan engine can reach the AI endpoint.
try:
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
except Exception:
    pass

from engine.sl_scan import scan_file_for_footnotes
from engine.sl_types import ScanOptions


SUPPORTED_BOOK_EXTS = {".epub", ".pdf", ".txt", ".md", ".html", ".htm", ".xhtml"}
GOLDEN_NAME_RE = re.compile(r"^(?P<label>.+)_\d{8}_\d{6}\.json$", re.IGNORECASE)
IGNORED_ROW_KEYS = {
    "id",
    "confidence",
    "confidence_score",
    "confidence_score_base",
    "match_method",
    "chapter_index",
    "chapter_name",
    "chapter_group",
}


def _default_scan_options() -> ScanOptions:
    return ScanOptions(marker_profile="auto_ai", allow_ai_pairing=True, ai_debug=False)


def _safe_book_label(name: str) -> str:
    stem = Path(name).stem.strip() or "scan"
    cleaned = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_", " ", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    label = "".join(cleaned).strip().replace(" ", "_")
    return label or "scan"


def _discover_books(root: Path) -> Dict[str, Path]:
    books: Dict[str, Path] = {}
    for entry in root.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in SUPPORTED_BOOK_EXTS:
            continue
        books[_safe_book_label(entry.name)] = entry
    return books


def _discover_golden_files(tests_dir: Path) -> List[Path]:
    files: List[Path] = []
    for entry in sorted(tests_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".json":
            continue
        if GOLDEN_NAME_RE.match(entry.name):
            files.append(entry)
    return files


def _golden_label(path: Path) -> str:
    m = GOLDEN_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"not a golden file: {path.name}")
    return m.group("label")


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows") or []
    return [row for row in raw if isinstance(row, dict)]


def _load_scan_options(path: Path) -> ScanOptions:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return _default_scan_options()

    opt = raw.get("scan_options")
    if not isinstance(opt, dict):
        return _default_scan_options()

    return ScanOptions(
        marker_profile=str(opt.get("marker_profile") or opt.get("markerProfile") or "auto_ai"),
        allow_ai_pairing=bool(
            opt.get("allow_ai_pairing")
            if opt.get("allow_ai_pairing") is not None
            else opt.get("allowAIPairing")
            if opt.get("allowAIPairing") is not None
            else True
        ),
        ai_debug=bool(
            opt.get("ai_debug")
            if opt.get("ai_debug") is not None
            else opt.get("aiDebug")
            if opt.get("aiDebug") is not None
            else False
        ),
    )


def _format_scan_options(options: ScanOptions) -> str:
    return (
        f"marker_profile={options.marker_profile}, "
        f"allow_ai_pairing={options.allow_ai_pairing}, "
        f"ai_debug={options.ai_debug}"
    )


def _scan_rows(book_path: Path, options: ScanOptions) -> Tuple[str, List[Dict[str, Any]]]:
    raw = scan_file_for_footnotes(str(book_path), options=options)
    rows = [row for row in json.loads(raw) if isinstance(row, dict)]
    return raw, rows


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value


def _canonical_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _normalize_value(value)
        for key, value in sorted(row.items())
        if key not in IGNORED_ROW_KEYS
    }


def _row_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("order_key"),
        row.get("order_key"),
        row.get("chapter_label"),
        row.get("marker"),
        row.get("position"),
        row.get("context"),
        row.get("suggested_definition"),
    )


def _display_group_key(row: Dict[str, Any]) -> str:
    source = row.get("source")
    if source == "epub":
        return f"epub:{row.get('chapter_group') or row.get('chapter_label') or row.get('chapter_index')}"
    if source == "pdf":
        return f"pdf:{row.get('page_index')}"
    if source == "text":
        return f"text:{row.get('file_name') or ''}"
    return f"other:{source or ''}"


def _display_group_label(rows: Sequence[Dict[str, Any]]) -> str:
    for row in rows:
        label = row.get("chapter_label")
        if label:
            return str(label)
    first = rows[0] if rows else {}
    return str(_display_group_key(first))


def _group_by_chapter(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_display_group_key(row)].append(row)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=_row_sort_key)
    return dict(grouped)


def _format_row(row: Dict[str, Any]) -> str:
    marker = row.get("marker")
    label = row.get("chapter_label")
    name = row.get("chapter_name")
    method = row.get("match_method")
    position = row.get("position")
    return f"marker={marker!r} label={label!r} chapter_name={name!r} match={method!r} position={position!r}"


def _diff_chapter(expected_rows: Sequence[Dict[str, Any]], actual_rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    exp = [_canonical_row(row) for row in expected_rows]
    act = [_canonical_row(row) for row in actual_rows]

    if len(exp) != len(act):
        return f"row count mismatch: expected {len(exp)}, got {len(act)}"

    for idx, (left, right) in enumerate(zip(exp, act)):
        if left == right:
            continue

        differing = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
        preview = ", ".join(differing[:8])
        return (
            f"first row mismatch at offset {idx}: differing keys [{preview}]\n"
            f"  expected {_format_row(left)}\n"
            f"  actual   {_format_row(right)}"
        )

    return None


def compare_against_golden(golden_path: Path, book_path: Path) -> List[str]:
    expected_rows = _load_rows(golden_path)
    options = _load_scan_options(golden_path)
    _actual_raw, actual_rows = _scan_rows(book_path, options)

    failures: List[str] = []

    expected_grouped = _group_by_chapter(expected_rows)
    actual_grouped = _group_by_chapter(actual_rows)

    expected_chapters = set(expected_grouped)
    actual_chapters = set(actual_grouped)
    missing = sorted(expected_chapters - actual_chapters)
    extra = sorted(actual_chapters - expected_chapters)

    if missing:
        missing_labels = [_display_group_label(expected_grouped[key]) for key in missing]
        failures.append(f"missing chapter displays: {missing_labels}")
    if extra:
        extra_labels = [_display_group_label(actual_grouped[key]) for key in extra]
        failures.append(f"new chapter displays detected: {extra_labels}")

    for chapter_key in sorted(expected_chapters & actual_chapters):
        diff = _diff_chapter(expected_grouped[chapter_key], actual_grouped[chapter_key])
        if diff is not None:
            label = _display_group_label(expected_grouped[chapter_key])
            failures.append(f"chapter_display {label!r}: {diff}")
            break

    return failures


def refresh_golden(golden_path: Path, book_path: Path) -> ScanOptions:
    options = _load_scan_options(golden_path)
    actual_raw, _actual_rows = _scan_rows(book_path, options)
    payload = {
        "scan_options": {
            "marker_profile": options.marker_profile,
            "allow_ai_pairing": options.allow_ai_pairing,
            "ai_debug": options.ai_debug,
        },
        "rows": json.loads(actual_raw),
    }
    golden_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return options


def _parse_args(argv: Sequence[str]) -> Tuple[bool, List[str]]:
    refresh = False
    selectors: List[str] = []
    for arg in argv:
        if arg == "--refresh-goldens":
            refresh = True
            continue
        selectors.append(arg)
    return refresh, selectors


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refresh_goldens, selectors = _parse_args(argv)

    tests_dir = ROOT / "tests"
    books = _discover_books(ROOT)
    golden_files = _discover_golden_files(tests_dir)

    if selectors:
        requested = set(selectors)
        golden_files = [path for path in golden_files if path.name in requested or _golden_label(path) in requested]

    if not golden_files:
        print("No golden regression JSON files found in /tests.")
        return 1

    failures_total = 0
    for golden_path in golden_files:
        label = _golden_label(golden_path)
        book_path = books.get(label)
        if book_path is None:
            print(f"FAIL {golden_path.name}: no matching source book found for label {label!r}")
            failures_total += 1
            continue

        options = _load_scan_options(golden_path)

        if refresh_goldens:
            used = refresh_golden(golden_path, book_path)
            print(f"UPDATED {golden_path.name} from {book_path.name} ({_format_scan_options(used)})")
            continue

        failures = compare_against_golden(golden_path, book_path)
        if failures:
            print(f"FAIL {golden_path.name} vs {book_path.name} ({_format_scan_options(options)})")
            for item in failures:
                print(f"  - {item}")
            failures_total += 1
            continue

        print(f"PASS {golden_path.name} vs {book_path.name} ({_format_scan_options(options)})")

    return 1 if failures_total else 0


if __name__ == "__main__":
    raise SystemExit(main())