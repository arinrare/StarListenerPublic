# StarListener Project Guidelines

## Build & Test
- `npm start` — launch the Electron app.
- There is no test suite yet. All verification is manual.

## Python Engine
- The engine lives in `engine/`. It is invoked from Node via `spawn`.
- All engine modules are prefixed `sl_` (e.g., `sl_ai.py`, `sl_scan.py`).
- Use internal relative imports (`from .sl_utility import ...`) for all engine modules.

## Code Style
- JavaScript (src/): CommonJS (`require`/`module.exports`), no build step.
- Python (engine/): standard library + `ebooklib`, `bs4`, `requests`, `numpy`, `onnxruntime`, `kokoro_onnx`. No type-checking pipeline.

## Copyright Compliance
- Do **not** include copyrighted proper nouns, character names, place names,
  invented-language names, book titles, chapter titles, or quoted passages from
  copyrighted works in any source code, comments, docstrings, string literals,
  or debug log messages.
- Use generic placeholder names instead of real ones. Examples:
  - Instead of a specific Tolkien chapter like *The Scouring of the Shire*,
    use `CHAPTER NAME` or `Chapter Name.`
  - Instead of a specific character like *Frodo*, use `Character Name`.
  - Instead of a specific place like *Mordor*, use `Place Name`.
- This applies to every file in the repository (`src/`, `engine/`, `assets/`,
  config files, etc.).
- When writing tests or examples, fabricate plausible but non-infringing names.
