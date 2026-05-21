# Star Listener v0.0.2

### Note: This Electron application has been coded with heavy AI assistance. It has been tested locally by me (i use it daily). If you are not comfortable with this, then it is your perogative to not use it.

### Requiremenmts

- NodeJS
https://nodejs.org/en/download

- Python (3.12 required — newer versions lack pre-built wheels for dependency packages)
https://www.python.org/downloads/

- espeak-ng (Windows only)
https://github.com/espeak-ng/espeak-ng/releases — download and run the `.msi` installer

## To set up and run

### 1. Install Electron
```
npm install
```

### 2. Create the environment
```
python -m venv .venv
```

### 3. Activate it

**Windows (PowerShell):**
```
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```
.\.venv\Scripts\activate.bat
```

### 4. Install the dependencies in order

**Core (engine + EPUB support):**

```
.\.venv\Scripts\python.exe -m pip install ebooklib beautifulsoup4 requests ffmpeg-python
```

```
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Kokoro TTS (PyTorch with word-level timestamps):**

```
.\.venv\Scripts\python.exe -m pip install "kokoro>=0.9.4" soundfile
```

### Troubleshooting GPU Detection

**Verify CUDA is detected:**
```
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```
You should see `True` in the output.

### 5. Model files (auto-downloaded, offline after first run)

The Kokoro PyTorch model and voice files (~330 MB total) are downloaded automatically from HuggingFace on first use. No manual downloads required.

**IMPORTANT — First run must be online.** The model weights and voice files need to be downloaded once. After the initial download, enable offline mode by adding to `.env`:

```env
HF_HUB_OFFLINE = "1"
HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

**Quick first-run tip:** Limit words to download faster:
```env
TTS_WORD_LIMIT = "50"
```
This processes ~50 words (under a minute), downloads everything, then you can remove or increase the limit for subsequent runs.

**Manual download (for fully offline setups):** If the target machine has no internet, download on another machine using the HuggingFace CLI:

1. Install: `pip install huggingface_hub`
2. Run: `huggingface-cli download hexgrad/Kokoro-82M --local-dir ./kokoro_model`
3. Copy the `kokoro_model` folder to your target machine
4. Set the model path before running: `set KOKORO_MODEL_DIR=C:\path\to\kokoro_model`
5. Enable offline mode in `.env`: `HF_HUB_OFFLINE = "1"`

Alternatively, run the following Python script on the online machine to populate the standard cache, then copy `~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/` to the target machine:
```python
from huggingface_hub import snapshot_download
snapshot_download("hexgrad/Kokoro-82M")
```

### Custom Pronunciations

### Optional: PDF support
```
.\.venv\Scripts\python.exe -m pip install pymupdf
```

### Optional: AI-assisted disambiguation

If you run a local OpenAI-compatible server at `http://localhost:8080/v1/chat/completions`, the engine will use it to disambiguate cases where multiple definitions could match the same marker.

You will need to shut down this AI after you run the note markers scan, BEFORE you run the TTS scan, as both make use of the GPU if you are using GPU Execution for the TTS.

### 6. To start the app
```
npm run start
```

---

## Environment Variables (.env)

Create a `.env` file in the root folder:

```env
# App
NODE_ENV = "development"

# Footnote scanning AI
STARLISTENER_AI_ENDPOINT = "http://localhost:1234/v1"
STARLISTENER_AI_DISABLED = "false"

# Regression testing — writes scan results to /tests for golden-file comparison
REGRESSION_TESTING = "false"

# Text-to-Speech

# TTS audio output
TTS_BITRATE = "128k"
TTS_WORD_LIMIT = "1000"

# Custom pronunciation dictionary (JSON file in /assets)
TTS_PRONUNCIATIONS = "pronunciations.json"
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `NODE_ENV` | `"development"` | Electron mode |
| `STARLISTENER_AI_ENDPOINT` | — | OpenAI-compatible endpoint for AI disambiguation |
| `STARLISTENER_AI_DISABLED` | `"false"` | Skip all AI HTTP calls (`"true"` = disabled) |
| `REGRESSION_TESTING` | `"false"` | Write scan results JSON to `/tests` on each scan |
| `TTS_BITRATE` | `128k` | MP3 audio bitrate (ffmpeg `-b:a` value) |
| `TTS_WORD_LIMIT` | (none = all words) | Max words to process from EPUB for TTS |
| `TTS_PRONUNCIATIONS` | `pronunciations.json` | Custom IPA pronunciation dictionary in `/assets` |

---

## Custom Pronunciations

Create `assets/pronunciations.json` with word-to-IPA mappings:

```json
{
  "tolkien": "tˈɒlkɪn",
  "Tolkien": "tˈɒlkɪn",
  "TOLKIEN": "tˈɒlkɪn",
  "feanor": "fˈeɪənɔː",
  "Feanor": "fˈeɪənɔː",
  "FEANOR": "fˈeɪənɔː",
  "noldor": "nˈɒldɔː",
  "Noldor": "nˈɒldɔː",
  "NOLDOR": "nˈɒldɔː"
}
```

Entries are case-sensitive. Both cased and lowercase variants can be included for reliable matching. The pronunciations must use standard IPA (International Phonetic Alphabet) tokens compatible with espeak-ng / Kokoro.

# Supported document types

Currently only ePub. The file selector will let you choose more types, but ePub are currently the only types supported.

# Footnote detection

This feature is a work in progress. There are a lot of different types of docucments out there. This has been tested on some ePub publications both with structured and unstructured formatting. Well structured ePubs work a lot better than badly structured ones. Footnotes can appear at the end of a page, or at the end of a chapter. They can be sequential numbers, daggers, asterixes. Various types have been tested. But be aware, this project is fairly new and a wide range of publications have not yet been tested. It is highly possible you may need to add code or heuristics yourself if footnotes are not being dtected correctly for a specific publication, or use a different ePub that is better or differently formatted.

I will continue to work on documents and improve the footnote detection as time goes on.

I also plan to possibly add chapter detection for splitting mp3 files by chapter. This is already partially implemented within the footnote detection.

# Images in documents

Images are not currently supported and the processing will skip over them. It's possible in the future i may add a feature to describe an image, though that would need a third AI model in the project.

# AI models

Curently local Kokoro v1.0 for Text to Speech is supported. Any local OpenAI compatible model is supported for footnote ambiguity detection - most of the testing has been done using Qwen3-Coder-Next. I plan to add support for more TTS models of varying quality and hardware requirements in the future.

For the TTS, processing time is about 20-50x realtime speed, and a 500 page publication will generate in about 1 hour. The specs of the testing system are AMD 7800x3d and Nvidia RTX4090. The bottleneck curretnly is python running on the CPU, the CPU reaches 99% utilisation at times. The GPU utilisation on the system while generating ia about 40%, and the VRAM utilisation is about 30-40%.

# Output

Ouput file are stored in the /ouput folder. If you leave these here the app will load them on launch of file open, saving you having to rescan and reprocess the book each time. Reprocessing will overwrite files in this folder for the same publication name - so copy them out after processing if you want to save a particular version of a file.

# DISCLAIMER

This product is only to be used on legally owned, DRM free, non-copyrighted publications. Copyrighted material must not be used, and the generation of audio, or sharing of output files of copyrighted material is stricty against terms of use. The author bears no responsibility for the consequnces of such actions, and explicitly adivses against it.