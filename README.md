# Star Listener v0.0.1

### Note: This Electron application has been coded with heavy AI assistance. It has been tested locally by me (i use it daily). If you are not comfortable with this, then it is your perogative to not use it.

### Requiremenmts

- NodeJS
https://nodejs.org/en/download

- Python
https://www.python.org/downloads/

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

**CPU-Accelerated Runtime (Must install):**
```
.\.venv\Scripts\python.exe -m pip install kokoro-onnx soundfile onnxruntime
```

**GPU-Accelerated Runtime (Optional install: Recommended for NVIDIA RTX):**
```
.\.venv\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]"
```

> The `[cuda,cudnn]` extras automatically install the correct NVIDIA CUDA 12.x and cuDNN 9.x packages. No separate `nvidia-*` installs needed.

### Troubleshooting GPU Detection

**Verify CUDA is detected:**
```
.\.venv\Scripts\python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```
You should see `'CUDAExecutionProvider'` in the output.

**If CUDA is NOT listed, try these fixes in order:**

1. **CPU package conflict** (most common): If you accidentally installed  `onnxruntime` after `onnxruntime-gpu`, run:
   ```
   .\.venv\Scripts\python.exe -m pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]"
   ```
2. **Force CUDA provider**: In your `.env` file, set `ONNX_PROVIDER = "CUDAExecutionProvider"` to skip auto-detection.
3. **Driver**: Ensure NVIDIA Game Ready or Studio drivers are installed (not Windows DCH). Run `nvidia-smi` to confirm.

### 5. Download the Kokoro ONNX model files

Place them in the `/assets` folder:
- `kokoro-v1.0.onnx` (CPU)
- `kokoro-v1.0.fp16.onnx` (GPU, FP16)
- `kokoro-v1.0.fp16-gpu.onnx` (GPU, FP16)
- `kokoro-v1.0.int8.onnx` (GPU, INT8)
- `voices-v1.0.bin`

Download from: https://github.com/thewh1teagle/kokoro-onnx/releases

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

# Text-to-Speech model
TTS_MODEL = "kokoro-v1.0.fp16-gpu.onnx"
TTS_VOICES = "voices-v1.0.bin"
ONNX_PROVIDER = "CUDAExecutionProvider"

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
| `TTS_MODEL` | `kokoro-v1.0.fp16-gpu.onnx` | Kokoro ONNX model file in `/assets` |
| `TTS_VOICES` | `voices-v1.0.bin` | Voice styles file in `/assets` |
| `ONNX_PROVIDER` | `CUDAExecutionProvider` | ONNX execution provider for TTS. Set to `CPUExecutionProvider` if GPU acceleration fails. |
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
  "Feanor": "fˈeɪənɔː",
  "Noldor": "nˈɒldɔː"
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