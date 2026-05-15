# MKV Subtitle Translator

Translate MKV subtitle tracks or standalone subtitle files to Latin American Spanish.

## What It Does

- Translates MKV subtitles to Latin American Spanish
- Works with MKV subtitle tracks, standalone text subtitles (`ASS`, `SSA`, `SRT`), and OCR for hardcoded or image subtitle tracks like `PGS`
- Supports Gemini and Ollama
- Can mux translated subtitles back into MKVs

## Defaults

- Provider: `gemini`
- Gemini translation model: `models/gemma-4-26b-a4b-it`
- Gemini audio fallback model: `models/gemini-3.1-flash-lite-preview`
- Local Ollama default model: `llama3.2`

If the main translation model does not support audio input, Gemini audio fallback is used automatically for gender hints.

## Requirements

### System

- `mkvmerge` and `mkvextract` for MKV workflows
- `ffmpeg` and `ffprobe` for audio-aware translation or OCR mode

Install examples:

```bash
# Arch
sudo pacman -S mkvtoolnix-cli ffmpeg

# Debian/Ubuntu
sudo apt-get install mkvtoolnix ffmpeg
```

### Python

```bash
pip install -r requirements.txt
```

## Providers

### Gemini

Use one of:

- `--api-key YOUR_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

### Ollama Local

- Run an Ollama server locally or on another host
- Optional env override: `OLLAMA_HOST`

### Ollama Cloud

- `--provider ollama-cloud`
- `--api-key YOUR_KEY` or `OLLAMA_API_KEY`
- pass an explicit `--model`

### Shared Env Vars

- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_AUDIO_MODEL`
- `LLM_BASE_URL`

## Basic Usage

Translate one file:

```bash
python3 translator.py --api-key YOUR_KEY video.mkv
```

Translate every supported file in a directory:

```bash
python3 translator.py --api-key YOUR_KEY /path/to/videos
```

Translate a standalone subtitle file directly:

```bash
python3 translator.py --api-key YOUR_KEY episode01.ass
```

## Common Examples

Use audio-aware translation with automatic extraction:

```bash
python3 translator.py --api-key YOUR_KEY --extract-audio video.mkv
```

Use Gemini explicitly:

```bash
python3 translator.py --provider gemini --api-key YOUR_KEY video.mkv
```

Use an extra context file:

```bash
python3 translator.py --provider gemini --api-key YOUR_KEY --extra-context notes.txt video.mkv
```

Use local Ollama:

```bash
python3 translator.py --provider ollama-local --model llama3.2 video.mkv
```

Use Ollama Cloud:

```bash
python3 translator.py --provider ollama-cloud --api-key YOUR_KEY --model YOUR_MODEL video.mkv
```

OCR hardcoded subtitles with Gemini:

```bash
python3 translator.py --provider gemini --api-key YOUR_KEY --ocr --ocr-lang eng video.mkv
```

Skip the OCR review UI:

```bash
python3 translator.py --provider gemini --api-key YOUR_KEY --ocr --skip-ocr-review video.mkv
```

Run diagnostics:

```bash
python3 translator.py --doctor --provider ollama-local --model llama3.2
```

Add `{Original: ...}` comments to existing translated ASS output:

```bash
python3 translator.py --add-original-only episode01.mkv
```

Remux corrected ASS files back into translated MKVs:

```bash
python3 tools/remux_corrected_subs.py translated_subs
```

Dry-run remux:

```bash
python3 tools/remux_corrected_subs.py translated_subs --dry-run
```

## All Flags

- `INPUT_PATH` - single supported file (`.mkv`, `.ass`, `.ssa`, `.srt`) or a directory containing them
- `--provider {gemini,ollama-local,ollama-cloud}` - select the LLM provider
- `--base-url URL` - override the API base URL
- `--api-key KEY` - primary API key for the selected provider
- `--api-key2 KEY` - secondary Gemini API key for quota failover
- `--model NAME` - translation model
- `--audio-model NAME` - Gemini fallback model for audio-based gender analysis
- `--list-models` - list available models for the selected provider and exit
- `--doctor` - print config and provider/tool diagnostics
- `--output-dir DIR` - output directory for translated files
- `--batch-size N` - subtitle lines per batch
- `--extra-context FILE` - optional UTF-8 text file with extra translation context (`.txt` recommended)
- `--no-thinking` - disable thinking mode
- `--thinking-budget N` - token budget for thinking mode
- `--progress-log` - save progress details to a log file
- `--thoughts-log` - save thinking output to a separate log file
- `--no-colors` - disable colored terminal output
- `--keep-original` - keep original text as hidden ASS comments during translation
- `--add-original-only` - inject `{Original: ...}` into existing translated ASS output and rebuild the MKV
- `--ocr` - extract burned-in subtitles with OCR for MKVs
- `--ocr-lang CODE` - source language code for OCR subtitles before translation
- `--skip-ocr-review` - skip the browser OCR review step
- `-a, --audio-file FILE` - use an existing audio file for gender-aware translation
- `--extract-audio` - extract audio from the MKV for gender-aware translation
- `--strip-sdh` - remove SDH elements like speaker names and sound-effect captions
- `--review` - force the AI review pass after translation (skip the interactive prompt)
- `--no-review` - skip the AI review pass entirely (no prompt, no review)
- `--review-batch-size N` - lines per review batch (default 150)
- `--temperature FLOAT` - sampling temperature
- `--top-p FLOAT` - nucleus sampling value
- `--top-k INT` - top-k sampling value

## AI Review Pass

After translation, the tool can optionally run a second pass to check for quality issues:

- **Gender agreement** — mismatched adjectives, verbs, or pronouns vs. established speaker gender
- **Untranslated source text** — words or phrases left in the source language
- **Terminology consistency** — same term translated differently across lines
- **Register consistency** — unjustified tú/usted switches for the same character
- **Pronoun and reference errors** — wrong gender on pronouns, dangling references

By default you'll be asked interactively whether to run the review. Use `--review` to always run it or `--no-review` to skip the prompt entirely.

The review pass uses the same provider and model as the translation pass. For Ollama cloud models, review calls use streaming to avoid timeouts.

## Output

By default, files are written to `translated_subs/`:

- `video.translated.mkv`
- `video.es-419.ass` or `.srt` or `.ssa`
- `subtitles.es-419.ass` or `.srt` or `.ssa` for standalone subtitle inputs
- `video.translation.log` if `--progress-log` is enabled
- `video.thoughts.log` if `--thoughts-log` is enabled

## Layout

- `translator.py` - main CLI entrypoint
- `tools/` - support modules and helper scripts
- `tools/remux_corrected_subs.py` - remux corrected ASS files back into translated MKVs

## Notes

- The tool resumes interrupted work from `tmp/*.progress`
- ASS formatting is preserved mechanically, not just by prompt instructions
- Audio-aware translation is mainly useful for gendered languages like Spanish
- Ollama translation is text-only; Gemini still handles audio/gender hints, while OCR mode supports Gemini and Ollama vision models
- Thinking is enabled by default for supported models; use `--no-thinking` to disable it
- OCR review sessions can be resumed from `tmp/<name>.ocr-review/`
- OCR frame extraction and OCR review sessions are cached under `tmp/` for reuse
- The OCR review web UI pauses translation so you can correct OCR text before translation continues
- GPU decode can help the frame-extraction side of OCR, but the dominant runtime cost is usually the OCR model requests
- Secondary subtitle context is optional and mainly useful when primary and reference subtitles are in different languages
