# MKV Subtitle Translator

A Python-based command-line tool that automatically extracts, translates, and re-embeds subtitles into MKV video files. Designed specifically to translate subtitles from various languages to Latin American Spanish using Google's Gemini AI API.

## Features

- **Automatic subtitle extraction** from MKV files using MKVToolNix
- **AI-powered translation** to Latin American Spanish (es-419) via Google Gemini API
- **Gender-aware translation** - analyzes audio to apply correct grammatical gender forms (verb conjugations, adjectives, pronouns)
- **Batch processing** with configurable batch sizes for efficient translation
- **Resume capability** - automatically saves progress and resumes interrupted translations
- **Dual API key support** - failover to secondary key when quota is reached
- **Real-time progress tracking** with animated progress bars and colored output
- **Thinking mode** - leverages Gemini's thinking capabilities for improved translation quality
- **Format preservation** - maintains original subtitle format (.ass/.srt/.ssa) and ASS formatting, colors, and positioning
- **Advanced ASS processing** - mechanical preservation of formatting directives (\N, \n, \h) and automatic color code normalization
- **Language-aware filtering** - handles Latin, CJK (Chinese/Japanese/Korean), and RTL (Right-to-Left) scripts intelligently
- **Error recovery** - handles API failures, quota errors, and partial batch successes
- **Optional logging** - saves detailed translation progress and AI thinking processes

## Requirements

### System Dependencies

**MKVToolNix** (required) - must be installed and available in system PATH:
- `mkvmerge` - for track detection and subtitle merging
- `mkvextract` - for subtitle extraction

**Installation:**
- Ubuntu/Debian: `sudo apt-get install mkvtoolnix`
- Arch Linux: `sudo pacman -S mkvtoolnix-cli`
- macOS: `brew install mkvtoolnix`
- Windows: Download from [MKVToolNix website](https://mkvtoolnix.download/)

**FFmpeg/FFprobe** (optional, required for gender-aware translation) - must be installed and available in system PATH:
- `ffmpeg` - for audio extraction and compression
- `ffprobe` - for audio stream analysis

**Installation:**
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Arch Linux: `sudo pacman -S ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

### Python Dependencies

Python 3.6 or higher is required. Install dependencies using:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `google-genai` - Google Gemini API client
- `pysubs2` - Subtitle file parsing (.ass, .srt, .ssa)
- `json-repair` - Robust JSON parsing for API responses

### API Requirements

**Google Gemini API Key** is required. Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

You can provide the API key via:
1. Command-line argument: `--api-key YOUR_KEY`
2. Environment variable: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install MKVToolNix for your platform (see System Dependencies above)
4. Obtain a Google Gemini API key

## Usage

### Basic Usage

Translate a single MKV file:
```bash
python3 translator.py --api-key YOUR_API_KEY video.mkv
```

Translate all MKV files in a directory:
```bash
python3 translator.py --api-key YOUR_API_KEY /path/to/videos/
```

### Command-Line Options

```
python3 translator.py [OPTIONS] INPUT_PATH
```

**Positional Arguments:**
- `INPUT_PATH` - Single .mkv file or directory containing .mkv files

**API Configuration:**
- `--api-key KEY` - Primary Gemini API key (required if not set as environment variable)
- `--api-key2 KEY` - Secondary API key for automatic failover when quota is reached
- `--model NAME` - Gemini model to use (default: `gemini-2.5-pro`)
- `--list-models` - List all available Gemini models and exit

**Translation Options:**
- `--batch-size N` - Number of subtitle lines per batch (default: 300)
- `--thinking` - Enable thinking mode for better translation quality (default: enabled)
- `--no-thinking` - Disable thinking mode
- `--thinking-budget N` - Token budget for thinking process, 0-24576 (default: 2048)
- `--keep-original` - Preserve original text as hidden comments in ASS format

**Gender-Aware Translation Options:**
- `-a, --audio-file FILE` - Audio file for gender-aware translation (MP3 format recommended)
- `--extract-audio` - Extract audio from MKV video for gender-aware translation (requires ffmpeg)

**Output Options:**
- `--output-dir DIR` - Output directory for translated files (default: `translated_subs/`)
- `--progress-log` - Save translation progress to log file
- `--thoughts-log` - Save AI thinking process to separate log file
- `--no-colors` - Disable colored terminal output

**Help:**
- `--help` - Show help message and exit

### Examples

**Single file with default settings:**
```bash
python3 translator.py --api-key YOUR_KEY movie.mkv
```

**Directory with dual API keys for continuous processing:**
```bash
python3 translator.py \
  --api-key PRIMARY_KEY \
  --api-key2 SECONDARY_KEY \
  /path/to/anime/series/
```

**Custom batch size with full logging:**
```bash
python3 translator.py \
  --api-key YOUR_KEY \
  --batch-size 150 \
  --progress-log \
  --thoughts-log \
  video.mkv
```

**Without thinking mode (faster, but potentially lower quality):**
```bash
python3 translator.py \
  --api-key YOUR_KEY \
  --no-thinking \
  video.mkv
```

**Use specific model:**
```bash
python3 translator.py \
  --api-key YOUR_KEY \
  --model gemini-1.5-flash \
  video.mkv
```

**Keep original text alongside translations:**
```bash
python3 translator.py \
  --api-key YOUR_KEY \
  --keep-original \
  video.mkv
```

**Gender-aware translation with automatic audio extraction:**
```bash
python3 translator.py \
  --api-key YOUR_KEY \
  --extract-audio \
  video.mkv
```

**Gender-aware translation with existing audio file:**
```bash
python3 translator.py \
  --api-key YOUR_KEY \
  --audio-file video_audio.mp3 \
  video.mkv
```

## How It Works

### Processing Pipeline

1. **File Detection** - Scans for .mkv files in the provided path
2. **Track Selection** - Detects available subtitle tracks and prompts for language selection
3. **Subtitle Extraction** - Extracts the selected subtitle track using `mkvextract`
4. **Audio Processing** (optional) - Extracts and compresses audio to ~20MB MP3 using ffmpeg
5. **ASS Preprocessing** - Normalizes malformed color codes and protects formatting directives
6. **Parsing** - Parses subtitle file using pysubs2 library
7. **Filtering** - Filters out short lines and metadata, applies language-aware rules
8. **Batch Translation** - Groups lines into batches and sends to Gemini API with streaming (includes audio if available)
9. **Real-time Processing** - Applies translations as they arrive from the API
10. **Format Restoration** - Restores original ASS formatting tags and styles
11. **Merging** - Creates new MKV file with translated subtitles using `mkvmerge`
12. **Cleanup** - Removes temporary files and saves logs if enabled

### Resume Functionality

If translation is interrupted (Ctrl+C, network error, quota limit), the tool automatically saves progress. On next run:
- Progress file is detected in `tmp/` directory
- You'll be prompted to resume from where you left off
- Completed batches are skipped automatically

### API Quota Management

When using dual API keys (`--api-key` and `--api-key2`):
- Primary key is used initially
- When quota is reached, automatically switches to secondary key
- No manual intervention required
- If both keys are exhausted, waits 60 seconds before retry

### Gender-Aware Translation

When audio is provided (`--extract-audio` or `--audio-file`), the tool analyzes speaker voices to apply correct grammatical gender:

**How it works:**
- Audio is sent to Gemini API alongside subtitle text
- AI analyzes voice characteristics at each subtitle timestamp
- Applies gender-appropriate verb conjugations, adjectives, and pronouns
- Essential for languages with grammatical gender (Spanish, French, etc.)

**Use cases:**
- First-person dialogue: "I am tired" → "Estoy cansado" (male) vs "Estoy cansada" (female)
- Adjectives: "I am happy" → "Estoy feliz" (neutral) vs "Estoy contento/a" (gendered)
- Past participles: "I have finished" → "He terminado" (male) vs "He terminada" (female in some contexts)
- Addressing others: Adapts based on who the speaker is talking to

**Audio processing:**
- Extracts primary audio stream (default track)
- Converts multi-channel to mono (dialogue-focused)
- Compresses to ~20MB MP3 for API upload efficiency
- Automatically cleans up temporary audio files

### Output Structure

```
translated_subs/
├── video.translated.mkv          # Final MKV with translated subtitles
├── video.es-419.ass/.srt/.ssa    # Translated subtitle file (standalone, same format as input)
├── video.translation.log         # Translation progress log (if --progress-log)
└── video.thoughts.log            # AI thinking process log (if --thoughts-log)
```

## Supported Subtitle Formats

**Input formats:**
- `.ass` (Advanced SubStation Alpha)
- `.srt` (SubRip)
- `.ssa` (SubStation Alpha)

**Output format:**
- Same format as input - the tool preserves the original subtitle format
- `.ass` files output as `.ass` (preserves formatting, colors, positioning)
- `.srt` files output as `.srt`
- `.ssa` files output as `.ssa`

**Detected languages:**
- English
- German
- Japanese
- French
- (Other languages can be selected manually)

## Language-Aware Processing

The tool applies intelligent filtering based on script type:

**Latin scripts** (English, Spanish, French, etc.):
- Minimum 2 characters required
- Short lines filtered out

**CJK scripts** (Chinese, Japanese, Korean):
- No minimum length requirement
- All lines processed

**RTL scripts** (Arabic, Hebrew, Farsi):
- Detected automatically
- Special wrapping markers applied

## Troubleshooting

**"mkvmerge not found"**
- Install MKVToolNix and ensure it's in your system PATH
- Test with: `mkvmerge --version`

**"ffmpeg not found" or "ffprobe not found"**
- Install FFmpeg and ensure it's in your system PATH
- Test with: `ffmpeg -version` and `ffprobe -version`
- Only required if using `--extract-audio` flag

**"Failed to extract audio" or audio-related errors**
- Ensure FFmpeg is properly installed
- Check that the MKV file has an audio track
- Try providing audio manually with `--audio-file` instead of `--extract-audio`
- Verify disk space for temporary audio files (~20-50MB per video)

**"API key not provided"**
- Set `GEMINI_API_KEY` environment variable, or
- Use `--api-key` command-line argument

**"Quota exceeded" errors**
- Use `--api-key2` to provide a secondary API key
- Wait for quota reset (usually 60 seconds to 1 minute)
- Consider using smaller `--batch-size`

**Translation interrupted**
- Run the same command again
- Tool will detect progress file and offer to resume

**Poor translation quality**
- Ensure `--thinking` mode is enabled (default)
- Increase `--thinking-budget` (e.g., `--thinking-budget 4096`)
- Try a more capable model (e.g., `--model gemini-1.5-pro`)

**Colors not working in terminal**
- Some terminals don't support ANSI colors
- Use `--no-colors` flag to disable

## Advanced Usage

### Environment Variables

Set API key permanently:
```bash
export GEMINI_API_KEY="your_api_key_here"
python3 translator.py video.mkv
```

### Batch Processing Multiple Directories

Process all MKV files in multiple directories:
```bash
for dir in /path/to/series/Season*; do
  python3 translator.py --api-key YOUR_KEY "$dir"
done
```

Process with gender-aware translation for entire series:
```bash
for dir in /path/to/series/Season*; do
  python3 translator.py --api-key YOUR_KEY --extract-audio "$dir"
done
```

### Integration with Scripts

The tool returns exit code 0 on success, non-zero on failure, making it suitable for automation:
```bash
if python3 translator.py --api-key KEY video.mkv; then
  echo "Translation successful"
  rm video.mkv  # Remove original if needed
fi
```

## Technical Details

### Architecture

- **Modular design** with separate modules for logging, progress display, and translation
- **Streaming API responses** for real-time progress updates
- **Batch-based processing** optimizes API usage and token limits
- **Signal handling** for graceful interruption (Ctrl+C)
- **Progress persistence** via JSON checkpoints in `tmp/` directory

### File Structure

```
.
├── translator.py               # Main application with translation engine
├── audio_utils.py              # Audio extraction and compression utilities
├── progress_display.py         # Real-time progress UI
├── logger.py                   # Colored logging system
├── requirements.txt            # Python dependencies
├── tmp/                        # Runtime files (created automatically)
│   ├── *.progress             # Progress checkpoint files
│   └── *_audio.mp3            # Extracted audio (if using --extract-audio)
└── translated_subs/           # Output directory (created automatically)
```

### Translation Process

The tool uses a sophisticated prompt system that:
- Instructs Gemini to translate to Latin American Spanish (es-419)
- Analyzes audio (when provided) for gender-aware translation
- Preserves line breaks and special characters via token replacement
- Maintains subtitle formatting and timing
- Protects ASS directives (\N, \n, \h) through mechanical substitution
- Normalizes malformed ASS color codes proactively
- Handles mature content appropriately
- Uses thinking mode for complex contexts
- Returns structured JSON for reliable parsing

**ASS Format Protection:**
- Token-based preservation (industry-standard i18n approach)
- `\N` (hard line break) → placeholder → restored after translation
- `\n` (soft line break) → placeholder → restored after translation
- `\h` (hard space) → placeholder → restored after translation
- More reliable than prompt-based instructions alone

## License

This project is provided as-is for personal use. Check the license file if available, or contact the repository owner for usage terms.

## Credits

- Built with [Google Gemini API](https://ai.google.dev/)
- Uses [MKVToolNix](https://mkvtoolnix.download/) for subtitle processing
- Audio extraction via [FFmpeg](https://ffmpeg.org/)
- Subtitle parsing via [pysubs2](https://github.com/tkarabela/pysubs2)
- Architecture inspired by gemini-translator-srt

## Support

For issues, bugs, or feature requests, please check the project repository or contact the maintainer.
