#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MKV Subtitle Translator
Refactored with gemini-translator-srt's proven batch-based architecture.
Extracts subtitles from MKV files, translates them using Google Gemini, and merges back.
"""

import argparse
import logging
import sys
import subprocess
import json
import re
import time
from pathlib import Path
from collections import Counter
import unicodedata

from google import genai
from google.genai import types
import pysubs2

try:
    import json_repair
except ImportError:
    logging.error("json_repair module not found. Install it with: pip install json-repair")
    sys.exit(1)

try:
    from audio_utils import prepare_audio
except ImportError:
    logging.error("audio_utils module not found. Please ensure audio_utils.py is in the same directory.")
    sys.exit(1)

# --- API Manager for Dual API Key Support ---

class APIManager:
    """
    Manages dual API key support for handling quota limitations.
    Matches gemini-srt-translator's _switch_api and _get_client pattern.
    """
    def __init__(self, api_key, api_key2=None):
        """
        Initialize API manager with primary and optional secondary API key.

        Args:
            api_key: Primary API key
            api_key2: Secondary API key (optional, for quota failover)
        """
        self.api_key = api_key
        self.api_key2 = api_key2
        self.current_api_key = api_key
        self.current_api_number = 1
        self.backup_api_number = 2

    def get_client(self):
        """
        Create and return a Gemini client using the currently active API key.

        Returns:
            genai.Client: Client configured with current API key
        """
        return genai.Client(api_key=self.current_api_key)

    def switch_api(self):
        """
        Switch to the alternate API key if available.
        Matches gemini-srt-translator's _switch_api pattern (lines 622-639).

        Returns:
            bool: True if switched successfully, False if no alternative available
        """
        # If currently on API 1 and API 2 exists → switch to API 2
        if self.current_api_number == 1 and self.api_key2:
            self.current_api_key = self.api_key2
            self.current_api_number = 2
            self.backup_api_number = 1
            return True

        # If currently on API 2 and API 1 exists → switch back to API 1
        if self.current_api_number == 2 and self.api_key:
            self.current_api_key = self.api_key
            self.current_api_number = 1
            self.backup_api_number = 2
            return True

        # No alternative API key available
        return False

    def has_secondary_key(self):
        """Check if a secondary API key is configured."""
        return self.api_key2 is not None


# Import progress display module
try:
    from progress_display import (
        progress_bar, progress_complete, clear_progress,
        info_with_progress, warning_with_progress, error_with_progress, success_with_progress
    )
except ImportError:
    logging.error("progress_display module not found. Make sure progress_display.py is in the same directory.")
    sys.exit(1)

# Import enhanced logger module
try:
    import logger
except ImportError:
    logging.error("logger module not found. Make sure logger.py is in the same directory.")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress verbose HTTP logging from google/urllib and all submodules
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('google.genai').setLevel(logging.ERROR)
logging.getLogger('google.ai').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

# --- ASS Format Protection ---
# Use token replacement to guarantee preservation of ASS directives
# This is more reliable than prompt instructions
#
# ASS Text Control Directives:
# \N - Hard line break (forces new line, not wrappable)
# \n - Soft line break (wrappable, renderer can adjust)
# \h - Hard space (non-breaking space)
ASS_HARD_LINEBREAK = "\\N"
ASS_SOFT_LINEBREAK = "\\n"
ASS_HARD_SPACE = "\\h"

ASS_HARD_LINEBREAK_PLACEHOLDER = "<<<ASS_HLB>>>"
ASS_SOFT_LINEBREAK_PLACEHOLDER = "<<<ASS_SLB>>>"
ASS_HARD_SPACE_PLACEHOLDER = "<<<ASS_HSP>>>"


def protect_ass_directives(text):
    """
    Replace ASS format directives with placeholders before translation.
    This guarantees preservation regardless of model behavior.

    Protects:
    - \\N (hard line break) → <<<ASS_HLB>>>
    - \\n (soft line break) → <<<ASS_SLB>>>
    - \\h (hard space) → <<<ASS_HSP>>>

    More reliable than prompt-based instructions because:
    - Mechanical preservation (not AI-dependent)
    - Model never sees the directive (can't misinterpret)
    - Industry standard approach for i18n systems
    - Easy to extend for other ASS directives
    """
    # Order matters: replace longer sequences first to avoid conflicts
    text = text.replace(ASS_HARD_LINEBREAK, ASS_HARD_LINEBREAK_PLACEHOLDER)
    text = text.replace(ASS_SOFT_LINEBREAK, ASS_SOFT_LINEBREAK_PLACEHOLDER)
    text = text.replace(ASS_HARD_SPACE, ASS_HARD_SPACE_PLACEHOLDER)
    return text


def restore_ass_directives(text):
    """
    Restore ASS format directives from placeholders after translation.

    Restores:
    - <<<ASS_HLB>>> → \\N (hard line break)
    - <<<ASS_SLB>>> → \\n (soft line break)
    - <<<ASS_HSP>>> → \\h (hard space)
    """
    text = text.replace(ASS_HARD_LINEBREAK_PLACEHOLDER, ASS_HARD_LINEBREAK)
    text = text.replace(ASS_SOFT_LINEBREAK_PLACEHOLDER, ASS_SOFT_LINEBREAK)
    text = text.replace(ASS_HARD_SPACE_PLACEHOLDER, ASS_HARD_SPACE)
    return text


# --- ASS Formatting Helper Functions ---

def remove_formatting(text):
    """Remove ASS formatting tags from text."""
    return re.sub(r'\{.*?\}', '', text).strip()

def restore_formatting(original_text, translated_plain_text):
    """
    Restore ASS formatting tags from original text to translated text.
    Preserves the structure of formatting tags from the original.
    """
    formatting_tags = re.findall(r'\{[^}]+\}', original_text)

    if not formatting_tags:
        return translated_plain_text

    # Prepend all formatting tags to the translated text
    formatting_prefix = ''.join(formatting_tags)
    return f"{formatting_prefix}{translated_plain_text}"


def normalize_ass_colors(ass_path):
    """
    Normalize ALL ASS color codes to spec-compliant format in a single pass.

    Industry-standard approach (used by Aegisub, professional subtitle tools):
    - Proactive normalization BEFORE parsing
    - Preserves color values where possible
    - Single-pass efficiency
    - Never fails - always produces valid ASS
    - Tolerant of all malformed patterns

    ASS Color Format Spec:
    - Inline tags: \\c&HBBGGRR& or \\c&HAABBGGRR& (BGR order!)
    - Style values: &HAABBGGRR& (with alpha channel)
    - Required: &H prefix and trailing &

    Common malformations fixed:
    - Missing &H prefix: \\cFFFFFF& → \\c&HFFFFFF&
    - Missing trailing &: \\c&HFFFFFF → \\c&HFFFFFF&
    - Double &&: \\c&HFFFFFF&& → \\c&HFFFFFF&
    - Wrong prefix: \\cH00FFFFFF& → \\c&H00FFFFFF&
    - Partial hex: \\cFFF& → \\c&H000FFF& (padded to 6 digits)
    """
    try:
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        original_content = content

        # === Inline Color Tag Normalization ===
        # Matches: \c, \1c, \2c, \3c, \4c followed by color in any format
        # Pattern breakdown:
        # - \\(\d?c) : Matches \c or \1c-\4c
        # - (?:&H?)?([0-9A-Fa-f]+)&? : Matches color with optional &H and trailing &

        def normalize_inline_color(match):
            """Normalize a single inline color tag."""
            tag = match.group(1)  # 'c' or '1c' or '2c' or '3c' or '4c'
            color_hex = match.group(2)  # Just the hex digits

            # Remove any non-hex characters that slipped through
            clean_hex = re.sub(r'[^0-9A-Fa-f]', '', color_hex)

            if not clean_hex:
                # Invalid/empty color - remove the tag entirely
                return ''

            # Parse hex value
            try:
                color_value = int(clean_hex, 16)
            except ValueError:
                # Should never happen after cleaning, but be safe
                return ''

            # Normalize to proper length (pad with zeros if needed)
            # 6 digits = RGB, 8 digits = ARGB
            if len(clean_hex) <= 6:
                # RGB format - pad to 6 digits
                normalized = f"&H{color_value:06X}&"
            else:
                # ARGB format - pad to 8 digits
                normalized = f"&H{color_value:08X}&"

            return f"\\{tag}{normalized}"

        # Replace all inline color tags
        # This pattern matches all variations: \c..., \1c..., \2c..., \3c..., \4c...
        content = re.sub(
            r'\\(\d?c)(?:&H?)?([0-9A-Fa-f]+)&?(?![0-9A-Fa-f])',
            normalize_inline_color,
            content,
            flags=re.IGNORECASE
        )

        # === Style Line Color Normalization ===
        # Style lines have colors in specific comma-separated positions
        # Format: Style: Name,Font,Size,PrimaryColour,SecondaryColour,OutlineColour,BackColour,...

        def normalize_style_line(match):
            """Normalize colors in a Style: line."""
            line = match.group(0)

            # Find and normalize each color value in the style
            # Pattern: color values start with &H or H or just hex digits
            def fix_style_color(color_match):
                color_str = color_match.group(0)

                # Extract just hex digits
                clean_hex = re.sub(r'[^0-9A-Fa-f]', '', color_str)

                if not clean_hex or len(clean_hex) > 8:
                    # Invalid - use white with full opacity as safe default
                    return '&H00FFFFFF'

                try:
                    color_value = int(clean_hex, 16)
                    # Style colors should be 8 digits (AABBGGRR)
                    # If shorter, assume RGB and add full opacity (00)
                    if len(clean_hex) <= 6:
                        return f"&H00{color_value:06X}"
                    else:
                        return f"&H{color_value:08X}"
                except ValueError:
                    return '&H00FFFFFF'  # Safe default

            # Match color values in style (anywhere in the line after "Style:")
            # These are typically &HAABBGGRR or malformed versions
            # CRITICAL: &? before lookahead to match trailing & (e.g., FFFFFF&,)
            line = re.sub(
                r'(?:&H?|H)?[0-9A-Fa-f]{6,8}&?(?=\s*,|\s*$)',
                fix_style_color,
                line,
                flags=re.IGNORECASE
            )

            return line

        # Normalize all Style: lines
        content = re.sub(
            r'^Style:.*$',
            normalize_style_line,
            content,
            flags=re.MULTILINE | re.IGNORECASE
        )

        # === Cleanup Pass ===
        # Remove any orphaned/incomplete color tags that might cause issues
        # These are patterns that look like color tags but are too malformed to fix

        # Remove standalone \c or \Xc without any hex following
        content = re.sub(r'\\(\d?c)(?![&0-9A-Fa-f])', '', content)

        # Fix any remaining double ampersands
        content = re.sub(r'&&+', '&', content)

        # === Write if changed ===
        if content != original_content:
            with open(ass_path, 'w', encoding='utf-8-sig') as f:
                f.write(content)
            logging.debug(f"Normalized ASS color codes in {ass_path.name}")
            return True

        logging.debug(f"No color normalization needed for {ass_path.name}")
        return True

    except Exception as e:
        logging.warning(f"Failed to normalize ASS colors in {ass_path}: {e}")
        # Don't fail - let pysubs2 handle it
        return False

# --- MKVToolNix Functions ---

def check_mkvtoolnix():
    """Checks if mkvmerge and mkvextract are installed and in the PATH."""
    try:
        subprocess.run(["mkvmerge", "--version"], check=True, capture_output=True, text=True)
        subprocess.run(["mkvextract", "--version"], check=True, capture_output=True, text=True)
        logging.debug("MKVToolNix command-line tools found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("MKVToolNix not found. Please install it from https://mkvtoolnix.download/")
        logger.error("Ensure 'mkvmerge' and 'mkvextract' are in your system's PATH.")
        return False

def select_subtitle_track(tracks, remembered_lang=None):
    """
    Identifies English, German, Japanese, and French subtitle tracks.
    If a language has been previously selected, it defaults to that.
    Otherwise, prompts the user for selection if multiple are present.
    """
    track_map = {'eng': [], 'de': [], 'ja': [], 'fr': []}

    for track in tracks:
        if track.get("type") == "subtitles":
            lang = track.get("properties", {}).get("language")
            if lang == "eng":
                track_map['eng'].append(track)
            elif lang in ["de", "ger"]:
                track_map['de'].append(track)
            elif lang in ["ja", "jpn"]:
                track_map['ja'].append(track)
            elif lang in ["fr", "fre", "fra"]:
                track_map['fr'].append(track)

    found_tracks = {lang: tracks for lang, tracks in track_map.items() if tracks}

    if remembered_lang and remembered_lang in found_tracks:
        logger.info(f"Automatically selecting {remembered_lang} based on previous choice.")
        return found_tracks[remembered_lang][0], remembered_lang

    if not found_tracks:
        return None, None

    if len(found_tracks) == 1:
        lang = list(found_tracks.keys())[0]
        logging.debug(f"Found single subtitle track: {lang}.")
        return found_tracks[lang][0], lang

    # Multiple tracks found, prompt user
    lang_options = list(found_tracks.keys())
    print(f"Found multiple subtitle languages: {', '.join(lang_options)}")
    while True:
        default_lang = 'eng' if 'eng' in lang_options else lang_options[0]
        choice = input(f"Select language to process ({'/'.join(lang_options)}) [default: {default_lang}]: ").strip().lower()

        if not choice:
            choice = default_lang

        if choice in lang_options:
            logger.info(f"User selected {choice}.")
            return found_tracks[choice][0], choice

        print(f"Invalid choice. Please enter one of: {', '.join(lang_options)}.")

# --- Progress Management Functions ---

def save_progress(progress_file_path, current_line, total_lines, input_file):
    """
    Save translation progress to disk.

    Args:
        progress_file_path: Path to .progress file
        current_line: Current line number (1-indexed)
        total_lines: Total number of lines
        input_file: Path to input file being processed
    """
    try:
        progress_data = {
            "line": current_line,
            "total": total_lines,
            "input_file": str(input_file),
            "timestamp": time.time()
        }
        with open(progress_file_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
        logging.debug(f"Progress saved: line {current_line}/{total_lines}")
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def load_progress(progress_file_path, input_file):
    """
    Load saved progress from disk.

    Args:
        progress_file_path: Path to .progress file
        input_file: Current input file path

    Returns:
        tuple: (should_resume, start_line) or (False, 1) if no valid progress
    """
    if not progress_file_path.exists():
        return False, 1

    try:
        with open(progress_file_path, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)

        saved_line = progress_data.get("line", 1)
        saved_file = progress_data.get("input_file")
        saved_total = progress_data.get("total", 0)
        saved_timestamp = progress_data.get("timestamp", 0)

        # Validate progress matches current file
        if saved_file != str(input_file):
            logger.warning(f"Progress file is for different subtitle: {saved_file}")
            logger.warning(f"Current file: {input_file}")
            logger.warning("Ignoring saved progress.")
            return False, 1

        # Calculate age of progress
        age_hours = (time.time() - saved_timestamp) / 3600

        if saved_line > 1:
            logger.info(f"Found saved progress from {age_hours:.1f} hours ago")
            logger.info(f"Progress: {saved_line}/{saved_total} lines completed")
            return True, saved_line

        return False, 1

    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted progress file: {e}")
        return False, 1
    except Exception as e:
        logger.warning(f"Error reading progress file: {e}")
        return False, 1


def prompt_resume(saved_line, total_lines):
    """
    Prompt user whether to resume from saved progress.

    Args:
        saved_line: Number of completed translatable lines (NOT dialogue line index)
        total_lines: Total number of translatable lines (after MIN_TRANSLATION_LENGTH filtering)

    Returns:
        bool: True if user wants to resume, False otherwise
    """
    percentage = (saved_line / total_lines) * 100 if total_lines > 0 else 0

    print(f"\n{'='*60}")
    print(f"Previous translation was interrupted")
    print(f"Progress: {saved_line}/{total_lines} translatable lines ({percentage:.1f}% complete)")
    print(f"{'='*60}")

    while True:
        response = input("Resume from where you left off? (y/n) [default: y]: ").strip().lower()

        if response in ['', 'y', 'yes']:
            logger.info("Resuming from saved progress...")
            return True
        elif response in ['n', 'no']:
            logger.info("Starting from beginning...")
            return False
        else:
            print("Please enter 'y' or 'n'")

# --- Translation Helper Functions ---

def get_system_instruction(source_lang, target_lang="Latin American Spanish", thinking=True, audio_file=None):
    """
    Generate system instruction for translation.
    Adapted from gemini-translator-srt's approach with thinking mode support.
    """
    thinking_instruction = (
        "\n\nThink deeply and reason as much as possible before returning the response."
        if thinking
        else "\n\nDo NOT think or reason."
    )

    # Field definitions (conditional based on audio_file)
    fields = (
        "- index: a string identifier\n"
        "- content: the text to translate\n"
        "- time_start: the start time of the segment\n"
        "- time_end: the end time of the segment\n"
    ) if audio_file else (
        "- index: a string identifier\n"
        "- content: the text to translate\n"
    )

    instruction = f"""You are an assistant that translates subtitles from {source_lang} to {target_lang}.

You will receive a list of objects, each with these fields:
{fields}
Translate the 'content' field of each object.
If the 'content' field is empty, leave it as is.
Preserve line breaks, formatting, and special characters.
Do NOT move or merge 'content' between objects.
Do NOT add or remove any objects.
Do NOT alter the 'index' field."""

    # Audio-specific instructions (conditional)
    if audio_file:
        instruction += f"""

You will also receive an audio file.
Use the time_start and time_end of each object to analyze the audio.
Analyze the speaker's voice in the audio to determine gender, then apply grammatical gender rules for {target_lang}:
1. Listen for voice characteristics to identify if speaker is male/female:
   - Use masculine verb forms/adjectives if speaker sounds male
   - Use feminine verb forms/adjectives if speaker sounds female
   - Apply gender agreement to: verbs, adjectives, past participles, pronouns
   - Example: French 'I am tired' -> 'Je suis fatigué' (male) vs 'Je suis fatiguée' (female)
2. In some cases you also need to identify who the current speaker is talking to:
   - If the speaker is talking to a male, use masculine forms.
   - If the speaker is talking to a female, use feminine forms.
   - If the speaker is talking to a group, use plural forms.
   - Example: Portuguese 'You are tired' -> 'Você está cansado' (male) vs 'Você está cansada' (female)
   - Example: Spanish 'You are talking to a group' -> 'Ustedes están cansados' (male/general group) vs 'Ustedes están cansadas' (female group)"""

    instruction += thinking_instruction

    return instruction


def get_translation_config(system_instruction, model_name, thinking=True, thinking_budget=2048):
    """
    Build API configuration.
    Based on gemini-translator-srt's config builder with thinking mode support.
    """
    # Response schema: array of {index, content} objects
    response_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "index": types.Schema(type=types.Type.STRING),
                "content": types.Schema(type=types.Type.STRING)
            },
            required=["index", "content"]
        )
    )

    # Safety settings: allow all content (subtitles may contain mature content)
    safety_settings = [
        types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
        types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
    ]

    # Determine thinking mode compatibility
    thinking_compatible = "2.5" in model_name or "2.0" in model_name
    thinking_budget_compatible = "flash" in model_name

    # Build thinking config if compatible
    # Flash models: Use thinking_budget for controlled thinking
    # Pro models: Enable thinking without budget (handled by timeout/retry mechanism)
    thinking_config = None
    if thinking_compatible and thinking:
        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=thinking_budget if thinking_budget_compatible else None
        )

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        safety_settings=safety_settings,
        system_instruction=system_instruction,
        thinking_config=thinking_config
    )


def is_rtl(text):
    """
    Detect if text is right-to-left.
    From gemini-translator-srt's RTL detection.
    """
    if not text:
        return False

    count = Counter([unicodedata.bidirectional(c) for c in text])
    rtl_count = count.get("R", 0) + count.get("AL", 0) + count.get("RLE", 0) + count.get("RLI", 0)
    ltr_count = count.get("L", 0) + count.get("LRE", 0) + count.get("LRI", 0)

    return rtl_count > ltr_count


def is_primarily_latin(text):
    """
    Detect if text is primarily Latin/ASCII characters.
    Returns True for Latin script text, False for CJK, Arabic, Cyrillic, etc.

    This is used to determine if MIN_TRANSLATION_LENGTH should apply.
    Non-Latin scripts (like Japanese, Chinese, Korean) can convey complete
    meaning in 1-2 characters, so we don't filter them by length.
    """
    if not text:
        return True

    text = text.strip()
    if not text:
        return True

    # Count characters by script
    latin_count = 0
    other_count = 0

    for char in text:
        # Skip whitespace and punctuation
        if char.isspace() or unicodedata.category(char).startswith('P'):
            continue

        # Check Unicode script
        try:
            script_name = unicodedata.name(char).split()[0]
            # Latin includes basic ASCII and extended Latin characters
            if ord(char) < 128 or 'LATIN' in script_name:
                latin_count += 1
            else:
                other_count += 1
        except (ValueError, IndexError):
            # If we can't determine, assume Latin for ASCII range
            if ord(char) < 128:
                latin_count += 1
            else:
                other_count += 1

    # Text is primarily Latin if more than 50% of non-punctuation chars are Latin
    total = latin_count + other_count
    if total == 0:
        return True  # Only punctuation, treat as Latin

    return latin_count > other_count


def validate_batch_tokens(client, batch, model_name):
    """
    Validate batch doesn't exceed token limit.
    Uses actual model limits based on Gemini model specifications.
    """
    try:
        # Use the ACTUAL model for token counting
        token_count = client.models.count_tokens(
            model=model_name,
            contents=json.dumps(batch, ensure_ascii=False)
        )

        # Set token limits based on model (conservative estimates)
        # Source: https://ai.google.dev/gemini-api/docs/models/gemini
        if "pro" in model_name:
            token_limit = 2_000_000  # Pro models: ~2M tokens
        else:
            token_limit = 1_000_000  # Flash models: ~1M tokens

        if token_count.total_tokens > token_limit * 0.9:
            # Token limit exceeded - will be shown after clearing progress bar
            logger.error(f"Token count ({token_count.total_tokens}) exceeds 90% of limit ({token_limit})")
            return False

        logging.debug(f"Token validation passed: {token_count.total_tokens} tokens")
        return True

    except Exception as e:
        logger.warning(f"Token validation failed: {e}. Proceeding anyway.")
        return True


def prompt_new_batch_size(current_size):
    """
    Prompt user for new batch size when token limit exceeded.
    Progress bar is cleared before calling this, so use regular output.
    """
    while True:
        try:
            user_prompt = input(f"Enter new batch size (current: {current_size}): ")
            if user_prompt.strip():
                new_size = int(user_prompt)
                if new_size > 0:
                    return new_size
                print("Batch size must be a positive integer.")
            else:
                print("Please enter a valid number.")
        except ValueError:
            print("Invalid input. Batch size must be a positive integer.")
        except KeyboardInterrupt:
            logger.warning("\nUser interrupted batch size prompt")
            return current_size // 2  # Default to half


def build_resume_context(dialogue_lines, translated_subtitle, start_line, batch_size):
    """
    Build conversation context when resuming.
    Provides continuity for translation consistency.

    Args:
        start_line: Next line to translate (1-indexed dialogue line number)
    """
    # Include up to one batch of previous context (up to and including last translated line)
    context_start = max(0, start_line - batch_size - 1)
    context_end = start_line  # Exclusive end, so includes start_line - 1

    if context_start >= context_end:
        return []

    # Build original batch
    original_batch = [
        {"index": str(i), "content": dialogue_lines[i]}
        for i in range(context_start, context_end)
    ]

    # Build translated batch
    translated_batch = [
        {"index": str(i), "content": translated_subtitle[i]}
        for i in range(context_start, context_end)
    ]

    return [
        types.Content(role="user", parts=[types.Part(text=json.dumps(original_batch, ensure_ascii=False))]),
        types.Content(role="model", parts=[types.Part(text=json.dumps(translated_batch, ensure_ascii=False))])
    ]


# Global variable to track last successful chunk size (like gemini-translator-srt)
_last_chunk_size = 0

def get_last_chunk_size():
    """Get the number of lines successfully translated in the last batch."""
    global _last_chunk_size
    return _last_chunk_size


def process_batch_streaming(client, model_name, batch, previous_message, translated_subtitle, config,
                           current_line, total_lines, batch_number=1, keep_original=False, original_format=".ass", audio_part=None, audio_file=None,
                           dialogue_lines=None, unique_text_indices=None):
    """
    Process a batch with streaming responses and real-time progress display.
    Implements retry loop matching gemini-srt-translator's _process_batch pattern.

    Args:
        current_line: Current line number in the file (not batch index)
        total_lines: Total lines to translate

    Returns:
        previous_message context for next batch
    """
    global _last_chunk_size
    batch_size = len(batch)
    _last_chunk_size = 0  # Reset

    # Build request
    parts = [types.Part(text=json.dumps(batch, ensure_ascii=False))]

    # Add audio if available (for gender-aware translation)
    if audio_part:
        parts.append(audio_part)

    current_message = types.Content(role="user", parts=parts)

    # Build full conversation
    contents = previous_message + [current_message]

    # Temporarily suppress ALL logging during API call to prevent disrupting progress bar
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)

    # Retry loop matching gemini-srt-translator pattern
    done = False
    retry = -1
    final_response_text = ""
    final_thoughts_text = ""
    max_retries_on_timeout = 3  # Retry up to 3 times if thinking times out

    # Create lookup dict for original texts (for --keep-original feature)
    original_texts = {int(item["index"]): item["content"] for item in batch}

    try:
        while done == False:
            retry += 1
            response_text = ""
            thoughts_text = ""
            chunk_count = 0
            translated_batch = []
            blocked = False

            # Timeout tracking for thinking phase
            thinking_start_time = None
            thinking_timeout_seconds = 300  # 5 minutes
            timed_out = False

            # Stream response
            if blocked:
                break  # Exit retry loop if previously blocked

            response = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            )

            for chunk in response:
                # Check for blocking
                if chunk.prompt_feedback:
                    blocked = True
                    break

                # Extract text - check if parts exist before iterating
                if chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if not part.text:
                            continue
                        elif part.thought:
                            thoughts_text += part.text

                            # Track thinking start time
                            if thinking_start_time is None:
                                thinking_start_time = time.time()

                            # Check for thinking timeout (5 minutes)
                            thinking_elapsed = time.time() - thinking_start_time
                            if thinking_elapsed > thinking_timeout_seconds:
                                warning_with_progress(
                                    f"Thinking exceeded {thinking_timeout_seconds//60} minutes. "
                                    f"Retrying batch (attempt {retry + 1}/{max_retries_on_timeout})..."
                                )
                                timed_out = True
                                break  # Break out of chunk loop

                            # Show thinking indicator with elapsed time
                            thinking_minutes = int(thinking_elapsed // 60)
                            thinking_seconds = int(thinking_elapsed % 60)
                            progress_bar(
                                current=current_line,
                                total=total_lines,
                                model_name=model_name,
                                chunk_size=chunk_count,
                                is_thinking=True,
                                thinking_time=f"({thinking_minutes}m {thinking_seconds}s)"
                            )
                        else:
                            response_text += part.text

                            # Try to parse partial JSON and apply translations immediately
                            try:
                                partial_batch = json_repair.loads(response_text)
                                if isinstance(partial_batch, list):
                                    prev_chunk_count = chunk_count
                                    chunk_count = len(partial_batch)

                                    # Apply new translations as they arrive
                                    for i in range(prev_chunk_count, chunk_count):
                                        if i < len(partial_batch):
                                            item = partial_batch[i]
                                            idx = int(item["index"])
                                            content = item["content"]

                                            # Detect and wrap RTL text
                                            if is_rtl(content):
                                                content = f"\u202b{content}\u202c"

                                            # Add original text as hidden comment if --keep-original flag is enabled (ASS only)
                                            if keep_original and original_format == ".ass":
                                                original_text = original_texts.get(idx, "")
                                                if original_text:
                                                    content = f"{{Original: {original_text}}}{content}"

                                            translated_subtitle[idx] = content

                                            # Apply translation to all duplicates of this text
                                            if dialogue_lines is not None and unique_text_indices is not None:
                                                original_text_content = dialogue_lines[idx].strip()
                                                if original_text_content in unique_text_indices:
                                                    for duplicate_idx in unique_text_indices[original_text_content]:
                                                        if duplicate_idx != idx:  # Skip the one we just translated
                                                            translated_subtitle[duplicate_idx] = content

                                    # Update global chunk size for error recovery
                                    _last_chunk_size = chunk_count

                                    # Update progress bar with real-time chunk progress
                                    progress_bar(
                                        current=current_line,
                                        total=total_lines,
                                        model_name=model_name,
                                        chunk_size=chunk_count,
                                        is_loading=True
                                    )
                            except:
                                # Can't parse yet, just show loading
                                progress_bar(
                                    current=current_line,
                                    total=total_lines,
                                    model_name=model_name,
                                    chunk_size=chunk_count,
                                    is_loading=True
                                )

                # If timeout occurred during part processing, break chunk loop
                if timed_out:
                    break

            # Handle thinking timeout - retry if within limit
            if timed_out:
                if retry < max_retries_on_timeout:
                    clear_progress()
                    warning_with_progress(f"Thinking timeout. Retrying (attempt {retry + 1}/{max_retries_on_timeout})...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    # Max retries exceeded - give up on this batch
                    clear_progress()
                    error_with_progress(
                        f"Thinking timeout after {max_retries_on_timeout} retries. "
                        f"Skipping batch {batch_number}. Consider using --no-thinking or a Flash model."
                    )
                    # Return empty context to skip this batch
                    logging.getLogger().setLevel(old_level)
                    return previous_message

            # Check if blocked - exit retry loop
            if blocked:
                break

            # Check for empty response - retry
            if not response_text or not response_text.strip():
                clear_progress()
                error_with_progress("Gemini returned an empty response.")
                info_with_progress("Sending last batch again...")
                continue

            # Parse final response
            try:
                translated_batch = json_repair.loads(response_text)
            except Exception as e:
                clear_progress()
                warning_with_progress(f"Failed to parse response: {e}")
                info_with_progress("Sending last batch again...")
                continue

            # Validate response length
            if len(translated_batch) != len(batch):
                clear_progress()
                warning_with_progress(
                    f"Response length mismatch: expected {len(batch)}, got {len(translated_batch)}"
                )
                info_with_progress("Sending last batch again...")
                continue

            # Final application of translations
            for item in translated_batch:
                idx = int(item["index"])
                content = item["content"]

                # Detect and wrap RTL text
                if is_rtl(content):
                    content = f"\u202b{content}\u202c"

                # Add original text as hidden comment if --keep-original flag is enabled (ASS only)
                if keep_original and original_format == ".ass":
                    original_text = original_texts.get(idx, "")
                    if original_text:
                        content = f"{{Original: {original_text}}}{content}"

                translated_subtitle[idx] = content

                # Apply translation to all duplicates of this text
                if dialogue_lines is not None and unique_text_indices is not None:
                    original_text_content = dialogue_lines[idx].strip()
                    if original_text_content in unique_text_indices:
                        for duplicate_idx in unique_text_indices[original_text_content]:
                            if duplicate_idx != idx:  # Skip the one we just translated
                                translated_subtitle[duplicate_idx] = content

            _last_chunk_size = len(translated_batch)
            final_response_text = response_text
            final_thoughts_text = thoughts_text

            # Success - exit retry loop
            done = True

        # After retry loop - check if blocked
        if blocked:
            clear_progress()
            error_with_progress(
                "Gemini has blocked the translation for unknown reasons. "
                "Try changing your description (if you have one) and/or the batch size and try again."
            )
            import signal
            signal.raise_signal(signal.SIGINT)

        # Save thoughts to file if enabled
        if final_thoughts_text:
            logger.save_thoughts(final_thoughts_text, batch_number, retry=retry)

        # Build context for next batch
        response_parts = []
        if final_thoughts_text:
            response_parts.append(types.Part(thought=True, text=final_thoughts_text))
        response_parts.append(types.Part(text=final_response_text))

        return [
            types.Content(role="user", parts=[types.Part(text=json.dumps(batch, ensure_ascii=False))]),
            types.Content(role="model", parts=response_parts)
        ]
    finally:
        # Restore logging level
        logging.getLogger().setLevel(old_level)


def save_incremental_output(subs, dialogue_events, translated_subtitle, original_texts, output_path):
    """
    Save partial output after each batch.
    Allows recovery even if final save fails.
    """
    # Restore formatting to translated lines
    for i, event in enumerate(dialogue_events):
        if i < len(translated_subtitle):
            # First restore ASS directives from placeholders
            restored_directives = restore_ass_directives(translated_subtitle[i])
            # Then restore ASS formatting tags
            event.text = restore_formatting(original_texts[i], restored_directives)

    # Save
    subs.save(str(output_path))
    logging.debug(f"Incremental output saved to {output_path}")


# --- Core Translation Function ---

def translate_ass_file(ass_path, api_manager, model_name, output_dir, original_mkv_stem, lang_code, original_format=".ass", batch_size=300, thinking=True, thinking_budget=2048, keep_original=False, audio_file=None, extract_audio=False, video_path=None, free_quota=True):
    """
    Translates subtitle file using batch processing (simplified from multi-tier approach).
    Adapted from gemini-translator-srt's proven architecture.

    Args:
        ass_path: Path to extracted subtitle file
        client: Gemini API client
        model_name: Model to use for translation
        output_dir: Directory to save translated file
        original_mkv_stem: Original MKV filename stem
        lang_code: Source language code
        original_format: Original subtitle format (.ass, .srt, .ssa)
        batch_size: Number of lines to translate per batch

    Returns:
        Path to translated subtitle file, or None on failure
    """
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    # Progress and output files
    progress_file_path = tmp_dir / f"{original_mkv_stem}.progress"
    output_ass_path = output_dir / f"{original_mkv_stem}.es-419{original_format}"

    # Set log file paths
    logger.set_log_file_path(str(output_dir / f"{original_mkv_stem}.translation.log"))
    logger.set_thoughts_file_path(str(output_dir / f"{original_mkv_stem}.thoughts.log"))

    # Import progress display functions
    from progress_display import clear_progress

    # Audio handling for gender-aware translation
    audio_part = None
    audio_extracted = False

    try:
        # Extract audio from video if requested
        if video_path and extract_audio:
            if video_path.exists():
                logger.info("Extracting audio from video for gender-aware translation...")
                audio_file = prepare_audio(str(video_path))
                if audio_file:
                    audio_extracted = True
                else:
                    logger.warning("Failed to extract audio. Continuing without audio context.")
            else:
                logger.error(f"Video file {video_path} does not exist.")

        # Read audio file if provided
        if audio_file and Path(audio_file).exists():
            logger.info(f"Loading audio file: {Path(audio_file).name}")
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
            logger.info("Audio loaded successfully. Gender-aware translation enabled.")
        elif audio_file:
            logger.error(f"Audio file {audio_file} does not exist.")

        # Normalize ASS color codes to spec-compliant format before parsing
        normalize_ass_colors(ass_path)

        # Load and parse subtitle file
        # Color normalization already applied, so parsing should succeed
        try:
            subs = pysubs2.load(str(ass_path))
        except Exception as e:
            logger.error(f"Failed to parse ASS file {ass_path.name}: {e}")
            logger.error("Note: Color normalization was already applied. This may be a different parsing error.")
            return None, batch_size

        if not subs:
            logger.warning(f"No subtitle events found in {ass_path.name}. Skipping.")
            return None, batch_size

        # Extract dialogue events (keep your existing logic)
        dialogue_events = []
        dialogue_lines = []
        original_texts = []

        all_dialogue_events = [line for line in subs if hasattr(line, 'type') and line.type == "Dialogue"]

        if not all_dialogue_events:
            event_types = set(getattr(line, 'type', 'Unknown') for line in subs)
            logger.warning(f"No Dialogue events found in {ass_path.name}. Found event types: {event_types}")
            return None, batch_size

        # Styles to exclude from translation (romanized lyrics - already readable)
        EXCLUDE_STYLES = [
            r'.*Romaji$',       # Matches: OP-Romaji, ED-Romaji, Insert-Romaji
            r'.*Romaji[- ]',    # Matches: Romaji-Top, Romaji Bottom
        ]

        def should_exclude_style(style_name):
            """Check if style should be excluded from translation (Romaji styles)."""
            if not style_name:
                return False
            for pattern in EXCLUDE_STYLES:
                if re.match(pattern, style_name, re.IGNORECASE):
                    return True
            return False

        ass_header_keywords = ["[Script Info]", "[V4+ Styles]", "[Events]", "[Aegisub", "Format:", "Style:",
                               "ScriptType:", "PlayResX:", "PlayResY:", "WrapStyle:", "Title:", "Collisions:"]

        excluded_count = 0
        vector_drawing_count = 0
        for event in all_dialogue_events:
            # Skip Romaji styles (romanized lyrics - already readable, don't need translation)
            if should_exclude_style(event.style):
                excluded_count += 1
                continue

            # Skip vector drawings (lines with \p1, \p2, etc. - no text to translate)
            # These are shapes/animations like Sign - Mask
            if r'\p1' in event.text or r'\p2' in event.text or r'\p3' in event.text or r'\p4' in event.text:
                vector_drawing_count += 1
                continue

            if not any(keyword in event.text for keyword in ass_header_keywords):
                plain_text = remove_formatting(event.text)
                if plain_text:
                    # Protect ASS directives (like \N) before translation
                    protected_text = protect_ass_directives(plain_text)
                    dialogue_events.append(event)
                    dialogue_lines.append(protected_text)
                    original_texts.append(event.text)

        if not dialogue_lines:
            logger.warning(f"No valid dialogue lines found in {ass_path.name}.")
            return None, batch_size

        total_lines = len(dialogue_lines)

        # Deduplicate lines (same text with different effects/layers)
        # Maps unique text -> list of indices with that text
        text_to_indices = {}
        for i, line in enumerate(dialogue_lines):
            stripped = line.strip()
            if stripped not in text_to_indices:
                text_to_indices[stripped] = []
            text_to_indices[stripped].append(i)

        duplicate_count = total_lines - len(text_to_indices)
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count} duplicate lines (same text with different effects)")

        # Filter out lines too short to translate meaningfully
        # Only apply length filter to Latin/ASCII text (CJK can be meaningful in 1-2 chars)
        MIN_TRANSLATION_LENGTH = 2
        lines_to_translate = []  # List of indices that need translation
        translation_map = {}  # Maps index to original text for short lines
        unique_texts_to_translate = []  # Unique texts that need translation
        unique_text_indices = {}  # Maps unique text -> first occurrence index

        for unique_text, indices in text_to_indices.items():
            first_idx = indices[0]  # Use first occurrence as representative

            # Only apply MIN_TRANSLATION_LENGTH to Latin scripts
            # CJK, Arabic, Cyrillic, etc. can convey meaning in 1-2 characters
            if is_primarily_latin(unique_text) and len(unique_text) < MIN_TRANSLATION_LENGTH:
                # Keep very short Latin text as-is (e.g., "OK", "Hi", "!")
                for idx in indices:
                    translation_map[idx] = dialogue_lines[idx]
            else:
                # This text needs translation (either non-Latin or long enough)
                unique_texts_to_translate.append(unique_text)
                unique_text_indices[unique_text] = indices
                lines_to_translate.append(first_idx)  # Track first occurrence for progress

        # Calculate total kept-as-is: Romaji lines + vector drawings + short Latin lines
        total_kept_as_is = excluded_count + vector_drawing_count + len(translation_map)
        total_original_lines = len(all_dialogue_events)

        # Show clean summary (total original lines = unique texts to translate + duplicates + kept as-is)
        unique_count = len(unique_texts_to_translate)
        print(f"Found {total_original_lines} dialogue lines ({unique_count} unique texts to translate, {total_kept_as_is} kept as-is)\n")

        if duplicate_count > 0:
            logger.info(f"Deduplication: {duplicate_count} duplicate lines (same text with different effects)")
        if vector_drawing_count > 0:
            logger.info(f"Excluded {vector_drawing_count} vector drawing lines (shapes/animations)")

        if not lines_to_translate:
            logger.warning(f"No lines to translate in {ass_path.name} (all lines too short). Skipping.")
            # Just use the original lines - no translation needed
            for i, event in enumerate(dialogue_events):
                # Still need to restore ASS directives even for untranslated lines
                restored_directives = restore_ass_directives(dialogue_lines[i])
                event.text = restore_formatting(original_texts[i], restored_directives)
            subs.save(str(output_ass_path))
            logger.info(f"Saved (untranslated) output to {output_ass_path}")
            return output_ass_path, batch_size

        # Check for saved progress
        start_line = 0  # ASS dialogue events are 0-indexed
        translated_subtitle = dialogue_lines.copy()  # Start with original text (short lines stay as-is)

        if progress_file_path.exists():
            has_progress, saved_line = load_progress(progress_file_path, ass_path)
            if has_progress:
                # Calculate how many translatable lines have been completed
                completed_translatable = sum(1 for idx in lines_to_translate if idx < saved_line)

                # Only prompt to resume if we've actually translated something
                if completed_translatable > 0 and prompt_resume(completed_translatable, len(lines_to_translate)):
                    start_line = saved_line

                    # Load partial output if it exists
                    if output_ass_path.exists():
                        try:
                            partial_subs = pysubs2.load(str(output_ass_path))
                            partial_events = [e for e in partial_subs if hasattr(e, 'type') and e.type == "Dialogue"]

                            # Extract already translated lines (load ALL events, not just [:start_line])
                            # This matches gemini-srt-translator's approach (line 352)
                            for i, event in enumerate(partial_events):
                                if i < len(translated_subtitle):
                                    translated_subtitle[i] = remove_formatting(event.text)

                            logger.info(f"Loaded {completed_translatable} previously translated lines")
                        except Exception as e:
                            logger.warning(f"Failed to load partial output: {e}")
                else:
                    # User chose to start over
                    if output_ass_path.exists():
                        output_ass_path.unlink()
                    progress_file_path.unlink()

        # Build system instruction (with audio context if available)
        system_instruction = get_system_instruction(lang_code, target_lang="Latin American Spanish", thinking=thinking, audio_file=audio_file)

        # Configure API - get client from manager
        client = api_manager.get_client()
        config = get_translation_config(system_instruction, model_name, thinking, thinking_budget)

        # Process in batches (only translatable lines)
        # Use i to track current position in lines_to_translate (like gemini-translator-srt)
        i = 0
        previous_message = []
        total = len(lines_to_translate)
        batch_number = 1  # For thoughts logging

        # Skip to start position if resuming
        # Skip until we find the dialogue line index that matches start_line
        while i < total and lines_to_translate[i] < start_line:
            i += 1

        # Build context if resuming
        if start_line > 0:
            previous_message = build_resume_context(
                dialogue_lines,
                translated_subtitle,
                start_line,
                batch_size
            )

        # Signal handler for graceful interruption (matching gemini-srt-translator)
        import signal
        import sys

        def handle_interrupt(signal_received, frame):
            """Handle Ctrl+C or blocked content by saving progress and exiting cleanly."""
            last_chunk_size = get_last_chunk_size()
            clear_progress()
            warning_with_progress(
                f"Translation interrupted. Saving partial results to file. Progress saved."
            )

            # Save incremental output with current progress
            save_incremental_output(
                subs=subs,
                dialogue_events=dialogue_events,
                translated_subtitle=translated_subtitle,
                original_texts=original_texts,
                output_path=output_ass_path
            )

            # Save logs
            logger.save_logs()

            # Save progress (calculate current position accounting for partial success)
            if i > 0:
                current_position = max(1, i - len(batch) + max(0, last_chunk_size - 1))
                if current_position < len(lines_to_translate):
                    current_dialogue_line = lines_to_translate[current_position]
                    save_progress(progress_file_path, current_dialogue_line, total_lines, ass_path)

            sys.exit(0)

        signal.signal(signal.SIGINT, handle_interrupt)

        # Track quota error timing for smart API switching (gemini-srt-translator line 487)
        last_time = 0

        # Rate limiting for free tier users with pro models (gemini-srt-translator lines 395-404)
        delay = False
        delay_time = 30

        if "pro" in model_name:
            if free_quota:
                delay = True
                if not api_manager.has_secondary_key():
                    logger.info("Pro model and free user quota detected.\n")
                else:
                    delay_time = 15
                    logger.info("Pro model and free user quota detected, using secondary API key if needed.\n")
            else:
                logger.info("Paid quota mode enabled - no artificial rate limiting.\n")

        # Show initial progress bar (matching gemini-srt-translator line 460)
        progress_bar(
            current=i,
            total=total,
            model_name=model_name,
            is_sending=True
        )

        # Main translation loop (like gemini-translator-srt lines 489-606)
        batch = []  # Initialize batch outside loop for signal handler access
        while i < total:
            batch = []
            validated = False

            # Build batch up to batch_size
            batch_start_i = i
            while i < total and len(batch) < batch_size:
                line_idx = lines_to_translate[i]
                batch_item = {
                    "index": str(line_idx),
                    "content": dialogue_lines[line_idx]
                }
                # Add time codes if audio is present (for gender-aware translation)
                if audio_file:
                    batch_item["time_start"] = str(dialogue_events[line_idx].start)
                    batch_item["time_end"] = str(dialogue_events[line_idx].end)
                batch.append(batch_item)
                i += 1

            # Validate batch size against token limit
            while not validated:
                if not validate_batch_tokens(client, batch, model_name):
                    # Clear progress bar for cleaner token validation display
                    clear_progress()

                    # Reduce batch size and retry
                    new_batch_size = prompt_new_batch_size(batch_size)
                    decrement = batch_size - new_batch_size
                    if decrement > 0:
                        for _ in range(decrement):
                            i -= 1
                            batch.pop()
                    batch_size = new_batch_size

                    # Continue silently - user already confirmed in prompt
                    continue
                # Token validation passed, continue silently
                validated = True

            # Translate batch with partial success tracking
            try:
                # Show sending indicator
                progress_bar(
                    current=batch_start_i,
                    total=total,
                    model_name=model_name,
                    is_sending=True
                )

                # Track batch processing time for rate limiting (gemini-srt-translator line 537)
                start_time = time.time()

                previous_message = process_batch_streaming(
                    client=client,
                    model_name=model_name,
                    batch=batch,
                    previous_message=previous_message,
                    translated_subtitle=translated_subtitle,
                    config=config,
                    current_line=batch_start_i,
                    total_lines=total,
                    batch_number=batch_number,
                    keep_original=keep_original,
                    original_format=original_format,
                    audio_part=audio_part,
                    audio_file=audio_file,
                    dialogue_lines=dialogue_lines,
                    unique_text_indices=unique_text_indices
                )
                batch_number += 1  # Increment for next batch

                # Save progress after successful batch
                # Save index of next dialogue line to process (not position in lines_to_translate)
                if i < total:
                    current_dialogue_line = lines_to_translate[i]
                else:
                    current_dialogue_line = total_lines
                save_progress(progress_file_path, current_dialogue_line, total_lines, ass_path)

                # Save logs incrementally after each batch
                logger.save_logs()

                # Save incremental output (silently, without disrupting progress bar)
                save_incremental_output(
                    subs=subs,
                    dialogue_events=dialogue_events,
                    translated_subtitle=translated_subtitle,
                    original_texts=original_texts,
                    output_path=output_ass_path
                )

                batch.clear()

                # Apply rate limiting delay for free tier users (gemini-srt-translator lines 547-548)
                end_time = time.time()
                if delay and (end_time - start_time < delay_time) and i < total:
                    time.sleep(delay_time - (end_time - start_time))

            except Exception as e:
                error_msg = str(e)
                last_chunk_size = get_last_chunk_size()

                # Clear progress bar before logging
                clear_progress()

                # Handle quota errors with API switching or wait (gemini-srt-translator lines 553-564)
                if "quota" in error_msg.lower() or "503" in error_msg or "UNAVAILABLE" in error_msg:
                    current_time = time.time()

                    # Try switching API if:
                    # 1. More than 60 seconds since last quota error
                    # 2. Alternative API key is available
                    if current_time - last_time > 60 and api_manager.switch_api():
                        # Successfully switched to alternative API
                        info_with_progress(
                            f"API {api_manager.backup_api_number} quota exceeded! "
                            f"Switching to API {api_manager.current_api_number}..."
                        )

                        # Create new client with switched API key
                        client = api_manager.get_client()

                        # Save logs
                        logger.save_logs()

                        # Reset to batch start to retry with new API
                        i = batch_start_i
                        batch.clear()
                    else:
                        # Either no alternative API OR < 60 seconds since last quota error
                        # Wait 60 seconds before retrying
                        warning_with_progress(f"API quota exceeded: {error_msg}")
                        if not api_manager.has_secondary_key():
                            info_with_progress("No secondary API key configured.")
                        info_with_progress("Waiting 60 seconds before retry...")

                        # Save logs before waiting
                        logger.save_logs()

                        # Reset to batch start
                        i = batch_start_i
                        batch.clear()

                        # Countdown wait
                        for j in range(60, 0, -1):
                            progress_bar(batch_start_i, total, model_name, is_retrying=True, retry_countdown=j)
                            time.sleep(1)

                    # Update last quota error timestamp
                    last_time = current_time
                else:
                    # For other errors, retry the entire batch from the beginning
                    # This matches gemini-srt-translator's approach: reset and retry full batch
                    error_with_progress(f"Error: {error_msg}")
                    info_with_progress("Retrying last batch...")

                    # Reset to batch start (this will retry the ENTIRE batch)
                    i = batch_start_i
                    batch.clear()

                    # DO NOT ADVANCE i - we retry the full batch from batch_start_i
                    # The partial success in translated_subtitle stays, but we re-translate all items
                    # to ensure consistency

                    # Save logs after error
                    logger.save_logs()

                    # Save progress
                    if i < len(lines_to_translate):
                        current_dialogue_line = lines_to_translate[i]
                        save_progress(progress_file_path, current_dialogue_line, total_lines, ass_path)

                    # Save incremental output with partial success
                    save_incremental_output(
                        subs=subs,
                        dialogue_events=dialogue_events,
                        translated_subtitle=translated_subtitle,
                        original_texts=original_texts,
                        output_path=output_ass_path
                    )

                # Resume progress bar
                progress_bar(
                    current=i,
                    total=total,
                    model_name=model_name
                )

        # Show completion
        progress_complete(total, total, model_name)

        # Final validation
        if len(translated_subtitle) != len(dialogue_lines):
            logger.error(f"Line count mismatch: {len(translated_subtitle)} vs {len(dialogue_lines)}")
            return None, batch_size

        # Restore ASS formatting to final translations
        for i, event in enumerate(dialogue_events):
            # First restore ASS directives from placeholders
            restored_directives = restore_ass_directives(translated_subtitle[i])
            # Then restore ASS formatting tags
            event.text = restore_formatting(original_texts[i], restored_directives)

        # Save final output
        subs.save(str(output_ass_path))
        logger.success(f"Successfully created translated subtitle file: {output_ass_path.name}")

        # Clean up progress file
        if progress_file_path.exists():
            progress_file_path.unlink()
            logging.debug("Progress file deleted (translation complete)")

        # Clean up extracted audio if we extracted it
        if audio_file and audio_extracted and Path(audio_file).exists():
            Path(audio_file).unlink()
            logging.debug(f"Extracted audio file deleted: {Path(audio_file).name}")

        # Save logs if enabled
        logger.save_logs()

        return output_ass_path, batch_size

    except KeyboardInterrupt:
        clear_progress()
        logger.warning("\nTranslation interrupted by user!")
        if 'i' in locals() and i < len(lines_to_translate):
            # Save the current dialogue line index
            current_dialogue_line = lines_to_translate[i]
            save_progress(progress_file_path, current_dialogue_line, total_lines, ass_path)
            logger.info("Progress saved. Run again to resume.")
        logger.save_logs()
        return None, batch_size

    except Exception as e:
        clear_progress()
        logger.error(f"Translation failed: {e}")
        logger.save_logs()
        if 'i' in locals() and i < len(lines_to_translate):
            # Save the current dialogue line index
            current_dialogue_line = lines_to_translate[i]
            save_progress(progress_file_path, current_dialogue_line, total_lines, ass_path)
        return None, batch_size


def merge_subtitles_to_mkv(mkv_path, translated_subtitle_path, output_mkv_dir):
    """
    Merges the translated subtitle file (ASS, SRT, or SSA) into a copy of the original .mkv file.
    """
    try:
        output_mkv_name = f"{mkv_path.stem}.translated.mkv"
        output_mkv_path = output_mkv_dir / output_mkv_name

        logger.info(f"Merging translated subtitles into new file: {output_mkv_path.name}")

        mkvmerge_cmd = [
            "mkvmerge",
            "-o", str(output_mkv_path),
            "--language", f"0:es-419",
            "--track-name", "0:Spanish (Latin America)",
            "--default-track-flag", "0:yes",
            str(translated_subtitle_path),
            str(mkv_path)
        ]

        result = subprocess.run(mkvmerge_cmd, capture_output=True, text=True, encoding='utf-8')

        # mkvmerge exit codes: 0=success, 1=warnings (still successful), 2+=error
        if result.returncode == 0:
            logger.success(f"Successfully created merged MKV file: {output_mkv_path.name}")
            return output_mkv_path
        elif result.returncode == 1:
            logger.success(f"Successfully created merged MKV file: {output_mkv_path.name}")
            if result.stderr:
                logger.warning(f"mkvmerge warnings: {result.stderr.strip()}")
            return output_mkv_path
        else:
            logger.error(f"Failed to merge subtitles for {mkv_path.name}: {result.stderr}")
            return None

    except subprocess.CalledProcessError as e:
        # Should not reach here since we removed check=True
        logger.error(f"Failed to merge subtitles for {mkv_path.name}: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during merging for {mkv_path.name}: {e}")
        return None


def process_mkv_file(mkv_path, output_dir, api_manager, model_name, remembered_lang=None, batch_size=300, thinking=True, thinking_budget=2048, keep_original=False, audio_file=None, extract_audio=False, free_quota=True):
    """
    Processes a single MKV file: detects subtitles, prompts for selection (if needed),
    extracts, translates, and merges the chosen track.
    Returns tuple: (language code, final batch size) to be remembered for subsequent files.

    Args:
        free_quota: If True (default), apply rate limiting for free tier users.
                   If False (--paid-quota), remove artificial delays for paid users.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {mkv_path.name}")
    print(f"{'='*60}\n")

    # Pre-flight check
    expected_output_name = f"{mkv_path.stem}.translated.mkv"
    expected_output_path = output_dir / expected_output_name

    if expected_output_path.exists():
        logger.info(f"Output file \'{expected_output_name}\' already exists. Skipping.")
        return None, batch_size

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # 1. Identify tracks using mkvmerge
        mkvmerge_cmd = ["mkvmerge", "-J", str(mkv_path)]
        result = subprocess.run(mkvmerge_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        mkv_info = json.loads(result.stdout)

        # 2. Select subtitle track
        selected_track, lang_code = select_subtitle_track(mkv_info.get("tracks", []), remembered_lang)

        if selected_track is None:
            logger.warning(f"No suitable subtitle track found in {mkv_path.name}. Skipping.")
            return None, batch_size

        # 3. Check for supported subtitle format and extract the track
        codec_id = selected_track.get("properties", {}).get("codec_id")
        supported_codecs = {
            "S_TEXT/ASS": ".ass",
            "S_TEXT/SSA": ".ssa",
            "S_TEXT/UTF8": ".srt"
        }

        if codec_id not in supported_codecs:
            logger.warning(f"Unsupported subtitle format \'{codec_id}\' in {mkv_path.name}. Skipping.")
            return None, batch_size

        subtitle_track_id = selected_track['id']
        subtitle_extension = supported_codecs[codec_id]

        extracted_ass_path = tmp_dir / f"{mkv_path.stem}.{lang_code}{subtitle_extension}"
        mkvextract_cmd = ["mkvextract", "tracks", str(mkv_path), f"{subtitle_track_id}:{extracted_ass_path}"]
        logging.debug(f"Extracting track {subtitle_track_id} ({lang_code}, {codec_id}) to: {extracted_ass_path}")

        result = subprocess.run(mkvextract_cmd, capture_output=True, text=True, encoding='utf-8')

        # Check for MKV corruption
        if "Error in the Matroska file structure" in result.stdout or "Resync failed" in result.stdout:
            logger.warning(f"MKV file {mkv_path.name} appears to be corrupted. Skipping.")
            if extracted_ass_path.is_file():
                extracted_ass_path.unlink()
            return None, batch_size

        # Check if the file was actually created
        if not extracted_ass_path.is_file() or extracted_ass_path.stat().st_size == 0:
            logger.error(f"Extraction failed for {mkv_path.name}")
            return None, batch_size

        logging.debug(f"Successfully extracted subtitle track to {extracted_ass_path}")

        # 4. Translate the extracted file (preserving original format)
        translated_ass_path, final_batch_size = translate_ass_file(
            extracted_ass_path,
            api_manager,
            model_name,
            tmp_dir,
            mkv_path.stem,
            lang_code,
            subtitle_extension,
            batch_size,
            thinking,
            thinking_budget,
            keep_original,
            audio_file,
            extract_audio,
            mkv_path,  # video_path for audio extraction
            free_quota
        )

        # 5. Merge the translated subtitle back into a new MKV
        if translated_ass_path:
            merge_subtitles_to_mkv(mkv_path, translated_ass_path, output_dir)

        return lang_code, final_batch_size

    except FileNotFoundError:
        logger.error("mkvmerge or mkvextract not found. Please ensure MKVToolNix is installed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command-line tool failed for {mkv_path.name}: {e}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON output from mkvmerge for {mkv_path.name}.")
    except Exception as e:
        logger.error(f"Unexpected error while processing {mkv_path.name}: {e}")

    return None, batch_size


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Detects subtitles in .mkv files and translates them to Spanish using Google Gemini.\n\n"
                    "Usage:\n"
                    "  %(prog)s <file.mkv>              # Process a single file\n"
                    "  %(prog)s <directory>             # Process all .mkv files in directory",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--api-key", help="Primary API key for Google Gemini (or set GEMINI_API_KEY env var).")
    parser.add_argument("--api-key2", help="Secondary API key for additional quota (optional).")
    parser.add_argument("--model", default="gemini-2.5-pro",
                       help="The model to use for translation (default: 'gemini-2.5-pro'). "
                            "Note: Pro models may take longer for thinking, but have automatic timeout/retry.")
    parser.add_argument("--list-models", action="store_true",
                       help="List available Gemini models and exit.")
    parser.add_argument("--output-dir", type=Path, default=Path("translated_subs"),
                       help="Directory to save translated files.")
    parser.add_argument("--batch-size", type=int, default=300,
                       help="Number of lines to translate per batch (default: 300).")
    parser.add_argument("--thinking", action="store_true", default=True,
                       help="Enable thinking mode for better translations (default: enabled for Gemini 2.5+).")
    parser.add_argument("--no-thinking", action="store_true",
                       help="Disable thinking mode.")
    parser.add_argument("--thinking-budget", type=int, default=2048,
                       help="Token budget for thinking process (0-24576, only for flash models, default: 2048).")
    parser.add_argument("--progress-log", action="store_true",
                       help="Save translation progress to a log file.")
    parser.add_argument("--thoughts-log", action="store_true",
                       help="Save thinking process to a separate log file (requires thinking mode).")
    parser.add_argument("--no-colors", action="store_true",
                       help="Disable colored output.")
    parser.add_argument("--keep-original", action="store_true",
                       help="Keep original text as hidden comments in ASS subtitles (format: {Original: text}translation).")
    parser.add_argument("-a", "--audio-file", type=Path, default=None,
                       help="Audio file for gender-aware translation (MP3 format recommended).")
    parser.add_argument("--extract-audio", action="store_true",
                       help="Extract audio from video for gender-aware translation.")
    parser.add_argument("--paid-quota", action="store_true",
                       help="Remove artificial rate limits for paid quota users (allows faster processing).")
    parser.add_argument("input_path", nargs="?", default=None, type=Path,
                       help="Path to a single .mkv file or directory containing .mkv files.")

    args = parser.parse_args()

    # Handle thinking mode flags
    if args.no_thinking:
        args.thinking = False

    # Validate thinking_budget
    if args.thinking_budget < 0 or args.thinking_budget > 24576:
        logger.error("thinking-budget must be between 0 and 24576")
        sys.exit(1)

    # Info message for Pro models with thinking
    if args.thinking and "pro" in args.model.lower() and "flash" not in args.model.lower():
        logger.info(
            f"Using {args.model} with thinking mode enabled.\n"
            f"Pro models may take longer to think (5+ minutes per batch is normal).\n"
            f"Automatic timeout/retry enabled - will retry if thinking exceeds 5 minutes."
        )

    # Initialize enhanced logger settings
    logger.set_color_mode(not args.no_colors)
    logger.enable_file_logging(args.progress_log)
    logger.enable_thoughts_logging(args.thoughts_log)

    # Initialize API manager (with dual API key support)
    try:
        if args.api_key:
            api_manager = APIManager(args.api_key, args.api_key2)
        else:
            # Use environment variable for primary key
            import os
            env_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not env_key:
                raise ValueError("No API key provided")
            api_manager = APIManager(env_key, args.api_key2)

        # Create initial client
        client = api_manager.get_client()
        logging.debug(f"Initialized Google Gemini client")

        # Show dual API status
        if api_manager.has_secondary_key():
            logger.info(f"Dual API keys configured - automatic quota failover enabled")

    except Exception as e:
        logger.error(f"Failed to initialize Google Gemini client: {e}")
        logger.error("Make sure you provide --api-key or set GEMINI_API_KEY/GOOGLE_API_KEY environment variable")
        sys.exit(1)

    # Handle --list-models before checking other args
    if args.list_models:
        try:
            print("Available Gemini models for translation:")
            for m in client.models.list():
                print(f"  {m.name}")
        except Exception as e:
            logger.error(f"Failed to list models. Error: {e}")
            sys.exit(1)
        sys.exit(0)

    # Pre-execution checks
    if not args.input_path:
        logger.error("You must provide a path to an .mkv file or directory.")
        parser.print_help()
        sys.exit(1)

    if not check_mkvtoolnix():
        sys.exit(1)

    args.output_dir.mkdir(exist_ok=True)

    logging.debug(f"Using model: {args.model}")
    logging.debug(f"Batch size: {args.batch_size}")

    # File processing loop
    files_to_process = []
    if args.input_path.is_file():
        if args.input_path.suffix == ".mkv":
            files_to_process = [args.input_path]
            logging.debug(f"Processing single file: {args.input_path.resolve()}")
        else:
            logger.error(f"File must be an .mkv file: {args.input_path}")
            sys.exit(1)
    elif args.input_path.is_dir():
        files_to_process = sorted(list(args.input_path.glob("*.mkv")))
        logger.info(f"Searching for .mkv files in: {args.input_path.resolve()}")
    else:
        logger.error(f"Path does not exist: {args.input_path}")
        sys.exit(1)

    if not files_to_process:
        logger.warning("No .mkv files found to process.")
        return

    remembered_lang = None
    remembered_batch_size = args.batch_size
    for file_path in files_to_process:
        if file_path.suffix == ".mkv":
            # Set free_quota based on paid_quota flag (inverted logic)
            free_quota = not args.paid_quota

            chosen_lang, final_batch_size = process_mkv_file(
                file_path,
                args.output_dir,
                api_manager,
                args.model,
                remembered_lang,
                remembered_batch_size,
                args.thinking,
                args.thinking_budget,
                args.keep_original,
                args.audio_file,
                args.extract_audio,
                free_quota
            )
            # Remember language selection for subsequent files
            if chosen_lang and not remembered_lang:
                remembered_lang = chosen_lang
            # Remember batch size adjustment for subsequent files
            if final_batch_size and final_batch_size != remembered_batch_size:
                logger.info(f"Batch size adjusted to {final_batch_size} - will use for remaining files")
                remembered_batch_size = final_batch_size

    logger.success("--- All files processed. ---")


if __name__ == "__main__":
    main()
