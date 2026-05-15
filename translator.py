#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MKV Subtitle Translator
Refactored with gemini-translator-srt's proven batch-based architecture.
Extracts subtitles from MKV files, translates them using Google Gemini, and merges back.
"""

import argparse
import base64
import contextlib
import io
import logging
import math
import os
import statistics
import sys
import subprocess
import json
import re
import time
import threading
from pathlib import Path
from collections import Counter
import unicodedata


OLLAMA_LOCAL_TIMEOUT_SECONDS = 120
OLLAMA_CLOUD_TIMEOUT_SECONDS = 180

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None

import pysubs2

try:
    import json_repair
except ImportError:
    logging.error(
        "json_repair module not found. Install it with: pip install json-repair"
    )
    sys.exit(1)

try:
    from tools.audio_utils import prepare_audio
except ImportError:
    logging.error(
        "audio_utils module not found. Please ensure tools/audio_utils.py is available."
    )
    sys.exit(1)

try:
    from tools.ocr_utils import (
        build_srt_from_ocr_results,
        check_ffmpeg_tools,
        count_subtitle_bitmap_frames_full_stream,
        extract_ocr_frames,
        extract_subtitle_bitmap_frames_full_stream,
        extract_subtitle_bitmap_frames_at_timestamps,
        get_video_info,
        resolve_ocr_crop_filter,
        select_distinct_frame_samples,
        select_distinct_image_samples,
        _load_ocr_extract_cache,
        _save_ocr_extract_cache,
    )
except ImportError:
    logging.error(
        "ocr_utils module not found. Please ensure tools/ocr_utils.py is available."
    )
    sys.exit(1)

try:
    from tools.ocr_review_webui import review_ocr_text_in_webui
except ImportError:
    logging.error(
        "ocr_review_webui module not found. Please ensure tools/ocr_review_webui.py is available."
    )
    sys.exit(1)

try:
    from tools.process_utils import cleanup_tracked_processes
except ImportError:
    logging.error(
        "process_utils module not found. Please ensure tools/process_utils.py is available."
    )
    sys.exit(1)

# --- API Manager for Dual API Key Support ---

SUPPORTED_PROVIDERS = ("gemini", "ollama-local", "ollama-cloud")
GEMINI_DEFAULT_MODEL = "models/gemma-4-26b-a4b-it"
GEMINI_DEFAULT_AUDIO_MODEL = "models/gemini-3.1-flash-lite-preview"
OLLAMA_DEFAULT_MODEL = "llama3.2"
GEMINI_FREE_TIER_MIN_INTERVAL_SECONDS = 5.0
OCR_PROMPT_VERSION = 2
OCR_DEFAULT_FPS = 2.0
OCR_DEFAULT_FRAME_DIFF = 5.0
OCR_DEFAULT_RECHECK_EVERY = 3
OCR_DEFAULT_REQUEST_BATCH_SIZE = 24

_gemini_request_lock = threading.Lock()
_last_gemini_request_time = 0.0


def is_gemini_provider(provider):
    return provider == "gemini"


def is_ollama_provider(provider):
    return provider in {"ollama-local", "ollama-cloud"}


def is_ocr_capable_provider(provider):
    return is_gemini_provider(provider) or is_ollama_provider(provider)


def get_provider_display_name(provider):
    return {
        "gemini": "Google Gemini",
        "ollama-local": "Ollama (local)",
        "ollama-cloud": "Ollama Cloud",
    }.get(provider, provider)


def get_default_model(provider):
    if is_gemini_provider(provider):
        return GEMINI_DEFAULT_MODEL
    if provider == "ollama-local":
        return OLLAMA_DEFAULT_MODEL
    return None


def get_default_audio_model(provider):
    return GEMINI_DEFAULT_AUDIO_MODEL if is_gemini_provider(provider) else None


def get_default_base_url(provider):
    if provider == "ollama-cloud":
        return "https://ollama.com"
    if provider == "ollama-local":
        return os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    return None


def wait_for_gemini_request_slot(
    provider,
    min_interval=GEMINI_FREE_TIER_MIN_INTERVAL_SECONDS,
):
    """Throttle Gemini requests so OCR and translation stay paced."""
    global _last_gemini_request_time

    if not is_gemini_provider(provider):
        return 0.0

    with _gemini_request_lock:
        now = time.time()
        delay = max(0.0, min_interval - (now - _last_gemini_request_time))
        if delay > 0:
            time.sleep(delay)
        _last_gemini_request_time = time.time()
        return delay


def get_ocr_session_metadata(provider, model_name):
    return {
        "provider": provider,
        "model": model_name,
        "prompt_version": OCR_PROMPT_VERSION,
    }


def infer_image_mime_type(image_bytes):
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "application/octet-stream"


def group_ocr_samples(frame_samples, distinct_samples):
    """Group every extracted sample under the distinct sample that represents it."""
    if not frame_samples or not distinct_samples:
        return []

    distinct_by_index = {sample.index: sample for sample in distinct_samples}
    groups = []
    current_representative = None
    current_members = []

    for sample in frame_samples:
        if sample.index in distinct_by_index:
            if current_representative is not None and current_members:
                groups.append((current_representative, current_members))
            current_representative = distinct_by_index[sample.index]
            current_members = [sample]
            continue

        if current_representative is None:
            current_representative = distinct_samples[0]
        current_members.append(sample)

    if current_representative is not None and current_members:
        groups.append((current_representative, current_members))

    return groups


def expand_grouped_ocr_text(grouped_samples, representative_text_cache):
    """Copy OCR text from each representative sample to every grouped frame."""
    expanded_cache = {}
    for representative, members in grouped_samples:
        text = representative_text_cache.get(representative.index, "")
        for sample in members:
            expanded_cache[sample.index] = text
    return expanded_cache


def extract_ollama_model_names(response):
    """Extract model names from Ollama list() responses."""
    models = (
        response.get("models", [])
        if isinstance(response, dict)
        else getattr(response, "models", [])
    )
    names = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(name)
    return names


def extract_ollama_chunk_text(chunk):
    """Extract streamed text content from Ollama chat responses."""
    if isinstance(chunk, dict):
        return (chunk.get("message") or {}).get("content", "") or chunk.get(
            "response", ""
        )

    message = getattr(chunk, "message", None)
    if message is not None:
        if isinstance(message, dict):
            return message.get("content", "") or ""
        return getattr(message, "content", "") or ""

    response_text = getattr(chunk, "response", None)
    return response_text or ""


def extract_ollama_chunk_thinking(chunk):
    """Extract Ollama thinking trace content when available."""
    if isinstance(chunk, dict):
        return (chunk.get("message") or {}).get("thinking", "") or chunk.get(
            "thinking", ""
        )

    message = getattr(chunk, "message", None)
    if message is not None:
        if isinstance(message, dict):
            return message.get("thinking", "") or ""
        return getattr(message, "thinking", "") or ""

    return getattr(chunk, "thinking", "") or ""


def get_ollama_think_value(model_name, thinking, thinking_budget=2048):
    """Map generic thinking flags to the Ollama think parameter."""
    if not thinking:
        if "gpt-oss" in model_name.lower():
            return "low"
        return False

    if "gpt-oss" in model_name.lower():
        if thinking_budget >= 8192:
            return "high"
        if thinking_budget >= 2048:
            return "medium"
        return "low"

    return True


def load_dotenv_file(env_path):
    """Load simple KEY=VALUE pairs from a .env file without overriding real env vars."""
    loaded = False

    if not env_path.exists() or not env_path.is_file():
        return loaded

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded = True
    except Exception as e:
        logging.warning(f"Failed to load {env_path}: {e}")

    return loaded


def load_extra_context_file(context_path):
    """Load optional user-supplied translation context from a text file."""
    if not context_path:
        return None

    with open(context_path, "r", encoding="utf-8") as handle:
        loaded = handle.read().strip()
    return loaded or None


def mask_secret(value):
    """Return a masked representation of a secret value."""
    if not value:
        return "not set"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:3]}...{value[-3:]}"


def is_configured_model_available(configured_model, available_models):
    """Check whether a configured model matches an available model entry."""
    if not configured_model:
        return False

    for available_model in available_models:
        if available_model == configured_model:
            return True
        if available_model.startswith(f"{configured_model}:"):
            return True
        if configured_model.startswith(f"{available_model}:"):
            return True

    return False


def get_available_model_names(client, provider):
    """Return model names visible to the current provider client."""
    if is_gemini_provider(provider):
        return [model.name for model in client.models.list()]
    return extract_ollama_model_names(client.list())


def test_provider_roundtrip(client, provider, model_name):
    """Run a minimal API call against the configured provider/model."""
    if is_gemini_provider(provider):
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly OK.",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=8,
            ),
        )
        return (response.text or "").strip()

    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with exactly OK."}],
        options={"temperature": 0, "num_predict": 8},
    )
    return extract_ollama_chunk_text(response).strip()


def check_command_version(command):
    """Check whether a CLI tool is available and return a short status string."""
    try:
        version_flag = "-version" if command == "ffmpeg" else "--version"
        result = subprocess.run(
            [command, version_flag], capture_output=True, text=True, encoding="utf-8"
        )
        if result.returncode == 0:
            output = (result.stdout or result.stderr or "").strip().splitlines()
            return True, output[0] if output else "OK"
        return False, (result.stderr or result.stdout or "failed").strip()
    except FileNotFoundError:
        return False, "not found"
    except Exception as e:
        return False, str(e)


def print_subtitle_track_report(mkv_path):
    """Print subtitle track metadata for a single MKV file."""
    print("\nSubtitle tracks:")

    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(mkv_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        mkv_info = json.loads(result.stdout)
        subtitle_tracks = [
            track
            for track in mkv_info.get("tracks", [])
            if track.get("type") == "subtitles"
        ]

        if not subtitle_tracks:
            print("  No subtitle tracks found.")
            return True

        for track in subtitle_tracks:
            props = track.get("properties", {})
            track_id = track.get("id")
            lang = props.get("language") or "unknown"
            name = props.get("track_name") or "(no name)"
            codec = props.get("codec_id") or "unknown"
            default_flag = props.get("default_track", False)
            forced_flag = props.get("forced_track", False)

            print(
                f"  - id={track_id} lang={lang} codec={codec} default={default_flag} forced={forced_flag} name={name}"
            )

        return True
    except Exception as e:
        print(f"  Failed to inspect subtitle tracks: {e}")
        return False


def run_doctor(
    args, api_manager=None, client=None, init_error=None, dotenv_loaded=False
):
    """Print runtime configuration and perform basic provider/tool health checks."""
    provider_name = get_provider_display_name(args.provider)
    api_key_value = args.api_key
    if not api_key_value:
        if is_gemini_provider(args.provider):
            api_key_value = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        elif args.provider == "ollama-cloud":
            api_key_value = os.environ.get("OLLAMA_API_KEY")

    effective_base_url = (
        api_manager.base_url
        if api_manager is not None
        else args.base_url
        or os.environ.get("LLM_BASE_URL")
        or get_default_base_url(args.provider)
    )

    print("=== mkv-translator doctor ===")
    print(f"Provider: {provider_name}")
    print(f"Model: {args.model or '(not set)'}")
    print(f"Audio model: {args.audio_model or '(not set)'}")
    print(f"Base URL: {effective_base_url or '(n/a)'}")
    print(f"Primary API key: {mask_secret(api_key_value)}")
    print(f"Secondary API key: {mask_secret(args.api_key2)}")
    print(f".env loaded: {'yes' if dotenv_loaded else 'no'}")
    print(f"Working directory: {Path.cwd()}")

    mkvmerge_ok, mkvmerge_info = check_command_version("mkvmerge")
    mkvextract_ok, mkvextract_info = check_command_version("mkvextract")
    ffmpeg_ok, ffmpeg_info = check_command_version("ffmpeg")

    print("\nTools:")
    print(f"  mkvmerge: {'OK' if mkvmerge_ok else 'FAIL'} - {mkvmerge_info}")
    print(f"  mkvextract: {'OK' if mkvextract_ok else 'FAIL'} - {mkvextract_info}")
    print(f"  ffmpeg: {'OK' if ffmpeg_ok else 'FAIL'} - {ffmpeg_info}")

    available_models = []
    provider_ok = init_error is None and client is not None

    print("\nProvider checks:")
    if not provider_ok:
        print(f"  Client init: FAIL - {init_error}")
    else:
        print("  Client init: OK")
        try:
            available_models = get_available_model_names(client, args.provider)
            print(f"  Model listing: OK - {len(available_models)} models visible")
            if available_models:
                preview = ", ".join(available_models[:10])
                suffix = " ..." if len(available_models) > 10 else ""
                print(f"  Model preview: {preview}{suffix}")
        except Exception as e:
            print(f"  Model listing: FAIL - {e}")

        if args.model:
            model_available = is_configured_model_available(
                args.model, available_models
            )
            availability_text = "yes" if model_available else "no/unknown"
            print(f"  Configured model visible: {availability_text}")

            try:
                roundtrip_text = test_provider_roundtrip(
                    client, args.provider, args.model
                )
                print(f"  API roundtrip: OK - {roundtrip_text or '(empty text)'}")
            except Exception as e:
                print(f"  API roundtrip: FAIL - {e}")
                provider_ok = False

    if (
        args.input_path
        and args.input_path.is_file()
        and args.input_path.suffix == ".mkv"
    ):
        tracks_ok = print_subtitle_track_report(args.input_path)
    else:
        tracks_ok = True
        if args.input_path:
            print(
                "\nSubtitle tracks: skipped (provide a single .mkv file to inspect tracks)"
            )

    overall_ok = mkvmerge_ok and mkvextract_ok and provider_ok and tracks_ok
    print(f"\nOverall status: {'OK' if overall_ok else 'FAIL'}")
    return overall_ok


def resolve_mkv_input_files(input_path):
    """Resolve a file or directory input into a list of MKV files."""
    if not input_path:
        raise ValueError("You must provide a path to an .mkv file or directory.")

    if input_path.is_file():
        if input_path.suffix == ".mkv":
            logging.debug(f"Processing single file: {input_path.resolve()}")
            return [input_path]
        raise ValueError(f"File must be an .mkv file: {input_path}")

    if input_path.is_dir():
        logger.info(f"Searching for .mkv files in: {input_path.resolve()}")
        return sorted(list(input_path.glob("*.mkv")))

    raise ValueError(f"Path does not exist: {input_path}")


def resolve_input_files(input_path):
    """Resolve a file or directory input into supported MKV/raw subtitle files."""
    if not input_path:
        raise ValueError(
            "You must provide a path to an .mkv, .ass, .ssa, or .srt file or directory."
        )

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS:
            logging.debug(f"Processing single input file: {input_path.resolve()}")
            return [input_path]
        raise ValueError(
            f"Unsupported input file: {input_path}. Supported: .mkv, .ass, .ssa, .srt"
        )

    if input_path.is_dir():
        logger.info(f"Searching for supported input files in: {input_path.resolve()}")
        return sorted(
            [
                entry
                for entry in input_path.iterdir()
                if entry.is_file() and entry.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
            ]
        )

    raise ValueError(f"Path does not exist: {input_path}")


def is_permanent_ollama_error(error_msg):
    """Detect non-retryable Ollama errors that should fail fast."""
    lower_msg = error_msg.lower()

    auth_markers = [
        "unauthorized",
        "forbidden",
        "authentication",
        "invalid api key",
        "permission denied",
    ]
    request_markers = ["bad request", "unsupported format", "invalid format"]
    missing_model = "model" in lower_msg and (
        "not found" in lower_msg
        or "does not exist" in lower_msg
        or "no such" in lower_msg
        or "pull" in lower_msg
    )

    return (
        any(marker in lower_msg for marker in auth_markers)
        or any(marker in lower_msg for marker in request_markers)
        or missing_model
    )


class APIManager:
    """
    Manages dual API key support for handling quota limitations.
    Matches gemini-srt-translator's _switch_api and _get_client pattern.
    """

    def __init__(self, provider, api_key=None, api_key2=None, base_url=None):
        """
        Initialize API manager with primary and optional secondary API key.

        Args:
            provider: Provider name (gemini, ollama-local, ollama-cloud)
            api_key: Primary API key
            api_key2: Secondary API key (optional, for quota failover)
        """
        self.provider = provider
        self.api_key = api_key
        self.api_key2 = api_key2
        self.current_api_key = api_key
        self.current_api_number = 1
        self.backup_api_number = 2
        self.base_url = base_url or get_default_base_url(provider)

    def get_client(self):
        """
        Create and return a client using the currently active provider config.

        Returns:
            Provider client configured with current settings
        """
        if is_gemini_provider(self.provider):
            if genai is None:
                raise ImportError(
                    "google-genai is not installed. Run: pip install -r requirements.txt"
                )
            return genai.Client(api_key=self.current_api_key)

        if is_ollama_provider(self.provider):
            if OllamaClient is None:
                raise ImportError(
                    "ollama is not installed. Run: pip install -r requirements.txt"
                )

            headers = None
            if self.current_api_key:
                headers = {"Authorization": f"Bearer {self.current_api_key}"}

            timeout = (
                OLLAMA_CLOUD_TIMEOUT_SECONDS
                if self.provider == "ollama-cloud"
                else OLLAMA_LOCAL_TIMEOUT_SECONDS
            )

            return OllamaClient(host=self.base_url, headers=headers, timeout=timeout)

        raise ValueError(f"Unsupported provider: {self.provider}")

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


@contextlib.contextmanager
def suppress_stderr_output():
    """Temporarily silence stderr noise from third-party clients."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield


def handle_process_termination(signum, frame):
    """Ensure tracked external tools are stopped on abrupt termination."""
    cleanup_tracked_processes()
    raise SystemExit(1)


# Import progress display module
try:
    from tools.progress_display import (
        progress_bar,
        progress_complete,
        clear_progress,
        info_with_progress,
        warning_with_progress,
        error_with_progress,
        success_with_progress,
    )
except ImportError:
    logging.error(
        "progress_display module not found. Make sure tools/progress_display.py is available."
    )
    sys.exit(1)

# Import enhanced logger module
try:
    from tools import logger
except ImportError:
    logging.error(
        "logger module not found. Make sure tools/logger.py is available."
    )
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress verbose HTTP logging from google/urllib and all submodules
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)
logging.getLogger("google.genai._api_client").setLevel(logging.ERROR)
logging.getLogger("google.genai._automatic_function_calling_util").setLevel(
    logging.ERROR
)
logging.getLogger("google.ai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

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

    Also handles truncated/corrupted placeholders from LLM output:
    - <<<ASS_HLB, <<<ASS_HL, etc. → \\N
    """
    # First try exact matches
    text = text.replace(ASS_HARD_LINEBREAK_PLACEHOLDER, ASS_HARD_LINEBREAK)
    text = text.replace(ASS_SOFT_LINEBREAK_PLACEHOLDER, ASS_SOFT_LINEBREAK)
    text = text.replace(ASS_HARD_SPACE_PLACEHOLDER, ASS_HARD_SPACE)

    # Handle truncated/corrupted placeholders with regex
    # Gemini sometimes truncates placeholders at various points
    # Match from most specific to least specific to avoid partial replacements

    # Full or partial hard linebreak: <<<ASS_HLB>>>, <<<ASS_HL, <<<ASS_H, <<<ASS_, <<<ASS, <<<
    text = re.sub(r"<<<(?:ASS_?(?:H(?:LB?)?)?)?(?:>>>|>>|>)?", r"\\N", text)

    # Note: After the above regex, soft linebreak and hard space patterns would already
    # be consumed. But we keep them for cases where S or HS appears (unlikely truncation path)
    # These would only match if somehow <<<ASS_S or <<<ASS_HS survived the first pass
    text = re.sub(r"<<<ASS_S(?:LB?)?(?:>>>|>>|>)?", r"\\n", text)
    text = re.sub(r"<<<ASS_HS(?:P)?(?:>>>|>>|>)?", r"\\h", text)

    return text


# --- ASS Formatting Helper Functions ---


def remove_formatting(text):
    """Remove ASS formatting tags from text."""
    return re.sub(r"\{.*?\}", "", text).strip()


def restore_formatting(original_text, translated_plain_text):
    """
    Restore ASS formatting tags from original text to translated text.
    Preserves the structure of formatting tags from the original.
    """
    formatting_tags = re.findall(r"\{[^}]+\}", original_text)

    if not formatting_tags:
        return translated_plain_text

    # Prepend all formatting tags to the translated text
    formatting_prefix = "".join(formatting_tags)
    return f"{formatting_prefix}{translated_plain_text}"


def strip_sdh_elements(text):
    """
    Remove SDH (Subtitles for Deaf and Hard of Hearing) elements from subtitle text.

    Removes:
    - Sound effects in brackets/parentheses: [door slams], (thunder rumbling), （日本語）, *footsteps*, /rustling/
    - Speaker identification: JOHN:, Mary:, [narrator]:
    - Music symbols (preserves lyrics): ♪ lyrics ♪ → lyrics
    - Watermarks and credits: www.site.com, Subtitled by..., OpenSubtitles

    Preserves:
    - Actual dialogue content
    - Music lyrics text (only symbols removed)
    - ASS formatting tags {...} (handled separately by remove_formatting)
    - Time patterns like 12:30 (not mistaken for speaker names)

    Args:
        text: Subtitle text to process

    Returns:
        Text with SDH elements removed
    """
    if not text:
        return text

    # 1. Remove sound effects in various bracket styles (including Japanese fullwidth)
    text = re.sub(r"\[[^\]]*\]", "", text)  # [sound effect]
    text = re.sub(r"\([^\)]*\)", "", text)  # (sound effect)
    text = re.sub(r"（[^）]*）", "", text)  # （Japanese fullwidth parentheses）
    text = re.sub(r"\*[^\*]+\*", "", text)  # *sound effect*
    text = re.sub(r"/[^/]+/", "", text)  # /sound effect/

    # 2. Remove music symbols but preserve lyrics
    text = re.sub(r"♪([^♪]*)♪", r"\1", text)  # ♪ lyrics ♪ → lyrics
    text = re.sub(r"[#♪♫]+", "", text)  # Remove standalone music symbols

    # 3. Remove speaker names (ALL CAPS or Title Case) with colon
    # Pattern explanation:
    # - (?:\[[^\]]+\]\s*)? : Optional bracketed speaker like [narrator]
    # - (?:-\s?)? : Optional leading dash
    # - ([A-Z][A-Z0-9'\s]*|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*) : ALL CAPS or Title Case name
    # - :\s? : Colon followed by optional space
    # - (?![0-9]) : NOT followed by a number (avoids matching time like 12:30)
    text = re.sub(
        r"^(?:\[[^\]]+\]\s*)?(?:-\s?)?([A-Z][A-Z0-9\'\s]*|[A-Z][a-z]+(?:\s[A-Z][a-z]+)*):(?![0-9])\s?",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 4. Remove watermarks and credits
    text = re.sub(r"(?i)\b(subtitle|caption|sync)s?\s+(by|from)\b.*", "", text)
    text = re.sub(r"(?i)www\.|https?://|\.(?:com|org|net)", "", text)
    text = re.sub(r"(?i)(opensubtitles|subscene|addic7ed|podnapisi)", "", text)

    # 5. Remove Unicode directional marks (LRM, RLM, etc.) and clean up whitespace
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_sdh_only_line(text):
    """
    Detect if a subtitle line contains ONLY SDH content (no actual dialogue).

    Returns True for lines that should be deleted entirely:
    - Entire line is bracketed: [MUSIC PLAYING], （音）
    - Only music symbols: ♪♪♪, ###
    - Only dashes: ---, ———
    - Speaker name only: JOHN:, MARY:
    - Empty or whitespace-only (including after Unicode marks removed)

    Args:
        text: Subtitle text to check

    Returns:
        True if line is SDH-only, False if it contains dialogue
    """
    if not text:
        return True

    # Remove Unicode directional marks before checking
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]+", "", text)
    text = text.strip()

    if not text:
        return True

    # Check for full-line SDH patterns
    sdh_only_patterns = [
        r"^[\[\(（].*[\]\)）]$",  # Entire line is bracketed (ASCII or Japanese)
        r"^\s*[♪#♫\s]+$",  # Only music symbols and whitespace
        r"^\s*[-–—]+\s*$",  # Only dashes
        r"^[A-Z\s]+:$",  # Speaker name only (ALL CAPS with colon)
    ]

    for pattern in sdh_only_patterns:
        if re.match(pattern, text):
            return True

    return False


def _normalize_inline_color(match):
    """Normalize a single inline color tag (\\c, \\1c, etc.)."""
    tag = match.group(1)  # 'c' or '1c' or '2c' or '3c' or '4c'
    color_hex = match.group(2)  # Just the hex digits

    # Remove any non-hex characters that slipped through
    clean_hex = re.sub(r"[^0-9A-Fa-f]", "", color_hex)

    if not clean_hex:
        # Invalid/empty color - remove the tag entirely
        return ""

    # Parse hex value
    try:
        color_value = int(clean_hex, 16)
    except ValueError:
        # Should never happen after cleaning, but be safe
        return ""

    # Normalize to proper length (pad with zeros if needed)
    # 6 digits = RGB, 8 digits = ARGB
    if len(clean_hex) <= 6:
        # RGB format - pad to 6 digits
        normalized = f"&H{color_value:06X}&"
    else:
        # ARGB format - pad to 8 digits
        normalized = f"&H{color_value:08X}&"

    return f"\\{tag}{normalized}"


def _normalize_style_line(match):
    """Normalize colors in a Style: line."""
    line = match.group(0)

    # Find and normalize each color value in the style
    # Pattern: color values start with &H or H or just hex digits
    def fix_style_color(color_match):
        color_str = color_match.group(0)

        # Extract just hex digits
        clean_hex = re.sub(r"[^0-9A-Fa-f]", "", color_str)

        if not clean_hex or len(clean_hex) > 8:
            # Invalid - use white with full opacity as safe default
            return "&H00FFFFFF"

        try:
            color_value = int(clean_hex, 16)
            # Style colors should be 8 digits (AABBGGRR)
            # If shorter, assume RGB and add full opacity (00)
            if len(clean_hex) <= 6:
                return f"&H00{color_value:06X}"
            else:
                return f"&H{color_value:08X}"
        except ValueError:
            return "&H00FFFFFF"  # Safe default

    # Match color values in style (anywhere in the line after "Style:")
    # These are typically &HAABBGGRR or malformed versions
    # CRITICAL: &? before lookahead to match trailing & (e.g., FFFFFF&,)
    line = re.sub(
        r"(?:&H?|H)?[0-9A-Fa-f]{6,8}&?(?=\s*,|\s*$)",
        fix_style_color,
        line,
        flags=re.IGNORECASE,
    )

    return line


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
        with open(ass_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for line in lines:
            # Skip lines that contain vector drawing commands (\p1, \p2, etc.)
            # Drawing data contains numeric coordinates that can be confused with color values
            if re.search(r'\\p[1-4](?!\d)', line):
                # This line has drawing commands - skip color normalization entirely
                # to avoid corrupting coordinate data
                new_lines.append(line)
                continue

            new_line = line

            # === Inline Color Tag Normalization ===
            new_line = re.sub(
                r"\\(\d?c)(?:&H?)?([0-9A-Fa-f]+)&?(?![0-9A-Fa-f])",
                _normalize_inline_color,
                new_line,
                flags=re.IGNORECASE,
            )

            # === Style Line Color Normalization ===
            if new_line.startswith("Style:"):
                new_line = re.sub(
                    r"^Style:.*$",
                    _normalize_style_line,
                    new_line,
                    flags=re.MULTILINE | re.IGNORECASE,
                )

            if new_line != line:
                changed = True
            new_lines.append(new_line)

        # === Cleanup Pass ===
        # Remove any orphaned/incomplete color tags that might cause issues
        # These are patterns that look like color tags but are too malformed to fix
        result = "".join(new_lines)

        # Remove standalone \c or \Xc without any hex following
        result = re.sub(r"\\(\d?c)(?![&0-9A-Fa-f])", "", result)

        # Fix any remaining double ampersands
        result = re.sub(r"&&+", "&", result)

        # === Write if changed ===
        original_content = "".join(lines)
        if result != original_content:
            with open(ass_path, "w", encoding="utf-8-sig") as f:
                f.write(result)
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
        subprocess.run(
            ["mkvmerge", "--version"], check=True, capture_output=True, text=True
        )
        subprocess.run(
            ["mkvextract", "--version"], check=True, capture_output=True, text=True
        )
        logging.debug("MKVToolNix command-line tools found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "MKVToolNix not found. Please install it from https://mkvtoolnix.download/"
        )
        logger.error("Ensure 'mkvmerge' and 'mkvextract' are in your system's PATH.")
        return False


SUPPORTED_SUBTITLE_CODECS = {
    "S_TEXT/ASS": ".ass",
    "S_TEXT/SSA": ".ssa",
    "S_TEXT/UTF8": ".srt",
}

OCR_SUBTITLE_CODECS = {"S_HDMV/PGS"}
RAW_TEXT_SUBTITLE_EXTENSIONS = {".ass", ".ssa", ".srt"}
SUPPORTED_INPUT_EXTENSIONS = {".mkv"} | RAW_TEXT_SUBTITLE_EXTENSIONS


def normalize_track_language(lang_code):
    """Normalize language names used in prompts and track grouping."""
    normalized = (lang_code or "").strip().lower()
    aliases = {
        "en": "eng",
        "eng": "eng",
        "english": "eng",
        "de": "de",
        "ger": "de",
        "deu": "de",
        "german": "de",
        "ja": "ja",
        "jpn": "ja",
        "jp": "ja",
        "japanese": "ja",
        "fr": "fr",
        "fra": "fr",
        "fre": "fr",
        "french": "fr",
    }
    return aliases.get(normalized, normalized)


def is_mkv_path(input_path):
    return input_path.suffix.lower() == ".mkv"


def is_raw_text_subtitle_path(input_path):
    return input_path.suffix.lower() in RAW_TEXT_SUBTITLE_EXTENSIONS


def prompt_yes_no(prompt, default=False):
    """Prompt the user for a yes/no answer."""
    default_text = "Y/n" if default else "y/N"

    while True:
        try:
            choice = input(f"{prompt} [{default_text}]: ").strip().lower()
        except EOFError:
            print()
            return default
        if not choice:
            return default
        if choice in {"y", "yes", "s", "si"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_subtitle_language(
    found_tracks, prompt_text, default_lang=None, exclude=None
):
    """Prompt the user to choose a subtitle language from discovered tracks."""
    exclude = exclude or set()
    lang_options = [lang for lang in found_tracks.keys() if lang not in exclude]

    if not lang_options:
        return None

    if default_lang not in lang_options:
        default_lang = lang_options[0]

    while True:
        choice = (
            input(
                f"{prompt_text} ({'/'.join(lang_options)}) [default: {default_lang}]: "
            )
            .strip()
            .lower()
        )

        if not choice:
            choice = default_lang

        if choice in lang_options:
            return choice

        print(f"Invalid choice. Please enter one of: {', '.join(lang_options)}.")


def get_track_display_name(track, supported_codecs=None):
    """Build a short human-readable label for a subtitle track."""
    props = track.get("properties", {})
    codec = props.get("codec_id") or "unknown"
    name = props.get("track_name") or "(no name)"
    default_flag = props.get("default_track", False)
    forced_flag = props.get("forced_track", False)
    supported_codecs = supported_codecs or SUPPORTED_SUBTITLE_CODECS
    supported = codec in supported_codecs

    flags = []
    if default_flag:
        flags.append("default")
    if forced_flag:
        flags.append("forced")
    flags.append("supported" if supported else "unsupported")

    return f"id={track.get('id')} codec={codec} name={name} flags={','.join(flags)}"


def choose_track_for_language(lang_code, lang_tracks, supported_codecs=None):
    """Choose a specific track within one language bucket."""
    if not lang_tracks:
        return None

    supported_codecs = supported_codecs or SUPPORTED_SUBTITLE_CODECS

    supported_tracks = [
        track
        for track in lang_tracks
        if track.get("properties", {}).get("codec_id") in supported_codecs
    ]

    if len(lang_tracks) == 1:
        only_track = lang_tracks[0]
        codec = only_track.get("properties", {}).get("codec_id")
        if codec not in supported_codecs:
            logger.warning(
                f"The only '{lang_code}' subtitle track is unsupported ({codec})."
            )
            return None
        return only_track

    print(f"Found multiple '{lang_code}' subtitle tracks:")
    for idx, track in enumerate(lang_tracks, start=1):
        print(f"  {idx}. {get_track_display_name(track, supported_codecs=supported_codecs)}")

    default_track = supported_tracks[0] if supported_tracks else lang_tracks[0]
    default_choice = str(lang_tracks.index(default_track) + 1)

    while True:
        choice = input(
            f"Select {lang_code} track number [default: {default_choice}]: "
        ).strip()

        if not choice:
            choice = default_choice

        if choice.isdigit() and 1 <= int(choice) <= len(lang_tracks):
            selected_track = lang_tracks[int(choice) - 1]
            codec = selected_track.get("properties", {}).get("codec_id")
            if codec not in supported_codecs:
                print(
                    "That track format is not supported for translation. Choose another."
                )
                continue
            return selected_track

        print(
            f"Invalid choice. Please enter a number between 1 and {len(lang_tracks)}."
        )


def get_supported_language_options(found_tracks, exclude=None):
    """Return languages that have at least one supported subtitle track."""
    exclude = exclude or set()
    supported_langs = []

    for lang, lang_tracks in found_tracks.items():
        if lang in exclude:
            continue
        if any(
            track.get("properties", {}).get("codec_id") in SUPPORTED_SUBTITLE_CODECS
            for track in lang_tracks
        ):
            supported_langs.append(lang)

    return supported_langs


def get_language_options_for_codecs(found_tracks, supported_codecs, exclude=None):
    """Return languages that have at least one track using the requested codecs."""
    exclude = exclude or set()
    supported_langs = []

    for lang, lang_tracks in found_tracks.items():
        if lang in exclude:
            continue
        if any(
            track.get("properties", {}).get("codec_id") in supported_codecs
            for track in lang_tracks
        ):
            supported_langs.append(lang)

    return supported_langs


def build_found_subtitle_tracks(tracks):
    """Group recognized subtitle tracks by language bucket."""
    track_map = {"eng": [], "de": [], "ja": [], "fr": []}

    for track in tracks:
        if track.get("type") == "subtitles":
            lang = track.get("properties", {}).get("language")
            if lang == "eng":
                track_map["eng"].append(track)
            elif lang in ["de", "ger"]:
                track_map["de"].append(track)
            elif lang in ["ja", "jpn"]:
                track_map["ja"].append(track)
            elif lang in ["fr", "fre", "fra"]:
                track_map["fr"].append(track)

    return {lang: lang_tracks for lang, lang_tracks in track_map.items() if lang_tracks}


def select_original_injection_track(tracks, remembered_lang=None):
    """Select one source subtitle track to inject as {Original: ...}."""
    found_tracks = build_found_subtitle_tracks(tracks)
    if not found_tracks:
        return None, None

    supported_langs = get_supported_language_options(found_tracks)
    if not supported_langs:
        logger.warning(
            "No supported text subtitle tracks found for original injection."
        )
        return None, None

    if remembered_lang in supported_langs:
        lang_code = remembered_lang
        logger.info(
            f"Automatically selecting original-comment language {lang_code} based on previous choice."
        )
    elif len(supported_langs) == 1:
        lang_code = supported_langs[0]
    else:
        default_lang = "eng" if "eng" in supported_langs else supported_langs[0]
        print(
            f"Available languages for Original injection: {', '.join(supported_langs)}"
        )
        lang_code = prompt_subtitle_language(
            found_tracks,
            "Select language to inject as Original",
            default_lang=default_lang,
            exclude={lang for lang in found_tracks if lang not in supported_langs},
        )

    track = choose_track_for_language(lang_code, found_tracks[lang_code])
    if track is None:
        return None, None

    return track, lang_code


def select_subtitle_tracks(
    tracks, remembered_lang=None, remembered_secondary_lang=None
):
    """
    Select primary subtitle language and optional secondary context language.
    """
    found_tracks = build_found_subtitle_tracks(tracks)

    if not found_tracks:
        return None, None, None, None

    if remembered_lang and remembered_lang in found_tracks:
        primary_lang = remembered_lang
        logger.info(
            f"Automatically selecting {remembered_lang} based on previous choice."
        )
    elif len(found_tracks) == 1:
        primary_lang = list(found_tracks.keys())[0]
        logging.debug(f"Found single subtitle track: {primary_lang}.")
    else:
        print(f"Found multiple subtitle languages: {', '.join(found_tracks.keys())}")
        primary_default = (
            "fr"
            if "fr" in found_tracks
            else ("eng" if "eng" in found_tracks else list(found_tracks.keys())[0])
        )
        primary_lang = prompt_subtitle_language(
            found_tracks,
            "Select primary language to translate from",
            default_lang=primary_default,
        )
        logger.info(f"User selected primary language {primary_lang}.")

    primary_track = choose_track_for_language(primary_lang, found_tracks[primary_lang])
    if primary_track is None:
        supported_primary_langs = get_supported_language_options(found_tracks)
        fallback_langs = [
            lang for lang in supported_primary_langs if lang != primary_lang
        ]

        if fallback_langs:
            logger.warning(
                f"Primary language '{primary_lang}' has no supported text subtitle track. Choose another language."
            )
            primary_lang = prompt_subtitle_language(
                found_tracks,
                "Select primary language to translate from",
                default_lang=fallback_langs[0],
                exclude={lang for lang in found_tracks if lang not in fallback_langs},
            )
            primary_track = choose_track_for_language(
                primary_lang, found_tracks[primary_lang]
            )

    if primary_track is None:
        return None, None, None, None

    secondary_lang = None
    secondary_track = None
    remaining_langs = [lang for lang in found_tracks.keys() if lang != primary_lang]

    if remaining_langs:
        if remembered_secondary_lang and remembered_secondary_lang in remaining_langs:
            secondary_lang = remembered_secondary_lang
            logger.info(
                f"Automatically selecting secondary language {secondary_lang} based on previous choice."
            )
        else:
            default_use_secondary = primary_lang == "fr" and "eng" in remaining_langs
            if prompt_yes_no(
                "Use a secondary subtitle language as translation context?",
                default=default_use_secondary,
            ):
                secondary_default = (
                    "eng" if "eng" in remaining_langs else remaining_langs[0]
                )
                secondary_lang = prompt_subtitle_language(
                    found_tracks,
                    "Select secondary context language",
                    default_lang=secondary_default,
                    exclude={primary_lang},
                )
                logger.info(
                    f"User selected secondary context language {secondary_lang}."
                )

        if secondary_lang:
            secondary_track = choose_track_for_language(
                secondary_lang, found_tracks[secondary_lang]
            )
            if secondary_track is None:
                logger.warning(
                    f"Secondary language '{secondary_lang}' has no supported text subtitle track. Skipping secondary context."
                )
                secondary_lang = None

    return primary_track, primary_lang, secondary_track, secondary_lang


def get_ffmpeg_subtitle_stream_index(tracks, selected_track):
    """Map an mkvmerge subtitle track to ffmpeg subtitle stream order."""
    subtitle_tracks = [track for track in tracks if track.get("type") == "subtitles"]
    for subtitle_index, track in enumerate(subtitle_tracks):
        if track.get("id") == selected_track.get("id"):
            return subtitle_index
    raise ValueError("Could not map selected subtitle track to ffmpeg subtitle stream index")


def select_ocr_subtitle_track(tracks, preferred_lang=None):
    """Select an image subtitle track for OCR rendering when available."""
    found_tracks = build_found_subtitle_tracks(tracks)
    if not found_tracks:
        return None, None, None

    supported_langs = get_language_options_for_codecs(found_tracks, OCR_SUBTITLE_CODECS)
    if not supported_langs:
        return None, None, None

    normalized_preferred = normalize_track_language(preferred_lang)
    if normalized_preferred in supported_langs:
        lang_code = normalized_preferred
        logger.info(f"Using OCR subtitle language {lang_code} from --ocr-lang.")
    elif len(supported_langs) == 1:
        lang_code = supported_langs[0]
    else:
        default_lang = (
            normalized_preferred
            if normalized_preferred in supported_langs
            else ("ja" if "ja" in supported_langs else supported_langs[0])
        )
        print(f"Found OCR-capable subtitle languages: {', '.join(supported_langs)}")
        lang_code = prompt_subtitle_language(
            found_tracks,
            "Select subtitle language to OCR from",
            default_lang=default_lang,
            exclude={lang for lang in found_tracks if lang not in supported_langs},
        )

    track = choose_track_for_language(
        lang_code, found_tracks[lang_code], supported_codecs=OCR_SUBTITLE_CODECS
    )
    if track is None:
        return None, None, None

    subtitle_stream_index = get_ffmpeg_subtitle_stream_index(tracks, track)
    return track, lang_code, subtitle_stream_index


def get_ocr_review_session_file(mkv_path):
    return Path("tmp") / f"{mkv_path.stem}.ocr-review" / "session.json"


def load_saved_ocr_text_cache(mkv_path, samples, session_metadata=None):
    session_file = get_ocr_review_session_file(mkv_path)
    if not session_file.exists():
        return {}

    expected_source_indexes = [getattr(sample, "index", position) for position, sample in enumerate(samples)]

    try:
        with open(session_file, "r", encoding="utf-8") as handle:
            session_data = json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to load saved OCR review session: {exc}")
        return {}

    if session_data.get("input_file") != str(mkv_path):
        return {}

    if session_metadata:
        saved_metadata = session_data.get("ocr_metadata") or {}
        for key, expected_value in session_metadata.items():
            if saved_metadata.get(key) != expected_value:
                return {}

    items = session_data.get("items", [])
    saved_source_indexes = []
    for position, item in enumerate(items):
        source_indexes = item.get("source_indexes")
        if source_indexes:
            saved_source_indexes.extend(source_indexes)
        else:
            saved_source_indexes.append(item.get("source_index", position))

    if saved_source_indexes != expected_source_indexes:
        return {}

    saved_text_cache = {}
    for position, item in enumerate(items):
        source_indexes = item.get("source_indexes") or [item.get("source_index", position)]
        item_text = item.get("original_text", item.get("text", ""))
        for source_index in source_indexes:
            if source_index is None:
                continue
            saved_text_cache[source_index] = item_text

    return saved_text_cache


def extract_subtitle_track(mkv_path, track, tmp_dir, lang_code, label=""):
    """Extract a specific subtitle track from an MKV file."""
    codec_id = track.get("properties", {}).get("codec_id")
    if codec_id not in SUPPORTED_SUBTITLE_CODECS:
        logger.warning(f"Unsupported subtitle format '{codec_id}' in {mkv_path.name}.")
        return None, None

    subtitle_extension = SUPPORTED_SUBTITLE_CODECS[codec_id]
    suffix = f".{label}" if label else ""
    extracted_subtitle_path = (
        tmp_dir / f"{mkv_path.stem}.{lang_code}{suffix}{subtitle_extension}"
    )
    subtitle_track_id = track["id"]

    mkvextract_cmd = [
        "mkvextract",
        "tracks",
        str(mkv_path),
        f"{subtitle_track_id}:{extracted_subtitle_path}",
    ]
    logging.debug(
        f"Extracting track {subtitle_track_id} ({lang_code}, {codec_id}) to: {extracted_subtitle_path}"
    )

    result = subprocess.run(
        mkvextract_cmd, capture_output=True, text=True, encoding="utf-8"
    )

    if (
        "Error in the Matroska file structure" in result.stdout
        or "Resync failed" in result.stdout
    ):
        logger.warning(f"MKV file {mkv_path.name} appears to be corrupted. Skipping.")
        if extracted_subtitle_path.is_file():
            extracted_subtitle_path.unlink()
        return None, None

    if (
        not extracted_subtitle_path.is_file()
        or extracted_subtitle_path.stat().st_size == 0
    ):
        logger.error(f"Extraction failed for {mkv_path.name}")
        return None, None

    logging.debug(f"Successfully extracted subtitle track to {extracted_subtitle_path}")
    return extracted_subtitle_path, subtitle_extension


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
            "timestamp": time.time(),
        }
        with open(progress_file_path, "w", encoding="utf-8") as f:
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
        with open(progress_file_path, "r", encoding="utf-8") as f:
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

    print(f"\n{'=' * 60}")
    print(f"Previous translation was interrupted")
    print(
        f"Progress: {saved_line}/{total_lines} translatable lines ({percentage:.1f}% complete)"
    )
    print(f"{'=' * 60}")

    while True:
        response = (
            input("Resume from where you left off? (y/n) [default: y]: ")
            .strip()
            .lower()
        )

        if response in ["", "y", "yes"]:
            logger.info("Resuming from saved progress...")
            return True
        elif response in ["n", "no"]:
            logger.info("Starting from beginning...")
            return False
        else:
            print("Please enter 'y' or 'n'")


# --- Translation Helper Functions ---


def get_system_instruction(
    source_lang,
    target_lang="Latin American Spanish",
    thinking=True,
    audio_file=None,
    gender_hints=False,
    reference_lang=None,
    extra_context_text=None,
):
    """
    Generate system instruction for translation.
    Adapted from gemini-translator-srt's approach with thinking mode support.
    """
    thinking_instruction = (
        "\n\nThink deeply and reason as much as possible before returning the response."
        if thinking
        else "\n\nDo NOT think or reason."
    )

    fields = ["- index: a string identifier", "- content: the text to translate"]
    if audio_file:
        fields.extend(
            [
                "- time_start: the start time of the segment",
                "- time_end: the end time of the segment",
            ]
        )
    if gender_hints:
        fields.extend(
            [
                "- speaker_gender: optional hint (male/female/unknown)",
                "- addressee_gender: optional hint (male/female/mixed/unknown)",
                "- addressee_number: optional hint (singular/plural/unknown)",
                "- gender_confidence: optional hint (low/medium/high)",
            ]
        )
    if reference_lang:
        fields.extend(
            [
                f"- reference_content: optional aligned subtitle text in {reference_lang}",
            ]
        )
    fields_text = "\n".join(fields) + "\n"

    instruction = f"""You are an assistant that translates subtitles from {source_lang} to {target_lang}.

You will receive a list of objects, each with these fields:
{fields_text}
Treat the full batch as one continuous scene: infer context across neighboring lines first, then write the final translation for each object.
Translate the 'content' field of each object into natural {target_lang}.
If the 'content' field is empty, leave it as is.
Preserve line breaks, formatting, and special characters.
Keep the same segmentation: think with full-scene context, but return one translated line per original object.
Never leave dialogue in {source_lang} unless the text is already a proper noun, a deliberate loanword, or an expression that should remain unchanged in Spanish.
When the main source line contains gendered wording or pronoun cues, preserve that gender in Spanish.
Prefer gender/person cues from the main {source_lang} line over any auxiliary reference.
Maintain gender continuity across neighboring lines when they appear to belong to the same speaker or same conversational thread.
If nearby context in the main {source_lang} subtitles establishes a feminine speaker/addressee, keep feminine agreement in later ambiguous lines unless the main source clearly changes it.
Do not default to masculine just because a later line is ambiguous.
If gender is truly uncertain, prefer a neutral Spanish rephrase over forcing masculine agreement.
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

    if gender_hints:
        instruction += f"""

Some objects may include precomputed gender hints.
Use them as guidance for grammatical agreement in {target_lang}:
- speaker_gender applies to first-person or self-descriptive lines
- addressee_gender and addressee_number apply when the speaker is addressing someone directly
- gender_confidence tells you how strongly to trust the hints
- If a hint is unknown or low confidence, rely on dialogue context or prefer neutral wording when natural"""

    if reference_lang:
        instruction += f"""

Some objects may include reference_content from an aligned {reference_lang} subtitle track.
Use it as a semantic cross-check to confirm meaning, speaker intent, names, and ambiguous lines.
If the main line in {source_lang} seems underspecified, use reference_content to understand the scene before translating.
Do not copy reference_content literally unless that is also the correct meaning in Spanish.
Do not let reference_content override explicit gender, pronoun, or speaker-role cues already present in the main {source_lang} line.
Do not let reference_content push an ambiguous line toward masculine if the main {source_lang} scene context was already feminine.
Always prioritize the main {source_lang} content when there is a conflict.
Before finalizing the batch, mentally verify each translation against reference_content and the surrounding batch context.
Do not translate the reference_content field itself into the output."""

    if extra_context_text:
        instruction += f"""

Additional user-requested context:
{extra_context_text}

Use this context only as supporting guidance.
Do not invent facts that are not present in the source line, nearby lines, or this extra context."""

    instruction += thinking_instruction

    return instruction


def get_review_system_instruction(source_lang, target_lang="Latin American Spanish", extra_context_text=None):
    """
    Generate system instruction for the translation review pass.
    Reviews gender agreement, consistency, untranslated leaks, and register issues.
    """
    instruction = f"""You are a meticulous translation proofreader. You will review subtitles translated from {source_lang} to {target_lang} and fix every real error you find.

INPUT FORMAT — a JSON array where each object has:
- index: line number (string)
- original: source text
- translated: current translation

Carefully compare each original against its translation. Check for:
1. **Gender agreement**: Adjectives, past participles, and verbs must match the speaker's established gender. "Ella está cansada" is correct for a woman; "Ella está cansado" is an error. Use surrounding lines and context to determine who is speaking or being spoken about.
2. **Untranslated source text**: Any word or phrase left in {source_lang} that belongs in {target_lang}. Proper nouns and intentional loanwords are fine; everything else must be translated.
3. **Terminology consistency**: The same concept, name, or term should be translated the same way every time. "Chairman Park" should not become "Director Park" halfway through.
4. **Register consistency**: If a character uses tú with someone, keep tú. If usted, keep usted. A sudden switch without reason is an error.
5. **Pronoun and reference errors**: Wrong gender on pronouns, dangling references, or subject-verb disagreements.

Be thorough. A single wrong gender ending or missed untranslated word is a real, noticeable error — flag it. It is better to over-correct than to let mistakes through.

OUTPUT FORMAT — a JSON array of objects:
- index: the line index (string) to correct
- corrected: the full corrected line with all formatting preserved
- reason: short explanation of what was wrong

Preserve all ASS override tags exactly ({{\\\\an8}}, {{\\\\i1}}, {{\\\\b0}}, colors, positions, etc.). Preserve \\\\N line breaks. Only change the translation text itself.

Do NOT return an empty array unless every single line is truly correct. Most translations have at least a few real errors."""

    if extra_context_text:
        instruction += f"""

Additional context about this subtitle file:
{extra_context_text}

Use this to identify character genders, relationships, and proper terminology."""

    return instruction


def get_review_config(system_instruction, model_name, thinking=True, thinking_budget=2048, temperature=None, top_p=None, top_k=None, provider="gemini"):
    """Build API configuration for the translation review pass."""
    if is_gemini_provider(provider):
        response_schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "index": types.Schema(type=types.Type.STRING),
                    "corrected": types.Schema(type=types.Type.STRING),
                    "reason": types.Schema(type=types.Type.STRING),
                },
                required=["index", "corrected", "reason"],
            ),
        )

        thinking_compatible = (
            "2.5" in model_name or "2.0" in model_name or "gemini-3" in model_name
        )
        thinking_budget_compatible = "flash" in model_name

        thinking_config = None
        if thinking_compatible and thinking:
            thinking_config = types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=thinking_budget if thinking_budget_compatible else None,
            )

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            safety_settings=get_safety_settings(),
            system_instruction=system_instruction,
            thinking_config=thinking_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    return {
        "provider": provider,
        "system_instruction": system_instruction,
        "think": get_ollama_think_value(model_name, thinking, thinking_budget),
        "format": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "string"},
                    "corrected": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "corrected", "reason"],
            },
        },
        "options": {
            key: value
            for key, value in {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }.items()
            if value is not None
        },
    }


def review_translation(
    dialogue_events,
    original_texts,
    translated_subtitle,
    source_lang,
    target_lang,
    api_manager,
    model_name,
    thinking=True,
    thinking_budget=2048,
    extra_context_text=None,
    temperature=None,
    top_p=None,
    top_k=None,
    review_batch_size=150,
):
    """
    Review translated subtitles for gender, consistency, and quality issues.

    Sends batches of (original, translated) pairs to the LLM and applies
    any corrections returned.

    Args:
        dialogue_events: List of ASS subtitle events
        original_texts: List of original (stripped) texts for each event
        translated_subtitle: List of translated texts for each event
        source_lang: Source language name/code
        target_lang: Target language name (default "Latin American Spanish")
        api_manager: APIManager instance for LLM calls
        model_name: Model to use for review
        thinking: Whether to enable thinking mode
        thinking_budget: Token budget for thinking
        extra_context_text: Optional extra context from user
        temperature: Sampling temperature
        top_p: Nucleus sampling
        top_k: Top-K sampling
        review_batch_size: Lines per review batch (default 150)

    Returns:
        Number of corrections applied
    """
    from tools.progress_display import progress_bar, progress_complete, clear_progress

    provider = api_manager.provider
    client = api_manager.get_client()

    system_instruction = get_review_system_instruction(
        source_lang, target_lang, extra_context_text=extra_context_text
    )
    config = get_review_config(
        system_instruction, model_name,
        thinking=thinking, thinking_budget=thinking_budget,
        temperature=temperature, top_p=top_p, top_k=top_k,
        provider=provider,
    )

    total_lines = len(dialogue_events)
    total_corrections = 0

    logger.info(f"Reviewing {total_lines} translated lines for quality issues...")

    # Build index->position map for fast lookup
    # Process in batches for context
    for batch_start in range(0, total_lines, review_batch_size):
        batch_end = min(batch_start + review_batch_size, total_lines)
        batch_indices = list(range(batch_start, batch_end))

        # Build the review payload
        review_items = []
        for i in batch_indices:
            orig = original_texts[i] if i < len(original_texts) else ""
            trans = translated_subtitle[i] if i < len(translated_subtitle) else ""
            review_items.append({
                "index": str(i),
                "original": orig,
                "translated": trans,
            })

        # Include overlap context: add a few lines before and after for cross-batch consistency
        overlap_before = max(0, batch_start - 5)
        overlap_after = min(total_lines, batch_end + 5)
        # Overlap lines are sent as context but not marked for correction
        context_items = []
        for i in range(overlap_before, batch_start):
            orig = original_texts[i] if i < len(original_texts) else ""
            trans = translated_subtitle[i] if i < len(translated_subtitle) else ""
            context_items.append({
                "index": str(i),
                "original": orig,
                "translated": trans,
            })
        for i in range(batch_end, overlap_after):
            orig = original_texts[i] if i < len(original_texts) else ""
            trans = translated_subtitle[i] if i < len(translated_subtitle) else ""
            context_items.append({
                "index": str(i),
                "original": orig,
                "translated": trans,
            })

        # Combine: context + review items
        all_items = context_items[:5] + review_items + context_items[5:]
        # Mark which indices are reviewable (batch range)
        reviewable_indices = set(str(i) for i in batch_indices)

        progress_bar(current=batch_start, total=total_lines, model_name=model_name, task_label="Reviewing")

        json_payload = json.dumps(all_items, ensure_ascii=False)

        # Build the user message
        user_message = f"Review the following subtitle translation batch. Lines with indices {min(batch_indices)}-{max(batch_indices)} are the ones to check. Surrounding lines are context only — do NOT correct context lines, only lines in the target range.\n\n{json_payload}"

        try:
            if is_gemini_provider(provider):
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_message,
                    config=config,
                )
                response_text = response.text
            else:
                # Ollama path — use streaming to avoid timeouts on cloud models
                ollama_messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message},
                ]
                ollama_options = config.get("options", {})
                think = config.get("think", False)

                chat_kwargs = {
                    "model": model_name,
                    "messages": ollama_messages,
                    "stream": True,
                    "format": config.get("format", "json"),
                    "think": think,
                }
                if ollama_options:
                    chat_kwargs["options"] = ollama_options

                response = client.chat(**chat_kwargs)
                response_text = ""
                for chunk in response:
                    chunk_text = extract_ollama_chunk_text(chunk)
                    if chunk_text:
                        response_text += chunk_text

            # Parse the corrections
            try:
                corrections = json_repair.loads(response_text)
            except Exception:
                corrections = json.loads(response_text)

            if not isinstance(corrections, list):
                corrections = []

            # Apply corrections (only for indices in the reviewable range)
            batch_corrections = 0
            for correction in corrections:
                idx_str = str(correction.get("index", ""))
                if idx_str not in reviewable_indices:
                    continue
                idx = int(idx_str)
                corrected_text = correction.get("corrected", "")
                reason = correction.get("reason", "")
                if not corrected_text:
                    continue
                if idx < 0 or idx >= len(translated_subtitle):
                    continue

                old_text = translated_subtitle[idx]
                if corrected_text != old_text:
                    translated_subtitle[idx] = corrected_text
                    # Also update the event text (restore formatting will be re-applied)
                    if dialogue_events and idx < len(dialogue_events):
                        # Re-apply ASS formatting restoration to the corrected text
                        restored_directives = restore_ass_directives(corrected_text)
                        dialogue_events[idx].text = restore_formatting(
                            original_texts[idx], restored_directives
                        )
                    batch_corrections += 1
                    logging.info(
                        f"  Review corrected line {idx}: {reason}"
                    )
                    logging.debug(
                        f"    Old: {old_text!r}\n    New: {corrected_text!r}"
                    )

            total_corrections += batch_corrections
            if batch_corrections > 0:
                logger.info(
                    f"  Review batch {batch_start // review_batch_size + 1}: "
                    f"{batch_corrections} correction(s) applied"
                )

        except Exception as e:
            logger.warning(f"Review batch failed (non-fatal): {e}")
            # Review is best-effort; continue with the next batch

    progress_complete(total_lines, total_lines, model_name)

    if total_corrections > 0:
        logger.success(
            f"Review complete: {total_corrections} correction(s) applied across {total_lines} lines"
        )
    else:
        logger.info("Review complete: no corrections needed")

    return total_corrections


def get_safety_settings():
    """Build permissive safety settings for subtitle translation content."""
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
        ),
    ]


def get_audio_analysis_instruction(source_lang, target_lang="Latin American Spanish"):
    """Build system instruction for audio-only gender analysis fallback."""
    return f"""You analyze subtitle-timed dialogue audio to provide grammatical gender hints for translation from {source_lang} to {target_lang}.

You will receive a list of objects with these fields:
- index: a string identifier
- content: the subtitle text
- time_start: subtitle start timestamp in milliseconds
- time_end: subtitle end timestamp in milliseconds

You will also receive an audio file.
For each object, inspect the audio around the provided timestamps and return a JSON array with exactly one object per input item.

Return only these fields:
- index: same identifier from input
- speaker_gender: male, female, or unknown
- addressee_gender: male, female, mixed, or unknown
- addressee_number: singular, plural, or unknown
- confidence: low, medium, or high

Rules:
- Use the subtitle text and the timed audio together.
- Return unknown when the audio does not make the answer clear.
- Do not guess.
- Do not translate the text.
- Do not add explanations outside the JSON array."""


def get_audio_analysis_config(system_instruction):
    """Build API configuration for structured audio analysis fallback."""
    response_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "index": types.Schema(type=types.Type.STRING),
                "speaker_gender": types.Schema(type=types.Type.STRING),
                "addressee_gender": types.Schema(type=types.Type.STRING),
                "addressee_number": types.Schema(type=types.Type.STRING),
                "confidence": types.Schema(type=types.Type.STRING),
            },
            required=[
                "index",
                "speaker_gender",
                "addressee_gender",
                "addressee_number",
                "confidence",
            ],
        ),
    )

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        safety_settings=get_safety_settings(),
        system_instruction=system_instruction,
        temperature=0.0,
        top_p=0.1,
        top_k=1,
    )


def get_translation_config(
    system_instruction,
    model_name,
    thinking=True,
    thinking_budget=2048,
    temperature=None,
    top_p=None,
    top_k=None,
    provider="gemini",
):
    """
    Build API configuration.
    Based on gemini-translator-srt's config builder with thinking mode support.

    Args:
        system_instruction: System instruction for the model
        model_name: Name of the model to use
        thinking: Whether to enable thinking mode
        thinking_budget: Token budget for thinking (flash models only)
        temperature: Controls randomness (0.0-2.0). Lower = more deterministic
        top_p: Nucleus sampling parameter (0.0-1.0)
        top_k: Top-K sampling parameter (integer >= 0)
    """
    if is_gemini_provider(provider):
        # Response schema: array of {index, content} objects
        response_schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "index": types.Schema(type=types.Type.STRING),
                    "content": types.Schema(type=types.Type.STRING),
                },
                required=["index", "content"],
            ),
        )

        # Determine thinking mode compatibility
        # Supports: Gemini 2.0, 2.5, and 3.x models
        thinking_compatible = (
            "2.5" in model_name or "2.0" in model_name or "gemini-3" in model_name
        )
        thinking_budget_compatible = "flash" in model_name

        # Build thinking config if compatible
        # Flash models: Use thinking_budget for controlled thinking
        # Pro models: Enable thinking without budget (handled by timeout/retry mechanism)
        thinking_config = None
        if thinking_compatible and thinking:
            thinking_config = types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=thinking_budget if thinking_budget_compatible else None,
            )

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            safety_settings=get_safety_settings(),
            system_instruction=system_instruction,
            thinking_config=thinking_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    return {
        "provider": provider,
        "system_instruction": system_instruction,
        "think": get_ollama_think_value(model_name, thinking, thinking_budget),
        "format": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["index", "content"],
            },
        },
        "options": {
            key: value
            for key, value in {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }.items()
            if value is not None
        },
    }


def is_rtl(text):
    """
    Detect if text is right-to-left.
    From gemini-translator-srt's RTL detection.
    """
    if not text:
        return False

    count = Counter([unicodedata.bidirectional(c) for c in text])
    rtl_count = (
        count.get("R", 0)
        + count.get("AL", 0)
        + count.get("RLE", 0)
        + count.get("RLI", 0)
    )
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
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue

        # Check Unicode script
        try:
            script_name = unicodedata.name(char).split()[0]
            # Latin includes basic ASCII and extended Latin characters
            if ord(char) < 128 or "LATIN" in script_name:
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


def validate_batch_tokens(client, batch, model_name, provider="gemini"):
    """
    Validate batch doesn't exceed token limit.
    Uses actual model limits based on Gemini model specifications.
    """
    if not is_gemini_provider(provider):
        return True

    try:
        # Use the ACTUAL model for token counting
        token_count = client.models.count_tokens(
            model=model_name, contents=json.dumps(batch, ensure_ascii=False)
        )

        # Set token limits based on model (conservative estimates)
        # Source: https://ai.google.dev/gemini-api/docs/models/gemini
        if "pro" in model_name:
            token_limit = 2_000_000  # Pro models: ~2M tokens
        else:
            token_limit = 1_000_000  # Flash models: ~1M tokens

        if token_count.total_tokens > token_limit * 0.9:
            # Token limit exceeded - will be shown after clearing progress bar
            logger.error(
                f"Token count ({token_count.total_tokens}) exceeds 90% of limit ({token_limit})"
            )
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


def build_resume_context(
    dialogue_lines,
    translated_subtitle,
    start_line,
    batch_size,
    provider="gemini",
    reference_contexts=None,
):
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
    original_batch = []
    for i in range(context_start, context_end):
        item = {"index": str(i), "content": dialogue_lines[i]}
        if reference_contexts and reference_contexts.get(i):
            item["reference_content"] = reference_contexts[i]
        original_batch.append(item)

    # Build translated batch
    translated_batch = [
        {"index": str(i), "content": translated_subtitle[i]}
        for i in range(context_start, context_end)
    ]

    if is_ollama_provider(provider):
        return [
            {"role": "user", "content": json.dumps(original_batch, ensure_ascii=False)},
            {
                "role": "assistant",
                "content": json.dumps(translated_batch, ensure_ascii=False),
            },
        ]

    return [
        types.Content(
            role="user",
            parts=[types.Part(text=json.dumps(original_batch, ensure_ascii=False))],
        ),
        types.Content(
            role="model",
            parts=[types.Part(text=json.dumps(translated_batch, ensure_ascii=False))],
        ),
    ]


def _parse_ass_timestamp(ts_str):
    """Parse ASS timestamp (H:MM:SS.CC or 0:00:00.00) to milliseconds."""
    ts_str = ts_str.strip()
    # Remove any trailing non-numeric junk (e.g., color codes mixed in)
    ts_str = re.sub(r'[^0-9.:].*$', '', ts_str).strip()
    try:
        parts = ts_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600000 + int(m) * 60000 + int(float(s) * 1000)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60000 + int(float(s) * 1000)
    except (ValueError, IndexError):
        pass
    return None


def _parse_ass_file_manually(ass_path):
    """
    Fallback ASS parser for when pysubs2 fails (e.g., malformed drawing commands,
    corrupted timestamps from color normalization, etc.).
    Returns a list of pysubs2.SSAEvent-like dicts with start, end, text fields.
    """
    entries = []
    try:
        with open(ass_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line.lower().startswith('dialogue:'):
                    continue
                # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
                # Remove the "Dialogue:" prefix and split
                rest = line[len('dialogue:'):].lstrip()
                # Split on commas, but preserve the text field (which may contain commas)
                fields = rest.split(',', 9)
                if len(fields) < 10:
                    continue
                start_ms = _parse_ass_timestamp(fields[1])
                end_ms = _parse_ass_timestamp(fields[2])
                if start_ms is None or end_ms is None:
                    continue
                text = fields[9]
                # Skip drawing commands
                if r'\p1' in text or r'\p2' in text or r'\p3' in text or r'\p4' in text:
                    continue
                entries.append({
                    'start': start_ms,
                    'end': end_ms,
                    'text': text,
                })
    except Exception as e:
        logging.warning(f"Manual ASS fallback parser also failed for {ass_path}: {e}")
    return entries


def load_reference_subtitle_entries(subtitle_path, strip_sdh=False):
    """Load simplified dialogue entries from a secondary subtitle file."""
    if not subtitle_path:
        return []

    normalize_ass_colors(subtitle_path)

    ass_header_keywords = [
        "[Script Info]",
        "[V4+ Styles]",
        "[Events]",
        "[Aegisub",
        "Format:",
        "Style:",
        "ScriptType:",
        "PlayResX:",
        "PlayResY:",
        "WrapStyle:",
        "Title:",
        "Collisions:",
    ]

    entries = []
    try:
        subs = pysubs2.load(str(subtitle_path))

        for event in subs:
            if not hasattr(event, "type") or event.type != "Dialogue":
                continue

            if (
                r"\p1" in event.text
                or r"\p2" in event.text
                or r"\p3" in event.text
                or r"\p4" in event.text
            ):
                continue

            if any(keyword in event.text for keyword in ass_header_keywords):
                continue

            plain_text = remove_formatting(event.text)
            if not plain_text:
                continue

            if strip_sdh:
                plain_text = strip_sdh_elements(plain_text)
                if not plain_text or is_sdh_only_line(plain_text):
                    continue

            entries.append(
                {
                    "start": event.start,
                    "end": event.end,
                    "content": plain_text.strip(),
                }
            )
    except Exception as e:
        logging.warning(
            f"pysubs2 failed to parse {subtitle_path.name}: {e}. Trying manual fallback parser."
        )
        # Fallback: parse ASS manually to extract what we can
        fallback_entries = _parse_ass_file_manually(subtitle_path)
        for entry in fallback_entries:
            text = entry['text']
            if any(keyword in text for keyword in ass_header_keywords):
                continue
            plain_text = remove_formatting(text)
            if not plain_text:
                continue
            if strip_sdh:
                plain_text = strip_sdh_elements(plain_text)
                if not plain_text or is_sdh_only_line(plain_text):
                    continue
            entries.append(
                {
                    "start": entry['start'],
                    "end": entry['end'],
                    "content": plain_text.strip(),
                }
            )
        if fallback_entries and not entries:
            logging.warning(
                f"Manual fallback parser found {len(fallback_entries)} lines but none passed filters"
            )
        elif entries:
            logging.info(
                f"Manual fallback parser recovered {len(entries)} dialogue entries from {subtitle_path.name}"
            )

    return entries


def build_reference_context_map(dialogue_events, reference_entries):
    """Align reference subtitle entries to primary dialogue events by timing."""
    if not reference_entries:
        return {}

    context_map = {}

    for idx, event in enumerate(dialogue_events):
        primary_start = event.start
        primary_end = event.end
        primary_mid = (primary_start + primary_end) / 2
        candidates = []

        for entry in reference_entries:
            reference_start = entry["start"]
            reference_end = entry["end"]
            reference_mid = (reference_start + reference_end) / 2
            overlap = min(primary_end, reference_end) - max(
                primary_start, reference_start
            )
            midpoint_delta = abs(primary_mid - reference_mid)

            if overlap > 0 or midpoint_delta <= 1500:
                candidates.append((max(overlap, 0), midpoint_delta, entry["content"]))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (-item[0], item[1]))
        unique_texts = []
        for _, _, text in candidates:
            if text and text not in unique_texts:
                unique_texts.append(text)
            if len(unique_texts) >= 2:
                break

        if unique_texts:
            context_map[idx] = " | ".join(unique_texts)

    return context_map


def get_dialogue_event_signature(event):
    """Build a stable signature for matching dialogue events across saves."""
    return (
        getattr(event, "start", None),
        getattr(event, "end", None),
        getattr(event, "style", None),
        getattr(event, "name", None),
        getattr(event, "layer", None),
    )


def strip_original_comment_prefix(text):
    """Remove an existing leading {Original: ...} comment if present."""
    return re.sub(r"^\{Original: .*?\}", "", text, count=1)


def normalize_translation_comparison_text(text):
    """Normalize text for rough source-vs-output comparison."""
    if not text:
        return ""
    text = remove_formatting(text)
    text = restore_ass_directives(text)
    text = re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)
    return text.strip()


def has_strong_source_language_signal(text, source_lang=None):
    """Heuristic: detect whether unchanged text still strongly looks like source language."""
    if not text:
        return False

    stripped = text.strip()
    lower = stripped.lower()

    if re.search(r"[àâæçéèêëîïôœùûüÿ]", lower):
        return True

    if source_lang == "fr":
        french_markers = [
            r"\b(?:je|tu|il|elle|on|nous|vous|ils|elles)\b",
            r"\b(?:le|la|les|un|une|des|du|de|d'|au|aux)\b",
            r"\b(?:est|suis|es|sommes|êtes|sont|avait|avec|pour|pas|plus|que|qui)\b",
            r"\b(?:bonjour|merci|monsieur|madame|mademoiselle|oui)\b",
        ]
        return any(re.search(pattern, lower) for pattern in french_markers)

    if source_lang == "eng":
        english_markers = [
            r"\b(?:the|and|you|are|is|was|were|with|for|this|that|what|why|don't|can't)\b",
        ]
        return any(re.search(pattern, lower) for pattern in english_markers)

    return len(stripped.split()) >= 4


def find_source_script_leaks(translated_batch, source_lang=None):
    """Detect translated lines that still contain source-script characters."""
    if source_lang not in {"ja", "jpn", "japanese"}:
        return []

    leaked = []
    japanese_pattern = r"[ぁ-ゟ゠-ヿㇰ-ㇿ一-鿿々〆ヵヶ]"

    for translated_item in translated_batch:
        item_index = str(translated_item.get("index", "")) or "?"
        translated_text = translated_item.get("content", "")
        normalized_text = restore_ass_directives(remove_formatting(translated_text))
        if re.search(japanese_pattern, normalized_text):
            leaked.append(item_index)

    return leaked


def find_suspicious_unchanged_translations(batch, translated_batch, source_lang=None):
    """Detect lines that were returned essentially unchanged from the source."""
    suspicious = []
    batch_by_index = {str(item.get("index", "")): item for item in batch}

    for translated_item in translated_batch:
        item_index = str(translated_item.get("index", ""))
        source_item = batch_by_index.get(item_index)
        if not source_item:
            continue

        source_text = source_item.get("content", "")
        translated_text = translated_item.get("content", "")

        source_norm = normalize_translation_comparison_text(source_text)
        translated_norm = normalize_translation_comparison_text(translated_text)

        if len(source_norm) < 4 or not re.search(r"[A-Za-zÀ-ÿ]", source_text):
            continue

        # Skip probable proper nouns / titles / single-token labels that can naturally stay unchanged.
        if " " not in source_text.strip() and not re.search(r"[.!?¿¡,:;]", source_text):
            continue

        if (
            source_norm
            and source_norm == translated_norm
            and has_strong_source_language_signal(source_text, source_lang)
        ):
            suspicious.append(item_index or "?")

    return suspicious


def normalize_translated_batch(batch, translated_batch):
    """Normalize model output to the expected batch order by index.

    Accepts harmless extras/duplicates if every expected index is present exactly once
    after normalization. Raises ValueError if required indices are missing or items are invalid.
    """
    if not isinstance(translated_batch, list):
        raise ValueError("Model response is not a JSON array")

    expected_indices = [str(item.get("index", "")) for item in batch]
    expected_index_set = set(expected_indices)
    normalized_map = {}
    duplicate_indices = []
    unexpected_indices = []

    for item in translated_batch:
        if not isinstance(item, dict):
            raise ValueError("Model response contains a non-object item")

        item_index = str(item.get("index", ""))
        if not item_index:
            raise ValueError("Model response contains an item without index")

        if item_index not in expected_index_set:
            unexpected_indices.append(item_index)
            continue

        if item_index in normalized_map:
            duplicate_indices.append(item_index)
            continue

        normalized_map[item_index] = item

    missing_indices = [idx for idx in expected_indices if idx not in normalized_map]
    if missing_indices:
        raise ValueError(
            f"Response missing expected indices: {', '.join(missing_indices[:8])}"
        )

    if unexpected_indices:
        logger.log_only(
            f"Ignoring unexpected response indices: {', '.join(unexpected_indices[:8])}"
        )
    if duplicate_indices:
        logger.log_only(
            f"Ignoring duplicate response indices: {', '.join(duplicate_indices[:8])}"
        )

    return [normalized_map[idx] for idx in expected_indices]


def inject_original_comments_into_ass(translated_ass_path, reference_subtitle_path):
    """Inject {Original: ...} comments into an existing translated ASS file."""
    if translated_ass_path.suffix.lower() != ".ass":
        logger.warning(
            f"Skipping {translated_ass_path.name}: Original comments are only supported for ASS output."
        )
        return False

    translated_subs = pysubs2.load(str(translated_ass_path))
    translated_events = [
        event
        for event in translated_subs
        if hasattr(event, "type") and event.type == "Dialogue"
    ]
    if not translated_events:
        logger.warning(f"No Dialogue events found in {translated_ass_path.name}.")
        return False

    reference_entries = load_reference_subtitle_entries(reference_subtitle_path)
    reference_contexts = build_reference_context_map(
        translated_events, reference_entries
    )
    if not reference_contexts:
        logger.warning(
            f"No aligned source lines found to inject into {translated_ass_path.name}."
        )
        return False

    updated_count = 0
    for idx, event in enumerate(translated_events):
        original_text = reference_contexts.get(idx)
        if not original_text:
            continue

        cleaned_text = strip_original_comment_prefix(event.text)
        event.text = f"{{Original: {original_text}}}{cleaned_text}"
        updated_count += 1

    translated_subs.save(str(translated_ass_path))
    logger.success(
        f"Added Original comments to {updated_count} dialogue lines in {translated_ass_path.name}"
    )
    return True


def find_translated_subtitle_for_mkv(output_dir, mkv_stem):
    """Find the translated subtitle file corresponding to an MKV stem."""
    candidates = [
        output_dir / f"{mkv_stem}.es-419.ass",
        output_dir / f"{mkv_stem}.es-419.ssa",
        output_dir / f"{mkv_stem}.es-419.srt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def add_original_comments_to_existing_output(
    mkv_path, output_dir, remembered_lang=None
):
    """Post-process an existing translated subtitle/MKV to add {Original: ...} comments."""
    translated_subtitle_path = find_translated_subtitle_for_mkv(
        output_dir, mkv_path.stem
    )
    if not translated_subtitle_path:
        logger.warning(
            f"No translated subtitle file found for {mkv_path.name} in {output_dir}."
        )
        return None

    if translated_subtitle_path.suffix.lower() != ".ass":
        logger.warning(
            f"Found {translated_subtitle_path.name}, but Original comments only work with ASS output."
        )
        return None

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    try:
        result = subprocess.run(
            ["mkvmerge", "-J", str(mkv_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        mkv_info = json.loads(result.stdout)
        selected_track, lang_code = select_original_injection_track(
            mkv_info.get("tracks", []), remembered_lang
        )
        if selected_track is None:
            logger.warning(
                f"No suitable subtitle track found to inject as Original in {mkv_path.name}."
            )
            return None

        reference_subtitle_path, _ = extract_subtitle_track(
            mkv_path, selected_track, tmp_dir, lang_code, label="original"
        )
        if not reference_subtitle_path:
            return None

        updated = inject_original_comments_into_ass(
            translated_subtitle_path, reference_subtitle_path
        )
        if not updated:
            return None

        merge_subtitles_to_mkv(mkv_path, translated_subtitle_path, output_dir)
        return lang_code
    except Exception as e:
        logger.error(f"Failed to add Original comments for {mkv_path.name}: {e}")
        return None


def is_audio_capability_error(error_msg):
    """Detect model errors caused by unsupported audio input."""
    lower_msg = error_msg.lower()
    audio_markers = [
        "audio",
        "audio/mpeg",
        "modality",
        "mime",
        "inline_data",
        "file_data",
    ]
    capability_markers = [
        "unsupported",
        "does not support",
        "not support",
        "not enabled",
        "invalid argument",
        "invalid_argument",
        "only supports",
        "input type",
    ]
    return any(marker in lower_msg for marker in audio_markers) and any(
        marker in lower_msg for marker in capability_markers
    )


def analyze_audio_batch(
    client,
    audio_model_name,
    batch,
    audio_part,
    source_lang,
    target_lang="Latin American Spanish",
    progress_current=0,
    progress_total=0,
    progress_model_name="",
):
    """Analyze audio for gender hints when the translation model cannot accept audio."""
    system_instruction = get_audio_analysis_instruction(source_lang, target_lang)
    config = get_audio_analysis_config(system_instruction)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=json.dumps(batch, ensure_ascii=False)), audio_part],
        )
    ]
    start_time = time.time()
    stop_event = threading.Event()
    progress_bar(
        current=progress_current,
        total=progress_total,
        model_name=progress_model_name,
        is_loading=True,
        status_detail="Analyzing audio",
    )

    def audio_analysis_heartbeat():
        heartbeat_count = 0
        while not stop_event.wait(1):
            elapsed = int(time.time() - start_time)
            progress_bar(
                current=progress_current,
                total=progress_total,
                model_name=progress_model_name,
                is_loading=True,
                status_detail="Analyzing audio",
            )
            heartbeat_count += 1
            if heartbeat_count % 15 == 0:
                logger.log_only(
                    f"Audio analysis still running with {audio_model_name}... {elapsed}s elapsed"
                )

    heartbeat_thread = threading.Thread(target=audio_analysis_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        wait_for_gemini_request_slot("gemini")
        with suppress_stderr_output():
            response = client.models.generate_content(
                model=audio_model_name, contents=contents, config=config
            )
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=0.2)

    response_text = response.text or ""
    analyzed_batch = json_repair.loads(response_text)

    if not isinstance(analyzed_batch, list) or len(analyzed_batch) != len(batch):
        raise ValueError(
            f"Audio analysis response length mismatch: expected {len(batch)}, got {len(analyzed_batch) if isinstance(analyzed_batch, list) else 'invalid'}"
        )

    hints_by_index = {}
    valid_speaker_genders = {"male", "female", "unknown"}
    valid_addressee_genders = {"male", "female", "mixed", "unknown"}
    valid_addressee_numbers = {"singular", "plural", "unknown"}
    valid_confidences = {"low", "medium", "high"}

    for item in analyzed_batch:
        idx = str(item.get("index", ""))
        hints_by_index[idx] = {
            "speaker_gender": (
                item.get("speaker_gender", "unknown").lower()
                if item.get("speaker_gender", "unknown").lower()
                in valid_speaker_genders
                else "unknown"
            ),
            "addressee_gender": (
                item.get("addressee_gender", "unknown").lower()
                if item.get("addressee_gender", "unknown").lower()
                in valid_addressee_genders
                else "unknown"
            ),
            "addressee_number": (
                item.get("addressee_number", "unknown").lower()
                if item.get("addressee_number", "unknown").lower()
                in valid_addressee_numbers
                else "unknown"
            ),
            "confidence": (
                item.get("confidence", "low").lower()
                if item.get("confidence", "low").lower() in valid_confidences
                else "low"
            ),
        }

    return hints_by_index


def probe_audio_input_support(client, model_name, sample_batch, audio_part):
    """Check whether a translation model accepts audio input."""
    system_instruction = (
        "Return the same JSON array you receive, preserving index and content."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "index": types.Schema(type=types.Type.STRING),
                    "content": types.Schema(type=types.Type.STRING),
                },
                required=["index", "content"],
            ),
        ),
        safety_settings=get_safety_settings(),
        system_instruction=system_instruction,
        temperature=0.0,
        top_p=0.1,
        top_k=1,
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=json.dumps(sample_batch, ensure_ascii=False)),
                audio_part,
            ],
        )
    ]
    wait_for_gemini_request_slot("gemini")
    with suppress_stderr_output():
        client.models.generate_content(
            model=model_name, contents=contents, config=config
        )


def attach_gender_hints_to_batch(batch, hints_by_index=None):
    """Build a translation batch with optional gender hints attached."""
    request_batch = []

    for item in batch:
        request_item = dict(item)
        if hints_by_index:
            hints = hints_by_index.get(str(item["index"]))
            if hints:
                request_item["speaker_gender"] = hints["speaker_gender"]
                request_item["addressee_gender"] = hints["addressee_gender"]
                request_item["addressee_number"] = hints["addressee_number"]
                request_item["gender_confidence"] = hints["confidence"]
        request_batch.append(request_item)

    return request_batch


# Global variable to track last successful chunk size (like gemini-translator-srt)
_last_chunk_size = 0


def get_last_chunk_size():
    """Get the number of lines successfully translated in the last batch."""
    global _last_chunk_size
    return _last_chunk_size


def process_batch_streaming_ollama(
    client,
    model_name,
    batch,
    previous_message,
    translated_subtitle,
    config,
    current_line,
    total_lines,
    batch_number=1,
    keep_original=False,
    original_format=".ass",
    audio_part=None,
    audio_file=None,
    dialogue_lines=None,
    unique_text_indices=None,
    deduplication_keys=None,
    provider="ollama-local",
    source_lang=None,
):
    """Process a translation batch using Ollama's streaming chat API."""
    global _last_chunk_size
    _last_chunk_size = 0
    provider_name = get_provider_display_name(provider)

    original_texts = {int(item["index"]): item["content"] for item in batch}
    done = False
    corrective_message = None
    source_leak_retry_count = 0
    max_source_leak_retries = 2
    final_thoughts_text = ""

    while done == False:
        response_text = ""
        thoughts_text = ""
        chunk_count = 0

        messages = [{"role": "system", "content": config["system_instruction"]}]
        messages.extend(previous_message)
        messages.append(
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}
        )
        if corrective_message:
            messages.append({"role": "user", "content": corrective_message})

        chat_kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "format": config.get("format", "json"),
            "think": config.get("think", False),
        }
        if config.get("options"):
            chat_kwargs["options"] = config["options"]

        response = client.chat(**chat_kwargs)

        for chunk in response:
            thinking_text = extract_ollama_chunk_thinking(chunk)
            if thinking_text:
                thoughts_text += thinking_text
                progress_bar(
                    current=current_line,
                    total=total_lines,
                    model_name=model_name,
                    chunk_size=min(chunk_count, max(0, total_lines - current_line)),
                    is_thinking=True,
                )

            part_text = extract_ollama_chunk_text(chunk)
            if not part_text:
                continue

            response_text += part_text

            try:
                partial_batch = json_repair.loads(response_text)
                if isinstance(partial_batch, list):
                    prev_chunk_count = chunk_count
                    chunk_count = len(partial_batch)

                    for i in range(prev_chunk_count, chunk_count):
                        if i < len(partial_batch):
                            item = partial_batch[i]
                            idx = int(item["index"])
                            content = item["content"]

                            if is_rtl(content):
                                content = f"\u202b{content}\u202c"

                            if keep_original and original_format == ".ass":
                                original_text = original_texts.get(idx, "")
                                if original_text:
                                    content = f"{{Original: {original_text}}}{content}"

                            translated_subtitle[idx] = content

                            if (
                                dialogue_lines is not None
                                and unique_text_indices is not None
                                and deduplication_keys is not None
                            ):
                                dedup_key = deduplication_keys[idx]
                                if dedup_key in unique_text_indices:
                                    for duplicate_idx in unique_text_indices[dedup_key]:
                                        if duplicate_idx != idx:
                                            translated_subtitle[duplicate_idx] = content

                    _last_chunk_size = chunk_count
                    effective_chunk = min(
                        chunk_count, max(0, total_lines - current_line)
                    )
                    progress_bar(
                        current=current_line,
                        total=total_lines,
                        model_name=model_name,
                        chunk_size=effective_chunk,
                        is_loading=True,
                    )
            except Exception:
                effective_chunk = min(chunk_count, max(0, total_lines - current_line))
                progress_bar(
                    current=current_line,
                    total=total_lines,
                    model_name=model_name,
                    chunk_size=effective_chunk,
                    is_loading=True,
                )

        if not response_text or not response_text.strip():
            clear_progress()
            error_with_progress(f"{provider_name} returned an empty response.")
            info_with_progress("Sending last batch again...")
            continue

        try:
            translated_batch = json_repair.loads(response_text)
        except Exception as e:
            clear_progress()
            warning_with_progress(f"Failed to parse response: {e}")
            info_with_progress("Sending last batch again...")
            continue

        try:
            translated_batch = normalize_translated_batch(batch, translated_batch)
        except Exception as e:
            clear_progress()
            warning_with_progress(f"Invalid response structure: {e}")
            info_with_progress("Sending last batch again...")
            continue

        suspicious_indices = find_suspicious_unchanged_translations(
            batch, translated_batch, source_lang=source_lang
        )
        script_leak_indices = find_source_script_leaks(
            translated_batch, source_lang=source_lang
        )
        leak_threshold = max(3, (len(batch) * 8 + 99) // 100)
        if script_leak_indices:
            if source_leak_retry_count < max_source_leak_retries:
                source_leak_retry_count += 1
                corrective_message = (
                    f"Your previous answer left Japanese characters in lines {', '.join(map(str, script_leak_indices[:8]))}. "
                    "Translate those lines fully into Latin American Spanish. Do not leave any Japanese characters in the output. "
                    "If a name must stay recognizable, romanize it into Latin script instead of keeping Japanese glyphs."
                )
                clear_progress()
                warning_with_progress(
                    "Detected untranslated Japanese characters in the batch. Retrying..."
                )
                continue
            raise ValueError(
                f"Persistent untranslated Japanese characters detected in batch lines: {', '.join(map(str, script_leak_indices[:8]))}"
            )
        if len(suspicious_indices) >= leak_threshold:
            if source_leak_retry_count < max_source_leak_retries:
                source_leak_retry_count += 1
                corrective_message = (
                    f"Your previous answer left lines {', '.join(map(str, suspicious_indices[:8]))} effectively unchanged in {source_lang or 'the source language'}. "
                    "Re-evaluate the whole batch using scene context and reference_content, then return only corrected Latin American Spanish JSON."
                )
                clear_progress()
                warning_with_progress(
                    "Detected source-language leakage in the batch. Retrying with stricter context review..."
                )
                continue
            raise ValueError(
                f"Persistent source language leak detected in batch lines: {', '.join(map(str, suspicious_indices[:8]))}"
            )
        elif suspicious_indices:
            logger.log_only(
                f"Minor unchanged-source warning ignored for batch: {', '.join(map(str, suspicious_indices[:8]))}"
            )

        for item in translated_batch:
            idx = int(item["index"])
            content = item["content"]

            if is_rtl(content):
                content = f"\u202b{content}\u202c"

            if keep_original and original_format == ".ass":
                original_text = original_texts.get(idx, "")
                if original_text:
                    content = f"{{Original: {original_text}}}{content}"

            translated_subtitle[idx] = content

            if (
                dialogue_lines is not None
                and unique_text_indices is not None
                and deduplication_keys is not None
            ):
                dedup_key = deduplication_keys[idx]
                if dedup_key in unique_text_indices:
                    for duplicate_idx in unique_text_indices[dedup_key]:
                        if duplicate_idx != idx:
                            translated_subtitle[duplicate_idx] = content

        _last_chunk_size = len(translated_batch)
        final_thoughts_text = thoughts_text
        done = True

        if final_thoughts_text:
            logger.save_thoughts(final_thoughts_text, batch_number)

        return [
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            {"role": "assistant", "content": response_text},
        ]


def process_batch_streaming(
    client,
    model_name,
    batch,
    previous_message,
    translated_subtitle,
    config,
    current_line,
    total_lines,
    batch_number=1,
    keep_original=False,
    original_format=".ass",
    audio_part=None,
    audio_file=None,
    dialogue_lines=None,
    unique_text_indices=None,
    deduplication_keys=None,
    provider="gemini",
    source_lang=None,
):
    """
    Process a batch with streaming responses and real-time progress display.
    Implements retry loop matching gemini-srt-translator's _process_batch pattern.

    Args:
        current_line: Current line number in the file (not batch index)
        total_lines: Total lines to translate

    Returns:
        previous_message context for next batch
    """
    if is_ollama_provider(provider):
        return process_batch_streaming_ollama(
            client=client,
            model_name=model_name,
            batch=batch,
            previous_message=previous_message,
            translated_subtitle=translated_subtitle,
            config=config,
            current_line=current_line,
            total_lines=total_lines,
            batch_number=batch_number,
            keep_original=keep_original,
            original_format=original_format,
            audio_part=audio_part,
            audio_file=audio_file,
            dialogue_lines=dialogue_lines,
            unique_text_indices=unique_text_indices,
            deduplication_keys=deduplication_keys,
            provider=provider,
            source_lang=source_lang,
        )

    global _last_chunk_size
    batch_size = len(batch)
    _last_chunk_size = 0  # Reset
    provider_name = get_provider_display_name(provider)

    # Build request
    parts = [types.Part(text=json.dumps(batch, ensure_ascii=False))]

    # Add audio if available (for gender-aware translation)
    if audio_part:
        parts.append(audio_part)

    current_message = types.Content(role="user", parts=parts)

    # Temporarily suppress ALL logging during API call to prevent disrupting progress bar
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)

    # Retry loop matching gemini-srt-translator pattern
    done = False
    retry = -1
    final_response_text = ""
    final_thoughts_text = ""
    max_retries_on_timeout = 3  # Retry up to 3 times if thinking times out
    corrective_message = None
    source_leak_retry_count = 0
    max_source_leak_retries = 2

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

            contents = previous_message + [current_message]
            if corrective_message:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=corrective_message)],
                    )
                )

            # Timeout tracking for thinking phase
            thinking_start_time = None
            thinking_timeout_seconds = 300  # 5 minutes
            timed_out = False

            # Stream response
            if blocked:
                break  # Exit retry loop if previously blocked

            wait_for_gemini_request_slot(provider)
            response = client.models.generate_content_stream(
                model=model_name, contents=contents, config=config
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
                                    f"Thinking exceeded {thinking_timeout_seconds // 60} minutes. "
                                    f"Retrying batch (attempt {retry + 1}/{max_retries_on_timeout})..."
                                )
                                timed_out = True
                                break  # Break out of chunk loop

                            # Show thinking indicator with elapsed time
                            thinking_minutes = int(thinking_elapsed // 60)
                            thinking_seconds = int(thinking_elapsed % 60)
                            # Cap chunk_size to prevent exceeding total (defensive check)
                            effective_chunk = min(
                                chunk_count, max(0, total_lines - current_line)
                            )
                            progress_bar(
                                current=current_line,
                                total=total_lines,
                                model_name=model_name,
                                chunk_size=effective_chunk,
                                is_thinking=True,
                                thinking_time=f"({thinking_minutes}m {thinking_seconds}s)",
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
                                            if (
                                                keep_original
                                                and original_format == ".ass"
                                            ):
                                                original_text = original_texts.get(
                                                    idx, ""
                                                )
                                                if original_text:
                                                    content = f"{{Original: {original_text}}}{content}"

                                            translated_subtitle[idx] = content

                                            # Apply translation to all duplicates of this text
                                            if (
                                                dialogue_lines is not None
                                                and unique_text_indices is not None
                                                and deduplication_keys is not None
                                            ):
                                                dedup_key = deduplication_keys[idx]
                                                if dedup_key in unique_text_indices:
                                                    for (
                                                        duplicate_idx
                                                    ) in unique_text_indices[dedup_key]:
                                                        if (
                                                            duplicate_idx != idx
                                                        ):  # Skip the one we just translated
                                                            translated_subtitle[
                                                                duplicate_idx
                                                            ] = content

                                    # Update global chunk size for error recovery
                                    _last_chunk_size = chunk_count

                                    # Update progress bar with real-time chunk progress
                                    # Cap chunk_size to prevent exceeding total (defensive check)
                                    effective_chunk = min(
                                        chunk_count, max(0, total_lines - current_line)
                                    )
                                    progress_bar(
                                        current=current_line,
                                        total=total_lines,
                                        model_name=model_name,
                                        chunk_size=effective_chunk,
                                        is_loading=True,
                                    )
                            except:
                                # Can't parse yet, just show loading
                                # Cap chunk_size to prevent exceeding total (defensive check)
                                effective_chunk = min(
                                    chunk_count, max(0, total_lines - current_line)
                                )
                                progress_bar(
                                    current=current_line,
                                    total=total_lines,
                                    model_name=model_name,
                                    chunk_size=effective_chunk,
                                    is_loading=True,
                                )

                # If timeout occurred during part processing, break chunk loop
                if timed_out:
                    break

            # Handle thinking timeout - retry if within limit
            if timed_out:
                if retry < max_retries_on_timeout:
                    clear_progress()
                    warning_with_progress(
                        f"Thinking timeout. Retrying (attempt {retry + 1}/{max_retries_on_timeout})..."
                    )
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
                error_with_progress(f"{provider_name} returned an empty response.")
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

            try:
                translated_batch = normalize_translated_batch(batch, translated_batch)
            except Exception as e:
                clear_progress()
                warning_with_progress(f"Invalid response structure: {e}")
                info_with_progress("Sending last batch again...")
                continue

            suspicious_indices = find_suspicious_unchanged_translations(
                batch, translated_batch, source_lang=source_lang
            )
            script_leak_indices = find_source_script_leaks(
                translated_batch, source_lang=source_lang
            )
            leak_threshold = max(3, (len(batch) * 8 + 99) // 100)
            if script_leak_indices:
                if source_leak_retry_count < max_source_leak_retries:
                    source_leak_retry_count += 1
                    corrective_message = (
                        f"Your previous answer left Japanese characters in lines {', '.join(map(str, script_leak_indices[:8]))}. "
                        "Translate those lines fully into Latin American Spanish. Do not leave any Japanese characters in the output. "
                        "If a name must stay recognizable, romanize it into Latin script instead of keeping Japanese glyphs."
                    )
                    clear_progress()
                    warning_with_progress(
                        "Detected untranslated Japanese characters in the batch. Retrying..."
                    )
                    continue
                raise ValueError(
                    f"Persistent untranslated Japanese characters detected in batch lines: {', '.join(map(str, script_leak_indices[:8]))}"
                )
            if len(suspicious_indices) >= leak_threshold:
                if source_leak_retry_count < max_source_leak_retries:
                    source_leak_retry_count += 1
                    corrective_message = (
                        f"Your previous answer left lines {', '.join(map(str, suspicious_indices[:8]))} effectively unchanged in {source_lang or 'the source language'}. "
                        "Re-evaluate the whole batch using scene context and reference_content, then return only corrected Latin American Spanish JSON."
                    )
                    clear_progress()
                    warning_with_progress(
                        "Detected source-language leakage in the batch. Retrying with stricter context review..."
                    )
                    continue
                raise ValueError(
                    f"Persistent source language leak detected in batch lines: {', '.join(map(str, suspicious_indices[:8]))}"
                )
            elif suspicious_indices:
                logger.log_only(
                    f"Minor unchanged-source warning ignored for batch: {', '.join(map(str, suspicious_indices[:8]))}"
                )

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
                if (
                    dialogue_lines is not None
                    and unique_text_indices is not None
                    and deduplication_keys is not None
                ):
                    dedup_key = deduplication_keys[idx]
                    if dedup_key in unique_text_indices:
                        for duplicate_idx in unique_text_indices[dedup_key]:
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
            types.Content(
                role="user",
                parts=[types.Part(text=json.dumps(batch, ensure_ascii=False))],
            ),
            types.Content(role="model", parts=response_parts),
        ]
    finally:
        # Restore logging level
        logging.getLogger().setLevel(old_level)


def save_incremental_output(
    subs, dialogue_events, translated_subtitle, original_texts, output_path
):
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


def translate_ass_file(
    ass_path,
    api_manager,
    model_name,
    audio_model_name,
    reference_subtitle_path,
    reference_lang_code,
    output_dir,
    original_mkv_stem,
    lang_code,
    original_format=".ass",
    batch_size=300,
    thinking=True,
    thinking_budget=2048,
    keep_original=False,
    audio_file=None,
    extract_audio=False,
    video_path=None,
    extra_context_text=None,
    temperature=None,
    top_p=None,
    top_k=None,
    strip_sdh=False,
    review=None,
    review_batch_size=150,
):
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
    provider = api_manager.provider
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    # Progress and output files
    progress_file_path = tmp_dir / f"{original_mkv_stem}.progress"
    output_ass_path = output_dir / f"{original_mkv_stem}.es-419{original_format}"

    # Set log file paths
    logger.set_log_file_path(str(output_dir / f"{original_mkv_stem}.translation.log"))
    logger.set_thoughts_file_path(str(output_dir / f"{original_mkv_stem}.thoughts.log"))

    # Import progress display functions
    from tools.progress_display import clear_progress

    # Audio handling for gender-aware translation
    audio_part = None
    audio_extracted = False

    try:
        if is_ollama_provider(provider) and (audio_file or extract_audio):
            logger.warning(
                "Audio-assisted translation is currently only supported with Gemini. Continuing without audio context."
            )
            audio_file = None
            extract_audio = False

        # Extract audio from video if requested
        if video_path and extract_audio:
            if video_path.exists():
                logger.info(
                    "Extracting audio from video for gender-aware translation..."
                )
                audio_file = prepare_audio(str(video_path))
                if audio_file:
                    audio_extracted = True
                else:
                    logger.warning(
                        "Failed to extract audio. Continuing without audio context."
                    )
            else:
                logger.error(f"Video file {video_path} does not exist.")

        # Read audio file if provided
        if audio_file and Path(audio_file).exists():
            logger.info(f"Loading audio file: {Path(audio_file).name}")
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
                audio_part = types.Part.from_bytes(
                    data=audio_bytes, mime_type="audio/mpeg"
                )
            logger.info("Audio loaded successfully. Gender-aware translation enabled.")
        elif audio_file:
            logger.error(f"Audio file {audio_file} does not exist.")

        reference_contexts = {}

        # Normalize ASS color codes to spec-compliant format before parsing
        normalize_ass_colors(ass_path)

        # Load and parse subtitle file
        # Color normalization already applied, so parsing should succeed
        try:
            subs = pysubs2.load(str(ass_path))
        except Exception as e:
            logger.error(f"Failed to parse ASS file {ass_path.name}: {e}")
            logger.error(
                "Note: Color normalization was already applied. This may be a different parsing error."
            )
            return None, batch_size

        if not subs:
            logger.warning(f"No subtitle events found in {ass_path.name}. Skipping.")
            return None, batch_size

        # Extract dialogue events (keep your existing logic)
        dialogue_events = []
        dialogue_lines = []
        original_texts = []
        sdh_events_to_remove = []  # Track SDH-only events to remove from output

        all_dialogue_events = [
            line for line in subs if hasattr(line, "type") and line.type == "Dialogue"
        ]

        if not all_dialogue_events:
            event_types = set(getattr(line, "type", "Unknown") for line in subs)
            logger.warning(
                f"No Dialogue events found in {ass_path.name}. Found event types: {event_types}"
            )
            return None, batch_size

        # Styles to exclude from translation (romanized lyrics - already readable)
        EXCLUDE_STYLES = [
            r".*Romaji$",  # Matches: OP-Romaji, ED-Romaji, Insert-Romaji
            r".*Romaji[- ]",  # Matches: Romaji-Top, Romaji Bottom
        ]

        def should_exclude_style(style_name):
            """Check if style should be excluded from translation (Romaji styles)."""
            if not style_name:
                return False
            for pattern in EXCLUDE_STYLES:
                if re.match(pattern, style_name, re.IGNORECASE):
                    return True
            return False

        ass_header_keywords = [
            "[Script Info]",
            "[V4+ Styles]",
            "[Events]",
            "[Aegisub",
            "Format:",
            "Style:",
            "ScriptType:",
            "PlayResX:",
            "PlayResY:",
            "WrapStyle:",
            "Title:",
            "Collisions:",
        ]

        excluded_count = 0
        vector_drawing_count = 0
        for event in all_dialogue_events:
            # Skip Romaji styles (romanized lyrics - already readable, don't need translation)
            if should_exclude_style(event.style):
                excluded_count += 1
                continue

            # Skip vector drawings (lines with \p1, \p2, etc. - no text to translate)
            # These are shapes/animations like Sign - Mask
            if (
                r"\p1" in event.text
                or r"\p2" in event.text
                or r"\p3" in event.text
                or r"\p4" in event.text
            ):
                vector_drawing_count += 1
                continue

            if not any(keyword in event.text for keyword in ass_header_keywords):
                plain_text = remove_formatting(event.text)
                if plain_text:
                    # Protect ASS directives (like \N) before translation
                    protected_text = protect_ass_directives(plain_text)

                    # Apply SDH stripping if enabled
                    if strip_sdh:
                        protected_text = strip_sdh_elements(protected_text)
                        # Skip events that become empty or SDH-only after stripping
                        if not protected_text or is_sdh_only_line(protected_text):
                            sdh_events_to_remove.append(event)
                            continue

                    dialogue_events.append(event)
                    dialogue_lines.append(protected_text)
                    original_texts.append(event.text)

        # Log SDH stripping stats if enabled
        if strip_sdh and sdh_events_to_remove:
            logger.info(
                f"Removed {len(sdh_events_to_remove)} SDH-only lines (sound effects, speaker names, etc.)"
            )

        if not dialogue_lines:
            logger.warning(f"No valid dialogue lines found in {ass_path.name}.")
            return None, batch_size

        if reference_subtitle_path:
            try:
                reference_entries = load_reference_subtitle_entries(
                    reference_subtitle_path, strip_sdh=strip_sdh
                )
                reference_contexts = build_reference_context_map(
                    dialogue_events, reference_entries
                )
                logger.info(
                    f"Loaded aligned {reference_lang_code} context for {len(reference_contexts)} dialogue lines"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load secondary subtitle context from {reference_subtitle_path.name}: {e}"
                )

        total_lines = len(dialogue_lines)

        if reference_contexts and batch_size > 120:
            adjusted_batch_size = min(batch_size, 120)
            logger.info(
                f"Secondary subtitle context enabled. Reducing batch size from {batch_size} to {adjusted_batch_size} for stability."
            )
            batch_size = adjusted_batch_size

        # Deduplicate lines (same text with different effects/layers)
        # Maps (unique text + aligned reference context) -> list of indices
        text_to_indices = {}
        deduplication_keys = []
        for i, line in enumerate(dialogue_lines):
            stripped = line.strip()
            dedup_key = (stripped, reference_contexts.get(i, ""))
            deduplication_keys.append(dedup_key)
            if dedup_key not in text_to_indices:
                text_to_indices[dedup_key] = []
            text_to_indices[dedup_key].append(i)

        duplicate_count = total_lines - len(text_to_indices)
        if duplicate_count > 0:
            logger.info(
                f"Found {duplicate_count} duplicate lines (same text with different effects)"
            )

        # Filter out lines too short to translate meaningfully
        # Only apply length filter to Latin/ASCII text (CJK can be meaningful in 1-2 chars)
        MIN_TRANSLATION_LENGTH = 2
        lines_to_translate = []  # List of indices that need translation
        translation_map = {}  # Maps index to original text for short lines
        unique_texts_to_translate = []  # Unique texts/contexts that need translation
        unique_text_indices = {}  # Maps dedup key -> indices

        for dedup_key, indices in text_to_indices.items():
            unique_text = dedup_key[0]
            first_idx = indices[0]  # Use first occurrence as representative

            # Only apply MIN_TRANSLATION_LENGTH to Latin scripts
            # CJK, Arabic, Cyrillic, etc. can convey meaning in 1-2 characters
            if (
                is_primarily_latin(unique_text)
                and len(unique_text) < MIN_TRANSLATION_LENGTH
            ):
                # Keep very short Latin text as-is (e.g., "OK", "Hi", "!")
                for idx in indices:
                    translation_map[idx] = dialogue_lines[idx]
            else:
                # This text needs translation (either non-Latin or long enough)
                unique_texts_to_translate.append(dedup_key)
                unique_text_indices[dedup_key] = indices
                lines_to_translate.append(
                    first_idx
                )  # Track first occurrence for progress

        # Sort lines_to_translate to ensure consistent ordering for progress tracking
        # This is critical for resume logic to work correctly
        lines_to_translate.sort()

        # Calculate total kept-as-is: Romaji lines + vector drawings + short Latin lines
        total_kept_as_is = excluded_count + vector_drawing_count + len(translation_map)
        total_original_lines = len(all_dialogue_events)

        # Show clean summary (total original lines = unique texts to translate + duplicates + kept as-is)
        unique_count = len(unique_texts_to_translate)
        print(
            f"Found {total_original_lines} dialogue lines ({unique_count} unique texts to translate, {total_kept_as_is} kept as-is)\n"
        )

        if duplicate_count > 0:
            logger.info(
                f"Deduplication: {duplicate_count} duplicate lines (same text with different effects)"
            )
        if vector_drawing_count > 0:
            logger.info(
                f"Excluded {vector_drawing_count} vector drawing lines (shapes/animations)"
            )

        if not lines_to_translate:
            logger.warning(
                f"No lines to translate in {ass_path.name} (all lines too short). Skipping."
            )
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
        translated_subtitle = (
            dialogue_lines.copy()
        )  # Start with original text (short lines stay as-is)

        if progress_file_path.exists():
            has_progress, saved_line = load_progress(progress_file_path, ass_path)
            if has_progress:
                # Calculate how many translatable lines have been completed
                completed_translatable = sum(
                    1 for idx in lines_to_translate if idx < saved_line
                )

                # Only prompt to resume if we've actually translated something
                if completed_translatable > 0 and prompt_resume(
                    completed_translatable, len(lines_to_translate)
                ):
                    start_line = saved_line

                    # Load partial output if it exists
                    if output_ass_path.exists():
                        try:
                            partial_subs = pysubs2.load(str(output_ass_path))
                            partial_events = [
                                e
                                for e in partial_subs
                                if hasattr(e, "type") and e.type == "Dialogue"
                            ]

                            partial_event_map = {}
                            for partial_event in partial_events:
                                signature = get_dialogue_event_signature(partial_event)
                                partial_event_map.setdefault(signature, []).append(
                                    partial_event
                                )

                            for i, event in enumerate(dialogue_events):
                                signature = get_dialogue_event_signature(event)
                                matching_events = partial_event_map.get(signature, [])
                                if matching_events:
                                    translated_subtitle[i] = remove_formatting(
                                        matching_events.pop(0).text
                                    )

                            logger.info(
                                f"Loaded {completed_translatable} previously translated lines"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load partial output: {e}")
                else:
                    # User chose to start over
                    if output_ass_path.exists():
                        output_ass_path.unlink()
                    progress_file_path.unlink()

        use_audio_fallback = False
        client = api_manager.get_client()

        if is_gemini_provider(provider) and audio_part and lines_to_translate:
            sample_idx = lines_to_translate[0]
            audio_probe_batch = [
                {
                    "index": str(sample_idx),
                    "content": dialogue_lines[sample_idx],
                    "time_start": str(dialogue_events[sample_idx].start),
                    "time_end": str(dialogue_events[sample_idx].end),
                }
            ]
            try:
                probe_audio_input_support(
                    client=client,
                    model_name=model_name,
                    sample_batch=audio_probe_batch,
                    audio_part=audio_part,
                )
            except Exception as probe_error:
                if is_audio_capability_error(str(probe_error)):
                    use_audio_fallback = True
                    logger.info(
                        f"Model {model_name} does not accept audio input. Using {audio_model_name} for gender analysis fallback."
                    )
                else:
                    logger.warning(
                        f"Audio capability probe failed unexpectedly: {probe_error}. Proceeding with normal translation flow."
                    )

        # Build system instruction (with audio context if available)
        system_instruction = get_system_instruction(
            lang_code,
            target_lang="Latin American Spanish",
            thinking=thinking,
            audio_file=audio_file if not use_audio_fallback else None,
            gender_hints=use_audio_fallback,
            reference_lang=reference_lang_code,
            extra_context_text=extra_context_text,
        )

        config = get_translation_config(
            system_instruction,
            model_name,
            thinking,
            thinking_budget,
            temperature,
            top_p,
            top_k,
            provider,
        )

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

        # Verify resume position matches expected (for debugging)
        if start_line > 0:
            # The actual position i should match completed_translatable
            # If they differ, there may be an issue with how progress was saved/loaded
            expected_completed = sum(
                1 for idx in lines_to_translate if idx < start_line
            )
            if i != expected_completed:
                logger.warning(
                    f"Resume position mismatch: expected {expected_completed} but skipped to {i}"
                )
                logger.warning(
                    f"This may cause progress display issues. Using actual position {i}."
                )
            logging.debug(f"Resume: actual position i={i}, total={total}")

        # Build context if resuming
        if start_line > 0:
            previous_message = build_resume_context(
                dialogue_lines,
                translated_subtitle,
                start_line,
                batch_size,
                provider,
                reference_contexts,
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
                output_path=output_ass_path,
            )

            # Save logs
            logger.save_logs()

            cleanup_tracked_processes()

            # Save progress (calculate current position accounting for partial success)
            if i > 0:
                current_position = max(1, i - len(batch) + max(0, last_chunk_size - 1))
                if current_position < len(lines_to_translate):
                    current_dialogue_line = lines_to_translate[current_position]
                    save_progress(
                        progress_file_path, current_dialogue_line, total_lines, ass_path
                    )

            sys.exit(0)

        signal.signal(signal.SIGINT, handle_interrupt)

        # Track quota error timing for smart API switching (gemini-srt-translator line 487)
        last_time = 0

        # Show initial progress bar (matching gemini-srt-translator line 460)
        progress_bar(current=i, total=total, model_name=model_name, is_sending=True)

        # Main translation loop (like gemini-translator-srt lines 489-606)
        batch = []  # Initialize batch outside loop for signal handler access
        while i < total:
            batch = []

            # Build batch up to batch_size
            batch_start_i = i
            while i < total and len(batch) < batch_size:
                line_idx = lines_to_translate[i]
                batch_item = {
                    "index": str(line_idx),
                    "content": dialogue_lines[line_idx],
                }
                if reference_contexts.get(line_idx):
                    batch_item["reference_content"] = reference_contexts[line_idx]
                # Add time codes if audio is present (for gender-aware translation)
                if audio_file:
                    batch_item["time_start"] = str(dialogue_events[line_idx].start)
                    batch_item["time_end"] = str(dialogue_events[line_idx].end)
                batch.append(batch_item)
                i += 1

            batch_end_i = batch_start_i + len(batch)
            logger.log_only(
                f"Starting batch {batch_number}: items {batch_start_i + 1}-{batch_end_i}/{total}"
            )

            # Translate batch with partial success tracking
            try:
                request_batch = batch
                request_audio_part = audio_part
                request_audio_file = audio_file if audio_part else None

                if audio_part and use_audio_fallback:
                    try:
                        gender_hints = analyze_audio_batch(
                            client=client,
                            audio_model_name=audio_model_name,
                            batch=batch,
                            audio_part=audio_part,
                            source_lang=lang_code,
                            progress_current=batch_start_i,
                            progress_total=total,
                            progress_model_name=model_name,
                        )
                        request_batch = attach_gender_hints_to_batch(
                            batch, gender_hints
                        )
                    except Exception as analysis_error:
                        warning_with_progress(
                            f"Audio fallback analysis failed for this batch: {analysis_error}"
                        )
                        info_with_progress(
                            "Continuing without gender hints for this batch."
                        )
                        request_batch = attach_gender_hints_to_batch(batch)

                    request_audio_part = None
                    request_audio_file = None

                # Validate batch size against the actual request payload
                while not validate_batch_tokens(
                    client, request_batch, model_name, provider
                ):
                    clear_progress()
                    new_batch_size = prompt_new_batch_size(batch_size)
                    decrement = batch_size - new_batch_size
                    if decrement > 0:
                        for _ in range(decrement):
                            i -= 1
                            batch.pop()
                    batch_size = new_batch_size
                    request_batch = batch
                    request_audio_part = audio_part
                    request_audio_file = audio_file if audio_part else None
                    if audio_part and use_audio_fallback:
                        try:
                            gender_hints = analyze_audio_batch(
                                client=client,
                                audio_model_name=audio_model_name,
                                batch=batch,
                                audio_part=audio_part,
                                source_lang=lang_code,
                                progress_current=batch_start_i,
                                progress_total=total,
                                progress_model_name=model_name,
                            )
                            request_batch = attach_gender_hints_to_batch(
                                batch, gender_hints
                            )
                        except Exception:
                            request_batch = attach_gender_hints_to_batch(batch)
                        request_audio_part = None
                        request_audio_file = None

                # Show sending indicator
                progress_bar(
                    current=batch_start_i,
                    total=total,
                    model_name=model_name,
                    is_sending=True,
                )

                previous_message = process_batch_streaming(
                    client=client,
                    model_name=model_name,
                    batch=request_batch,
                    previous_message=previous_message,
                    translated_subtitle=translated_subtitle,
                    config=config,
                    current_line=batch_start_i,
                    total_lines=total,
                    batch_number=batch_number,
                    keep_original=keep_original,
                    original_format=original_format,
                    audio_part=request_audio_part,
                    audio_file=request_audio_file,
                    dialogue_lines=dialogue_lines,
                    unique_text_indices=unique_text_indices,
                    deduplication_keys=deduplication_keys,
                    provider=provider,
                    source_lang=lang_code,
                )
                batch_number += 1  # Increment for next batch

                # Lock in completed progress before any next-batch prep/status updates.
                progress_bar(current=i, total=total, model_name=model_name)
                logger.log_only(
                    f"Completed batch {batch_number - 1}: items {batch_start_i + 1}-{batch_end_i}/{total}"
                )

                # Save progress after successful batch
                # Save index of next dialogue line to process (not position in lines_to_translate)
                if i < total:
                    current_dialogue_line = lines_to_translate[i]
                else:
                    current_dialogue_line = total_lines
                save_progress(
                    progress_file_path, current_dialogue_line, total_lines, ass_path
                )

                # Save logs incrementally after each batch
                logger.save_logs()

                # Save incremental output (silently, without disrupting progress bar)
                save_incremental_output(
                    subs=subs,
                    dialogue_events=dialogue_events,
                    translated_subtitle=translated_subtitle,
                    original_texts=original_texts,
                    output_path=output_ass_path,
                )

                batch.clear()

            except Exception as e:
                error_msg = str(e)
                last_chunk_size = get_last_chunk_size()

                # Clear progress bar before logging
                clear_progress()

                if (
                    audio_part
                    and not use_audio_fallback
                    and is_audio_capability_error(error_msg)
                ):
                    warning_with_progress(
                        f"Model {model_name} rejected audio input. Switching to audio fallback model {audio_model_name}."
                    )
                    use_audio_fallback = True
                    system_instruction = get_system_instruction(
                        lang_code,
                        target_lang="Latin American Spanish",
                        thinking=thinking,
                        audio_file=None,
                        gender_hints=True,
                        reference_lang=reference_lang_code,
                        extra_context_text=extra_context_text,
                    )
                    config = get_translation_config(
                        system_instruction,
                        model_name,
                        thinking,
                        thinking_budget,
                        temperature,
                        top_p,
                        top_k,
                        provider,
                    )
                    i = batch_start_i
                    batch.clear()
                    logger.save_logs()
                    progress_bar(current=i, total=total, model_name=model_name)
                    continue

                if is_ollama_provider(provider) and is_permanent_ollama_error(
                    error_msg
                ):
                    raise

                if "source language leak" in error_msg.lower():
                    raise

                # Handle quota errors with API switching or wait (gemini-srt-translator lines 553-564)
                if is_gemini_provider(provider) and (
                    "quota" in error_msg.lower()
                    or "503" in error_msg
                    or "UNAVAILABLE" in error_msg
                ):
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
                            progress_bar(
                                batch_start_i,
                                total,
                                model_name,
                                is_retrying=True,
                                retry_countdown=j,
                            )
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
                        save_progress(
                            progress_file_path,
                            current_dialogue_line,
                            total_lines,
                            ass_path,
                        )

                    # Save incremental output with partial success
                    save_incremental_output(
                        subs=subs,
                        dialogue_events=dialogue_events,
                        translated_subtitle=translated_subtitle,
                        original_texts=original_texts,
                        output_path=output_ass_path,
                    )

                # Resume progress bar
                progress_bar(current=i, total=total, model_name=model_name)

        # Show completion
        progress_complete(total, total, model_name)

        # Final validation
        if len(translated_subtitle) != len(dialogue_lines):
            logger.error(
                f"Line count mismatch: {len(translated_subtitle)} vs {len(dialogue_lines)}"
            )
            return None, batch_size

        # Restore ASS formatting to final translations
        for i, event in enumerate(dialogue_events):
            # First restore ASS directives from placeholders
            restored_directives = restore_ass_directives(translated_subtitle[i])
            # Then restore ASS formatting tags
            event.text = restore_formatting(original_texts[i], restored_directives)

        # --- Review pass: check for gender, consistency, and quality issues ---
        # review: True=always, False=never, None=ask interactively
        should_review = review
        if should_review is None:
            clear_progress()
            should_review = prompt_yes_no(
                "Translation complete. Run AI review pass to check for gender agreement, "
                "consistency, and other issues?",
                default=False,
            )
        if should_review:
            try:
                review_corrections = review_translation(
                    dialogue_events=dialogue_events,
                    original_texts=original_texts,
                    translated_subtitle=translated_subtitle,
                    source_lang=lang_code,
                    target_lang="Latin American Spanish",
                    api_manager=api_manager,
                    model_name=model_name,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    extra_context_text=extra_context_text,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    review_batch_size=review_batch_size,
                )
                logger.info(f"Review applied {review_corrections} correction(s)")
            except Exception as e:
                logger.warning(f"Review pass failed (translation is still saved): {e}")

        # Remove SDH-only events from the subtitle file before saving
        if strip_sdh and sdh_events_to_remove:
            for event in sdh_events_to_remove:
                if event in subs:
                    subs.remove(event)

        # Save final output
        subs.save(str(output_ass_path))
        logger.success(
            f"Successfully created translated subtitle file: {output_ass_path.name}"
        )

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
        if "i" in locals() and i < len(lines_to_translate):
            # Save the current dialogue line index
            current_dialogue_line = lines_to_translate[i]
            save_progress(
                progress_file_path, current_dialogue_line, total_lines, ass_path
            )
            logger.info("Progress saved. Run again to resume.")
        logger.save_logs()
        return None, batch_size

    except Exception as e:
        clear_progress()
        logger.error(f"Translation failed: {e}")
        logger.save_logs()
        if "i" in locals() and i < len(lines_to_translate):
            # Save the current dialogue line index
            current_dialogue_line = lines_to_translate[i]
            save_progress(
                progress_file_path, current_dialogue_line, total_lines, ass_path
            )
        return None, batch_size


def merge_subtitles_to_mkv(mkv_path, translated_subtitle_path, output_mkv_dir):
    """
    Merges the translated subtitle file (ASS, SRT, or SSA) into a copy of the original .mkv file.
    """
    try:
        output_mkv_name = f"{mkv_path.stem}.translated.mkv"
        output_mkv_path = output_mkv_dir / output_mkv_name

        logger.info(
            f"Merging translated subtitles into new file: {output_mkv_path.name}"
        )

        mkvmerge_cmd = [
            "mkvmerge",
            "-o",
            str(output_mkv_path),
            "--language",
            f"0:es-419",
            "--track-name",
            "0:Spanish (Latin America)",
            "--default-track-flag",
            "0:yes",
            str(translated_subtitle_path),
            str(mkv_path),
        ]

        result = subprocess.run(
            mkvmerge_cmd, capture_output=True, text=True, encoding="utf-8"
        )

        # mkvmerge exit codes: 0=success, 1=warnings (still successful), 2+=error
        if result.returncode == 0:
            logger.success(
                f"Successfully created merged MKV file: {output_mkv_path.name}"
            )
            return output_mkv_path
        elif result.returncode == 1:
            logger.success(
                f"Successfully created merged MKV file: {output_mkv_path.name}"
            )
            if result.stderr:
                logger.warning(f"mkvmerge warnings: {result.stderr.strip()}")
            return output_mkv_path
        else:
            logger.error(
                f"Failed to merge subtitles for {mkv_path.name}: {result.stderr}"
            )
            return None

    except subprocess.CalledProcessError as e:
        # Should not reach here since we removed check=True
        logger.error(f"Failed to merge subtitles for {mkv_path.name}: {e.stderr}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during merging for {mkv_path.name}: {e}"
        )
        return None


def run_ocr_batch_ollama(
    client,
    model_name,
    frame_batch,
    temperature=None,
    top_p=None,
    top_k=None,
):
    """Extract burned-in subtitle text from one image batch using Ollama vision."""
    images = []
    for sample in frame_batch:
        images.append(base64.b64encode(sample.get_image_bytes()).decode("ascii"))

    system_prompt = (
        "You are an OCR extractor for burned-in subtitles. "
        "Read subtitle text exactly as it appears in each image. "
        "Do not translate, explain, or summarize. "
        "Ignore logos, watermarks, UI, credits, and background scene details."
    )
    user_prompt = (
        f"There are {len(frame_batch)} images attached in order from ordinal 0 to {len(frame_batch) - 1}. "
        "Return a JSON object with one field named results. "
        "results must be an array with exactly one object per image. "
        "Each object must contain: ordinal (integer) and text (string). "
        "If an image has no readable subtitle text, use an empty string for text. "
        "Preserve subtitle line breaks using \\n inside the text field."
    )
    response_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ordinal": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["ordinal", "text"],
                },
            }
        },
        "required": ["results"],
    }

    options = {"temperature": 0}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": images},
    ]

    with suppress_stderr_output():
        response = client.chat(
            model=model_name,
            messages=messages,
            stream=False,
            format=response_schema,
            think=False,
            options=options,
        )

    response_text = extract_ollama_chunk_text(response)
    if not response_text or not response_text.strip():
        raise ValueError("OCR model returned an empty response")

    parsed = json_repair.loads(response_text)
    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list):
        raise ValueError("OCR response did not contain a results array")

    by_ordinal = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        ordinal = item.get("ordinal")
        text = item.get("text")
        if isinstance(ordinal, int) and isinstance(text, str):
            by_ordinal[ordinal] = text

    return [by_ordinal.get(i, "") for i in range(len(frame_batch))]


def parse_gemini_ocr_payload(response):
    """Parse structured Gemini OCR responses with a text fallback."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    response_text = getattr(response, "text", "") or ""
    if not response_text.strip():
        raise ValueError("OCR model returned an empty response")

    parsed = json_repair.loads(response_text)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("OCR response did not contain a JSON object")


def run_ocr_batch_gemini(
    client,
    model_name,
    frame_batch,
    temperature=None,
    top_p=None,
    top_k=None,
):
    """Extract burned-in subtitle text from one image batch using Gemini."""
    response_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ordinal": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["ordinal", "text"],
                },
            }
        },
        "required": ["results"],
    }
    user_prompt = (
        f"There are {len(frame_batch)} images attached in order from ordinal 0 to {len(frame_batch) - 1}. "
        "Read subtitle text exactly as shown in each image. "
        "Do not translate, explain, or summarize. "
        "Ignore logos, watermarks, UI, credits, and background scene details. "
        "Return JSON with one field named results. "
        "results must contain exactly one object per image. "
        "Each object must contain ordinal (integer) and text (string). "
        "Use an empty string only when an image truly has no readable subtitle text. "
        "Preserve subtitle line breaks using \\n inside text."
    )
    parts = [types.Part.from_text(text=user_prompt)]
    for sample in frame_batch:
        image_bytes = sample.get_image_bytes()
        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=infer_image_mime_type(image_bytes),
            )
        )

    config_kwargs = {
        "response_mime_type": "application/json",
        "response_json_schema": response_schema,
        "temperature": 0 if temperature is None else temperature,
        "system_instruction": "You are an OCR extractor for burned-in subtitle images.",
    }
    if top_p is not None:
        config_kwargs["top_p"] = top_p
    if top_k is not None:
        config_kwargs["top_k"] = top_k

    wait_for_gemini_request_slot("gemini")
    with suppress_stderr_output():
        response = client.models.generate_content(
            model=model_name,
            contents=parts,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    parsed = parse_gemini_ocr_payload(response)
    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list):
        raise ValueError("OCR response did not contain a results array")

    by_ordinal = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        ordinal = item.get("ordinal")
        text = item.get("text")
        if isinstance(ordinal, int) and isinstance(text, str):
            by_ordinal[ordinal] = text

    return [by_ordinal.get(i, "") for i in range(len(frame_batch))]


def run_ocr_single_image_strict(
    client,
    model_name,
    sample,
    temperature=None,
    top_p=None,
    top_k=None,
):
    """Retry OCR for a single image with a stricter prompt when empty text looks suspicious."""
    image_b64 = base64.b64encode(sample.get_image_bytes()).decode("ascii")
    system_prompt = (
        "You are an OCR extractor for subtitle images. "
        "If there is any visible subtitle text at all, transcribe it exactly. "
        "Only return an empty string when the image truly contains no readable subtitle glyphs. "
        "Do not translate or explain anything."
    )
    response_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    options = {"temperature": 0}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Return JSON with one field named text. Preserve subtitle line breaks using \\n.",
            "images": [image_b64],
        },
    ]

    with suppress_stderr_output():
        response = client.chat(
            model=model_name,
            messages=messages,
            stream=False,
            format=response_schema,
            think=False,
            options=options,
        )

    response_text = extract_ollama_chunk_text(response)
    parsed = json_repair.loads(response_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return parsed["text"]
    raise ValueError("Strict OCR response did not contain a text field")


def run_ocr_single_image_strict_gemini(
    client,
    model_name,
    sample,
    temperature=None,
    top_p=None,
    top_k=None,
):
    """Retry one Gemini OCR image with a stricter prompt when empty text looks suspicious."""
    response_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    image_bytes = sample.get_image_bytes()
    config_kwargs = {
        "response_mime_type": "application/json",
        "response_json_schema": response_schema,
        "temperature": 0 if temperature is None else temperature,
        "system_instruction": (
            "You are an OCR extractor for subtitle images. "
            "If there is any visible subtitle text at all, transcribe it exactly. "
            "Only return an empty string when the image truly contains no readable subtitle glyphs."
        ),
    }
    if top_p is not None:
        config_kwargs["top_p"] = top_p
    if top_k is not None:
        config_kwargs["top_k"] = top_k

    contents = [
        types.Part.from_text(
            text="Return JSON with one field named text. Preserve subtitle line breaks using \\n."
        ),
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=infer_image_mime_type(image_bytes),
        ),
    ]

    wait_for_gemini_request_slot("gemini")
    with suppress_stderr_output():
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    parsed = parse_gemini_ocr_payload(response)
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return parsed["text"]
    raise ValueError("Strict OCR response did not contain a text field")


def repair_empty_ocr_results(
    client,
    provider,
    model_name,
    frame_batch,
    texts,
    temperature=None,
    top_p=None,
    top_k=None,
):
    """Re-check empty OCR outputs one-by-one so visible subtitle lines are not lost silently."""
    repaired = list(texts)
    for idx, text in enumerate(repaired):
        if text:
            continue
        try:
            if is_gemini_provider(provider):
                retried = run_ocr_single_image_strict_gemini(
                    client,
                    model_name,
                    frame_batch[idx],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                retried = run_ocr_single_image_strict(
                    client,
                    model_name,
                    frame_batch[idx],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            repaired[idx] = retried
        except Exception as exc:
            logger.log_only(f"Strict OCR retry failed for one image: {exc}")
    return repaired


def run_ocr_batch_resilient(
    client,
    provider,
    model_name,
    frame_batch,
    temperature=None,
    top_p=None,
    top_k=None,
    retries_remaining=2,
):
    """Run OCR with automatic batch splitting when the model returns bad output."""
    try:
        if is_gemini_provider(provider):
            texts = run_ocr_batch_gemini(
                client,
                model_name,
                frame_batch,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            texts = run_ocr_batch_ollama(
                client,
                model_name,
                frame_batch,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        return repair_empty_ocr_results(
            client,
            provider,
            model_name,
            frame_batch,
            texts,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as exc:
        if retries_remaining > 0:
            logger.log_only(
                f"OCR batch of {len(frame_batch)} failed ({exc}). Retrying {retries_remaining} more time(s)."
            )
            time.sleep(2)
            return run_ocr_batch_resilient(
                client,
                provider,
                model_name,
                frame_batch,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                retries_remaining=retries_remaining - 1,
            )

        if len(frame_batch) <= 1:
            raise RuntimeError(
                f"OCR failed for a single subtitle bitmap after retries: {exc}"
            )

        midpoint = max(1, len(frame_batch) // 2)
        logger.log_only(
            f"OCR batch of {len(frame_batch)} failed ({exc}). Retrying as {midpoint} + {len(frame_batch) - midpoint}."
        )
        left = run_ocr_batch_resilient(
            client,
            provider,
            model_name,
            frame_batch[:midpoint],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            retries_remaining=2,
        )
        right = run_ocr_batch_resilient(
            client,
            provider,
            model_name,
            frame_batch[midpoint:],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            retries_remaining=2,
        )
        return left + right


def run_ocr_batch_with_progress(
    client,
    provider,
    model_name,
    frame_batch,
    current,
    total,
    batch_index,
    total_batches,
    temperature=None,
    top_p=None,
    top_k=None,
):
    """Run one OCR batch while keeping the OCR progress bar animated."""
    stop_event = threading.Event()
    start_time = time.time()
    max_partial = max(0, len(frame_batch) - 1)

    def get_partial_progress():
        if max_partial <= 0:
            return 0
        elapsed = time.time() - start_time
        # Smoothly advance the visible count within the current batch while waiting.
        ramp_seconds = min(30.0, max(8.0, float(len(frame_batch))))
        return min(max_partial, int((elapsed / ramp_seconds) * max_partial))

    def heartbeat():
        while not stop_event.wait(1):
            progress_bar(
                current=current,
                total=total,
                chunk_size=get_partial_progress(),
                model_name=model_name,
                is_loading=True,
                status_detail="Reading subtitles",
                task_label="OCR",
            )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        return run_ocr_batch_resilient(
            client,
            provider,
            model_name,
            frame_batch,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=0.2)


def process_mkv_ocr_file(
    mkv_path,
    output_dir,
    api_manager,
    model_name,
    audio_model_name,
    ocr_lang,
    batch_size=300,
    thinking=True,
    thinking_budget=2048,
    keep_original=False,
    extra_context_text=None,
    temperature=None,
    top_p=None,
    top_k=None,
    strip_sdh=False,
    skip_ocr_review=False,
    review=None,
    review_batch_size=150,
):
    """Process a hard-subbed MKV by OCRing burned-in subtitles before translation."""
    print(f"\n{'=' * 60}")
    print(f"Processing with OCR: {mkv_path.name}")
    print(f"{'=' * 60}\n")

    expected_output_name = f"{mkv_path.stem}.translated.mkv"
    expected_output_path = output_dir / expected_output_name
    if expected_output_path.exists():
        logger.info(f"Output file '{expected_output_name}' already exists. Skipping.")
        return batch_size

    if not is_ocr_capable_provider(api_manager.provider):
        logger.error("--ocr requires a provider with image support.")
        return batch_size

    if not check_ffmpeg_tools():
        logger.error("ffmpeg and ffprobe are required for OCR mode.")
        return batch_size

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    ocr_dir = tmp_dir / f"{mkv_path.stem}.ocr"
    ocr_dir.mkdir(exist_ok=True)

    try:
        mkvmerge_cmd = ["mkvmerge", "-J", str(mkv_path)]
        result = subprocess.run(
            mkvmerge_cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        mkv_info = json.loads(result.stdout)
        tracks = mkv_info.get("tracks", [])

        ocr_track, selected_ocr_lang, subtitle_stream_index = select_ocr_subtitle_track(
            tracks, preferred_lang=ocr_lang
        )

        width, height, duration_s = get_video_info(mkv_path)
        crop_filter = resolve_ocr_crop_filter(width, height)
        grouped_ocr_samples = []
        if ocr_track is not None:
            logger.log_only(
                f"Rendering subtitle track id={ocr_track.get('id')} ({selected_ocr_lang}) directly into OCR bitmaps."
            )
            actual_frame_total = count_subtitle_bitmap_frames_full_stream(
                mkv_path,
                crop_filter,
                subtitle_stream_index,
            )
            logger.log_only(
                f"Detected {actual_frame_total} displayed subtitle frames for OCR extraction."
            )

            cached_samples = _load_ocr_extract_cache(
                mkv_path,
                crop_filter=crop_filter,
                ocr_mode="subtitle_bitmap",
                ocr_fps=None,
                subtitle_stream_index=subtitle_stream_index,
            )
            if cached_samples is not None:
                logger.info("Reusing cached OCR frame extraction for subtitle bitmap mode.")
                frame_samples = cached_samples
                progress_bar(
                    current=len(frame_samples),
                    total=max(len(frame_samples), 1),
                    model_name=model_name,
                    is_loading=True,
                    status_detail="Loading cached OCR frames",
                    task_label="OCR",
                    count_text=f"{len(frame_samples)}/{len(frame_samples)}",
                )
            else:
                progress_bar(
                    current=0,
                    total=max(1, actual_frame_total),
                    model_name=model_name,
                    is_loading=True,
                    status_detail="Extracting OCR frames",
                    task_label="OCR",
                    count_text=f"0/{max(1, actual_frame_total)}",
                )

                def log_extraction_progress(done, total):
                    display_total = max(total, done + 1)
                    progress_bar(
                        current=done,
                        total=max(display_total, 1),
                        model_name=model_name,
                        is_loading=True,
                        status_detail="Extracting OCR frames",
                        task_label="OCR",
                        count_text=f"{done}/{display_total}",
                    )

                frame_samples = extract_subtitle_bitmap_frames_full_stream(
                    mkv_path,
                    crop_filter,
                    subtitle_stream_index,
                    expected_total=actual_frame_total,
                    progress_callback=log_extraction_progress,
                )
                progress_bar(
                    current=len(frame_samples),
                    total=max(len(frame_samples), 1),
                    model_name=model_name,
                    is_loading=True,
                    status_detail="Extracting OCR frames",
                    task_label="OCR",
                    count_text=f"{len(frame_samples)}/{len(frame_samples)}",
                )
                _save_ocr_extract_cache(
                    mkv_path,
                    crop_filter=crop_filter,
                    ocr_mode="subtitle_bitmap",
                    ocr_fps=None,
                    subtitle_stream_index=subtitle_stream_index,
                    frame_samples=frame_samples,
                )
                logger.info("Saved OCR frame extraction to cache for future reuse.")

            ocr_samples = select_distinct_image_samples(
                frame_samples, recheck_every=OCR_DEFAULT_RECHECK_EVERY
            )
            grouped_ocr_samples = group_ocr_samples(frame_samples, ocr_samples)
            logger.log_only(
                f"Prepared {len(frame_samples)} subtitle bitmap frames and selected {len(ocr_samples)} distinct OCR samples."
            )
        else:
            selected_ocr_lang = ocr_lang
            logger.warning(
                "No OCR-capable subtitle track found for direct subtitle rendering."
            )
            if not prompt_yes_no(
                "Continue with slower full video-frame OCR extraction? This can take a while",
                default=False,
            ):
                logger.info("OCR cancelled before full video-frame extraction.")
                return batch_size

            logger.log_only(
                "Falling back to direct video-frame OCR after user confirmation."
            )
            estimated_frame_total = max(1, math.ceil(duration_s * OCR_DEFAULT_FPS))
            extraction_total = max(estimated_frame_total * 2, 1)

            cached_samples = _load_ocr_extract_cache(
                mkv_path,
                crop_filter=crop_filter,
                ocr_mode="video_frame",
                ocr_fps=OCR_DEFAULT_FPS,
                subtitle_stream_index=subtitle_stream_index,
            )
            if cached_samples is not None:
                logger.info("Reusing cached OCR frame extraction for video-frame mode.")
                frame_samples = cached_samples
                progress_bar(
                    current=extraction_total,
                    total=extraction_total,
                    model_name=model_name,
                    is_loading=True,
                    status_detail="Loading cached OCR frames",
                    task_label="OCR",
                    count_text=f"{extraction_total}/{extraction_total}",
                )
            else:
                progress_bar(
                    current=0,
                    total=extraction_total,
                    model_name=model_name,
                    is_loading=True,
                    status_detail="Extracting OCR frames",
                    task_label="OCR",
                    count_text=f"0/{extraction_total}",
                )
                fallback_progress_total = extraction_total

                def log_fallback_extraction_progress(done, total):
                    nonlocal fallback_progress_total
                    fallback_progress_total = max(total, done, 1)
                    progress_bar(
                        current=done,
                        total=fallback_progress_total,
                        model_name=model_name,
                        is_loading=True,
                        status_detail="Extracting OCR frames",
                        task_label="OCR",
                        count_text=f"{done}/{fallback_progress_total}",
                    )

                frame_samples = extract_ocr_frames(
                    mkv_path,
                    ocr_dir,
                    OCR_DEFAULT_FPS,
                    crop_filter,
                    subtitle_stream_index=subtitle_stream_index,
                    expected_total=estimated_frame_total,
                    progress_callback=log_fallback_extraction_progress,
                )
                progress_bar(
                    current=fallback_progress_total,
                    total=fallback_progress_total,
                    model_name=model_name,
                    is_loading=True,
                    status_detail="Extracting OCR frames",
                    task_label="OCR",
                    count_text=f"{fallback_progress_total}/{fallback_progress_total}",
                )
                _save_ocr_extract_cache(
                    mkv_path,
                    crop_filter=crop_filter,
                    ocr_mode="video_frame",
                    ocr_fps=OCR_DEFAULT_FPS,
                    subtitle_stream_index=subtitle_stream_index,
                    frame_samples=frame_samples,
                )
                logger.info("Saved OCR frame extraction to cache for future reuse.")

            ocr_samples = select_distinct_frame_samples(
                frame_samples,
                diff_threshold=OCR_DEFAULT_FRAME_DIFF,
                recheck_every=OCR_DEFAULT_RECHECK_EVERY,
            )
            grouped_ocr_samples = group_ocr_samples(frame_samples, ocr_samples)
            logger.log_only(
                f"Prepared {len(frame_samples)} sampled video frames and selected {len(ocr_samples)} distinct OCR samples."
            )
        if not ocr_samples:
            logger.error("No OCR frames were selected for analysis.")
            return batch_size

        logger.log_only(
            f"OCR extracted {len(frame_samples)} sampled frames and will OCR all {len(ocr_samples)} samples."
        )

        session_metadata = get_ocr_session_metadata(api_manager.provider, model_name)
        ocr_text_cache = {}
        if not skip_ocr_review:
            ocr_text_cache = load_saved_ocr_text_cache(
                mkv_path, ocr_samples, session_metadata
            )
        cached_distinct_count = len(ocr_text_cache)
        if cached_distinct_count:
            logger.info(
                f"Reusing saved OCR results for {cached_distinct_count}/{len(ocr_samples)} OCR samples."
            )

        progress_bar(
            current=min(cached_distinct_count, len(ocr_samples)),
            total=len(ocr_samples),
            model_name=model_name,
            is_loading=True,
            status_detail="Reading subtitles",
            task_label="OCR",
        )

        client = api_manager.get_client() if cached_distinct_count < len(ocr_samples) else None
        total_batches = (len(ocr_samples) + OCR_DEFAULT_REQUEST_BATCH_SIZE - 1) // OCR_DEFAULT_REQUEST_BATCH_SIZE
        for batch_index, start in enumerate(
            range(0, len(ocr_samples), OCR_DEFAULT_REQUEST_BATCH_SIZE), start=1
        ):
            frame_batch = ocr_samples[start : start + OCR_DEFAULT_REQUEST_BATCH_SIZE]
            progress_bar(
                current=start,
                total=len(ocr_samples),
                model_name=model_name,
                is_loading=True,
                status_detail="Reading subtitles",
                task_label="OCR",
            )
            uncached_pairs = [
                (sample, sample.index)
                for sample in frame_batch
                if sample.index not in ocr_text_cache
            ]

            if uncached_pairs:
                uncached_texts = run_ocr_batch_with_progress(
                    client,
                    api_manager.provider,
                    model_name,
                    [sample for sample, _ in uncached_pairs],
                    current=start,
                    total=len(ocr_samples),
                    batch_index=batch_index,
                    total_batches=total_batches,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                for (_, image_hash), text in zip(uncached_pairs, uncached_texts):
                    ocr_text_cache[image_hash] = text

            progress_bar(
                current=min(start + len(frame_batch), len(ocr_samples)),
                total=len(ocr_samples),
                model_name=model_name,
                is_loading=True,
                status_detail="Reading subtitles",
                task_label="OCR",
            )
            if batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0:
                logger.log_only(f"OCR batches completed: {batch_index}/{total_batches}")

        if skip_ocr_review:
            corrected_text_cache = dict(ocr_text_cache)
        else:
            logger.info("Opening OCR review UI before subtitle translation...")
            corrected_text_cache = review_ocr_text_in_webui(
                mkv_path=mkv_path,
                distinct_samples=ocr_samples,
                initial_text_by_hash=ocr_text_cache,
                logger=logger,
                session_metadata=session_metadata,
            )

        expanded_text_cache = expand_grouped_ocr_text(
            grouped_ocr_samples, corrected_text_cache
        )
        observed_text = [
            (sample.timestamp_s, expanded_text_cache.get(sample.index, ""))
            for sample in frame_samples
        ]

        positive_gaps = [
            later.timestamp_s - earlier.timestamp_s
            for earlier, later in zip(frame_samples, frame_samples[1:])
            if later.timestamp_s > earlier.timestamp_s
        ]
        sample_interval_s = statistics.median(positive_gaps) if positive_gaps else 0.25

        source_subtitle_path = tmp_dir / f"{mkv_path.stem}.{selected_ocr_lang}.ocr.srt"
        build_srt_from_ocr_results(
            observed_text,
            source_subtitle_path,
            sample_interval_s=sample_interval_s,
        )
        logger.success(f"Created OCR subtitle file: {source_subtitle_path.name}")

        translated_subtitle_path, final_batch_size = translate_ass_file(
            source_subtitle_path,
            api_manager,
            model_name,
            audio_model_name,
            None,
            None,
            tmp_dir,
            mkv_path.stem,
            selected_ocr_lang,
            ".srt",
            batch_size,
            thinking,
            thinking_budget,
            keep_original,
            None,
            False,
            mkv_path,
            extra_context_text,
            temperature,
            top_p,
            top_k,
            strip_sdh,
            review,
            review_batch_size,
        )

        if translated_subtitle_path:
            merge_subtitles_to_mkv(mkv_path, translated_subtitle_path, output_dir)

        return final_batch_size
    except Exception as e:
        logger.error(f"OCR processing failed for {mkv_path.name}: {e}")
        return batch_size


def process_raw_subtitle_file(
    subtitle_path,
    output_dir,
    api_manager,
    model_name,
    audio_model_name,
    source_lang,
    batch_size=300,
    thinking=True,
    thinking_budget=2048,
    keep_original=False,
    audio_file=None,
    extract_audio=False,
    extra_context_text=None,
    temperature=None,
    top_p=None,
    top_k=None,
    strip_sdh=False,
    review=None,
    review_batch_size=150,
):
    """Translate a standalone text subtitle file directly."""
    source_lang = normalize_track_language(source_lang)
    print(f"\n{'=' * 60}")
    print(f"Processing subtitle file: {subtitle_path.name}")
    print(f"{'=' * 60}\n")

    subtitle_extension = subtitle_path.suffix.lower()
    expected_output_name = f"{subtitle_path.stem}.es-419{subtitle_extension}"
    expected_output_path = output_dir / expected_output_name
    if expected_output_path.exists():
        logger.info(f"Output file '{expected_output_name}' already exists. Skipping.")
        return batch_size

    logger.info(f"Using source language {source_lang} for raw subtitle translation.")

    if extract_audio:
        logger.warning(
            "--extract-audio requires a video input. Ignoring it for raw subtitle files."
        )
        extract_audio = False

    translated_subtitle_path, final_batch_size = translate_ass_file(
        subtitle_path,
        api_manager,
        model_name,
        audio_model_name,
        None,
        None,
        output_dir,
        subtitle_path.stem,
        source_lang,
        subtitle_extension,
        batch_size,
        thinking,
        thinking_budget,
        keep_original,
        audio_file,
        extract_audio,
        None,
        extra_context_text,
        temperature,
        top_p,
        top_k,
        strip_sdh,
        review,
        review_batch_size,
    )

    if translated_subtitle_path:
        logger.success(f"Created translated subtitle file: {translated_subtitle_path.name}")

    return final_batch_size


def process_mkv_file(
    mkv_path,
    output_dir,
    api_manager,
    model_name,
    audio_model_name,
    remembered_lang=None,
    remembered_secondary_lang=None,
    batch_size=300,
    thinking=True,
    thinking_budget=2048,
    keep_original=False,
    audio_file=None,
    extract_audio=False,
    extra_context_text=None,
    temperature=None,
    top_p=None,
    top_k=None,
    strip_sdh=False,
    review=None,
    review_batch_size=150,
):
    """
    Processes a single MKV file: detects subtitles, prompts for selection (if needed),
    extracts, translates, and merges the chosen track.
    Returns tuple: (primary language code, secondary language code, final batch size)
    to be remembered for subsequent files.

    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {mkv_path.name}")
    print(f"{'=' * 60}\n")

    # Pre-flight check
    expected_output_name = f"{mkv_path.stem}.translated.mkv"
    expected_output_path = output_dir / expected_output_name

    if expected_output_path.exists():
        logger.info(f"Output file '{expected_output_name}' already exists. Skipping.")
        return None, None, batch_size

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # 1. Identify tracks using mkvmerge
        mkvmerge_cmd = ["mkvmerge", "-J", str(mkv_path)]
        result = subprocess.run(
            mkvmerge_cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        mkv_info = json.loads(result.stdout)

        # 2. Select subtitle track
        (
            selected_track,
            lang_code,
            secondary_track,
            secondary_lang_code,
        ) = select_subtitle_tracks(
            mkv_info.get("tracks", []), remembered_lang, remembered_secondary_lang
        )

        if selected_track is None:
            logger.warning(
                f"No suitable subtitle track found in {mkv_path.name}. Skipping."
            )
            return None, None, batch_size

        # 3. Extract primary and optional secondary subtitle tracks
        extracted_ass_path, subtitle_extension = extract_subtitle_track(
            mkv_path, selected_track, tmp_dir, lang_code
        )
        if not extracted_ass_path:
            return None, None, batch_size

        reference_subtitle_path = None
        if secondary_track and secondary_lang_code:
            reference_subtitle_path, _ = extract_subtitle_track(
                mkv_path,
                secondary_track,
                tmp_dir,
                secondary_lang_code,
                label="reference",
            )
            if reference_subtitle_path:
                logger.info(f"Using secondary subtitle context: {secondary_lang_code}")
            else:
                logger.warning(
                    "Failed to extract secondary subtitle context track. Continuing with primary subtitles only."
                )
                secondary_lang_code = None

        # 4. Translate the extracted file (preserving original format)
        translated_ass_path, final_batch_size = translate_ass_file(
            extracted_ass_path,
            api_manager,
            model_name,
            audio_model_name,
            reference_subtitle_path,
            secondary_lang_code,
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
            extra_context_text,
            temperature,
            top_p,
            top_k,
            strip_sdh,
            review,
            review_batch_size,
        )

        # 5. Merge the translated subtitle back into a new MKV
        if translated_ass_path:
            merge_subtitles_to_mkv(mkv_path, translated_ass_path, output_dir)

        return lang_code, secondary_lang_code, final_batch_size

    except FileNotFoundError:
        logger.error(
            "mkvmerge or mkvextract not found. Please ensure MKVToolNix is installed."
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command-line tool failed for {mkv_path.name}: {e}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON output from mkvmerge for {mkv_path.name}.")
    except Exception as e:
        logger.error(f"Unexpected error while processing {mkv_path.name}: {e}")

    return None, None, batch_size


# --- Main Execution ---


def main():
    import signal

    signal.signal(signal.SIGINT, handle_process_termination)
    signal.signal(signal.SIGTERM, handle_process_termination)

    parser = argparse.ArgumentParser(
        description="Detects subtitles in .mkv files or translates raw subtitle files to Spanish using Google Gemini or Ollama.\n\n"
        "Usage:\n"
        "  %(prog)s <file.mkv>              # Process a single MKV\n"
        "  %(prog)s <file.ass>              # Translate a raw text subtitle file\n"
        "  %(prog)s <directory>             # Process supported files in a directory",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        help="LLM provider to use: gemini, ollama-local, or ollama-cloud (default: env LLM_PROVIDER or gemini).",
    )
    parser.add_argument(
        "--base-url",
        help="Custom API base URL. Useful for Ollama (for example http://127.0.0.1:11434 or https://ollama.com).",
    )
    parser.add_argument(
        "--api-key",
        help="Primary API key for the selected provider. Gemini: GEMINI_API_KEY/GOOGLE_API_KEY. Ollama Cloud: OLLAMA_API_KEY.",
    )
    parser.add_argument(
        "--api-key2",
        help="Secondary Gemini API key for quota failover (optional).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="The model to use for translation. Defaults to 'models/gemma-4-26b-a4b-it' for Gemini and 'llama3.2' for local Ollama. Ollama Cloud requires an explicit model.",
    )
    parser.add_argument(
        "--audio-model",
        default=None,
        help="Fallback Gemini model to analyze audio for gender hints when the translation model cannot accept audio input (default: 'models/gemini-3.1-flash-lite-preview').",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models for the selected provider and exit.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Show loaded configuration, test provider connectivity, and inspect a single MKV's subtitle tracks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("translated_subs"),
        help="Directory to save translated files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Number of lines to translate per batch (default: 300).",
    )
    parser.add_argument(
        "--extra-context",
        type=Path,
        default=None,
        help="Optional UTF-8 text file with extra translation context (.txt recommended).",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Disable thinking mode."
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=2048,
        help="Token budget for thinking process (0-24576, only for flash models, default: 2048).",
    )
    parser.add_argument(
        "--progress-log",
        action="store_true",
        help="Save translation progress to a log file.",
    )
    parser.add_argument(
        "--thoughts-log",
        action="store_true",
        help="Save thinking process to a separate log file (requires thinking mode).",
    )
    parser.add_argument(
        "--no-colors", action="store_true", help="Disable colored output."
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original text as hidden comments in ASS subtitles (format: {Original: text}translation).",
    )
    parser.add_argument(
        "--add-original-only",
        action="store_true",
        help="Post-process existing translated ASS output and inject {Original: ...} comments from a chosen subtitle track, then rebuild the translated MKV.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use OCR for burned-in subtitles in MKVs.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Source language code to use for OCR subtitles before translation (default: eng).",
    )
    parser.add_argument(
        "--skip-ocr-review",
        action="store_true",
        help="Skip the browser OCR review step and continue with raw OCR text.",
    )
    parser.add_argument(
        "-a",
        "--audio-file",
        type=Path,
        default=None,
        help="Audio file for gender-aware translation (MP3 format recommended).",
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extract audio from video for gender-aware translation.",
    )
    parser.add_argument(
        "--strip-sdh",
        action="store_true",
        help="Remove SDH (Subtitles for Deaf/Hard of Hearing) elements like [sound effects], speaker names, and music symbols.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        default=False,
        help="After translation, always run the review pass (skip the prompt). "
        "If neither --review nor --no-review is given, you will be asked interactively.",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip the translation review pass entirely (never review).",
    )
    parser.add_argument(
        "--review-batch-size",
        type=int,
        default=150,
        help="Number of lines per review batch (default: 150). Only used when review is enabled.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Controls randomness in translation (0.0-2.0). "
        "Lower values = more deterministic/consistent. "
        "Higher values = more creative/varied. Default: model default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter (0.0-1.0). "
        "Consider tokens with cumulative probability up to this value. "
        "Default: model default.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K sampling parameter (integer >= 0). "
        "Consider only top K most likely tokens. "
        "Default: model default.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        type=Path,
        help="Path to a supported file (.mkv, .ass, .ssa, .srt) or a directory containing them.",
    )

    parser.set_defaults(thinking=True)
    args = parser.parse_args()

    dotenv_loaded = load_dotenv_file(Path(".env"))

    # Initialize logger settings early so utility modes also use them
    logger.set_color_mode(not args.no_colors)
    logger.enable_file_logging(args.progress_log)
    logger.enable_thoughts_logging(args.thoughts_log)

    if args.add_original_only:
        try:
            files_to_process = resolve_mkv_input_files(args.input_path)
        except ValueError as e:
            logger.error(str(e))
            parser.print_help()
            sys.exit(1)

        if not check_mkvtoolnix():
            sys.exit(1)

        args.output_dir.mkdir(exist_ok=True)

        if not files_to_process:
            logger.warning("No .mkv files found to process.")
            return

        remembered_original_lang = None
        for file_path in files_to_process:
            chosen_original_lang = add_original_comments_to_existing_output(
                file_path,
                args.output_dir,
                remembered_lang=remembered_original_lang,
            )
            if chosen_original_lang:
                remembered_original_lang = chosen_original_lang

        return

    args.provider = args.provider or os.environ.get("LLM_PROVIDER", "gemini")
    if args.provider not in SUPPORTED_PROVIDERS:
        logger.error(f"provider must be one of: {', '.join(SUPPORTED_PROVIDERS)}")
        sys.exit(1)

    args.base_url = args.base_url or os.environ.get("LLM_BASE_URL")

    extra_context_text = None
    if args.extra_context:
        try:
            extra_context_text = load_extra_context_file(args.extra_context)
        except Exception as e:
            logger.error(f"Failed to load extra context file: {e}")
            sys.exit(1)

    args.model = (
        args.model or os.environ.get("LLM_MODEL") or get_default_model(args.provider)
    )
    args.audio_model = (
        args.audio_model
        or os.environ.get("LLM_AUDIO_MODEL")
        or get_default_audio_model(args.provider)
    )

    if args.provider == "ollama-cloud" and not args.model:
        logger.error("ollama-cloud requires --model to be set explicitly")
        sys.exit(1)

    # Handle thinking mode flags
    if args.no_thinking:
        args.thinking = False

    # Handle review flags: --review=True, --no-review=False, neither=None (ask)
    if args.review and args.no_review:
        logger.error("--review and --no-review are mutually exclusive.")
        sys.exit(1)
    if args.review:
        args.review = True
    elif args.no_review:
        args.review = False
    else:
        args.review = None  # ask interactively

    if is_ollama_provider(args.provider):
        if (args.audio_file or args.extract_audio) and not args.ocr:
            logger.warning(
                "Audio-assisted translation is currently only supported with Gemini. Continuing without audio input."
            )
            args.audio_file = None
            args.extract_audio = False

        if args.audio_model:
            logger.info(
                "--audio-model is only used with Gemini and will be ignored for Ollama."
            )

        if args.api_key2:
            logger.info(
                "--api-key2 is only used for Gemini quota failover and will be ignored for Ollama."
            )

    # Validate thinking_budget
    if args.thinking_budget < 0 or args.thinking_budget > 24576:
        logger.error("thinking-budget must be between 0 and 24576")
        sys.exit(1)

    # Validate temperature (0.0 to 2.0)
    if args.temperature is not None and (args.temperature < 0 or args.temperature > 2):
        logger.error("temperature must be between 0.0 and 2.0")
        sys.exit(1)

    # Validate top_p (0.0 to 1.0)
    if args.top_p is not None and (args.top_p < 0 or args.top_p > 1):
        logger.error("top-p must be between 0.0 and 1.0")
        sys.exit(1)

    # Validate top_k (>= 0)
    if args.top_k is not None and args.top_k < 0:
        logger.error("top-k must be a non-negative integer")
        sys.exit(1)

    # Info message for Pro models with thinking
    if (
        is_gemini_provider(args.provider)
        and args.model
        and args.thinking
        and "pro" in args.model.lower()
        and "flash" not in args.model.lower()
    ):
        logger.info(
            f"Using {args.model} with thinking mode enabled.\n"
            f"Pro models may take longer to think (5+ minutes per batch is normal).\n"
            f"Automatic timeout/retry enabled - will retry if thinking exceeds 5 minutes."
        )

    api_manager = None
    client = None
    init_error = None

    # Initialize API manager (with dual API key support)
    try:
        env_key = None
        if is_gemini_provider(args.provider):
            env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        elif args.provider == "ollama-cloud":
            env_key = os.environ.get("OLLAMA_API_KEY")

        primary_key = args.api_key or env_key

        if is_gemini_provider(args.provider) and not primary_key:
            raise ValueError("No Gemini API key provided")

        if args.provider == "ollama-cloud" and not primary_key:
            raise ValueError("No Ollama Cloud API key provided")

        api_manager = APIManager(
            provider=args.provider,
            api_key=primary_key,
            api_key2=args.api_key2 if is_gemini_provider(args.provider) else None,
            base_url=args.base_url,
        )

        # Create initial client
        client = api_manager.get_client()
        logging.debug(f"Initialized {get_provider_display_name(args.provider)} client")

        # Show dual API status
        if is_gemini_provider(args.provider) and api_manager.has_secondary_key():
            logger.info(f"Dual API keys configured - automatic quota failover enabled")

        if is_ollama_provider(args.provider):
            logger.info(
                f"Using {get_provider_display_name(args.provider)} at {api_manager.base_url}"
            )

    except Exception as e:
        init_error = e

    if args.doctor:
        overall_ok = run_doctor(
            args,
            api_manager=api_manager,
            client=client,
            init_error=init_error,
            dotenv_loaded=dotenv_loaded,
        )
        sys.exit(0 if overall_ok else 1)

    if init_error is not None:
        logger.error(
            f"Failed to initialize {get_provider_display_name(args.provider)} client: {init_error}"
        )
        if is_gemini_provider(args.provider):
            logger.error("Provide --api-key or set GEMINI_API_KEY/GOOGLE_API_KEY.")
        elif args.provider == "ollama-cloud":
            logger.error("Provide --api-key or set OLLAMA_API_KEY.")
        else:
            logger.error(
                "Make sure Ollama is running, or set --base-url/OLLAMA_HOST/LLM_BASE_URL to the correct server."
            )
        sys.exit(1)

    if args.ocr and not is_ocr_capable_provider(args.provider):
        logger.error("--ocr requires a provider with image support.")
        sys.exit(1)

    # Handle --list-models before checking other args
    if args.list_models:
        try:
            print(f"Available {get_provider_display_name(args.provider)} models:")
            if is_gemini_provider(args.provider):
                for m in client.models.list():
                    print(f"  {m.name}")
            else:
                for model_name in extract_ollama_model_names(client.list()):
                    print(f"  {model_name}")
        except Exception as e:
            logger.error(f"Failed to list models. Error: {e}")
            sys.exit(1)
        sys.exit(0)

    # Pre-execution checks
    if not args.input_path:
        logger.error("You must provide a supported file path or directory.")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(exist_ok=True)

    logging.debug(f"Using model: {args.model}")
    logging.debug(f"Batch size: {args.batch_size}")

    # File processing loop
    try:
        files_to_process = resolve_input_files(args.input_path)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if any(is_mkv_path(file_path) for file_path in files_to_process) and not check_mkvtoolnix():
        sys.exit(1)

    if not files_to_process:
        logger.warning("No supported input files found to process.")
        return

    remembered_lang = None
    remembered_secondary_lang = None
    remembered_batch_size = args.batch_size
    for file_path in files_to_process:
        final_batch_size = remembered_batch_size

        if is_mkv_path(file_path):
            if args.ocr:
                final_batch_size = process_mkv_ocr_file(
                    mkv_path=file_path,
                    output_dir=args.output_dir,
                    api_manager=api_manager,
                    model_name=args.model,
                    audio_model_name=args.audio_model,
                    ocr_lang=args.ocr_lang,
                    batch_size=remembered_batch_size,
                    thinking=args.thinking,
                    thinking_budget=args.thinking_budget,
                    keep_original=args.keep_original,
                    extra_context_text=extra_context_text,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    strip_sdh=args.strip_sdh,
                    skip_ocr_review=args.skip_ocr_review,
                    review=args.review,
                    review_batch_size=args.review_batch_size,
                )
            else:
                chosen_lang, chosen_secondary_lang, final_batch_size = process_mkv_file(
                    file_path,
                    args.output_dir,
                    api_manager,
                    args.model,
                    args.audio_model,
                    remembered_lang,
                    remembered_secondary_lang,
                    remembered_batch_size,
                    args.thinking,
                    args.thinking_budget,
                    args.keep_original,
                    args.audio_file,
                    args.extract_audio,
                    extra_context_text,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.strip_sdh,
                    args.review,
                    args.review_batch_size,
                )
                # Remember language selection for subsequent files
                if chosen_lang:
                    remembered_lang = chosen_lang
                remembered_secondary_lang = chosen_secondary_lang
        elif is_raw_text_subtitle_path(file_path):
            if args.ocr:
                logger.warning(
                    f"Skipping {file_path.name}: --ocr is only supported for MKV inputs."
                )
                continue

            final_batch_size = process_raw_subtitle_file(
                subtitle_path=file_path,
                output_dir=args.output_dir,
                api_manager=api_manager,
                model_name=args.model,
                audio_model_name=args.audio_model,
                source_lang=args.ocr_lang,
                batch_size=remembered_batch_size,
                thinking=args.thinking,
                thinking_budget=args.thinking_budget,
                keep_original=args.keep_original,
                audio_file=args.audio_file,
                extract_audio=args.extract_audio,
                extra_context_text=extra_context_text,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                strip_sdh=args.strip_sdh,
                review=args.review,
                review_batch_size=args.review_batch_size,
            )
        # Remember batch size adjustment for subsequent files
        if final_batch_size and final_batch_size != remembered_batch_size:
            logger.info(
                f"Batch size adjusted to {final_batch_size} - will use for remaining files"
            )
            remembered_batch_size = final_batch_size

    logger.success("--- All files processed. ---")


if __name__ == "__main__":
    main()
