"""
Audio utilities for gender-aware translation.
Handles audio extraction, compression, and processing from video files.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Import from same directory
try:
    from logger import error, warning, info, success
except ImportError:
    print("Error: logger module not found. Please ensure logger.py is in the same directory.")
    sys.exit(1)


def get_file_size_mb(file_path):
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)


def get_audio_info(video_path):
    """
    Get audio channel information from video file, selecting the primary audio stream.
    Returns: (channels, channel_layout, stream_index)
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "a", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
        data = json.loads(result.stdout)

        if not data.get("streams"):
            raise ValueError("No audio stream found in video")

        # Select primary audio stream (with default disposition)
        primary_stream = None
        primary_index = 0

        for i, stream in enumerate(data["streams"]):
            disposition = stream.get("disposition", {})
            if disposition.get("default", 0) == 1:
                primary_stream = stream
                primary_index = i
                break

        # Fallback to first stream if no default
        if primary_stream is None:
            primary_stream = data["streams"][0]
            primary_index = 0

        channels = primary_stream.get("channels", 0)
        channel_layout = primary_stream.get("channel_layout", "")

        return channels, channel_layout, primary_index

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to analyze audio: {e}")
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse audio information")


def extract_audio(video_path, output_path, channels, stream_index=0):
    """
    Extract and process audio based on channel configuration.
    Converts multi-channel audio to mono using channel-specific mixing.
    """
    cmd = ["ffmpeg", "-i", str(video_path), "-map", f"0:a:{stream_index}", "-y"]

    audio_filters = []

    # Channel-specific processing (mix to mono)
    if channels == 1:
        # Already mono
        pass
    elif channels == 2:
        # Stereo to mono: average both channels
        audio_filters.append("pan=1c|c0=0.5*c0+0.5*c1")
    elif channels > 2:
        if channels >= 6:  # 5.1 or 7.1 surround
            # Extract center channel (where dialogue usually is)
            audio_filters.append("pan=1c|c0=1*c2")
        elif channels == 5:  # 5.0 surround
            audio_filters.append("pan=1c|c0=1*c2")
        elif channels == 4:  # Quad
            audio_filters.append("pan=1c|c0=0.25*c0+0.25*c1+0.25*c2+0.25*c3")
        elif channels == 3:  # 2.1 or 3.0
            audio_filters.append("pan=1c|c0=1*c2")
        else:
            audio_filters.append("pan=1c|c0=c0")

    # Build final ffmpeg command
    if audio_filters:
        filter_chain = ",".join(audio_filters)
        cmd.extend(["-vn", "-af", filter_chain, "-acodec", "pcm_s16le"])
    else:
        cmd.extend(["-vn", "-acodec", "pcm_s16le"])

    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")


def compress_audio(extracted_audio_path, target_size_mb=20):
    """
    Compress audio file to target size using MP3.
    Uses adaptive bitrate calculation to hit target size.
    """
    current_size = get_file_size_mb(extracted_audio_path)

    # Adjust target slightly lower if already small
    if current_size <= target_size_mb:
        target_size_mb = target_size_mb - target_size_mb * 0.1  # Reduce by 10%

    # Create compressed filename
    base_name = os.path.splitext(os.path.basename(extracted_audio_path))[0]
    dir_name = os.path.dirname(extracted_audio_path)
    compressed_path = os.path.join(dir_name, f"{base_name}_compressed.mp3")

    try:
        # Validate input
        if not os.path.exists(extracted_audio_path):
            raise FileNotFoundError(f"Input file not found: {extracted_audio_path}")

        # Get audio duration via ffprobe
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(extracted_audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
        data = json.loads(result.stdout)

        if "format" not in data or "duration" not in data["format"]:
            raise ValueError("Could not get duration information from audio file")

        duration = float(data["format"]["duration"])
        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}")

        # Calculate target bitrate with fallback options
        target_bitrate_kbps = int((target_size_mb * 8 * 1024) / duration * 0.9)

        # Bitrate fallback strategy
        if target_bitrate_kbps > 128:
            target_bitrate_kbps = 128
        elif target_bitrate_kbps > 96:
            target_bitrate_kbps = 96
        elif target_bitrate_kbps > 64:
            target_bitrate_kbps = 64
        elif target_bitrate_kbps > 48:
            target_bitrate_kbps = 48
        else:
            target_bitrate_kbps = 32  # Minimum

        # First compression attempt
        cmd = [
            "ffmpeg", "-i", str(extracted_audio_path), "-y",
            "-acodec", "libmp3lame",
            "-b:a", f"{target_bitrate_kbps}k",
            "-ac", "1",  # Mono
            "-ar", "22050",  # Sample rate: 22050 Hz
            str(compressed_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")

        # Verify and check if still too large
        if not os.path.exists(compressed_path):
            raise FileNotFoundError("Compressed file was not created")

        new_size = get_file_size_mb(compressed_path)

        # Fallback: Aggressive compression if still too large
        if new_size > target_size_mb * 1.2:
            lower_bitrate = max(16, target_bitrate_kbps // 2)  # Half bitrate, min 16 kbps
            aggressive_path = os.path.join(dir_name, f"{base_name}_compressed_low.mp3")

            cmd = [
                "ffmpeg", "-i", str(extracted_audio_path), "-y",
                "-acodec", "libmp3lame",
                "-b:a", f"{lower_bitrate}k",
                "-ac", "1",  # Mono
                "-ar", "16000",  # Sample rate: 16000 Hz (LOWER)
                str(aggressive_path)
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")

            # Replace initial compressed file with aggressive version
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            os.rename(aggressive_path, compressed_path)
            new_size = get_file_size_mb(compressed_path)

        # Final cleanup and renaming
        if extracted_audio_path != compressed_path and os.path.exists(extracted_audio_path):
            os.remove(extracted_audio_path)

        final_path = os.path.join(dir_name, f"{base_name}.mp3")
        if compressed_path != final_path:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(compressed_path, final_path)

        return final_path

    except subprocess.CalledProcessError as e:
        error(f"FFmpeg/FFprobe command failed: {e}")
        if e.stderr:
            error(f"Error output: {e.stderr}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return extracted_audio_path

    except json.JSONDecodeError as e:
        error(f"Failed to parse ffprobe output: {e}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return extracted_audio_path

    except Exception as e:
        error(f"Error during MP3 compression: {e}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return extracted_audio_path


def prepare_audio(video_path):
    """
    Main audio extraction pipeline:
    1. Get audio info (channels, layout)
    2. Extract audio from video (mono conversion)
    3. Compress to target size (~20MB MP3)

    Returns path to prepared audio file.
    """
    output_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_extracted.wav")
    final_path = os.path.join(output_dir, f"{base_name}_audio.mp3")

    # Skip if already extracted
    if os.path.exists(final_path):
        warning(f'"{os.path.basename(final_path)}" already exists. Skipping extraction.')
        return final_path

    info("Starting audio extraction and processing...")

    # Get audio information
    try:
        channels, channel_layout, stream_index = get_audio_info(video_path)
        info(f"Detected audio: {channels} channels ({channel_layout}), stream {stream_index}")
    except Exception as e:
        error(f"Failed to get audio info: {e}")
        return None

    # Extract audio
    try:
        info("Extracting audio from video...")
        extract_audio(video_path, output_path, channels, stream_index)
        info("Audio extraction complete.")
    except Exception as e:
        error(f"Failed to extract audio: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

    # Compress audio
    try:
        info("Compressing audio to ~20MB MP3...")
        result = compress_audio(output_path, target_size_mb=20)
        if result:
            final_size = get_file_size_mb(result)
            success(f"Success! Audio saved as: {os.path.basename(result)}")
            success(f"Final file size: {final_size:.2f}MB")
            return result
        else:
            error("Failed to compress audio.")
            return None
    except Exception as e:
        error(f"Failed to compress audio: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None
