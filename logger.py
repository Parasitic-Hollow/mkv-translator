#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced logging module with colors and file output support.
Adapted from gemini-translator-srt's logger.py for gemini-translator.
"""

import os
import sys
from enum import Enum
from typing import Any


# Global variables
_use_colors = True
_log_to_file = False
_thoughts_to_file = False
_log_file_path = "translation.log"
_thoughts_file_path = "thoughts.log"
_log_messages = []
_thoughts_list = []
_quiet_mode = False  # Suppress console output during translation


class Color(Enum):
    """ANSI color codes"""

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def supports_color() -> bool:
        """Check if the terminal supports color output"""
        # If NO_COLOR env var is set, disable color
        if os.environ.get("NO_COLOR"):
            return False

        # If FORCE_COLOR env var is set, enable color
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check if stdout is a TTY
        is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        return (
            is_a_tty
            or "ANSICON" in os.environ
            or "WT_SESSION" in os.environ
            or os.environ.get("TERM_PROGRAM") == "vscode"
        )


def set_color_mode(enabled: bool) -> None:
    """Set whether to use colors in output"""
    global _use_colors
    _use_colors = enabled


def set_quiet_mode(enabled: bool) -> None:
    """
    Set quiet mode - suppresses console output, only logs to file.
    Use during translation to keep output clean.
    """
    global _quiet_mode
    _quiet_mode = enabled


def set_log_file_path(path: str) -> None:
    """Set the log file path"""
    global _log_file_path
    _log_file_path = path


def set_thoughts_file_path(path: str) -> None:
    """Set the thoughts file path"""
    global _thoughts_file_path
    _thoughts_file_path = path


def enable_file_logging(enabled: bool = True) -> None:
    """Enable or disable logging to file"""
    global _log_to_file
    _log_to_file = enabled


def enable_thoughts_logging(enabled: bool = True) -> None:
    """Enable or disable thoughts logging to file"""
    global _thoughts_to_file
    _thoughts_to_file = enabled


def _log_message(message: Any, color: Color = None) -> None:
    """Internal function to log a message with optional color"""
    msg_str = str(message)

    # Store message for file logging
    if _log_to_file:
        _log_messages.append(msg_str)

    # Skip console output in quiet mode
    if _quiet_mode:
        return

    # Print with color if enabled
    if _use_colors and Color.supports_color() and color:
        print(f"{color.value}{msg_str}{Color.RESET.value}")
    else:
        print(msg_str)


def info(message: Any) -> None:
    """Print an information message in cyan color"""
    _log_message(message, Color.CYAN)


def warning(message: Any) -> None:
    """Print a warning message in yellow color"""
    _log_message(message, Color.YELLOW)


def error(message: Any) -> None:
    """Print an error message in red color"""
    _log_message(message, Color.RED)


def success(message: Any) -> None:
    """Print a success message in green color"""
    _log_message(message, Color.GREEN)


def highlight(message: Any) -> None:
    """Print an important message in magenta bold"""
    msg_str = str(message)

    # Store message for file logging
    if _log_to_file:
        _log_messages.append(msg_str)

    # Skip console output in quiet mode
    if _quiet_mode:
        return

    # Print with color if enabled
    if _use_colors and Color.supports_color():
        print(f"{Color.MAGENTA.value}{Color.BOLD.value}{msg_str}{Color.RESET.value}")
    else:
        print(msg_str)


def debug(message: Any) -> None:
    """Print a debug message in blue color"""
    _log_message(message, Color.BLUE)


def save_thoughts(thoughts: str, batch_number: int, retry: int = 0) -> bool:
    """
    Save thinking process to thoughts file.

    Args:
        thoughts (str): The thinking text to save
        batch_number (int): Current batch number
        retry (int): Retry attempt number (0 for first attempt)

    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not _thoughts_to_file:
        return False

    global _thoughts_list

    _thoughts_list.append({
        "batch": batch_number,
        "retry": retry,
        "text": thoughts
    })

    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(_thoughts_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(_thoughts_file_path, "w", encoding="utf-8") as f:
            for thought in _thoughts_list:
                f.write("=" * 80 + "\n\n")
                if thought["retry"] > 0:
                    f.write(f"Batch {thought['batch']}.{thought['retry']} thoughts (retry):\n\n")
                else:
                    f.write(f"Batch {thought['batch']} thoughts:\n\n")
                f.write("=" * 80 + "\n\n")
                f.write(thought["text"])
                f.write("\n\n")

        return True
    except (PermissionError, OSError) as e:
        warning(f"Failed to save thoughts to {_thoughts_file_path}: {e}")
        return False


def save_logs() -> bool:
    """
    Save accumulated log messages to file.

    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not _log_to_file or not _log_messages:
        return False

    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(_log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(_log_file_path, "w", encoding="utf-8") as f:
            for msg in _log_messages:
                f.write(msg + "\n")

        return True
    except (PermissionError, OSError) as e:
        warning(f"Failed to save logs to {_log_file_path}: {e}")
        return False


def clear_logs() -> None:
    """Clear accumulated log messages"""
    global _log_messages
    _log_messages = []


def clear_thoughts() -> None:
    """Clear accumulated thoughts"""
    global _thoughts_list
    _thoughts_list = []


if __name__ == "__main__":
    # Test color output
    info("This is an info message")
    warning("This is a warning message")
    error("This is an error message")
    success("This is a success message")
    highlight("This is a highlighted message")
    debug("This is a debug message")
