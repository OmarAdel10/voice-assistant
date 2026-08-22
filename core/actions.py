"""Voice Assistant — Action executors."""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.parse
import webbrowser
from datetime import datetime

import psutil

from core.exceptions import ActionError

logger = logging.getLogger(__name__)


def get_time() -> str:
    """Get current time formatted as HH:MM AM/PM.

    Returns:
        Formatted time string (e.g., "02:30 PM")
    """
    now = datetime.now()
    return now.strftime("%I:%M %p")


def get_date() -> str:
    """Get current date formatted as Weekday, Month DD, YYYY.

    Returns:
        Formatted date string (e.g., "Monday, January 15, 2024")
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")


def get_sysinfo() -> dict[str, float]:
    """Get system information (CPU, memory, disk usage).

    Returns:
        Dictionary with cpu_percent, memory_percent, disk_percent

    Raises:
        ActionError: If system info cannot be retrieved
    """
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent

        return {
            "cpu_percent": cpu,
            "memory_percent": memory,
            "disk_percent": disk,
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise ActionError(f"Failed to get system info: {e}") from e


def open_app(name: str) -> str:
    """Launch an application by name.

    Args:
        name: Application name (must be in PATH)

    Returns:
        Success message

    Raises:
        ActionError: If app not found or launch fails
    """
    # Validate app exists in PATH (allowlist via shutil.which)
    path = shutil.which(name)
    if path is None:
        logger.error(f"App not found in PATH: {name}")
        raise ActionError(f"Application '{name}' not found in PATH")

    try:
        # Launch with argv list ONLY, never shell=True
        subprocess.Popen([path], start_new_session=True)
        logger.info(f"Launched app: {name} ({path})")
        return f"Successfully launched {name}"
    except Exception as e:
        logger.error(f"Failed to launch {name}: {e}")
        raise ActionError(f"Failed to launch {name}: {e}") from e


def web_search(query: str) -> str:
    """Perform a web search by opening browser with query.

    Args:
        query: Search query string

    Returns:
        Success message

    Raises:
        ActionError: If browser cannot be opened
    """
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded}"
        webbrowser.open_new_tab(url)
        logger.info(f"Web search: {query}")
        return f"Successfully searched for {query}"
    except Exception as e:
        logger.error(f"Failed to search for {query}: {e}")
        raise ActionError(f"Failed to search for {query}: {e}") from e
