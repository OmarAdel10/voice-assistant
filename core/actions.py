"""Voice Assistant — Action executors."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path

import psutil
from rapidfuzz import process as rapidfuzz_process

from core.exceptions import ActionError, InstallError

logger = logging.getLogger(__name__)


# Common application directories
COMMON_APP_DIRS = [
    Path("/usr/bin"),
    Path("/usr/local/bin"),
    Path("/opt"),
    Path("/snap/bin"),
    Path("/var/lib/flatpak/exports/bin"),
    Path.home() / ".local/bin",
]

# Desktop file directories
DESKTOP_DIRS = [
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path.home() / ".local/share/applications",
]


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


def _parse_desktop_file(desktop_path: Path) -> dict[str, str] | None:
    """Parse a .desktop file and return app info.

    Args:
        desktop_path: Path to .desktop file

    Returns:
        Dictionary with name, exec, or None if invalid
    """
    try:
        content = desktop_path.read_text(encoding="utf-8")
        info = {}
        for line in content.splitlines():
            if line.startswith("Name="):
                info["name"] = line[5:].strip()
            elif line.startswith("Exec="):
                # Extract the executable command (first token before % args)
                exec_line = line[5:].strip()
                # Remove %f, %u, etc. placeholders
                exec_clean = re.sub(r"\s*%[a-zA-Z]", "", exec_line).split()[0]
                info["exec"] = exec_clean
            elif line.startswith("NoDisplay=true"):
                return None
        if "name" in info and "exec" in info:
            return info
    except Exception as e:
        logger.debug(f"Failed to parse {desktop_path}: {e}")
    return None


def _build_app_index() -> dict[str, str]:
    """Build index of known applications.

    Returns:
        Dictionary mapping app name (lowercase) -> executable path
    """
    apps = {}

    # 1. PATH executables
    for path_dir in COMMON_APP_DIRS:
        if path_dir.exists():
            for exe in path_dir.iterdir():
                if exe.is_file() and os.access(exe, os.X_OK):
                    apps[exe.name.lower()] = str(exe)

    # 2. .desktop files
    for desktop_dir in DESKTOP_DIRS:
        if desktop_dir.exists():
            for desktop_file in desktop_dir.glob("*.desktop"):
                info = _parse_desktop_file(desktop_file)
                if info:
                    exec_path = info["exec"]
                    # Resolve exec to full path
                    full_path = shutil.which(exec_path)
                    if full_path:
                        apps[info["name"].lower()] = full_path
                        # Also index by executable name
                        apps[Path(exec_path).name.lower()] = full_path

    return apps


# Global app index (lazy loaded)
_APP_INDEX: dict[str, str] | None = None


def _get_app_index() -> dict[str, str]:
    """Get cached app index, building if needed."""
    global _APP_INDEX
    if _APP_INDEX is None:
        logger.info("Building application index...")
        _APP_INDEX = _build_app_index()
        logger.info(f"Indexed {len(_APP_INDEX)} applications")
    return _APP_INDEX


def _fuzzy_match_app(name: str, threshold: int = 80) -> tuple[str, str] | None:
    """Find best fuzzy match for app name.

    Args:
        name: App name to match
        threshold: Minimum score (0-100)

    Returns:
        Tuple of (matched_name, executable_path) or None
    """
    apps = _get_app_index()
    if not apps:
        return None

    result = rapidfuzz_process.extractOne(
        name.lower(),
        apps.keys(),
        score_cutoff=threshold,
    )
    if result:
        matched_name, score, _ = result
        return matched_name, apps[matched_name]
    return None


def _suggest_install(app_name: str) -> list[str]:
    """Suggest install commands for app.

    Args:
        app_name: Application name to search for

    Returns:
        List of install command suggestions
    """
    suggestions = []

    # dnf search
    try:
        result = subprocess.run(
            ["dnf", "search", app_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Extract package names from output
            for line in result.stdout.splitlines():
                if line and not line.startswith("=") and ":" in line:
                    pkg = line.split(":")[0].strip()
                    if pkg:
                        suggestions.append(f"sudo dnf install {pkg}")
                        break
    except Exception as e:
        logger.debug(f"dnf search failed: {e}")

    # flatpak search
    try:
        result = subprocess.run(
            ["flatpak", "search", app_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        app_id = parts[1].strip()
                        suggestions.append(f"flatpak install {app_id}")
                        break
    except Exception as e:
        logger.debug(f"flatpak search failed: {e}")

    return suggestions


def _install_package_background(command: str, lang: str, callback=None) -> None:
    """Run package install in background thread."""
    import threading

    def run_install():
        try:
            logger.info(f"Running install: {command}")
            subprocess.run(command, shell=True, check=True, timeout=300)
            if callback:
                callback(True, f"Successfully installed via: {command}")
        except Exception as e:
            logger.error(f"Install failed: {e}")
            if callback:
                callback(False, f"Install failed: {e}")

    thread = threading.Thread(target=run_install, daemon=True)
    thread.start()


def open_app(name: str, lang: str = "en", confirm_install: bool = False) -> str:
    """Launch an application by name with smart resolution.

    Args:
        name: Application name (e.g., "vscode", "firefox", "chrome")
        lang: Language code for response ("en" or "ar")
        confirm_install: If True, user has confirmed installation

    Returns:
        Success or error message

    Raises:
        ActionError: If app not found and install not confirmed
        InstallError: If installation fails
    """
    # Normalize input
    name_clean = name.strip().lower()

    # 1. Try exact match in app index
    apps = _get_app_index()
    if name_clean in apps:
        path = apps[name_clean]
        try:
            subprocess.Popen([path], start_new_session=True)
            logger.info(f"Launched app: {name} ({path})")
            return _t("Successfully launched {app}.", lang, app=name)
        except Exception as e:
            logger.error(f"Failed to launch {name}: {e}")
            raise ActionError(f"Failed to launch {name}: {e}") from e

    # 2. Try fuzzy match
    fuzzy_result = _fuzzy_match_app(name_clean)
    if fuzzy_result:
        matched_name, path = fuzzy_result
        try:
            subprocess.Popen([path], start_new_session=True)
            logger.info(f"Launched app (fuzzy): {matched_name} ({path})")
            return _t("Successfully launched {app}.", lang, app=matched_name)
        except Exception as e:
            logger.error(f"Failed to launch {matched_name}: {e}")
            raise ActionError(f"Failed to launch {matched_name}: {e}") from e

    # 3. Not found - suggest installation
    if not confirm_install:
        suggestions = _suggest_install(name)
        if suggestions:
            msg = _t(
                "Application '{app}' not found. Install with: {cmd}?",
                lang,
                app=name,
                cmd=suggestions[0],
            )
            # Add all suggestions
            for _i, cmd in enumerate(suggestions[1:], 2):
                msg += f" Or: {cmd}"
            return msg
        else:
            return _t(
                "Application '{app}' not found and no install suggestion available.",
                lang,
                app=name,
            )

    # 4. User confirmed - install in background
    suggestions = _suggest_install(name)
    if not suggestions:
        raise InstallError(_t("No install method found for '{app}'.", lang, app=name))

    # Use first suggestion
    install_cmd = suggestions[0]

    # In a real implementation, we'd wait for voice confirmation here
    # For now, just start install and return status
    def on_complete(success: bool, message: str):
        # This would trigger a TTS notification in the main loop
        logger.info(f"Install callback: success={success}, msg={message}")

    _install_package_background(install_cmd, lang, on_complete)

    return _t("Installing {app} in background. Will notify when done.", lang, app=name)


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


def _t(template: str, lang: str, **kwargs) -> str:
    """Simple template localization."""
    if lang == "ar":
        ar_templates = {
            "Successfully launched {app}.": "تم تشغيل {app} بنجاح.",
            "Application '{app}' not found. Install with: {cmd}?": (
                "التطبيق '{app}' غير موجود. تثبيت عبر: {cmd}؟"
            ),
            "Application '{app}' not found and no install suggestion available.": (
                "التطبيق '{app}' غير موجود ولا يوجد اقتراح للتثبيت."
            ),
            "No install method found for '{app}'.": "لا توجد طريقة تثبيت لـ '{app}'.",
            "Installing {app} in background. Will notify when done.": (
                "جاري تثبيت {app} في الخلفية. سأخبرك عند الانتهاء."
            ),
        }
        template = ar_templates.get(template, template)
    return template.format(**kwargs)
