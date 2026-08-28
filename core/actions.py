"""Voice Assistant — Action executors."""

from __future__ import annotations

import json
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

# User config directory
CONFIG_DIR = Path.home() / ".config" / "voice-assistant"
ALIASES_FILE = CONFIG_DIR / "app_aliases.json"

# Layer 1: Built-in tech term transliteration (Arabic -> English)
TECH_TRANSLIT = {
    # Code editors / IDEs
    "كود": "code",
    "الكود": "code",
    "في إس كود": "vscode",
    "فيجوال ستوديو": "vscode",
    "فيجوال ستوديو كود": "vscode",
    "في اس كود": "vscode",
    "فيجوال": "visual",
    "ستوديو": "studio",
    "محرر": "editor",
    "المحرر": "editor",
    "فيم": "vim",
    "نيم": "nvim",
    "نافيم": "nvim",
    # Browsers
    "متصفح": "browser",
    "المتصفح": "browser",
    "كروم": "chrome",
    "الكروم": "chrome",
    "جوجل كروم": "chrome",
    # Terminals
    "تيرمينال": "terminal",
    "الترمينال": "terminal",
    "طرفية": "terminal",
    "الطرفية": "terminal",
    "كونسول": "console",
    "الكونسول": "console",
    "جنوم ترمينال": "gnome-terminal",
    "كيتتي": "kitty",
    "الاكريتي": "alacritty",
    "جوستي": "ghostty",
    # Dev tools
    "دوك": "docker",
    "الدوك": "docker",
    "دوكير": "docker",
    "جيت": "git",
    "الجيت": "git",
    "جيب": "github",
    "جيت هب": "github",
    "جيت لاب": "gitlab",
    "بايثون": "python",
    "بايثون3": "python3",
    "نود": "node",
    "نود جي اس": "node",
    "ان بي ام": "npm",
    "yarn": "yarn",
    "كارغو": "cargo",
    "رست": "rust",
    "جو": "go",
    "جولانج": "go",
    "دارت": "dart",
    "فلوتر": "flutter",
    # System
    "إعدادات": "settings",
    "الاعدادات": "settings",
    "ملفات": "files",
    "الملفات": "files",
    "ناوتيلوس": "nautilus",
    "الناوتيلوس": "nautilus",
    "ناوتيليس": "nautilus",
    "الناوتيليس": "nautilus",
    "دولفين": "dolphin",
    "الثونار": "thunar",
    "ثونار": "thunar",
    "مدراء الملفات": "file-manager",
    "محرر النصوص": "text-editor",
    "جيديت": "gedit",
    "كيت": "kate",
    # Media
    "في ال سي": "vlc",
    "فيإلسي": "vlc",
    "مشغل": "player",
    "المشغل": "player",
    "ام بي في": "mpv",
    "سبوتيفاي": "spotify",
    # Office
    "ليبر أوفيس": "libreoffice",
    "ليبري أوفيس": "libreoffice",
    "أوفيس": "office",
    "مايكروسوفت": "microsoft",
    "وورد": "word",
    "إكسل": "excel",
    "باور بوينت": "powerpoint",
    # Communication
    "ديسكورد": "discord",
    "الديسكورد": "discord",
    "تليجرام": "telegram",
    "التيليجرام": "telegram",
    "سيجنال": "signal",
    "سلاك": "slack",
    "السلوك": "slack",
    "زووم": "zoom",
    "الزووم": "zoom",
    # Misc
    "حاسبة": "calculator",
    "الالة الحاسبة": "calculator",
    "التقويم": "calendar",
    "ساعة": "clock",
    "الساعة": "clock",
    "لقطة شاشة": "screenshot",
}

# Layer 2: Arabic to Latin transliteration (simple char mapping)
ARABIC_TO_LATIN = {
    "ا": "a",
    "أ": "a",
    "إ": "a",
    "آ": "a",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ة": "h",
    "و": "w",
    "ي": "y",
    "ى": "y",
    " ": " ",
    "-": "-",
    "_": "_",
}

# Arabic normalization: map variant forms to canonical
ARABIC_NORMALIZE = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",  # alef variants
    "ى": "ي",  # alef maqsura -> ya
    "ة": "ه",  # ta marbuta -> ha
    "ؤ": "و",  # hamza on waw -> waw
    "ئ": "ي",  # hamza on ya -> ya
}


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text for consistent matching."""
    result = []
    for char in text:
        result.append(ARABIC_NORMALIZE.get(char, char))
    return "".join(result)


def _transliterate_arabic(text: str) -> str:
    """Simple Arabic to Latin transliteration."""
    # First normalize Arabic
    text = _normalize_arabic(text)
    result = []
    for char in text.lower():
        result.append(ARABIC_TO_LATIN.get(char, char))
    return "".join(result)


def _load_user_aliases() -> dict[str, str]:
    """Load user-defined app aliases from JSON config."""
    try:
        if ALIASES_FILE.exists():
            data = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
            # Normalize keys to lowercase
            return {k.lower(): v.lower() for k, v in data.items()}
    except Exception as e:
        logger.debug(f"Failed to load user aliases: {e}")
    return {}


def _save_default_aliases() -> None:
    """Create example aliases file if it doesn't exist."""
    try:
        if not ALIASES_FILE.exists():
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            example = {
                "البرنامج المحاسبة": "gnucash",
                "محرري": "nvim",
                "متصفحي": "firefox",
            }
            ALIASES_FILE.write_text(
                json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"Created example aliases file: {ALIASES_FILE}")
    except Exception as e:
        logger.debug(f"Failed to create default aliases: {e}")


def _normalize_app_name(name: str) -> str:
    """Normalize app name through all resolution layers."""
    name_clean = name.strip().lower()
    # Normalize Arabic characters for consistent matching
    name_clean = _normalize_arabic(name_clean)

    # Layer 1: Built-in tech transliteration
    if name_clean in TECH_TRANSLIT:
        return TECH_TRANSLIT[name_clean]

    # Layer 2: User aliases
    user_aliases = _load_user_aliases()
    if name_clean in user_aliases:
        return user_aliases[name_clean]

    # Layer 3: Transliteration + fuzzy will be handled in open_app
    return name_clean


def get_time(lang: str = "en") -> str:
    """Get current time formatted as HH:MM AM/PM.

    Args:
        lang: Language code ("en" or "ar")

    Returns:
        Formatted time string (e.g., "02:30 PM" or "02:30 م")
    """
    now = datetime.now()
    if lang == "ar":
        # Arabic time format: HH:MM ص/م
        hour = now.hour % 12
        if hour == 0:
            hour = 12
        period = "ص" if now.hour < 12 else "م"
        return f"{hour:02d}:{now.minute:02d} {period}"
    return now.strftime("%I:%M %p")


def get_date(lang: str = "en") -> str:
    """Get current date formatted as Weekday, Month DD, YYYY.

    Args:
        lang: Language code ("en" or "ar")

    Returns:
        Formatted date string (e.g., "Monday, January 15, 2024" or "الاثنين، 15 يناير 2024")
    """
    now = datetime.now()
    if lang == "ar":
        arabic_weekdays = [
            "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"
        ]
        arabic_months = [
            "يناير", "فبراير", "مارس", "إبريل", "مايو", "يونيو",
            "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
        ]
        weekday = arabic_weekdays[now.weekday()]
        month = arabic_months[now.month - 1]
        return f"{weekday}، {now.day} {month} {now.year}"
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
    # Ensure default aliases exist
    _save_default_aliases()

    # Normalize input through resolution layers
    name_clean = name.strip().lower()
    normalized_name = _normalize_app_name(name_clean)
    logger.debug(
        f"open_app: name={name}, name_clean={name_clean}, normalized_name={normalized_name}"
    )

    apps = _get_app_index()

    # 1. Try exact match with normalized name
    if normalized_name in apps:
        path = apps[normalized_name]
        try:
            subprocess.Popen([path], start_new_session=True)
            logger.info(f"Launched app: {name} -> {normalized_name} ({path})")
            return _t("Successfully launched {app}.", lang, app=name)
        except Exception as e:
            logger.error(f"Failed to launch {normalized_name}: {e}")
            raise ActionError(f"Failed to launch {normalized_name}: {e}") from e

    # 2. Try fuzzy match with normalized name (threshold 80%)
    fuzzy_result = _fuzzy_match_app(normalized_name, threshold=80)
    if fuzzy_result:
        matched_name, path = fuzzy_result
        try:
            subprocess.Popen([path], start_new_session=True)
            logger.info(f"Launched app (fuzzy): {name} -> {matched_name} ({path})")
            return _t("Successfully launched {app}.", lang, app=matched_name)
        except Exception as e:
            logger.error(f"Failed to launch {matched_name}: {e}")
            raise ActionError(f"Failed to launch {matched_name}: {e}") from e

    # 3. Try direct fuzzy match on original name (handles partial matches)
    # This helps with cases like "firefox" matching "mozilla-firefox" etc.
    fuzzy_result = _fuzzy_match_app(name_clean, threshold=80)
    if fuzzy_result:
        matched_name, path = fuzzy_result
        try:
            subprocess.Popen([path], start_new_session=True)
            logger.info(f"Launched app (fuzzy on original): {name} -> {matched_name} ({path})")
            return _t("Successfully launched {app}.", lang, app=matched_name)
        except Exception as e:
            logger.error(f"Failed to launch {matched_name}: {e}")
            raise ActionError(f"Failed to launch {matched_name}: {e}") from e

    # 4. Not found - suggest installation
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

    # 5. User confirmed - install in background
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
