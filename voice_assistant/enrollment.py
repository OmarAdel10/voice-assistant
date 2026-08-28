"""Voice Assistant — Speaker enrollment (phrase set, persistence, interactive flow)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from config.settings import Settings
from core.stt_engine import STTEngineBase, create_stt_engine

logger = logging.getLogger(__name__)

# Enrollment file path
ENROLLMENT_DIR = Path.home() / ".config" / "voice-assistant"
ENROLLMENT_FILE = ENROLLMENT_DIR / "enrollment.json"

ENROLLMENT_PHRASES = [
    "مرحبا، كيف حالك؟",
    "افتح الكود من فضلك",
    "ما الوقت الآن؟",
    "Hello, how are you?",
    "Open the browser please",
    "What time is it?",
]


def load_enrollment() -> str | None:
    """Load enrollment prompt from file."""
    try:
        if ENROLLMENT_FILE.exists():
            data = json.loads(ENROLLMENT_FILE.read_text(encoding="utf-8"))
            return data.get("initial_prompt")
    except Exception as e:
        logger.warning(f"Failed to load enrollment: {e}")
    return None


def save_enrollment(prompt: str) -> None:
    """Save enrollment prompt to file."""
    try:
        ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
        data = {"initial_prompt": prompt}
        ENROLLMENT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Enrollment saved to {ENROLLMENT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save enrollment: {e}")
        raise


def run_enrollment(settings: Settings) -> None:
    """Run voice enrollment process."""
    click.echo("\n🎤 Voice Enrollment")
    click.echo("=" * 50)
    click.echo("Please speak each phrase clearly when prompted.")
    click.echo("Press Enter when ready to record each phrase.\n")

    stt_engine: STTEngineBase = create_stt_engine(settings.stt)

    all_transcriptions = []

    for i, phrase in enumerate(ENROLLMENT_PHRASES, 1):
        click.echo(f"Phrase {i}/{len(ENROLLMENT_PHRASES)}:")
        click.echo(f'  📝 "{phrase}"')
        click.echo("  Press Enter to start recording (5 seconds)...")
        click.prompt("", prompt_suffix="", show_default=False)

        try:
            click.echo("  🔴 Recording...")
            audio = stt_engine.record_audio(settings.stt.max_listen_seconds)
            click.echo("  🎙️  Transcribing...")
            text, lang = stt_engine.transcribe(audio)

            if text:
                click.echo(f'  ✅ Heard: "{text}" (lang: {lang})')
                all_transcriptions.append(text)
            else:
                click.echo("  ⚠️  No speech detected, skipping...")

        except Exception as e:
            click.echo(f"  ❌ Error: {e}")
            logger.error(f"Enrollment error on phrase {i}: {e}")

        click.echo("")

    if not all_transcriptions:
        click.echo("❌ No successful transcriptions. Enrollment cancelled.")
        return

    # Create initial_prompt from all transcriptions
    initial_prompt = " ".join(all_transcriptions)

    click.echo("=" * 50)
    click.echo(f'📋 Combined prompt: "{initial_prompt}"')
    click.echo("")

    if click.confirm("Save this enrollment?", default=True):
        try:
            save_enrollment(initial_prompt)
            click.echo("✅ Enrollment saved successfully!")
            click.echo(f"   File: {ENROLLMENT_FILE}")
            click.echo("\n💡 Restart voice-assistant to use the new enrollment.")
        except Exception as e:
            click.echo(f"❌ Failed to save enrollment: {e}")
    else:
        click.echo("Enrollment cancelled.")
