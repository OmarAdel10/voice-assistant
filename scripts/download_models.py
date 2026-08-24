#!/usr/bin/env python3
"""Download STT and TTS models to local directory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from piper.download import download_voice, get_voices

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_whisper_model(model_size: str, target_dir: Path) -> None:
    """Download faster-whisper model from Hugging Face Hub."""
    repo_id = f"Systran/faster-whisper-{model_size}"
    local_dir = target_dir / model_size

    if local_dir.exists() and any(local_dir.iterdir()):
        logger.info(f"Model {model_size} already exists at {local_dir}, skipping")
        return

    logger.info(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"Successfully downloaded {model_size}")
    except Exception as e:
        logger.error(f"Failed to download {model_size}: {e}")
        raise


def download_piper_voice(voice_name: str, target_dir: Path) -> None:
    """Download piper TTS voice."""
    voice_dir = target_dir / voice_name

    if voice_dir.exists() and any(voice_dir.iterdir()):
        logger.info(f"Voice {voice_name} already exists at {voice_dir}, skipping")
        return

    logger.info(f"Downloading piper voice: {voice_name}...")
    try:
        voices = get_voices()
        if voice_name not in voices:
            logger.error(f"Voice {voice_name} not found in piper voices list")
            logger.info(f"Available voices: {', '.join(sorted(voices.keys()))}")
            raise ValueError(f"Voice {voice_name} not available")

        download_voice(voice_name, voices, target_dir)
        logger.info(f"Successfully downloaded {voice_name}")
    except Exception as e:
        logger.error(f"Failed to download voice {voice_name}: {e}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Download STT and TTS models")
    parser.add_argument(
        "--stt-model",
        default="large-v3",
        help="Whisper model size to download (default: large-v3)",
    )
    parser.add_argument(
        "--tts-voices",
        nargs="+",
        default=["ar_EG-medium", "en_US-medium"],
        help="Piper voices to download (default: ar_EG-medium en_US-medium)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Base directory for models (default: models)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    stt_dir = models_dir / "stt"
    tts_dir = models_dir / "tts"

    stt_dir.mkdir(parents=True, exist_ok=True)
    tts_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download STT model
        logger.info("=" * 50)
        logger.info("Downloading STT model")
        logger.info("=" * 50)
        download_whisper_model(args.stt_model, stt_dir)

        # Download TTS voices
        logger.info("=" * 50)
        logger.info("Downloading TTS voices")
        logger.info("=" * 50)
        for voice in args.tts_voices:
            download_piper_voice(voice, tts_dir)

        logger.info("=" * 50)
        logger.info("All models downloaded successfully!")
        logger.info(f"STT: {stt_dir}/{args.stt_model}")
        for voice in args.tts_voices:
            logger.info(f"TTS: {tts_dir}/{voice}")
        logger.info("=" * 50)

        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
