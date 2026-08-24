#!/usr/bin/env python3
"""Download STT and TTS models to local directory."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


PIPER_VOICES = {
    "ar_JO-kareem-medium": {
        "repo_id": "rhasspy/piper-voices",
        "files": [
            "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
            "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json",
        ],
        "folder": "ar_JO-kareem-medium",
    },
    "ar_JO-kareem-low": {
        "repo_id": "rhasspy/piper-voices",
        "files": [
            "ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx",
            "ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx.json",
        ],
        "folder": "ar_JO-kareem-low",
    },
    "en_US-lessac-medium": {
        "repo_id": "rhasspy/piper-voices",
        "files": [
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ],
        "folder": "en_US-lessac-medium",
    },
    "en_US-lessac-low": {
        "repo_id": "rhasspy/piper-voices",
        "files": [
            "en/en_US/lessac/low/en_US-lessac-low.onnx",
            "en/en_US/lessac/low/en_US-lessac-low.onnx.json",
        ],
        "folder": "en_US-lessac-low",
    },
}


def download_whisper_model(model_size: str, target_dir: Path) -> None:
    """Download faster-whisper model from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

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
    """Download piper TTS voice from rhasspy/piper-voices on HuggingFace."""
    if voice_name not in PIPER_VOICES:
        logger.error(f"Unknown voice: {voice_name}")
        raise ValueError(f"Voice {voice_name} not configured")

    voice_info = PIPER_VOICES[voice_name]
    voice_dir = target_dir / voice_info["folder"]

    if voice_dir.exists() and any(voice_dir.iterdir()):
        logger.info(f"Voice {voice_name} already exists at {voice_dir}, skipping")
        return

    logger.info(f"Downloading piper voice: {voice_name}...")
    voice_dir.mkdir(parents=True, exist_ok=True)

    try:
        for file_path in voice_info["files"]:
            filename = Path(file_path).name

            logger.info(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=voice_info["repo_id"],
                filename=file_path,
                local_dir=voice_dir,
                local_dir_use_symlinks=False,
            )
        logger.info(f"Successfully downloaded {voice_name}")
    except Exception as e:
        logger.error(f"Failed to download voice {voice_name}: {e}")
        raise


def download_piper_voices_parallel(voice_names: list[str], target_dir: Path) -> None:
    """Download multiple piper voices in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_piper_voice, name, target_dir): name for name in voice_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
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
        default=[
            "ar_JO-kareem-medium",
            "ar_JO-kareem-low",
            "en_US-lessac-medium",
            "en_US-lessac-low",
        ],
        help="Piper voices to download",
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

        # Download TTS voices in parallel
        logger.info("=" * 50)
        logger.info("Downloading TTS voices")
        logger.info("=" * 50)
        download_piper_voices_parallel(args.tts_voices, tts_dir)

        logger.info("=" * 50)
        logger.info("All models downloaded successfully!")
        logger.info(f"STT: {stt_dir}/{args.stt_model}")
        for voice in args.tts_voices:
            voice_info = PIPER_VOICES[voice]
            logger.info(f"TTS: {tts_dir}/{voice_info['folder']}")
        logger.info("=" * 50)

        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
