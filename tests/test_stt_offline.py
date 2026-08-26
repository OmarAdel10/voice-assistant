"""Offline-first STT loading: local snapshot resolution before Hub lookup."""

from __future__ import annotations

import sys
import types

import pytest

from core.stt_engine import STTError, STTEngine


class FakeWhisperModel:
    """Records constructor args instead of loading real weights."""

    instances: list["FakeWhisperModel"] = []

    def __init__(self, source, device=None, compute_type=None, download_root=None):
        self.source = source
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):  # pragma: no cover - not exercised here
        return iter(()), types.SimpleNamespace(language="en", language_probability=1.0)


@pytest.fixture()
def fake_faster_whisper(monkeypatch):
    FakeWhisperModel.instances.clear()
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeWhisperModel


def make_engine(model_dir, offline=False) -> STTEngine:
    return STTEngine(
        model_size="medium",
        device="cuda",
        compute_type="int8",
        model_dir=model_dir,
        offline=offline,
    )


def test_plain_folder_snapshot_is_used(tmp_path, fake_faster_whisper):
    model_dir = tmp_path / "models"
    (model_dir / "medium").mkdir(parents=True)
    (model_dir / "medium" / "model.bin").write_bytes(b"x")

    engine = make_engine(str(model_dir))
    engine.load_model()

    assert len(fake_faster_whisper.instances) == 1
    used = fake_faster_whisper.instances[0]
    assert used.source == str(model_dir / "medium")
    assert used.download_root is None  # local path: no hub machinery


def test_hf_cache_layout_snapshot_is_used(tmp_path, fake_faster_whisper):
    model_dir = tmp_path / "models"
    snap = model_dir / "models--Systran--faster-whisper-medium" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")

    engine = make_engine(str(model_dir))
    engine.load_model()

    assert fake_faster_whisper.instances[0].source == str(snap)


def test_missing_model_falls_back_to_hub_name(tmp_path, fake_faster_whisper):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    engine = make_engine(str(model_dir), offline=False)
    engine.load_model()

    used = fake_faster_whisper.instances[0]
    assert used.source == "medium"  # resolved by name via the Hub
    assert used.download_root == str(model_dir)


def test_offline_without_snapshot_raises(tmp_path, fake_faster_whisper):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    engine = make_engine(str(model_dir), offline=True)
    with pytest.raises(STTError, match="no local snapshot"):
        engine.load_model()

    assert fake_faster_whisper.instances == []  # never attempted a Hub load


def test_offline_with_snapshot_skips_hub(tmp_path, fake_faster_whisper):
    model_dir = tmp_path / "models"
    snap = model_dir / "models--Systran--faster-whisper-medium" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")

    engine = make_engine(str(model_dir), offline=True)
    engine.load_model()  # must not raise despite offline mode

    assert fake_faster_whisper.instances[0].source == str(snap)
