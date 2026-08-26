"""Project-wide filesystem anchors.

All default data/model/config locations are resolved relative to the
repository root so the CLI behaves identically regardless of the current
working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
