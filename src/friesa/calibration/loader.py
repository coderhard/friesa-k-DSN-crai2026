# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

from pathlib import Path

import yaml

from .models import CalibrationBundle


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = PACKAGE_ROOT / "calibration" / "bundles"


def load_bundle_from_path(path: str | Path) -> CalibrationBundle:
    bundle_path = Path(path)
    data = yaml.safe_load(bundle_path.read_text())
    return CalibrationBundle.model_validate(data)


def load_bundle(scenario_id: str | None = None, path: str | Path | None = None) -> CalibrationBundle:
    if path is not None:
        return load_bundle_from_path(path)
    if scenario_id is None:
        raise ValueError("provide either scenario_id or path")
    bundle_path = DEFAULT_BUNDLE_DIR / f"{scenario_id}.yaml"
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found for scenario_id={scenario_id}")
    return load_bundle_from_path(bundle_path)
