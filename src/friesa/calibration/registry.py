# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

from pathlib import Path

import yaml

from .loader import DEFAULT_BUNDLE_DIR


def list_bundles(bundle_dir: str | Path | None = None) -> list[dict]:
    root = Path(bundle_dir) if bundle_dir else DEFAULT_BUNDLE_DIR
    bundles: list[dict] = []
    for path in sorted(root.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        bundles.append(
            {
                "scenario_id": data.get("scenario_id"),
                "scenario_name": data.get("scenario_name"),
                "validation_status": data.get("validation_status"),
                "research_only": data.get("research_only"),
                "sector": data.get("sector"),
                "path": str(path),
            }
        )
    return bundles
