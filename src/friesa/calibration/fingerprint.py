# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

import hashlib
import json

from .models import CalibrationBundle


def bundle_fingerprint(bundle: CalibrationBundle) -> str:
    canonical = json.dumps(bundle.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
