from __future__ import annotations

import base64
from pathlib import Path

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def guess_mime_type(path: Path) -> str:
    return _MIME_TYPES.get(path.suffix.lower(), "image/png")


def encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")
