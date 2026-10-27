"""Dependency-light shared CLI helpers for the VLM examples."""
from __future__ import annotations
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError("Expected an integer >= 1")
    return number

def image_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path

def load_image(value: str):
    from PIL import Image
    path = image_path(value)
    with Image.open(path) as source:
        image = source.convert("RGB")
    return image

def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def metadata() -> dict[str, str]:
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version()}
