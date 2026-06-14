# Opus 4.7 Core Architectural Engine Pipeline
import os
import json
from typing import Dict, Any, Tuple

MEMORY_FILE = os.path.expanduser("~/eliteomni_opus_memory.json")

def load_persistent_memory() -> Dict[str, Any]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_persistent_memory(data: dict) -> None:
    try:
        current = load_persistent_memory()
        current.update(data)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception:
        pass

def process_vision_grid(image_metadata: Dict[str, Any]) -> Tuple[int, int]:
    """Downscales high-res image grids below the 3.75 Megapixel threshold."""
    width = image_metadata.get("width", 1000)
    height = image_metadata.get("height", 1000)

    megapixels = (width * height) / 1_000_000.0

    if megapixels > 3.75:
        scale_factor = (3.75 / megapixels) ** 0.5
        width = int(width * scale_factor)
        height = int(height * scale_factor)

    return width, height
