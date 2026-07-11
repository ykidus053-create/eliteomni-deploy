#!/usr/bin/env python3
"""Print multimodal configuration and optionally run live probes."""
from __future__ import annotations

import argparse
import json

from modules.multimodal_gateway import (
    multimodal_status,
    ocr_document_v22,
    vision_describe_v22,
)


TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    print(
        json.dumps(
            multimodal_status(),
            indent=2,
        )
    )

    if not args.live:
        return 0

    print("\nVISION LIVE PROBE")
    vision = vision_describe_v22(
        TINY_PNG,
        (
            "Describe this one-pixel test image "
            "in one short sentence."
        ),
    )
    print(vision)

    print("\nOCR LIVE PROBE")
    ocr = ocr_document_v22(
        TINY_PNG,
        "probe.png",
    )
    print(ocr[:1000])

    failed = (
        vision.startswith("[vision unavailable")
        or ocr.startswith("[OCR authentication failed")
        or ocr.startswith("[OCR unavailable")
        or ocr.startswith("[OCR failed")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
