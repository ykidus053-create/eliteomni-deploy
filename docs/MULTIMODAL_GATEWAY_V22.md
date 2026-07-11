# Multimodal Gateway V22

V22 repairs EliteOmni's active vision and document-reading paths.

## Fixes

- Loads `.env` from the repository root, while preserving explicit process
  environment variables.
- Sends Mistral vision images using Mistral's string `image_url` format.
- Uses Groq vision as an automatic fallback after a Mistral authentication,
  access, availability, or configuration failure.
- Uses Mistral OCR's `document_url` shape for PDFs/documents and `image_url`
  for images.
- Extracts ordered per-page Markdown and preserves returned tables,
  headers, and footers.
- Uses Mistral OCR automatically for scanned PDFs whose local extraction is
  empty or too short.
- Caches OCR results by content hash.
- Removes paid live vision calls from the normal offline unit-test suite.

## Configuration

```env
MISTRAL_API_KEY=
GROQ_API_KEY=

ELITE_VISION_PROVIDER=auto
MISTRAL_VISION_MODEL=mistral-small-latest
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
MISTRAL_OCR_MODEL=mistral-ocr-latest

ELITE_DOCUMENT_OCR_MODE=auto
ELITE_PDF_LOCAL_TEXT_MIN_CHARS=120
ELITE_DOCUMENT_MAX_CHARS=250000
ELITE_MULTIMODAL_MAX_BYTES=52428800
ELITE_OCR_TIMEOUT_SECONDS=120
ELITE_VISION_TIMEOUT_SECONDS=60
ELITE_OCR_CACHE_SECONDS=3600
ELITE_RUN_LIVE_MULTIMODAL_TESTS=0
```

Set `ELITE_DOCUMENT_OCR_MODE=always` when layout, equations, and tables are
more important than minimizing OCR calls.

## Checks

Offline configuration:

```bash
PYTHONPATH=. python3 scripts/check_multimodal.py
```

Live provider probes:

```bash
PYTHONPATH=. python3 scripts/check_multimodal.py --live
```

Live pytest:

```bash
ELITE_RUN_LIVE_MULTIMODAL_TESTS=1 \
PYTHONPATH=. python3 -m pytest -q \
smoke_test.py::test_vision_describe
```
