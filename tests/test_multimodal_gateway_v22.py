import base64
from pathlib import Path

import modules.multimodal_gateway as gateway


TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_mistral_vision_uses_string_image_url(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    captured = {}

    def fake_request_json(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "a tiny image",
                    }
                }
            ]
        }

    monkeypatch.setattr(
        gateway,
        "_request_json",
        fake_request_json,
    )
    result = gateway.vision_describe_v22(
        TINY_PNG,
        "describe",
    )

    image = (
        captured["payload"]["messages"][0]["content"][1]
    )
    assert result == "a tiny image"
    assert isinstance(image["image_url"], str)
    assert image["image_url"].startswith("data:image/")


def test_vision_falls_back_to_groq_after_mistral_401(
    monkeypatch,
):
    monkeypatch.setenv(
        "MISTRAL_API_KEY",
        "bad-mistral",
    )
    monkeypatch.setenv(
        "GROQ_API_KEY",
        "good-groq",
    )
    calls = []

    def fake_request_json(**kwargs):
        calls.append(kwargs)
        if kwargs["provider"] == "Mistral vision":
            raise gateway.ProviderHTTPError(
                "Mistral vision",
                401,
                "Unauthorized",
            )
        return {
            "choices": [
                {
                    "message": {
                        "content": "groq fallback",
                    }
                }
            ]
        }

    monkeypatch.setattr(
        gateway,
        "_request_json",
        fake_request_json,
    )
    result = gateway.vision_describe_v22(
        TINY_PNG,
        "describe",
    )

    assert result == "groq fallback"
    assert [
        call["provider"]
        for call in calls
    ] == [
        "Mistral vision",
        "Groq vision",
    ]

    groq_image = (
        calls[1]["payload"]["messages"][0]["content"][1]
    )
    assert isinstance(
        groq_image["image_url"],
        dict,
    )


def test_ocr_pdf_uses_document_url_and_renders_pages(
    monkeypatch,
):
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
    gateway._OCR_CACHE.clear()
    captured = {}

    def fake_request_json(**kwargs):
        captured.update(kwargs)
        return {
            "pages": [
                {
                    "index": 0,
                    "markdown": "# Title\nBody",
                },
                {
                    "index": 1,
                    "markdown": "Second page",
                },
            ]
        }

    monkeypatch.setattr(
        gateway,
        "_request_json",
        fake_request_json,
    )
    encoded = base64.b64encode(
        b"%PDF-1.4 fake"
    ).decode()
    result = gateway.ocr_document_v22(
        encoded,
        "sample.pdf",
    )

    document = captured["payload"]["document"]
    assert document["type"] == "document_url"
    assert document["document_url"].startswith(
        "data:application/pdf;base64,"
    )
    assert "## Page 1" in result
    assert "## Page 2" in result


def test_ocr_image_uses_image_url(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
    gateway._OCR_CACHE.clear()
    captured = {}

    def fake_request_json(**kwargs):
        captured.update(kwargs)
        return {
            "pages": [
                {
                    "index": 0,
                    "markdown": "receipt",
                }
            ]
        }

    monkeypatch.setattr(
        gateway,
        "_request_json",
        fake_request_json,
    )
    result = gateway.ocr_document_v22(
        TINY_PNG,
        "receipt.png",
    )

    assert result.endswith("receipt")
    document = captured["payload"]["document"]
    assert document["type"] == "image_url"
    assert document["image_url"].startswith(
        "data:image/png;base64,"
    )


def test_invalid_base64_is_graceful():
    result = gateway.ocr_document_v22(
        "not-valid-base64",
        "bad.pdf",
    )
    assert "invalid base64" in result.lower()


def test_missing_mistral_key_is_explicit(monkeypatch):
    for name in (
        "MISTRAL_OCR_API_KEY",
        "MISTRAL_VISION_API_KEY",
        "MISTRAL_API_KEY",
        "MISTRAL_KEY",
        "MISTRAL_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    gateway._ENV_LOADED = True
    result = gateway.ocr_document_v22(
        TINY_PNG,
        "image.png",
    )
    assert "configure MISTRAL_API_KEY" in result


def test_scanned_pdf_uses_ocr(monkeypatch):
    monkeypatch.setattr(
        gateway,
        "_local_pdf_text",
        lambda _: "",
    )
    monkeypatch.setattr(
        gateway,
        "ocr_document_v22",
        lambda *_: "## Page 1\nScanned text",
    )

    result = gateway.extract_uploaded_file_v22(
        "scan.pdf",
        b"%PDF scan",
    )
    assert "Scanned text" in result


def test_active_wiring_is_present():
    http_client = Path(
        "modules/core/http_client.py"
    ).read_text(encoding="utf-8")
    app = Path("app.py").read_text(
        encoding="utf-8"
    )

    assert "BEGIN MULTIMODAL GATEWAY V22" in http_client
    assert "vision_describe_v22" in http_client
    assert "BEGIN DOCUMENT VIEWING V22" in app
    assert "extract_uploaded_file_v22" in app
