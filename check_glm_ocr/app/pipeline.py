"""Orchestrates the upload -> per-page OCR -> SSE streaming pipeline."""

import json

import httpx

from app.ocr import call_glm_ocr, pdf_to_page_images

PAGE_SEPARATOR = "\n\n<!-- page {page} -->\n\n"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _is_pdf(filename: str, content: bytes) -> bool:
    return filename.lower().endswith(".pdf") or content[:5] == b"%PDF-"


async def run_conversion(filename: str, content: bytes):
    """Async generator yielding SSE-formatted strings as pages are OCR'd."""
    try:
        if _is_pdf(filename, content):
            page_images = pdf_to_page_images(content)
        else:
            page_images = [content]
    except Exception as exc:
        yield _sse("error", {"message": f"Failed to read document: {exc}"})
        return

    total = len(page_images)
    if total == 0:
        yield _sse("error", {"message": "Document has no pages."})
        return

    yield _sse("start", {"total": total})

    parts = []
    async with httpx.AsyncClient() as client:
        for index, image_bytes in enumerate(page_images, start=1):
            try:
                markdown = await call_glm_ocr(image_bytes, client)
            except Exception as exc:
                yield _sse("error", {"message": f"Page {index} failed: {exc}"})
                return

            if total > 1:
                parts.append(PAGE_SEPARATOR.format(page=index) + markdown)
            else:
                parts.append(markdown)

            yield _sse(
                "page",
                {"page": index, "total": total, "markdown": markdown},
            )

    full_markdown = "".join(parts).strip() + "\n"
    yield _sse("done", {"markdown": full_markdown})
