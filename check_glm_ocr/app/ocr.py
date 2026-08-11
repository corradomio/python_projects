"""PDF rasterization and GLM-OCR calls against a local Ollama instance."""

import base64
import re

import httpx
import pymupdf as fitz

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
GLM_OCR_MODEL = "glm-ocr:latest"

OCR_PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings/paragraphs/tables/formulas). "
    "Do not fabricate content that does not exist in the image."
)

RENDER_DPI = 200

_MARKDOWN_FENCE_RE = re.compile(r"```(?:markdown)?\s*\n(.*?)```", re.DOTALL)


def _extract_final_markdown(text: str) -> str:
    """GLM-OCR occasionally emits rambling self-corrections before settling on a
    final answer inside a ```markdown fenced block (since /api/generate bypasses
    the model's chat template). Prefer the last fenced block when present."""
    matches = _MARKDOWN_FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def pdf_to_page_images(pdf_bytes: bytes) -> list[bytes]:
    """Rasterize every page of a PDF to a PNG image."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap(dpi=RENDER_DPI)
            images.append(pixmap.tobytes("png"))
    return images


async def call_glm_ocr(image_bytes: bytes, client: httpx.AsyncClient) -> str:
    """Send a single page image to GLM-OCR via Ollama's native /api/generate endpoint."""
    b64_image = base64.b64encode(image_bytes).decode("ascii")
    response = await client.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": GLM_OCR_MODEL,
            "prompt": OCR_PROMPT,
            "images": [b64_image],
            "stream": False,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    data = response.json()
    return _extract_final_markdown(data.get("response", ""))
