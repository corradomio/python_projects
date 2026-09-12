"""Download the OpenRouter model list, save it to JSON, and print model names."""

import json
import os
import sys
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OUTPUT = "openrouter_models.json"
OPENROUTER_API_KEY = "..."

def download_openrouter_models(
    output_path: str = DEFAULT_OUTPUT,
    api_key: str | None = None,
    timeout: int = 30,
) -> list[dict]:
    """Fetch the list of models from the OpenRouter API.

    Args:
        output_path: Where to write the raw JSON response.
        api_key: OpenRouter API key. Falls back to the OPENROUTER_API_KEY
            environment variable.
        timeout: Request timeout in seconds.

    Returns:
        The list of model dicts contained in the response's "data" field.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key: set the OPENROUTER_API_KEY environment variable "
            "or pass api_key=..."
        )

    request = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"OpenRouter returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc

    # 2) save the raw response
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    models = payload.get("data", [])

    # 3) print each model's name
    for model in models:
        # print(model.get("name") or model.get("id", "<unnamed>"))
        print(model.get("id", "<unnamed>"))

    return models


if __name__ == "__main__":
    try:
        found = download_openrouter_models()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(found)} models saved to {DEFAULT_OUTPUT}", file=sys.stderr)
