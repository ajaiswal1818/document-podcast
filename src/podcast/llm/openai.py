"""OpenAI Responses API wrapper for podcast generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-4"


def _load_project_env() -> None:
    """Load local credentials without overriding environment-provided values."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def has_openai_api_key() -> bool:
    """Return whether an OpenAI API key is available from the environment or .env."""
    _load_project_env()
    return bool(os.getenv("OPENAI_API_KEY"))


class OpenAIChatGPT:
    """LLM adapter matching the project's existing generate/generate_json interface."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        _load_project_env()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.load_error: Exception | None = None
        self.client: Any | None = client
        if not self.api_key:
            self.load_error = RuntimeError("OPENAI_API_KEY is not set.")
        elif self.client is None:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as exc:  # pragma: no cover - environment-specific initialization failures
                self.load_error = exc

    @property
    def available(self) -> bool:
        return self.client is not None and self.load_error is None

    def release(self) -> None:
        """Close the HTTP client after script generation has completed."""
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()

    def generate(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate text using the Responses API without storing response state."""
        del temperature  # Reasoning models manage their own sampling controls.
        if not self.available:
            raise RuntimeError(f"OpenAI model is unavailable: {self.load_error}")
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
            max_output_tokens=max_tokens,
            store=False,
        )
        text = str(getattr(response, "output_text", "")).strip()
        if not text:
            raise RuntimeError("OpenAI returned no output text.")
        return text

    def generate_json(self, system: str, user: str, max_tokens: int = 2048, retries: int = 2) -> dict[str, Any]:
        """Generate and validate a JSON object, retrying malformed responses."""
        last_error: Exception | None = None
        for _attempt in range(retries + 1):
            try:
                raw = self.generate(
                    system,
                    f"{user}\n\nReturn only a valid JSON object, with no markdown or commentary.",
                    max_tokens=max_tokens,
                )
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
                raise ValueError("JSON output was not an object.")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"OpenAI did not return valid JSON after retries: {last_error}") from last_error
