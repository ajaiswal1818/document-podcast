"""Qwen model client wrapper for MLX-backed local inference."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
except Exception:  # pragma: no cover - optional dependency path
    generate = None
    load = None
    make_sampler = None

MODEL_VARIANTS = {
    "Qwen3-4B-4bit": "mlx-community/Qwen3-4B-4bit",
    "Qwen3-4B-6bit": "mlx-community/Qwen3-4B-6bit",
    "Qwen3-4B-8bit": "mlx-community/Qwen3-4B-8bit",
    "Qwen3-8B-4bit": "mlx-community/Qwen3-8B-4bit",
}


class Qwen:
    """Thin wrapper around the MLX Qwen model."""

    def __init__(
        self,
        model_name: str | None = None,
        model_version: str = "Qwen3-4B-4bit",
    ) -> None:
        self.model_version = model_version
        self.model_name = model_name or MODEL_VARIANTS.get(model_version, model_version)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.load_error: Exception | None = None

        if load is None:
            self.load_error = RuntimeError("mlx_lm is not installed or import failed")
            return

        try:
            self.model, self.tokenizer = load(self.model_name)
        except Exception as exc:  # pragma: no cover - model download may fail at runtime
            self.load_error = exc

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.load_error is None

    @staticmethod
    def _normalize_json_output(raw: str) -> str:
        text = raw.strip()

        text = re.sub(r"(?is)<think>.*?</think>", " ", text)
        # A generation that ran out of tokens mid-reasoning leaves an unclosed
        # <think> block; everything after it is internal reasoning, not output.
        text = re.sub(r"(?is)<think>.*$", " ", text)
        text = re.sub(r"(?is)```(?:json)?\s*", "", text)
        text = re.sub(r"(?is)\s*```\s*$", "", text)
        text = text.strip()

        if text.startswith("{"):
            return text

        start = text.find("{")
        if start != -1:
            end = text.rfind("}")
            if end > start:
                candidate = text[start : end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return text

    def generate(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response using the local Qwen model."""
        if not self.available:
            raise RuntimeError(f"Qwen model is unavailable: {self.load_error}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "verbose": False}
        if make_sampler is not None and temperature > 0:
            kwargs["sampler"] = make_sampler(temp=temperature, top_p=0.95)

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            **kwargs,
        )
        return self._normalize_json_output(str(response).strip())

    def generate_json(self, system: str, user: str, max_tokens: int = 2048, retries: int = 2) -> dict[str, Any]:
        """Generate JSON output from the model and validate it strictly."""
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                raw = self.generate(system, user, max_tokens=max_tokens, temperature=0.3)
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
                raise ValueError("JSON output was not an object.")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                user = (
                    f"{user}\n\nYour previous response was invalid JSON. "
                    "Return only raw JSON, no markdown fences, no code blocks, no commentary, no trailing commas."
                )

        if last_error is not None:
            raise ValueError(f"Qwen did not return valid JSON after retries: {last_error}") from last_error

        raise ValueError("Qwen did not return valid JSON.")
