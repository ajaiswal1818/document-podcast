"""Qwen model client wrapper for MLX-backed local inference."""

from __future__ import annotations

from typing import Any

try:
    from mlx_lm import generate, load
except Exception:  # pragma: no cover - optional dependency path
    generate = None
    load = None


class Qwen:
    """Thin wrapper around the MLX Qwen model."""

    def __init__(self, model_name: str = "mlx-community/Qwen3-4B-4bit") -> None:
        self.model_name = model_name
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.load_error: Exception | None = None

        if load is None:
            self.load_error = RuntimeError("mlx_lm is not installed or import failed")
            return

        try:
            self.model, self.tokenizer = load(model_name)
        except Exception as exc:  # pragma: no cover - model download may fail at runtime
            self.load_error = exc

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.load_error is None

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
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

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return str(response).strip()
