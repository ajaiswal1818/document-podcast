"""Configuration settings for the podcast pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Base configuration for document-podcast."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")
    input_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "input")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "output")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "models")

    def ensure_directories(self) -> None:
        """Create the required directories if they do not exist."""
        for directory in (self.data_dir, self.input_dir, self.output_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()
