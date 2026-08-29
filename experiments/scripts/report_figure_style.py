"""Shared publication style and file helpers for report figures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAVY = "#1F4E79"
BLUE = "#0072B2"
GREEN = "#009E73"
TEAL = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#7C3AED"
INK = "#20262E"
MUTED = "#5D6875"
GRID = "#D9E0E7"
GREY = "#6B7280"
LIGHT_GREY = "#B8C0CC"
WHITE = "#FFFFFF"


class FigureBuildError(ValueError):
    """Raised when a public report-figure package is invalid."""


def require(condition: bool, message: str) -> None:
    """Raise a stable validation error when a build requirement is unmet."""

    if not condition:
        raise FigureBuildError(message)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_png(path: Path) -> None:
    """Require one existing file with a valid PNG signature."""

    require(path.is_file(), f"Missing curated PNG source: {path}")
    require(
        path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
        f"Invalid PNG signature: {path}",
    )


def apply_experiment_style(plt_module) -> None:
    """Apply the common analytical-figure style used by all experiments."""

    plt_module.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": "#9CA3AF",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_experiment_figure(plt_module, figure, output_path: Path) -> None:
    """Atomically save and close one experiment report figure."""

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    figure.savefig(
        temporary_path,
        format="png",
        dpi=180,
        metadata={"Software": "face-identification-system"},
    )
    plt_module.close(figure)
    temporary_path.replace(output_path)


def project_style_context():
    """Return the deterministic style context for the project scope figure."""

    return plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": WHITE,
            "text.color": INK,
        }
    )
