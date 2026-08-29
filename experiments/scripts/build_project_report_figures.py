"""Assemble the curated PNG package used by the public project report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_figure_style import (  # noqa: E402
    BLUE,
    GRID,
    INK,
    MUTED,
    NAVY,
    TEAL,
    WHITE,
    project_style_context,
    require,
    sha256_file,
    validate_png,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "experiments" / "figures"
MANIFEST_NAME = "FIGURE_MANIFEST.json"

CURATED_SOURCES = (
    (
        "02_embedding_robustness",
        "01_embedding_robustness/01_checkpoint_comparison",
        "Embedding-checkpoint quality and stability comparison",
    ),
    (
        "03_face_preprocessing",
        "02_face_preprocessing/01_preprocessing_decision",
        "Paired face-preprocessing decision",
    ),
    (
        "04_candidate_list_configuration",
        "03_retrieval_configuration/01_candidate_list_decision",
        "Candidate-list coverage and review-burden trade-off",
    ),
    (
        "05_gallery_depth_configuration",
        "03_retrieval_configuration/02_gallery_depth_decision",
        "Gallery-depth quality and enrollment-cost trade-off",
    ),
)


def _system_scope_figure():
    """Build a simple responsibility view of the implemented service boundary."""

    panels = (
        {
            "heading": "Enrollment",
            "accent": NAVY,
            "bullets": (
                "Receive an identity label and gallery image",
                "Select the largest MTCNN face or use the defined fallback",
                "Create a 512-dimensional VGGFace2 embedding",
                "Normalize and append the image representation to the index",
            ),
        },
        {
            "heading": "Identification",
            "accent": BLUE,
            "bullets": (
                "Apply the same preparation to a probe image",
                "Search gallery-image embeddings by cosine similarity",
                "Collapse duplicate image matches into distinct identities",
                "Return five ranked candidates by default",
            ),
        },
        {
            "heading": "Operational boundary",
            "accent": TEAL,
            "bullets": (
                "Flask API packaged in a non-root Docker image",
                "Exact search by default; HNSW and LSH alternatives implemented",
                "Gallery state remains in process memory",
                "Unknown-person rejection and durable persistence are future work",
            ),
        },
    )

    fig = plt.figure(figsize=(13.6, 5.7))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    fig.text(
        0.055,
        0.965,
        "Implemented face-identification system scope",
        fontsize=19,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.055,
        0.915,
        "Enrollment and probe identification share one validated face-representation pipeline.",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    for left, panel in zip((0.055, 0.37, 0.685), panels):
        accent = panel["accent"]
        card = FancyBboxPatch(
            (left, 0.12),
            0.27,
            0.65,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=WHITE,
            edgecolor=GRID,
            linewidth=1.2,
            transform=axis.transAxes,
        )
        axis.add_patch(card)
        axis.add_patch(
            Rectangle(
                (left, 0.70),
                0.27,
                0.07,
                transform=axis.transAxes,
                facecolor=accent,
                edgecolor="none",
            )
        )
        axis.text(
            left + 0.018,
            0.735,
            str(panel["heading"]).upper(),
            transform=axis.transAxes,
            color=WHITE,
            fontsize=12,
            fontweight="bold",
            va="center",
        )

        y = 0.645
        for bullet in panel["bullets"]:
            axis.text(
                left + 0.022,
                y,
                "•",
                transform=axis.transAxes,
                color=accent,
                fontsize=14,
                va="top",
            )
            axis.text(
                left + 0.044,
                y,
                textwrap.fill(str(bullet), width=35),
                transform=axis.transAxes,
                color=INK,
                fontsize=10.3,
                va="top",
                linespacing=1.25,
            )
            y -= 0.145
    return fig


def _save_system_scope(staging: Path) -> dict[str, object]:
    path = staging / "01_system_scope.png"
    with project_style_context():
        figure = _system_scope_figure()
        figure.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
            facecolor=WHITE,
            metadata={"Software": "Matplotlib"},
        )
        plt.close(figure)
    validate_png(path)
    return {
        "target": path.name,
        "source_relative_to_experiment_figure_root": None,
        "role": "Implemented runtime and operational scope",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _copy_curated_png(
    source_root: Path,
    staging: Path,
    target_stem: str,
    source_stem: str,
    role: str,
) -> dict[str, object]:
    source = source_root / f"{source_stem}.png"
    validate_png(source)
    destination = staging / f"{target_stem}.png"
    shutil.copyfile(source, destination)
    require(
        sha256_file(source) == sha256_file(destination),
        f"Copy hash mismatch: {source}",
    )
    return {
        "target": destination.name,
        "source_relative_to_experiment_figure_root": f"{source_stem}.png",
        "role": role,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def build_project_figure_package(source_root: Path, output_dir: Path) -> Path:
    """Validate and atomically assemble a new public-report figure directory."""

    source_root = source_root.expanduser().absolute()
    destination = output_dir.expanduser().absolute()
    require(source_root.is_dir(), f"Experiment figure root not found: {source_root}")
    require(not destination.exists(), f"Refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete"
    require(not staging.exists(), f"Incomplete figure build already exists: {staging}")

    staging.mkdir()
    records = []
    try:
        records.append(_save_system_scope(staging))
        for target_stem, source_stem, role in CURATED_SOURCES:
            records.append(
                _copy_curated_png(
                    source_root,
                    staging,
                    target_stem,
                    source_stem,
                    role,
                )
            )

        manifest = {
            "schema_version": 1,
            "generator": "experiments/scripts/build_project_report_figures.py",
            "logical_figure_count": len(records),
            "asset_count": len(records),
            "files": sorted(records, key=lambda item: str(item["target"])),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        expected = {str(record["target"]) for record in records} | {MANIFEST_NAME}
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        require(actual == expected, "Project figure package has an unexpected file set.")
        os.replace(staging, destination)
    except Exception:
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to receive the atomic public-report figure package.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = build_project_figure_package(args.source_root, args.output_dir)
    print(f"[COMPLETE] Promoted project-report figure package: {destination}")
    for path in sorted(destination.iterdir()):
        print(f"[WRITE] {path}")


if __name__ == "__main__":
    main()
