"""Build publication-quality figures from verified Experiment 02 results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_SCRIPT_DIR = SCRIPT_DIR.parent
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))

from report_figure_style import (  # noqa: E402
    apply_experiment_style,
    save_experiment_figure,
    sha256_file,
)


CONDITIONS = ("full_image", "mtcnn_crop_fallback")
CONDITION_LABELS = {
    "full_image": "Full image",
    "mtcnn_crop_fallback": "MTCNN crop/fallback",
}
CONDITION_COLORS = {
    "full_image": "#6B7280",
    "mtcnn_crop_fallback": "#0072B2",
}
FIGURE_FILENAMES = (
    "01_preprocessing_decision.png",
    "02_detection_audit.png",
    "03_rank_movement.png",
)


def build_report_figures(
    *, input_dir: Path, output_dir: Path, force: bool = False
) -> tuple[Path, ...]:
    """Validate canonical results and build all Experiment 02 report figures."""

    results = load_verified_results(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = tuple(output_dir / name for name in FIGURE_FILENAMES)
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"figure output already exists ({names}); pass --force to replace it"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    apply_experiment_style(plt)
    _plot_preprocessing_decision(
        plt=plt,
        percent_formatter=PercentFormatter,
        metrics=results["condition_metrics"],
        outcomes=results["top1_outcome_summary"][0],
        output_path=output_paths[0],
    )
    _plot_detection_audit(
        plt=plt,
        percent_formatter=PercentFormatter,
        detection_summary=results["detection_summary"],
        detection_records=results["detection_records"],
        output_path=output_paths[1],
    )
    _plot_rank_movement(
        plt=plt,
        percent_formatter=PercentFormatter,
        cases=results["probe_case_analysis"],
        output_path=output_paths[2],
    )

    for path in output_paths:
        print(f"[WRITE] {path}")
        print(f"        sha256={sha256_file(path)}")
    return output_paths


def load_verified_results(input_dir: Path) -> dict[str, object]:
    """Load canonical tables after checking recorded hashes and shapes."""

    preprocessing_path = input_dir / "preprocessing_audit.json"
    comparison_path = input_dir / "comparison_run.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    if (
        preprocessing.get("status") != "complete"
        or preprocessing.get("run_scope") != "full"
    ):
        raise ValueError("preprocessing audit must describe a complete full run")
    if (
        preprocessing.get("results", {}).get("processed_images") != 3264
        or preprocessing.get("results", {}).get("coverage_rate") != 1.0
    ):
        raise ValueError("preprocessing audit contains unexpected coverage")
    if comparison.get("status") != "complete" or comparison.get("run_scope") != "full":
        raise ValueError("retrieval comparison must describe a complete full run")
    if comparison.get("model") != "vggface2":
        raise ValueError("retrieval comparison contains an unexpected checkpoint")
    if comparison.get("pipeline", {}).get("conditions") != list(CONDITIONS):
        raise ValueError("retrieval comparison contains unexpected conditions")
    if comparison.get("result", {}).get("recommendation") != "adopt_mtcnn_crop_fallback":
        raise ValueError("canonical preprocessing recommendation is unexpected")
    if comparison.get("experiment_01_baseline_check", {}).get("status") != "matched":
        raise ValueError("Experiment 01 baseline reproduction was not verified")

    for filename, metadata in preprocessing.get("artifacts", {}).items():
        path = input_dir / filename
        if sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"artifact hash does not match audit: {filename}")
    for filename, metadata in comparison.get("artifacts", {}).items():
        path = input_dir / filename
        if sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"artifact hash does not match comparison: {filename}")

    table_expectations = {
        "detection_summary": 3,
        "detection_records": 3264,
        "condition_metrics": 2,
        "metric_deltas": 1,
        "probe_rankings": 1998,
        "probe_case_analysis": 999,
        "top1_outcome_summary": 1,
        "rank_change_summary": 1,
    }
    results: dict[str, object] = {
        "preprocessing_audit": preprocessing,
        "comparison_run": comparison,
    }
    for table_name, expected_rows in table_expectations.items():
        rows = _read_csv(input_dir / f"{table_name}.csv")
        if len(rows) != expected_rows:
            raise ValueError(
                f"{table_name}.csv must contain {expected_rows} rows; "
                f"received {len(rows)}"
            )
        results[table_name] = rows

    metrics = {row["condition"]: row for row in results["condition_metrics"]}
    if set(metrics) != set(CONDITIONS):
        raise ValueError("condition metrics do not contain both preprocessing policies")
    outcomes = results["top1_outcome_summary"][0]
    if int(outcomes["probe_count"]) != 999:
        raise ValueError("paired outcome table must contain 999 probes")
    if sum(
        int(outcomes[field])
        for field in (
            "both_correct_count",
            "cropping_helps_count",
            "cropping_hurts_count",
            "neither_correct_count",
        )
    ) != 999:
        raise ValueError("paired Top-1 outcome counts must sum to 999")
    return results


def _plot_preprocessing_decision(
    *, plt, percent_formatter, metrics, outcomes, output_path: Path
) -> None:
    metrics_by_condition = {row["condition"]: row for row in metrics}
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))

    top_fields = ("top_1", "top_3", "top_5", "top_10")
    top_labels = ("Top-1", "Top-3", "Top-5", "Top-10")
    x = np.arange(len(top_fields))
    width = 0.36
    for index, condition in enumerate(CONDITIONS):
        values = [
            float(metrics_by_condition[condition][field]) for field in top_fields
        ]
        bars = axes[0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
        _label_percentage_bars(axes[0], bars, values, rotation=90)
    axes[0].set_title("A. Candidate-list accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("Retrieval accuracy")
    axes[0].set_xticks(x, top_labels)
    axes[0].set_ylim(0.78, 1.0)
    axes[0].yaxis.set_major_formatter(percent_formatter(1.0))
    axes[0].legend(frameon=False, loc="lower right")

    outcome_fields = (
        "both_correct_count",
        "cropping_helps_count",
        "cropping_hurts_count",
        "neither_correct_count",
    )
    outcome_labels = ("Both\ncorrect", "Cropping\nhelps", "Cropping\nhurts", "Neither\ncorrect")
    outcome_colors = ("#6B7280", "#009E73", "#D55E00", "#B9C0C9")
    counts = [int(outcomes[field]) for field in outcome_fields]
    bars = axes[1].bar(
        np.arange(len(counts)), counts, color=outcome_colors, width=0.68
    )
    _label_count_bars(axes[1], bars, counts, total=999)
    axes[1].set_title("B. Paired Top-1 outcomes", loc="left", fontweight="bold")
    axes[1].set_ylabel("Probe identities")
    axes[1].set_xticks(np.arange(len(counts)), outcome_labels)
    axes[1].set_ylim(0, max(counts) * 1.16)

    estimate = float(outcomes["top_1_delta"])
    low = float(outcomes["top_1_delta_ci_95_low"])
    high = float(outcomes["top_1_delta_ci_95_high"])
    axes[2].axvline(0.0, color="#6B7280", linewidth=1.2)
    axes[2].errorbar(
        estimate,
        0,
        xerr=np.asarray([[estimate - low], [high - estimate]]),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        markersize=9,
        capsize=7,
        linewidth=2.2,
    )
    axes[2].text(
        estimate,
        0.12,
        f"{estimate:+.2%}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    axes[2].text(
        estimate,
        -0.12,
        f"95% CI [{low:+.2%}, {high:+.2%}]",
        ha="center",
        va="top",
    )
    axes[2].set_title("C. Paired Top-1 effect", loc="left", fontweight="bold")
    axes[2].set_xlabel("Accuracy difference: crop/fallback minus full image")
    axes[2].set_yticks([])
    axes[2].set_ylim(-0.35, 0.35)
    axes[2].set_xlim(-0.025, 0.16)
    axes[2].xaxis.set_major_formatter(percent_formatter(1.0))
    axes[2].text(
        0.5,
        0.06,
        "Exact McNemar p = 2.38 × 10⁻¹⁶",
        transform=axes[2].transAxes,
        ha="center",
        color="#4B5563",
    )

    figure.suptitle(
        "Face cropping materially improves first-choice identification",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "The same 999 probes were evaluated under both preprocessing policies",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "VGGFace2 embeddings; 2,265 gallery images; exact cosine retrieval; 10,000 paired bootstrap samples.",
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.23, wspace=0.28)
    save_experiment_figure(plt, figure, output_path)


def _plot_detection_audit(
    *, plt, percent_formatter, detection_summary, detection_records, output_path: Path
) -> None:
    summaries = {row["split"]: row for row in detection_summary}
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))

    split_order = ("gallery", "probe", "all")
    split_labels = ("Gallery", "Probe", "Overall")
    detection_rates = [float(summaries[split]["detection_rate"]) for split in split_order]
    bars = axes[0].bar(
        np.arange(3), detection_rates, color=("#0072B2", "#009E73", "#6B7280")
    )
    axes[0].set_title("A. Detection coverage", loc="left", fontweight="bold")
    axes[0].set_ylabel("Detection rate (axis starts at 99.8%)")
    axes[0].set_xticks(np.arange(3), split_labels)
    axes[0].set_ylim(0.998, 1.00015)
    axes[0].yaxis.set_major_formatter(percent_formatter(1.0))
    for bar, value in zip(bars, detection_rates):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.00012,
            f"{value:.3%}",
            ha="center",
            va="top",
            fontsize=8,
        )

    confidence_fields = (
        "mean_detection_confidence",
        "median_detection_confidence",
        "minimum_detection_confidence",
    )
    confidence_labels = ("Mean", "Median", "Minimum")
    confidence_values = [float(summaries["all"][field]) for field in confidence_fields]
    bars = axes[1].bar(
        np.arange(3), confidence_values, color=("#0072B2", "#009E73", "#E69F00")
    )
    axes[1].set_title("B. Detection confidence", loc="left", fontweight="bold")
    axes[1].set_ylabel("MTCNN confidence")
    axes[1].set_xticks(np.arange(3), confidence_labels)
    axes[1].set_ylim(0.85, 1.01)
    axes[1].yaxis.set_major_formatter(percent_formatter(1.0))
    for bar, value in zip(bars, confidence_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.008,
            f"{value:.2%}",
            ha="center",
            va="top",
            fontsize=8,
        )

    crop_ratios = np.asarray(
        [
            float(row["crop_area_ratio"])
            for row in detection_records
            if row["face_detected"] == "True" and row["crop_area_ratio"]
        ],
        dtype=np.float64,
    )
    axes[2].hist(
        crop_ratios,
        bins=np.linspace(0.05, 0.85, 33),
        color="#0072B2",
        edgecolor="white",
        linewidth=0.4,
    )
    mean_ratio = float(np.mean(crop_ratios))
    median_ratio = float(np.median(crop_ratios))
    axes[2].axvline(mean_ratio, color="#D55E00", linewidth=2, label=f"Mean {mean_ratio:.1%}")
    axes[2].axvline(
        median_ratio,
        color="#009E73",
        linewidth=2,
        linestyle="--",
        label=f"Median {median_ratio:.1%}",
    )
    axes[2].set_title("C. Detected-face crop size", loc="left", fontweight="bold")
    axes[2].set_xlabel("Crop area as share of source image")
    axes[2].set_ylabel("Detected images")
    axes[2].xaxis.set_major_formatter(percent_formatter(1.0))
    axes[2].legend(frameon=False, loc="upper right")

    figure.suptitle(
        "MTCNN produced usable preprocessing for the complete corpus",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "3,263 faces detected; one controlled fallback; zero dropped images",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "Crop statistics exclude the one fallback image; 160 × 160 crops were stored losslessly for evaluation.",
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.23, wspace=0.34)
    save_experiment_figure(plt, figure, output_path)


def _plot_rank_movement(
    *, plt, percent_formatter, cases, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 6.15))
    full_ranks = np.asarray(
        [int(row["full_image_true_identity_rank"]) for row in cases], dtype=np.int32
    )
    crop_ranks = np.asarray(
        [int(row["crop_fallback_true_identity_rank"]) for row in cases], dtype=np.int32
    )
    rank_points = np.asarray((1, 3, 5, 10, 20, 50, 100, 250, 500, 999))
    for condition, ranks in zip(CONDITIONS, (full_ranks, crop_ranks)):
        values = np.asarray([(ranks <= rank).mean() for rank in rank_points])
        axes[0].plot(
            rank_points,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
    axes[0].set_title("A. Correct identity recovered by rank", loc="left", fontweight="bold")
    axes[0].set_xlabel("Candidate-list depth (log scale)")
    axes[0].set_ylabel("Cumulative retrieval accuracy")
    axes[0].set_xscale("log")
    axes[0].set_xticks(rank_points, [str(value) for value in rank_points])
    axes[0].set_ylim(0.80, 1.005)
    axes[0].yaxis.set_major_formatter(percent_formatter(1.0))
    axes[0].legend(frameon=False, loc="lower right")

    movement_counts = (
        int(np.sum(crop_ranks < full_ranks)),
        int(np.sum(crop_ranks == full_ranks)),
        int(np.sum(crop_ranks > full_ranks)),
    )
    movement_labels = ("Improved", "Unchanged", "Worsened")
    movement_colors = ("#009E73", "#6B7280", "#D55E00")
    bars = axes[1].bar(
        np.arange(3), movement_counts, color=movement_colors, width=0.68
    )
    _label_count_bars(axes[1], bars, movement_counts, total=len(cases))
    axes[1].set_title("B. True-identity rank movement", loc="left", fontweight="bold")
    axes[1].set_ylabel("Probe identities")
    axes[1].set_xticks(np.arange(3), movement_labels)
    axes[1].set_ylim(0, max(movement_counts) * 1.16)

    ordered = sorted(cases, key=lambda row: int(row["rank_delta_crop_minus_full"]))
    extremes = ordered[:5] + ordered[-5:]
    values = [int(row["rank_delta_crop_minus_full"]) for row in extremes]
    labels = [row["true_identity"].replace("_", " ") for row in extremes]
    colors = ["#009E73" if value < 0 else "#D55E00" for value in values]
    y = np.arange(len(extremes))
    bars = axes[2].barh(y, values, color=colors)
    axes[2].axvline(0, color="#6B7280", linewidth=1)
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    axes[2].set_title("C. Largest observed rank movements", loc="left", fontweight="bold")
    axes[2].set_xlabel("Crop rank minus full-image rank (negative is better)")
    axes[2].set_xlim(-1075, 1075)
    for bar, row, value in zip(bars, extremes, values):
        full_rank = int(row["full_image_true_identity_rank"])
        crop_rank = int(row["crop_fallback_true_identity_rank"])
        axes[2].text(
            value / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{full_rank}→{crop_rank}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    figure.suptitle(
        "Cropping improves short lists while exposing a smaller failure tail",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "Most ranks were unchanged, but a small number of probes experienced very large regressions",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "Rank movement is paired by probe; panel C shows the five largest gains and five largest losses.",
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.20, wspace=0.35)
    save_experiment_figure(plt, figure, output_path)


def _label_percentage_bars(
    axis, bars, values: Sequence[float], *, rotation: int = 0
) -> None:
    span = axis.get_ylim()[1] - axis.get_ylim()[0]
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(span * 0.018, 0.0015),
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def _label_count_bars(axis, bars, values: Sequence[int], *, total: int) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _add_figure_note(figure, text: str) -> None:
    figure.text(0.5, 0.055, text, ha="center", fontsize=9, color="#4B5563")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verified Experiment 02 report figures."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/outputs/02_face_preprocessing"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures/02_face_preprocessing"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report_figures(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
