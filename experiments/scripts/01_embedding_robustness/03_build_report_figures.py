"""Build publication-quality figures from verified Experiment 01 results."""

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


MODELS = ("vggface2", "casia-webface")
CONDITIONS = (
    "clean",
    "gaussian_blur_radius_2",
    "resize_down_64_up_250",
    "brightness_0_60",
    "brightness_1_40",
)
DEGRADED_CONDITIONS = CONDITIONS[1:]
CONDITION_LABELS = {
    "clean": "Clean",
    "gaussian_blur_radius_2": "Blur",
    "resize_down_64_up_250": "Reduced\nresolution",
    "brightness_0_60": "Dark",
    "brightness_1_40": "Bright",
}
MODEL_LABELS = {
    "vggface2": "VGGFace2",
    "casia-webface": "CASIA-WebFace",
}
MODEL_COLORS = {
    "vggface2": "#0072B2",
    "casia-webface": "#D55E00",
}
CONDITION_COLORS = (
    "#6B7280",
    "#009E73",
    "#CC79A7",
    "#4C566A",
    "#E69F00",
)
FIGURE_FILENAMES = (
    "01_checkpoint_comparison.png",
    "02_vggface2_robustness.png",
    "03_transformation_diagnostics.png",
)


def build_report_figures(
    *, input_dir: Path, output_dir: Path, force: bool = False
) -> tuple[Path, ...]:
    """Validate canonical results and build all Experiment 01 report figures."""

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
    _plot_checkpoint_comparison(
        plt=plt,
        percent_formatter=PercentFormatter,
        metrics=results["condition_metrics"],
        stability=results["rank_stability"],
        output_path=output_paths[0],
    )
    _plot_selected_checkpoint_robustness(
        plt=plt,
        percent_formatter=PercentFormatter,
        metrics=results["condition_metrics"],
        stability=results["rank_stability"],
        output_path=output_paths[1],
    )
    _plot_transformation_diagnostics(
        plt=plt,
        percent_formatter=PercentFormatter,
        quality=results["image_quality_summary"],
        output_path=output_paths[2],
    )

    for path in output_paths:
        print(f"[WRITE] {path}")
        print(f"        sha256={sha256_file(path)}")
    return output_paths


def load_verified_results(input_dir: Path) -> dict[str, object]:
    """Load canonical tables after checking their recorded hashes and shape."""

    manifest_path = input_dir / "comparison_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("run_scope") != "full":
        raise ValueError("comparison manifest must describe a complete full run")
    if manifest.get("models") != list(MODELS):
        raise ValueError("comparison manifest contains unexpected models")
    if manifest.get("conditions") != list(CONDITIONS):
        raise ValueError("comparison manifest contains unexpected conditions")
    if manifest.get("probe_images") != 999:
        raise ValueError("comparison manifest must contain 999 probe images")

    for filename, expected_hash in manifest.get("artifacts", {}).items():
        path = input_dir / filename
        if sha256_file(path) != expected_hash:
            raise ValueError(f"artifact hash does not match manifest: {filename}")

    results: dict[str, object] = {"manifest": manifest}
    table_expectations = {
        "condition_metrics": 10,
        "model_summary": 2,
        "rank_stability": 8,
        "probe_rankings": 9990,
        "image_quality_summary": 5,
    }
    for table_name, expected_rows in table_expectations.items():
        rows = _read_csv(input_dir / f"{table_name}.csv")
        if len(rows) != expected_rows:
            raise ValueError(
                f"{table_name}.csv must contain {expected_rows} rows; "
                f"received {len(rows)}"
            )
        results[table_name] = rows

    summaries = results["model_summary"]
    selected = [row for row in summaries if row["selected"] == "True"]
    if len(selected) != 1 or selected[0]["model"] != "vggface2":
        raise ValueError("canonical model selection must identify vggface2")
    return results


def _plot_checkpoint_comparison(
    *, plt, percent_formatter, metrics, stability, output_path: Path
) -> None:
    metrics_lookup = _lookup(metrics)
    stability_lookup = _lookup(stability)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))

    x_all = np.arange(len(CONDITIONS))
    x_degraded = np.arange(len(DEGRADED_CONDITIONS))
    width = 0.36

    for model_index, model in enumerate(MODELS):
        offset = (model_index - 0.5) * width
        top_1 = [
            float(metrics_lookup[(model, condition)]["top_1"])
            for condition in CONDITIONS
        ]
        mrr = [
            float(metrics_lookup[(model, condition)]["mrr"])
            for condition in CONDITIONS
        ]
        changed = [
            float(
                stability_lookup[(model, condition)]["rank_1_changed_rate"]
            )
            for condition in DEGRADED_CONDITIONS
        ]

        bars_top_1 = axes[0].bar(
            x_all + offset,
            top_1,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        bars_mrr = axes[1].bar(
            x_all + offset,
            mrr,
            width,
            color=MODEL_COLORS[model],
        )
        bars_changed = axes[2].bar(
            x_degraded + offset,
            changed,
            width,
            color=MODEL_COLORS[model],
        )
        _label_percentage_bars(axes[0], bars_top_1, top_1)
        _label_percentage_bars(axes[1], bars_mrr, mrr)
        _label_percentage_bars(axes[2], bars_changed, changed)

    _configure_percentage_axis(
        axes[0],
        percent_formatter,
        title="A. Correct first match",
        ylabel="Top-1 accuracy",
        labels=[CONDITION_LABELS[value] for value in CONDITIONS],
        x=x_all,
    )
    _configure_percentage_axis(
        axes[1],
        percent_formatter,
        title="B. Correct identity ranking",
        ylabel="Mean reciprocal rank",
        labels=[CONDITION_LABELS[value] for value in CONDITIONS],
        x=x_all,
    )
    _configure_percentage_axis(
        axes[2],
        percent_formatter,
        title="C. Rank-1 instability (lower is better)",
        ylabel="Rank-1 changed rate",
        labels=[CONDITION_LABELS[value] for value in DEGRADED_CONDITIONS],
        x=x_degraded,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Embedding checkpoint performance across probe conditions",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "VGGFace2 maintained substantially stronger retrieval quality and stability",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "999 probe identities per condition; 2,265 gallery images across 1,000 identities; exact cosine retrieval.",
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.23, wspace=0.27)
    save_experiment_figure(plt, figure, output_path)


def _plot_selected_checkpoint_robustness(
    *, plt, percent_formatter, metrics, stability, output_path: Path
) -> None:
    metrics_lookup = _lookup(metrics)
    stability_lookup = _lookup(stability)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))

    x_all = np.arange(len(CONDITIONS))
    x_degraded = np.arange(len(DEGRADED_CONDITIONS))
    top_fields = (
        ("top_1", "Top-1", "#0072B2", "o"),
        ("top_3", "Top-3", "#009E73", "s"),
        ("top_5", "Top-5", "#CC79A7", "^"),
        ("top_10", "Top-10", "#E69F00", "D"),
    )
    for field, label, color, marker in top_fields:
        values = [
            float(metrics_lookup[("vggface2", condition)][field])
            for condition in CONDITIONS
        ]
        axes[0].plot(
            x_all,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )

    axes[0].set_title("A. Candidate-list accuracy", loc="left", fontweight="bold")
    axes[0].set_ylabel("Retrieval accuracy")
    axes[0].set_xticks(x_all, [CONDITION_LABELS[value] for value in CONDITIONS])
    axes[0].set_ylim(0.68, 0.98)
    axes[0].yaxis.set_major_formatter(percent_formatter(1.0))
    axes[0].legend(frameon=False, ncol=2, loc="lower left")

    retention = [
        float(
            stability_lookup[("vggface2", condition)][
                "clean_correct_retention"
            ]
        )
        for condition in DEGRADED_CONDITIONS
    ]
    retention_bars = axes[1].bar(
        x_degraded,
        retention,
        color="#0072B2",
        width=0.65,
    )
    _label_percentage_bars(axes[1], retention_bars, retention)
    _configure_percentage_axis(
        axes[1],
        percent_formatter,
        title="B. Clean-correct matches retained",
        ylabel="Retention rate",
        labels=[CONDITION_LABELS[value] for value in DEGRADED_CONDITIONS],
        x=x_degraded,
    )

    width = 0.36
    for model_index, model in enumerate(MODELS):
        offset = (model_index - 0.5) * width
        drift = [
            float(
                stability_lookup[(model, condition)][
                    "mean_cosine_embedding_drift"
                ]
            )
            for condition in DEGRADED_CONDITIONS
        ]
        bars = axes[2].bar(
            x_degraded + offset,
            drift,
            width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        _label_decimal_bars(axes[2], bars, drift)

    axes[2].set_title("C. Embedding drift (lower is better)", loc="left", fontweight="bold")
    axes[2].set_ylabel("Mean cosine distance from clean embedding")
    axes[2].set_xticks(
        x_degraded,
        [CONDITION_LABELS[value] for value in DEGRADED_CONDITIONS],
    )
    axes[2].set_ylim(0.0, 0.15)
    axes[2].legend(frameon=False, loc="upper left")

    figure.suptitle(
        "VGGFace2 robustness profile",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "Reduced resolution was the most damaging tested condition, but Top-5 remained above 88%",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "Retention is measured only among the 822 probes VGGFace2 ranked correctly at Top-1 under clean input.",
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.23, wspace=0.27)
    save_experiment_figure(plt, figure, output_path)


def _plot_transformation_diagnostics(
    *, plt, percent_formatter, quality, output_path: Path
) -> None:
    quality_lookup = {row["condition"]: row for row in quality}
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    x = np.arange(len(CONDITIONS))
    labels = [CONDITION_LABELS[value] for value in CONDITIONS]

    panel_fields = (
        (
            "mean_intensity",
            "A. Mean pixel intensity",
            "Mean channel intensity",
        ),
        (
            "mean_zero_fraction",
            "B. Fully black pixels",
            "Pixel share equal to 0",
        ),
        (
            "mean_255_fraction",
            "C. Fully saturated pixels",
            "Pixel share equal to 255",
        ),
    )
    for axis, (field, title, ylabel) in zip(axes, panel_fields):
        values = [float(quality_lookup[condition][field]) for condition in CONDITIONS]
        bars = axis.bar(x, values, color=CONDITION_COLORS, width=0.68)
        _label_percentage_bars(axis, bars, values)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels)
        axis.yaxis.set_major_formatter(percent_formatter(1.0))
        upper = max(values) * 1.28 if max(values) else 0.1
        axis.set_ylim(0.0, min(1.0, upper))

    figure.suptitle(
        "Probe transformation diagnostics",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "Pixel statistics verify that each controlled condition materially changed the probe images",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "Statistics are calculated from transformed RGB images before the final 160 × 160 model resize.",
    )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.23, wspace=0.27)
    save_experiment_figure(plt, figure, output_path)


def _configure_percentage_axis(
    axis, percent_formatter, *, title: str, ylabel: str, labels, x
) -> None:
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(percent_formatter(1.0))


def _label_percentage_bars(axis, bars, values: Sequence[float]) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.018, 0.985),
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=90 if len(values) >= 5 else 0,
        )


def _label_decimal_bars(axis, bars, values: Sequence[float]) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=90,
        )


def _add_figure_note(figure, text: str) -> None:
    figure.text(0.5, 0.055, text, ha="center", fontsize=9, color="#4B5563")


def _lookup(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["model"], row["condition"]): row for row in rows}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verified Experiment 01 report figures."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/outputs/01_embedding_robustness"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures/01_embedding_robustness"),
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
