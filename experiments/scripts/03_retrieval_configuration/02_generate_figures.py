"""Generate publication-quality figures from verified Experiment 03 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GREY = "#6B7280"
LIGHT_GREY = "#B8C0CC"
PURPLE = "#7C3AED"

FIGURE_FILENAMES = (
    "01_candidate_list_decision.png",
    "02_gallery_depth_decision.png",
    "03_gallery_population_context.png",
)


def generate_figures(
    *, input_dir: Path, output_dir: Path, force: bool = False
) -> tuple[Path, ...]:
    """Validate canonical results and generate all Experiment 03 figures."""

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

    _apply_style(plt)
    _plot_candidate_list_decision(
        plt=plt,
        percent_formatter=PercentFormatter,
        curve=results["topn_curve"],
        selected=results["topn_selected_values"],
        output_path=output_paths[0],
    )
    _plot_gallery_depth_decision(
        plt=plt,
        percent_formatter=PercentFormatter,
        summary=results["gallery_m_summary"],
        trials=results["gallery_m_trial_metrics"],
        output_path=output_paths[1],
    )
    _plot_gallery_population_context(
        plt=plt,
        distribution=results["gallery_count_distribution"],
        fixed_set=results["fixed_identity_set_summary"][0],
        output_path=output_paths[2],
    )

    for path in output_paths:
        print(f"[WRITE] {path}")
        print(f"        sha256={_sha256_file(path)}")
    return output_paths


def load_verified_results(input_dir: Path) -> dict[str, object]:
    """Load canonical tables after checking hashes, row counts, and decisions."""

    manifest_path = input_dir / "retrieval_configuration_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("retrieval configuration manifest must be complete")
    if manifest.get("experiment") != "03_retrieval_configuration":
        raise ValueError("unexpected experiment identifier")
    if manifest.get("candidate_list", {}).get("selected_candidate_count") != 5:
        raise ValueError("canonical candidate decision must select N=5")
    if (
        manifest.get("gallery_depth", {}).get(
            "selected_gallery_images_per_identity"
        )
        != 2
    ):
        raise ValueError("canonical gallery-depth decision must select m=2")
    baseline = manifest.get("source_pipeline", {}).get(
        "preprocessing_baseline_check", {}
    )
    if baseline.get("status") != "matched":
        raise ValueError("Experiment 02 baseline reproduction must match")

    expected_rows = {
        "topn_curve": 50,
        "topn_selected_values": 7,
        "full_pipeline_rankings_top50": 999,
        "gallery_count_distribution": 10,
        "fixed_identity_set_summary": 1,
        "gallery_m_trial_metrics": 150,
        "gallery_m_summary": 5,
    }
    results: dict[str, object] = {"manifest": manifest}
    artifact_metadata = manifest.get("artifacts", {})
    if set(artifact_metadata) != {
        f"{table_name}.csv" for table_name in expected_rows
    }:
        raise ValueError("manifest contains an unexpected artifact set")

    for table_name, row_count in expected_rows.items():
        filename = f"{table_name}.csv"
        path = input_dir / filename
        metadata = artifact_metadata[filename]
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"artifact hash does not match: {filename}")
        rows = _read_csv(path)
        if len(rows) != row_count or metadata.get("rows") != row_count:
            raise ValueError(
                f"{filename} must contain {row_count} rows; received {len(rows)}"
            )
        results[table_name] = rows

    selected = results["topn_selected_values"]
    selected_defaults = [
        row for row in selected if row["is_recommended_default"] == "True"
    ]
    if (
        len(selected_defaults) != 1
        or int(selected_defaults[0]["candidate_count"]) != 5
    ):
        raise ValueError("selected candidate row must identify N=5")
    trials = results["gallery_m_trial_metrics"]
    trial_pairs = {
        (int(row["trial_index"]), int(row["gallery_images_per_identity"]))
        for row in trials
    }
    if trial_pairs != {(trial, m) for trial in range(30) for m in range(1, 6)}:
        raise ValueError("gallery trial grid must contain 30 trials for m=1..5")
    return results


def _plot_candidate_list_decision(
    *, plt, percent_formatter, curve, selected, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))
    curve_by_n = {int(row["candidate_count"]): row for row in curve}
    selected_by_n = {int(row["candidate_count"]): row for row in selected}

    short_n = np.arange(1, 11)
    short_accuracy = np.asarray(
        [float(curve_by_n[int(n)]["top_n_accuracy"]) for n in short_n]
    )
    axes[0].plot(
        short_n,
        short_accuracy,
        color=GREY,
        marker="o",
        linewidth=2.2,
        markersize=5,
    )
    axes[0].scatter([5], [short_accuracy[4]], color=BLUE, s=75, zorder=3)
    axes[0].axhline(
        0.95,
        color=PURPLE,
        linestyle="--",
        linewidth=1.8,
        label="95% operating target",
    )
    axes[0].annotate(
        "N=5\n95.10%",
        xy=(5, short_accuracy[4]),
        xytext=(5.7, 0.9518),
        arrowprops={"arrowstyle": "-", "color": BLUE},
        color=BLUE,
        fontweight="bold",
    )
    axes[0].set_title("A. Short-list coverage", loc="left", fontweight="bold")
    axes[0].set_xlabel("Returned candidate identities (N)")
    axes[0].set_ylabel("Correct identity in returned list")
    axes[0].set_xticks(short_n)
    axes[0].set_ylim(0.93, 0.957)
    axes[0].yaxis.set_major_formatter(percent_formatter(1.0))
    axes[0].legend(frameon=False, loc="lower right")

    intervals = ((1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50))
    interval_labels = [f"{start}→{stop}" for start, stop in intervals]
    recovered = [
        int(selected_by_n[stop]["correct_probes"])
        - int(selected_by_n[start]["correct_probes"])
        for start, stop in intervals
    ]
    colors = [BLUE if stop == 5 else GREY for _, stop in intervals]
    bars = axes[1].bar(
        np.arange(len(intervals)), recovered, color=colors, width=0.68
    )
    _label_integer_bars(axes[1], bars, recovered)
    axes[1].set_title(
        "B. Additional probes recovered", loc="left", fontweight="bold"
    )
    axes[1].set_xlabel("Candidate-list increase")
    axes[1].set_ylabel("Newly recovered probes")
    axes[1].set_xticks(np.arange(len(intervals)), interval_labels)
    axes[1].set_ylim(0, max(recovered) * 1.24)

    miss_counts = (
        (1, int(selected_by_n[1]["missed_probes"])),
        (3, int(selected_by_n[3]["missed_probes"])),
        (5, int(selected_by_n[5]["missed_probes"])),
        (10, int(selected_by_n[10]["missed_probes"])),
        (50, int(selected_by_n[50]["missed_probes"])),
    )
    miss_labels = [f"N={n}" for n, _ in miss_counts]
    misses = [value for _, value in miss_counts]
    miss_colors = [BLUE if n == 5 else LIGHT_GREY for n, _ in miss_counts]
    bars = axes[2].bar(
        np.arange(len(miss_counts)), misses, color=miss_colors, width=0.68
    )
    _label_integer_bars(axes[2], bars, misses)
    axes[2].set_title(
        "C. Probes still outside the list", loc="left", fontweight="bold"
    )
    axes[2].set_ylabel("Missed probe identities")
    axes[2].set_xticks(np.arange(len(miss_counts)), miss_labels)
    axes[2].set_ylim(0, max(misses) * 1.18)

    figure.suptitle(
        "Five candidates meet the target; longer lists recover little more",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "N=5 includes the correct identity for 950 of 999 probes; N=50 recovers only five additional probes",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "Closed-set evaluation; VGGFace2 embeddings; MTCNN crop/fallback; exact cosine retrieval over distinct identities.",
    )
    figure.subplots_adjust(
        left=0.06, right=0.99, top=0.82, bottom=0.22, wspace=0.30
    )
    _save_figure(plt, figure, output_path)


def _plot_gallery_depth_decision(
    *, plt, percent_formatter, summary, trials, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))
    summary_by_m = {
        int(row["gallery_images_per_identity"]): row for row in summary
    }
    m_values = np.arange(1, 6)
    means = np.asarray(
        [float(summary_by_m[int(m)]["top_1_mean"]) for m in m_values]
    )
    standard_deviations = np.asarray(
        [float(summary_by_m[int(m)]["top_1_std"]) for m in m_values]
    )
    reference = means[-1]
    threshold = reference - 0.01

    axes[0].fill_between(
        [0.7, 5.3],
        threshold,
        reference,
        color=GREEN,
        alpha=0.12,
        label="Within 1 pp of m=5",
    )
    axes[0].errorbar(
        m_values,
        means,
        yerr=standard_deviations,
        color=GREY,
        marker="o",
        linewidth=2.2,
        capsize=4,
        markersize=6,
    )
    axes[0].scatter([2], [means[1]], color=BLUE, s=75, zorder=3)
    axes[0].annotate(
        "m=2\n97.99%",
        xy=(2, means[1]),
        xytext=(2.2, 0.965),
        arrowprops={"arrowstyle": "-", "color": BLUE},
        color=BLUE,
        fontweight="bold",
    )
    axes[0].set_title("A. Mean Top-1 across trials", loc="left", fontweight="bold")
    axes[0].set_xlabel("Gallery images per identity (m)")
    axes[0].set_ylabel("Mean Top-1 accuracy")
    axes[0].set_xticks(m_values)
    axes[0].set_xlim(0.7, 5.3)
    axes[0].set_ylim(0.92, 1.002)
    axes[0].yaxis.set_major_formatter(percent_formatter(1.0))
    axes[0].legend(frameon=False, loc="lower right")

    incremental = np.asarray(
        [float(summary_by_m[int(m)]["incremental_top_1_gain"]) for m in m_values[1:]]
    )
    gain_labels = [f"{m - 1}→{m}" for m in m_values[1:]]
    gain_colors = [BLUE, GREY, GREY, GREY]
    bars = axes[1].bar(
        np.arange(len(incremental)), incremental, color=gain_colors, width=0.68
    )
    _label_percentage_point_bars(axes[1], bars, incremental)
    axes[1].set_title("B. Marginal gain", loc="left", fontweight="bold")
    axes[1].set_xlabel("Gallery-depth increase")
    axes[1].set_ylabel("Incremental Top-1 gain")
    axes[1].set_xticks(np.arange(len(incremental)), gain_labels)
    axes[1].set_ylim(0, max(incremental) * 1.24)
    axes[1].yaxis.set_major_formatter(percent_formatter(1.0))

    trial_values = [
        [
            float(row["top_1"])
            for row in trials
            if int(row["gallery_images_per_identity"]) == int(m)
        ]
        for m in m_values
    ]
    boxes = axes[2].boxplot(
        trial_values,
        positions=m_values,
        widths=0.58,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.8},
        whiskerprops={"color": GREY},
        capprops={"color": GREY},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": GREY},
    )
    for index, patch in enumerate(boxes["boxes"], start=1):
        patch.set_facecolor(BLUE if index == 2 else GREY)
        patch.set_alpha(0.95 if index == 2 else 0.72)
    axes[2].set_title("C. Trial variability", loc="left", fontweight="bold")
    axes[2].set_xlabel("Gallery images per identity (m)")
    axes[2].set_ylabel("Trial Top-1 accuracy")
    axes[2].set_xticks(m_values)
    axes[2].set_ylim(0.90, 1.002)
    axes[2].yaxis.set_major_formatter(percent_formatter(1.0))

    figure.suptitle(
        "Two gallery images capture most of the measured benefit",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "m=2 averaged 97.99% Top-1, within 0.76 percentage points of the m=5 reference",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "Same 123 identities and probes at every m; 30 deterministic trials with nested sampling within each trial.",
    )
    figure.subplots_adjust(
        left=0.06, right=0.99, top=0.82, bottom=0.22, wspace=0.30
    )
    _save_figure(plt, figure, output_path)


def _plot_gallery_population_context(
    *, plt, distribution, fixed_set, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.9))
    by_count = {
        int(row["gallery_images_per_identity"]): row for row in distribution
    }
    image_counts = np.arange(1, 11)
    identity_counts = np.asarray(
        [int(by_count[int(count)]["identity_count"]) for count in image_counts]
    )
    bars = axes[0].bar(image_counts, identity_counts, color=GREY, width=0.68)
    _label_integer_bars(axes[0], bars, identity_counts)
    axes[0].set_title(
        "A. Complete gallery distribution", loc="left", fontweight="bold"
    )
    axes[0].set_xlabel("Available gallery images per identity")
    axes[0].set_ylabel("Gallery identities")
    axes[0].set_xticks(image_counts)
    axes[0].set_ylim(0, max(identity_counts) * 1.16)

    tested_depths = np.arange(1, 6)
    eligible_counts = np.asarray(
        [
            int(by_count[int(depth)]["identities_with_at_least_this_many_images"])
            for depth in tested_depths
        ]
    )
    colors = [BLUE if depth == 5 else GREY for depth in tested_depths]
    bars = axes[1].bar(
        tested_depths, eligible_counts, color=colors, width=0.68
    )
    _label_integer_bars(axes[1], bars, eligible_counts)
    axes[1].set_title(
        "B. Identities eligible at each depth", loc="left", fontweight="bold"
    )
    axes[1].set_xlabel("Required gallery images per identity")
    axes[1].set_ylabel("Eligible identities")
    axes[1].set_xticks(tested_depths)
    axes[1].set_ylim(0, max(eligible_counts) * 1.16)

    fixed_counts = np.arange(5, 11)
    fixed_population = np.asarray(
        [int(by_count[int(count)]["identity_count"]) for count in fixed_counts]
    )
    bars = axes[2].bar(
        fixed_counts, fixed_population, color=BLUE, width=0.68
    )
    _label_integer_bars(axes[2], bars, fixed_population)
    axes[2].set_title(
        "C. Fixed evaluation population", loc="left", fontweight="bold"
    )
    axes[2].set_xlabel("Available gallery images per identity")
    axes[2].set_ylabel("Evaluated identities")
    axes[2].set_xticks(fixed_counts)
    axes[2].set_ylim(0, max(fixed_population) * 1.20)

    eligible = int(fixed_set["eligible_identities"])
    if int(np.sum(fixed_population)) != eligible:
        raise ValueError("fixed population does not match gallery distribution")
    figure.suptitle(
        "Gallery-depth comparison requires a fixed 123-identity population",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "Only identities with at least five gallery images and one probe support a fair m=1 through m=5 comparison",
        ha="center",
        fontsize=11,
        color="#4B5563",
    )
    _add_figure_note(
        figure,
        "The 123-identity gallery-depth subset is controlled across m but is not a full-population performance estimate.",
    )
    figure.subplots_adjust(
        left=0.06, right=0.99, top=0.82, bottom=0.22, wspace=0.30
    )
    _save_figure(plt, figure, output_path)


def _apply_style(plt) -> None:
    plt.rcParams.update(
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


def _label_integer_bars(axis, bars, values: Sequence[int]) -> None:
    maximum = max(values) if len(values) else 1
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + maximum * 0.025,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _label_percentage_point_bars(axis, bars, values: Sequence[float]) -> None:
    maximum = max(values) if len(values) else 0.01
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + maximum * 0.035,
            f"+{value * 100:.2f} pp",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _add_figure_note(figure, text: str) -> None:
    figure.text(0.5, 0.055, text, ha="center", fontsize=9, color="#4B5563")


def _save_figure(plt, figure, output_path: Path) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    figure.savefig(
        temporary_path,
        format="png",
        dpi=180,
        metadata={"Software": "face-identification-system"},
    )
    plt.close(figure)
    temporary_path.replace(output_path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verified Experiment 03 report figures."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/outputs/03_retrieval_configuration"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures/03_retrieval_configuration"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figures(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
