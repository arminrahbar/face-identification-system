"""Evaluate candidate-list length and gallery depth for identity retrieval.

Experiments 01 and 02 fixed the embedding checkpoint and preprocessing policy.
This final experiment therefore reuses their verified VGGFace2 embeddings with
MTCNN crop/fallback preprocessing and changes only retrieval configuration.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]
IntegerVector = NDArray[np.int32]
StringVector = NDArray[np.str_]

MODEL_NAME = "vggface2"
PREPROCESSING_CONDITION = "mtcnn_crop_fallback"
RETRIEVAL_METHOD = "exact_cosine"
RANKING_UNIT = "distinct_identity"
MAX_CANDIDATES = 50
SELECTED_CANDIDATE_COUNTS = (1, 2, 3, 5, 10, 20, 50)
CANDIDATE_ACCURACY_TARGET = 0.95
MIN_GALLERY_IMAGES = 1
MAX_GALLERY_IMAGES = 5
GALLERY_TRIALS = 30
GALLERY_SEED = 20_260_825
GALLERY_TOP_1_TOLERANCE = 0.01

ARTIFACT_NAMES = (
    "topn_curve.csv",
    "topn_selected_values.csv",
    "full_pipeline_rankings_top50.csv",
    "gallery_count_distribution.csv",
    "fixed_identity_set_summary.csv",
    "gallery_m_trial_metrics.csv",
    "gallery_m_summary.csv",
    "retrieval_configuration_run.json",
)


def run_analysis(
    *,
    dataset_manifest: Path,
    preprocessing_run: Path,
    cache_dir: Path,
    output_dir: Path,
    max_candidates: int = MAX_CANDIDATES,
    candidate_target: float = CANDIDATE_ACCURACY_TARGET,
    min_gallery_images: int = MIN_GALLERY_IMAGES,
    max_gallery_images: int = MAX_GALLERY_IMAGES,
    gallery_trials: int = GALLERY_TRIALS,
    gallery_seed: int = GALLERY_SEED,
    gallery_tolerance: float = GALLERY_TOP_1_TOLERANCE,
) -> dict[str, object]:
    """Run both retrieval-configuration analyses from verified embeddings."""

    validate_configuration(
        max_candidates=max_candidates,
        candidate_target=candidate_target,
        min_gallery_images=min_gallery_images,
        max_gallery_images=max_gallery_images,
        gallery_trials=gallery_trials,
        gallery_seed=gallery_seed,
        gallery_tolerance=gallery_tolerance,
    )
    inputs = load_verified_inputs(
        dataset_manifest=dataset_manifest,
        preprocessing_run=preprocessing_run,
        cache_dir=cache_dir,
    )
    gallery = inputs["gallery_cache"]
    probes = inputs["probe_cache"]

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [output_dir / name for name in ARTIFACT_NAMES]
    occupied = [path for path in final_paths if path.exists()]
    if occupied:
        raise FileExistsError(
            "retrieval-configuration outputs already exist: "
            + ", ".join(str(path) for path in occupied)
        )
    staging_dir = output_dir / ".analysis.incomplete"
    if staging_dir.exists():
        raise FileExistsError(
            "incomplete analysis staging directory already exists: "
            f"{staging_dir}"
        )
    staging_dir.mkdir()

    print("RETRIEVAL CONFIGURATION")
    print(f"Model: {MODEL_NAME}")
    print(f"Preprocessing: {PREPROCESSING_CONDITION}")
    print(f"Gallery images: {len(gallery['identities'])}")
    print(f"Gallery identities: {len(set(gallery['identities'].tolist()))}")
    print(f"Probe images: {len(probes['identities'])}")
    print(f"Candidate counts: 1 through {max_candidates}")
    print(
        "Gallery-depth design: "
        f"m={min_gallery_images} through {max_gallery_images}, "
        f"{gallery_trials} nested trials"
    )

    try:
        print("=" * 80)
        print("CANDIDATE-LIST ANALYSIS")
        print("=" * 80)
        full_evaluation = evaluate_identity_retrieval(
            gallery_embeddings=gallery["embeddings"],
            gallery_identities=gallery["identities"],
            probe_embeddings=probes["embeddings"],
            probe_identities=probes["identities"],
            probe_paths=probes["relative_paths"],
            max_candidates=max_candidates,
        )
        topn_curve = build_topn_curve(
            full_evaluation["true_ranks"],
            max_candidates=max_candidates,
        )
        baseline_check = verify_preprocessing_baseline(
            topn_curve=topn_curve,
            mrr=float(full_evaluation["mrr"]),
            expected_metrics=inputs["preprocessing_metrics"],
        )
        selected_candidate_count = select_candidate_count(
            topn_curve,
            target_accuracy=candidate_target,
        )
        selected_topn_rows = build_selected_topn_rows(
            topn_curve,
            selected_counts=SELECTED_CANDIDATE_COUNTS,
            selected_candidate_count=selected_candidate_count,
        )
        selected_topn_result = topn_curve[selected_candidate_count - 1]
        print(
            f"Selected N={selected_candidate_count}: "
            f"accuracy={float(selected_topn_result['top_n_accuracy']):.6f}, "
            f"correct={selected_topn_result['correct_probes']}/"
            f"{selected_topn_result['probe_count']}"
        )
        print(f"Full-pipeline MRR: {full_evaluation['mrr']:.6f}")

        print("=" * 80)
        print("GALLERY-DEPTH ANALYSIS")
        print("=" * 80)
        gallery_distribution = build_gallery_count_distribution(
            gallery["identities"]
        )
        gallery_trials_rows, fixed_set = run_gallery_depth_trials(
            gallery_embeddings=gallery["embeddings"],
            gallery_identities=gallery["identities"],
            probe_embeddings=probes["embeddings"],
            probe_identities=probes["identities"],
            probe_paths=probes["relative_paths"],
            min_gallery_images=min_gallery_images,
            max_gallery_images=max_gallery_images,
            trials=gallery_trials,
            seed=gallery_seed,
        )
        gallery_summary = summarize_gallery_trials(
            gallery_trials_rows,
            min_gallery_images=min_gallery_images,
            max_gallery_images=max_gallery_images,
        )
        selected_gallery_images = select_gallery_depth(
            gallery_summary,
            tolerance=gallery_tolerance,
        )
        selected_gallery_result = next(
            row
            for row in gallery_summary
            if int(row["gallery_images_per_identity"])
            == selected_gallery_images
        )
        print(
            f"Fixed identities: {fixed_set['eligible_identities']} | "
            f"Trials: {gallery_trials}"
        )
        print(
            f"Selected m={selected_gallery_images}: "
            f"mean Top-1={float(selected_gallery_result['top_1_mean']):.6f}"
        )

        artifact_rows: dict[str, Sequence[dict[str, object]]] = {
            "topn_curve.csv": topn_curve,
            "topn_selected_values.csv": selected_topn_rows,
            "full_pipeline_rankings_top50.csv": full_evaluation[
                "ranking_rows"
            ],
            "gallery_count_distribution.csv": gallery_distribution,
            "fixed_identity_set_summary.csv": [fixed_set],
            "gallery_m_trial_metrics.csv": gallery_trials_rows,
            "gallery_m_summary.csv": gallery_summary,
        }
        for name, rows in artifact_rows.items():
            _write_csv(staging_dir / name, rows)

        artifact_metadata = {
            name: _artifact_metadata(staging_dir / name, len(rows))
            for name, rows in artifact_rows.items()
        }
        run_manifest = {
            "status": "complete",
            "experiment": "03_retrieval_configuration",
            "source_pipeline": {
                "model": MODEL_NAME,
                "preprocessing": PREPROCESSING_CONDITION,
                "retrieval": RETRIEVAL_METHOD,
                "ranking_unit": RANKING_UNIT,
                "preprocessing_run_sha256": inputs[
                    "preprocessing_run_sha256"
                ],
                "preprocessing_metrics_sha256": inputs[
                    "preprocessing_metrics_sha256"
                ],
                "preprocessing_baseline_check": baseline_check,
                "dataset_manifest_sha256": inputs[
                    "dataset_manifest_sha256"
                ],
                "selection_fingerprint": inputs["selection_fingerprint"],
                "gallery_cache": inputs["gallery_cache_metadata"],
                "probe_cache": inputs["probe_cache_metadata"],
            },
            "dataset": {
                "gallery_images": len(gallery["identities"]),
                "gallery_identities": len(set(gallery["identities"].tolist())),
                "probe_images": len(probes["identities"]),
                "probe_identities": len(set(probes["identities"].tolist())),
                "closed_set": True,
            },
            "candidate_list": {
                "evaluated_counts": [1, max_candidates],
                "selected_report_counts": list(SELECTED_CANDIDATE_COUNTS),
                "decision_policy": {
                    "rule": "smallest_n_meeting_accuracy_target",
                    "target_accuracy": candidate_target,
                },
                "selected_candidate_count": selected_candidate_count,
                "selected_accuracy": selected_topn_result["top_n_accuracy"],
                "selected_correct_probes": selected_topn_result[
                    "correct_probes"
                ],
                "mrr": full_evaluation["mrr"],
            },
            "gallery_depth": {
                "evaluated_images_per_identity": [
                    min_gallery_images,
                    max_gallery_images,
                ],
                "trials": gallery_trials,
                "base_seed": gallery_seed,
                "sampling": "nested_without_replacement_within_each_trial",
                "eligible_identities": fixed_set["eligible_identities"],
                "decision_policy": {
                    "rule": "smallest_m_within_tolerance_of_max_m_mean_top_1",
                    "top_1_absolute_tolerance": gallery_tolerance,
                    "reference_m": max_gallery_images,
                },
                "selected_gallery_images_per_identity": (
                    selected_gallery_images
                ),
                "selected_mean_top_1": selected_gallery_result["top_1_mean"],
            },
            "dependencies": {"numpy": version("numpy")},
            "artifacts": artifact_metadata,
        }
        _write_json(staging_dir / "retrieval_configuration_run.json", run_manifest)

        for name in ARTIFACT_NAMES:
            (staging_dir / name).replace(output_dir / name)
        staging_dir.rmdir()
    except Exception:
        print(f"[INCOMPLETE] Preserved staging directory: {staging_dir}")
        raise

    for name in ARTIFACT_NAMES:
        print(f"[WRITE] {output_dir / name}")
    return run_manifest


def load_verified_inputs(
    *,
    dataset_manifest: Path,
    preprocessing_run: Path,
    cache_dir: Path,
) -> dict[str, object]:
    """Load and cross-check the selected pipeline's dataset and caches."""

    run = json.loads(preprocessing_run.read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("run_scope") != "full":
        raise ValueError("Experiment 02 comparison must be complete and full")
    if run.get("model") != MODEL_NAME:
        raise ValueError("Experiment 02 did not use the expected embedding model")
    if run.get("result", {}).get("recommendation") != (
        "adopt_mtcnn_crop_fallback"
    ):
        raise ValueError("Experiment 02 did not adopt MTCNN crop/fallback")
    pipeline = run.get("pipeline", {})
    if pipeline.get("retrieval") != RETRIEVAL_METHOD:
        raise ValueError("Experiment 02 retrieval method does not match")
    if pipeline.get("ranking_unit") != RANKING_UNIT:
        raise ValueError("Experiment 02 ranking unit does not match")

    manifest_hash = _sha256_file(dataset_manifest)
    dataset = run.get("dataset", {})
    if manifest_hash != dataset.get("manifest_sha256"):
        raise ValueError("dataset manifest hash does not match Experiment 02")
    rows = _read_csv(dataset_manifest)
    expected_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("gallery", "probe")
    }
    if not all(expected_rows.values()):
        raise ValueError("dataset manifest must contain gallery and probe rows")

    fingerprint = str(dataset.get("selection_fingerprint", ""))
    if len(fingerprint) != 64:
        raise ValueError("Experiment 02 selection fingerprint is invalid")
    preprocessing_hash = str(run.get("preprocessing_audit", {}).get("sha256", ""))
    metrics_path = preprocessing_run.parent / "condition_metrics.csv"
    expected_metrics_hash = str(
        run.get("artifacts", {})
        .get("condition_metrics.csv", {})
        .get("sha256", "")
    )
    if _sha256_file(metrics_path) != expected_metrics_hash:
        raise ValueError("Experiment 02 condition metrics hash does not match")
    metric_rows = _read_csv(metrics_path)
    selected_metric_rows = [
        row
        for row in metric_rows
        if row.get("model") == MODEL_NAME
        and row.get("condition") == PREPROCESSING_CONDITION
    ]
    if len(selected_metric_rows) != 1:
        raise ValueError("selected Experiment 02 metric row is ambiguous")
    caches: dict[str, dict[str, NDArray]] = {}
    cache_metadata: dict[str, dict[str, object]] = {}
    for split in ("gallery", "probe"):
        path = cache_dir / (
            f"{fingerprint[:12]}__{MODEL_NAME}__{split}__"
            f"{PREPROCESSING_CONDITION}.npz"
        )
        cache = load_embedding_cache(
            path=path,
            expected_rows=expected_rows[split],
            expected_split=split,
            expected_fingerprint=fingerprint,
            expected_preprocessing_hash=preprocessing_hash,
        )
        caches[split] = cache
        cache_metadata[split] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "configuration": json.loads(
                str(cache["configuration_json"].item())
            ),
        }

    gallery_identities = set(caches["gallery"]["identities"].tolist())
    missing = sorted(set(caches["probe"]["identities"].tolist()) - gallery_identities)
    if missing:
        raise ValueError(f"probe identities missing from gallery: {missing[:5]}")

    return {
        "gallery_cache": caches["gallery"],
        "probe_cache": caches["probe"],
        "gallery_cache_metadata": cache_metadata["gallery"],
        "probe_cache_metadata": cache_metadata["probe"],
        "dataset_manifest_sha256": manifest_hash,
        "preprocessing_run_sha256": _sha256_file(preprocessing_run),
        "preprocessing_metrics": selected_metric_rows[0],
        "preprocessing_metrics_sha256": expected_metrics_hash,
        "selection_fingerprint": fingerprint,
    }


def load_embedding_cache(
    *,
    path: Path,
    expected_rows: Sequence[dict[str, str]],
    expected_split: str,
    expected_fingerprint: str,
    expected_preprocessing_hash: str,
) -> dict[str, NDArray]:
    """Load one Experiment 02 cache after schema and provenance checks."""

    if not path.is_file():
        raise FileNotFoundError(f"required embedding cache not found: {path}")
    required = {
        "configuration_json",
        "embeddings",
        "identities",
        "relative_paths",
    }
    with np.load(path, allow_pickle=False) as cached:
        if set(cached.files) != required:
            raise ValueError(f"unexpected cache schema: {path.name}")
        payload = {name: cached[name].copy() for name in required}

    configuration = json.loads(str(payload["configuration_json"].item()))
    expected_configuration = {
        "selection_fingerprint": expected_fingerprint,
        "preprocessing_audit_sha256": expected_preprocessing_hash,
        "model": MODEL_NAME,
        "condition": PREPROCESSING_CONDITION,
        "split": expected_split,
        "image_size": 160,
        "standardization": "fixed_127_5_div_128",
    }
    for key, value in expected_configuration.items():
        if configuration.get(key) != value:
            raise ValueError(
                f"cache configuration mismatch for {key}: {path.name}"
            )

    expected_identities = np.asarray(
        [row["identity"] for row in expected_rows], dtype=np.str_
    )
    expected_paths = np.asarray(
        [row["relative_path"] for row in expected_rows], dtype=np.str_
    )
    if not np.array_equal(payload["identities"], expected_identities):
        raise ValueError(f"cache identity ordering mismatch: {path.name}")
    if not np.array_equal(payload["relative_paths"], expected_paths):
        raise ValueError(f"cache path ordering mismatch: {path.name}")
    embeddings = payload["embeddings"]
    if embeddings.shape != (len(expected_rows), 512):
        raise ValueError(f"cache embedding shape mismatch: {path.name}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"cache contains non-finite embeddings: {path.name}")
    return payload


def evaluate_identity_retrieval(
    *,
    gallery_embeddings: FloatMatrix,
    gallery_identities: StringVector,
    probe_embeddings: FloatMatrix,
    probe_identities: StringVector,
    probe_paths: StringVector,
    max_candidates: int,
) -> dict[str, object]:
    """Rank distinct identities using each identity's best image similarity."""

    gallery = _normalize_rows(gallery_embeddings)
    probes = _normalize_rows(probe_embeddings)
    if len(gallery) != len(gallery_identities):
        raise ValueError("gallery embeddings and identities must have equal length")
    if not (
        len(probes) == len(probe_identities) == len(probe_paths)
    ):
        raise ValueError("probe embeddings, identities, and paths must align")
    identity_count = len(set(gallery_identities.tolist()))
    if max_candidates > identity_count:
        raise ValueError("max_candidates exceeds available gallery identities")

    similarities = probes @ gallery.T
    image_orders = np.argsort(-similarities, axis=1, kind="stable")
    true_ranks: list[int] = []
    rows: list[dict[str, object]] = []

    for probe_index, image_order in enumerate(image_orders):
        true_identity = str(probe_identities[probe_index])
        seen: set[str] = set()
        top_identities: list[str] = []
        true_rank: int | None = None
        true_similarity: float | None = None
        rank_1_similarity: float | None = None

        for raw_position in image_order:
            position = int(raw_position)
            identity = str(gallery_identities[position])
            if identity in seen:
                continue
            seen.add(identity)
            similarity = float(similarities[probe_index, position])
            if rank_1_similarity is None:
                rank_1_similarity = similarity
            if len(top_identities) < max_candidates:
                top_identities.append(identity)
            if identity == true_identity:
                true_rank = len(seen)
                true_similarity = similarity
            if true_rank is not None and len(top_identities) == max_candidates:
                break

        if true_rank is None or true_similarity is None or rank_1_similarity is None:
            raise ValueError(
                f"true identity is absent from gallery ranking: {true_identity}"
            )
        true_ranks.append(true_rank)
        row: dict[str, object] = {
            "probe_relative_path": str(probe_paths[probe_index]),
            "true_identity": true_identity,
            "true_identity_rank": true_rank,
            "reciprocal_rank": 1.0 / true_rank,
            "rank_1_similarity": rank_1_similarity,
            "true_identity_similarity": true_similarity,
        }
        for rank in range(1, max_candidates + 1):
            row[f"rank_{rank}"] = top_identities[rank - 1]
        rows.append(row)

    ranks = np.asarray(true_ranks, dtype=np.int32)
    return {
        "true_ranks": ranks,
        "mrr": float(np.mean(1.0 / ranks)),
        "ranking_rows": rows,
    }


def build_topn_curve(
    true_ranks: IntegerVector, *, max_candidates: int
) -> list[dict[str, object]]:
    """Build cumulative candidate coverage and marginal-gain rows."""

    ranks = np.asarray(true_ranks, dtype=np.int32)
    if ranks.ndim != 1 or len(ranks) == 0 or np.any(ranks <= 0):
        raise ValueError("true ranks must be a nonempty positive vector")
    rows: list[dict[str, object]] = []
    previous_correct = 0
    for candidate_count in range(1, max_candidates + 1):
        correct = int(np.sum(ranks <= candidate_count))
        newly_recovered = correct - previous_correct
        rows.append(
            {
                "candidate_count": candidate_count,
                "probe_count": len(ranks),
                "correct_probes": correct,
                "missed_probes": len(ranks) - correct,
                "top_n_accuracy": correct / len(ranks),
                "newly_recovered_probes": newly_recovered,
                "accuracy_gain_from_previous_n": newly_recovered / len(ranks),
                "candidate_review_slots": candidate_count * len(ranks),
            }
        )
        previous_correct = correct
    return rows


def verify_preprocessing_baseline(
    *,
    topn_curve: Sequence[dict[str, object]],
    mrr: float,
    expected_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Require exact reproduction of Experiment 02 retrieval metrics."""

    curve = {int(row["candidate_count"]): row for row in topn_curve}
    differences = {
        f"top_{count}": float(curve[count]["top_n_accuracy"])
        - float(expected_metrics[f"top_{count}"])
        for count in (1, 3, 5, 10)
    }
    differences["mrr"] = mrr - float(expected_metrics["mrr"])
    if any(abs(value) > 1e-12 for value in differences.values()):
        raise ValueError(
            "Experiment 03 does not reproduce the selected Experiment 02 "
            f"baseline: {differences}"
        )
    return {"status": "matched", "metric_differences": differences}


def select_candidate_count(
    topn_curve: Sequence[dict[str, object]], *, target_accuracy: float
) -> int:
    """Return the shortest candidate list meeting the declared target."""

    for row in topn_curve:
        if float(row["top_n_accuracy"]) >= target_accuracy:
            return int(row["candidate_count"])
    return int(topn_curve[-1]["candidate_count"])


def build_selected_topn_rows(
    topn_curve: Sequence[dict[str, object]],
    *,
    selected_counts: Sequence[int],
    selected_candidate_count: int,
) -> list[dict[str, object]]:
    """Return practical candidate counts with interval-level trade-offs."""

    by_count = {int(row["candidate_count"]): row for row in topn_curve}
    available_counts = [count for count in selected_counts if count in by_count]
    if selected_candidate_count not in available_counts:
        available_counts.append(selected_candidate_count)
        available_counts.sort()
    rows: list[dict[str, object]] = []
    previous_count = 0
    previous_correct = 0
    for count in available_counts:
        source = by_count[count]
        correct = int(source["correct_probes"])
        rows.append(
            {
                **source,
                "candidate_increase_from_previous_selected_n": (
                    count - previous_count
                ),
                "recovered_probes_from_previous_selected_n": (
                    correct - previous_correct
                ),
                "accuracy_gain_from_previous_selected_n": (
                    (correct - previous_correct) / int(source["probe_count"])
                ),
                "is_recommended_default": count == selected_candidate_count,
            }
        )
        previous_count = count
        previous_correct = correct
    return rows


def build_gallery_count_distribution(
    gallery_identities: StringVector,
) -> list[dict[str, object]]:
    """Summarize how many enrolled images are available per identity."""

    counts = Counter(str(value) for value in gallery_identities)
    frequency = Counter(counts.values())
    total_identities = len(counts)
    rows: list[dict[str, object]] = []
    cumulative = 0
    for image_count in sorted(frequency):
        identity_count = frequency[image_count]
        cumulative += identity_count
        rows.append(
            {
                "gallery_images_per_identity": image_count,
                "identity_count": identity_count,
                "identity_share": identity_count / total_identities,
                "cumulative_identity_count": cumulative,
                "identities_with_at_least_this_many_images": sum(
                    count >= image_count for count in counts.values()
                ),
            }
        )
    return rows


def run_gallery_depth_trials(
    *,
    gallery_embeddings: FloatMatrix,
    gallery_identities: StringVector,
    probe_embeddings: FloatMatrix,
    probe_identities: StringVector,
    probe_paths: StringVector,
    min_gallery_images: int,
    max_gallery_images: int,
    trials: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate nested gallery samples on one fixed identity population."""

    gallery_positions: dict[str, list[int]] = defaultdict(list)
    probe_positions: dict[str, list[int]] = defaultdict(list)
    for position, raw_identity in enumerate(gallery_identities):
        gallery_positions[str(raw_identity)].append(position)
    for position, raw_identity in enumerate(probe_identities):
        probe_positions[str(raw_identity)].append(position)

    eligible = sorted(
        identity
        for identity, positions in gallery_positions.items()
        if len(positions) >= max_gallery_images
        and len(probe_positions.get(identity, [])) == 1
    )
    if not eligible:
        raise ValueError("no identities satisfy the fixed gallery-depth design")
    selected_probe_positions = np.asarray(
        [probe_positions[identity][0] for identity in eligible], dtype=np.int32
    )
    selected_probe_embeddings = probe_embeddings[selected_probe_positions]
    selected_probe_identities = probe_identities[selected_probe_positions]
    selected_probe_paths = probe_paths[selected_probe_positions]

    rows: list[dict[str, object]] = []
    for trial_index in range(trials):
        trial_seed = seed + trial_index
        samples = build_nested_gallery_samples(
            gallery_positions=gallery_positions,
            identities=eligible,
            max_gallery_images=max_gallery_images,
            rng=np.random.default_rng(trial_seed),
        )
        for image_count in range(min_gallery_images, max_gallery_images + 1):
            selected_gallery_positions = samples[image_count]
            evaluation = evaluate_identity_retrieval(
                gallery_embeddings=gallery_embeddings[selected_gallery_positions],
                gallery_identities=gallery_identities[selected_gallery_positions],
                probe_embeddings=selected_probe_embeddings,
                probe_identities=selected_probe_identities,
                probe_paths=selected_probe_paths,
                max_candidates=10,
            )
            ranks = evaluation["true_ranks"]
            rows.append(
                {
                    "trial_index": trial_index,
                    "seed": trial_seed,
                    "gallery_images_per_identity": image_count,
                    "identity_count": len(eligible),
                    "probe_count": len(eligible),
                    "gallery_image_count": len(selected_gallery_positions),
                    "top_1": float(np.mean(ranks <= 1)),
                    "top_3": float(np.mean(ranks <= 3)),
                    "top_5": float(np.mean(ranks <= 5)),
                    "top_10": float(np.mean(ranks <= 10)),
                    "mrr": float(evaluation["mrr"]),
                    "mean_true_identity_rank": float(np.mean(ranks)),
                    "median_true_identity_rank": float(np.median(ranks)),
                }
            )

    available_counts = [len(gallery_positions[identity]) for identity in eligible]
    fixed_set = {
        "eligibility_rule": (
            f"at_least_{max_gallery_images}_gallery_images_and_exactly_one_probe"
        ),
        "eligible_identities": len(eligible),
        "fixed_probes": len(eligible),
        "minimum_available_gallery_images": min(available_counts),
        "median_available_gallery_images": float(median(available_counts)),
        "mean_available_gallery_images": float(np.mean(available_counts)),
        "maximum_available_gallery_images": max(available_counts),
        "tested_minimum_m": min_gallery_images,
        "tested_maximum_m": max_gallery_images,
        "trials": trials,
        "sampling": "nested_without_replacement_within_each_trial",
    }
    return rows, fixed_set


def build_nested_gallery_samples(
    *,
    gallery_positions: Mapping[str, Sequence[int]],
    identities: Sequence[str],
    max_gallery_images: int,
    rng: np.random.Generator,
) -> dict[int, IntegerVector]:
    """Create nested per-identity gallery samples for one random trial."""

    shuffled: dict[str, IntegerVector] = {}
    for identity in identities:
        positions = np.asarray(gallery_positions[identity], dtype=np.int32)
        if len(positions) < max_gallery_images:
            raise ValueError(f"identity lacks required gallery images: {identity}")
        shuffled[identity] = rng.permutation(positions)

    samples: dict[int, IntegerVector] = {}
    for image_count in range(1, max_gallery_images + 1):
        samples[image_count] = np.asarray(
            [
                int(position)
                for identity in identities
                for position in shuffled[identity][:image_count]
            ],
            dtype=np.int32,
        )
    return samples


def summarize_gallery_trials(
    trial_rows: Sequence[dict[str, object]],
    *,
    min_gallery_images: int,
    max_gallery_images: int,
) -> list[dict[str, object]]:
    """Aggregate trial distributions and incremental gains for each m."""

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped[int(row["gallery_images_per_identity"])].append(row)
    expected = set(range(min_gallery_images, max_gallery_images + 1))
    if set(grouped) != expected:
        raise ValueError("gallery trial rows do not cover every configured m")

    metrics = (
        "top_1",
        "top_3",
        "top_5",
        "top_10",
        "mrr",
        "mean_true_identity_rank",
        "median_true_identity_rank",
    )
    rows: list[dict[str, object]] = []
    previous_top_1: float | None = None
    previous_mrr: float | None = None
    for image_count in sorted(grouped):
        trials = grouped[image_count]
        row: dict[str, object] = {
            "gallery_images_per_identity": image_count,
            "trials": len(trials),
            "identity_count": int(trials[0]["identity_count"]),
            "gallery_image_count": int(trials[0]["gallery_image_count"]),
        }
        for metric in metrics:
            values = np.asarray(
                [float(trial[metric]) for trial in trials], dtype=np.float64
            )
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        top_1_mean = float(row["top_1_mean"])
        mrr_mean = float(row["mrr_mean"])
        row["incremental_top_1_gain"] = (
            "" if previous_top_1 is None else top_1_mean - previous_top_1
        )
        row["incremental_mrr_gain"] = (
            "" if previous_mrr is None else mrr_mean - previous_mrr
        )
        previous_top_1 = top_1_mean
        previous_mrr = mrr_mean
        rows.append(row)
    return rows


def select_gallery_depth(
    summary_rows: Sequence[dict[str, object]], *, tolerance: float
) -> int:
    """Select the smallest m within tolerance of the max-m Top-1 mean."""

    if not summary_rows:
        raise ValueError("gallery summary cannot be empty")
    reference = float(summary_rows[-1]["top_1_mean"])
    threshold = reference - tolerance
    for row in summary_rows:
        if float(row["top_1_mean"]) >= threshold - 1e-12:
            return int(row["gallery_images_per_identity"])
    return int(summary_rows[-1]["gallery_images_per_identity"])


def validate_configuration(
    *,
    max_candidates: int,
    candidate_target: float,
    min_gallery_images: int,
    max_gallery_images: int,
    gallery_trials: int,
    gallery_seed: int,
    gallery_tolerance: float,
) -> None:
    """Reject invalid or methodologically inconsistent run parameters."""

    integer_values = {
        "max_candidates": max_candidates,
        "min_gallery_images": min_gallery_images,
        "max_gallery_images": max_gallery_images,
        "gallery_trials": gallery_trials,
        "gallery_seed": gallery_seed,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_candidates < max(SELECTED_CANDIDATE_COUNTS):
        raise ValueError("max_candidates must cover all selected report counts")
    if min_gallery_images != 1:
        raise ValueError("gallery-depth analysis must start at one image")
    if max_gallery_images < min_gallery_images:
        raise ValueError("maximum gallery depth cannot be below the minimum")
    if not 0 < candidate_target <= 1:
        raise ValueError("candidate_target must be in (0, 1]")
    if not 0 <= gallery_tolerance < 1:
        raise ValueError("gallery_tolerance must be in [0, 1)")


def _normalize_rows(values: FloatMatrix) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be two-dimensional and finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding matrix cannot contain zero vectors")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty artifact: {path.name}")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _artifact_metadata(path: Path, row_count: int) -> dict[str, object]:
    return {
        "rows": row_count,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-list length and gallery images per identity "
            "using the selected face-identification pipeline."
        )
    )
    experiment_01 = Path("experiments/outputs/01_embedding_robustness")
    experiment_02 = Path("experiments/outputs/02_face_preprocessing")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=experiment_01 / "dataset_manifest.csv",
    )
    parser.add_argument(
        "--preprocessing-run",
        type=Path,
        default=experiment_02 / "comparison_run.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/cache/02_face_preprocessing/embeddings"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/03_retrieval_configuration"),
    )
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument(
        "--candidate-target", type=float, default=CANDIDATE_ACCURACY_TARGET
    )
    parser.add_argument("--gallery-trials", type=int, default=GALLERY_TRIALS)
    parser.add_argument("--gallery-seed", type=int, default=GALLERY_SEED)
    parser.add_argument(
        "--gallery-tolerance", type=float, default=GALLERY_TOP_1_TOLERANCE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        dataset_manifest=args.dataset_manifest,
        preprocessing_run=args.preprocessing_run,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        candidate_target=args.candidate_target,
        min_gallery_images=MIN_GALLERY_IMAGES,
        max_gallery_images=MAX_GALLERY_IMAGES,
        gallery_trials=args.gallery_trials,
        gallery_seed=args.gallery_seed,
        gallery_tolerance=args.gallery_tolerance,
    )


if __name__ == "__main__":
    main()
