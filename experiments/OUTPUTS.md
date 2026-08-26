# Experiment output contract

This directory publishes the portable aggregate evidence used to verify the
three engineering decisions. Each numbered output folder is owned by the
experiment with the same number under `experiments/scripts/`,
`experiments/figures/`, and `experiments/reports/`.

## Retained canonical evidence

| Output stage | Published files | Purpose |
|---|---|---|
| `01_embedding_robustness` | `condition_metrics.csv`, `dataset_summary.csv`, `gallery_image_count_distribution.csv`, `image_quality_summary.csv`, `model_summary.csv`, `rank_stability.csv` | Defines the evaluation population and supports the checkpoint decision without publishing identity-level rankings |
| `02_face_preprocessing` | `condition_metrics.csv`, `detection_summary.csv`, `metric_deltas.csv`, `rank_change_summary.csv`, `top1_outcome_summary.csv` | Supports the paired preprocessing decision with population-level detection and retrieval measures |
| `03_retrieval_configuration` | `fixed_identity_set_summary.csv`, `gallery_count_distribution.csv`, `gallery_m_summary.csv`, `gallery_m_trial_metrics.csv`, `topn_curve.csv`, `topn_selected_values.csv` | Supports the candidate-list and gallery-depth decisions without publishing named candidate results |

The published files are deterministic aggregate result tables. They contain no
face images, model weights, embedding vectors, named identity rows, candidate
rankings, service requests, or machine-specific absolute paths. These folders
must not be used as a destination for production traces or user-provided
identity data.

## Local-only evidence

The following artifacts are reproducible but are intentionally excluded from
version control:

- `01_embedding_robustness/comparison_run.json`;
- `01_embedding_robustness/dataset_audit.json`;
- `01_embedding_robustness/dataset_manifest.csv`;
- `01_embedding_robustness/probe_rankings.csv`;
- `02_face_preprocessing/comparison_run.json`;
- `02_face_preprocessing/detection_records.csv`;
- `02_face_preprocessing/preprocessing_audit.json`;
- `02_face_preprocessing/probe_case_analysis.csv`;
- `02_face_preprocessing/probe_rankings.csv`;
- `03_retrieval_configuration/full_pipeline_rankings_top50.csv`;
- `03_retrieval_configuration/retrieval_configuration_run.json`;
- source face images and other external evaluation assets;
- preprocessed face crops;
- embedding matrices and model caches;
- temporary or incomplete run directories;
- execution logs and exploratory diagnostics;
- mutable service indexes and enrolled production data.

The excluded result files contain corpus identity labels, image or crop
references, content hashes, complete candidate rankings, or run-manifest
fingerprints. They remain available to the local analysis pipeline but are not
part of the public evidence package.

External image datasets and pretrained model files retain their own licenses
and terms. The repository's MIT License covers the project source code, not
those external assets.

## Reproduction ownership

Each numbered runner writes both public aggregate evidence and local-only
corpus-linked evidence to the matching output stage. A complete reproduction
requires the external gallery and probe layout documented in
`identification_service/storage/README.md` and the dependencies declared in
`requirements.txt`. The public reports and retained aggregate tables preserve
the decision evidence without redistributing the benchmark corpus or its
identity-level retrieval results.
