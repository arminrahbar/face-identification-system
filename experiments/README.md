# Experiments

The project contains exactly three experiment families:

1. `01_embedding_robustness`: compare embedding checkpoints under clean and
   degraded probe conditions.
2. `02_face_preprocessing`: evaluate face detection and crop/fallback
   preprocessing.
3. `03_retrieval_configuration`: evaluate candidate-list length and gallery
   images per identity.

## Status

- Experiment 01: complete; VGGFace2 selected from a verified full-corpus
  checkpoint comparison.
- Experiment 02: complete; the full-corpus audit and paired comparison support
  MTCNN crop/fallback preprocessing for first-choice identification.
- Experiment 03: complete; five returned candidates meet the declared 95%
  coverage target, and two gallery images per identity retain performance
  within one percentage point of the five-image reference.

Each experiment uses the same numbered folder name under:

```text
experiments/
├── scripts/
├── outputs/
├── figures/
└── reports/
```

Experiment scripts are placed inside their own folders. Shared dataset,
embedding-cache, and retrieval-evaluation utilities may live under
`experiments/scripts/shared/` when more than one experiment uses them.

Only verified outputs and publication-quality figures should be tracked. Raw run
logs, caches, temporary artifacts, and local development notes remain ignored.
