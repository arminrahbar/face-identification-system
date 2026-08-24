# Experiments

The project will contain exactly three experiment families:

1. `01_embedding_robustness`: compare embedding checkpoints under clean and
   degraded probe conditions.
2. `02_face_preprocessing`: evaluate face detection and crop/fallback
   preprocessing.
3. `03_retrieval_configuration`: evaluate candidate-list length and gallery
   images per identity.

Each experiment will use the same numbered folder name under:

```text
experiments/
├── scripts/
├── outputs/
├── figures/
└── reports/
```

Experiment scripts will be placed inside their own folders. Shared dataset,
embedding-cache, and retrieval-evaluation utilities may live under
`experiments/scripts/shared/` when more than one experiment uses them.

Only verified outputs and publication-quality figures should be tracked. Raw run
logs, caches, temporary artifacts, and private study material remain ignored.
