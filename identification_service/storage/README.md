# Local Storage

This directory is reserved for local face-identification assets. Dataset files
are intentionally excluded from version control.

Expected layout:

```text
storage/
├── multi_image_gallery/
│   └── <identity>/
│       └── <gallery-image>.jpg
└── probe/
    └── <identity>/
        └── <probe-image>.jpg
```

Dataset loaders must accept supported image extensions only and ignore hidden
files and macOS AppleDouble files whose names begin with `._`.

Do not commit face images, cached embeddings, generated indexes, or third-party
model weights.
