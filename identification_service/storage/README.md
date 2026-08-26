# Local Storage

This directory is reserved for local face-identification assets. Dataset files
are intentionally excluded from version control.

The externally managed benchmark corpus is not redistributed with this
repository. Anyone reproducing the experiments must obtain compatible assets
from an authorized source and independently confirm the applicable license,
terms, and permitted use.

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
