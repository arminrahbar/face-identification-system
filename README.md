# Face Identification System

A face-identification platform for enrolling gallery images, extracting facial
embeddings, indexing identities, and returning ranked identity candidates for a
probe image.

## Project status

This repository is under active development. The current runtime milestone
implements MTCNN-guided face cropping with a controlled full-image fallback,
VGGFace2-pretrained FaceNet embedding extraction, exact cosine retrieval, and
FAISS-backed HNSW and LSH retrieval over gallery-image embeddings. All three
backends feed a common ranking layer that consolidates image-level matches into
ranked, distinct identities. Service endpoints and experiment implementations
will be added incrementally with tests and validation.

## Intended system boundary

The initial backend component will support two primary workflows:

1. **Enrollment:** preprocess a gallery image, extract its embedding, and associate
   the indexed representation with an identity.
2. **Identification:** preprocess a probe image, extract its embedding, search the
   gallery, consolidate image-level matches into distinct identities, and return a
   ranked candidate list.

The repository name reflects the broader product boundary. Interfaces for
organizational enrollment, security review, persistent audit logging, and
operational monitoring will be added only when they are genuinely implemented.

## Planned repository areas

- `identification_service/`: backend enrollment and identification service.
- `experiments/`: three controlled evaluations with matching scripts, outputs,
  figures, and reports.
- `docs/`: integrated public project report and curated architecture figures.
- `tests/`: unit and service-level verification.

## Data policy

Face images and other external evaluation assets are not distributed with this
repository. Local assets belong under `identification_service/storage/` and are
excluded from version control. See
[`identification_service/storage/README.md`](identification_service/storage/README.md)
for the expected layout.

## License

The source code is available under the MIT License. External datasets and model
weights retain their own terms and are not covered by this repository's license.
