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
ranked, distinct identities. The Flask service exposes gallery enrollment and
probe-identification endpoints. Experiment 01 selected the VGGFace2 checkpoint
after a complete clean-and-degraded comparison over 999 probe identities; see
the [checkpoint report](experiments/reports/01_embedding_robustness/REPORT.md).
Experiment 02 then validated MTCNN crop/fallback preprocessing with 99.97% face
detection coverage and an 11.01-percentage-point Top-1 improvement; see the
[preprocessing report](experiments/reports/02_face_preprocessing/REPORT.md).
Experiment 03 measured retrieval configuration over the selected pipeline. Its
canonical analysis selected five returned candidates as the shortest list
meeting a 95% coverage target and two gallery images per identity as the
smallest tested depth within one percentage point of the five-image reference;
see the [retrieval configuration report](experiments/reports/03_retrieval_configuration/REPORT.md).

The integrated [engineering report](docs/PROJECT_REPORT.md) connects the
runtime architecture, experimental decisions, packaging, limitations, and next
engineering priorities.

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

## HTTP API

- `POST /add`: accepts an `image` file and `identity` value, then preprocesses,
  embeds, and adds the gallery image to the in-memory index. Multiple images may
  be enrolled for the same identity; the measured operating recommendation is
  two gallery images per identity when enrollment data is available.
- `POST /identify`: accepts a `probe` image and optional positive integer `k`,
  then returns up to `k` ranked, distinct identity candidates with match scores.
  When `k` is omitted, the measured default is five candidates.

The current service keeps gallery embeddings in process memory. Restarting the
service clears enrolled state; durable index persistence is not implemented yet.

## Local execution

Create a Python 3.11 environment, install the pinned dependencies, and start the
development server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --requirement requirements.txt
python -m identification_service.app
```

The first request may download the selected pretrained model into the current
user's cache. The service listens on `http://127.0.0.1:5000`.

Enroll a gallery image:

```bash
curl --request POST http://127.0.0.1:5000/add \
  --form "identity=example-person" \
  --form "image=@gallery-image.jpg"
```

Identify a probe using the measured five-candidate default:

```bash
curl --request POST http://127.0.0.1:5000/identify \
  --form "probe=@probe-image.jpg"
```

## Container execution

Build and run the service with a persistent cache for pretrained model files:

```bash
docker build \
  --file identification_service/Dockerfile \
  --tag face-identification-system .

docker run --rm \
  --publish 5000:5000 \
  --mount type=volume,src=face-model-cache,dst=/home/app/.cache \
  face-identification-system
```

The image runs as a non-root user. It intentionally uses one Gunicorn worker
because enrolled gallery state is currently held in process memory; four worker
threads share that service instance. Multiple processes require a durable shared
index, which is outside the current implementation. The container installs the
CPU-only PyTorch and torchvision wheels because the packaged service does not
require a CUDA runtime.

## Verification

Run the complete public test suite with:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same test suite, checks installed dependency
consistency, builds the service image, and validates its non-root runtime,
dependencies, routes, and Gunicorn configuration.

## Repository structure

- `identification_service/`: backend enrollment and identification service.
- `experiments/`: three controlled evaluations with matching scripts, outputs,
  figures, and reports.
- `docs/`: integrated engineering report and its curated publication figures.
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
