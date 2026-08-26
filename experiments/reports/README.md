# Experiment reports

The three experiments form one decision sequence for the face-identification
pipeline. Each stage holds the upstream policy constant and resolves the next
open engineering question.

1. [Experiment 01 — Embedding checkpoint robustness](01_embedding_robustness/REPORT.md)
   selects the embedding checkpoint under clean and controlled degraded probe
   conditions.
2. [Experiment 02 — Face preprocessing](02_face_preprocessing/REPORT.md) holds
   the selected checkpoint fixed and evaluates MTCNN-guided cropping with a
   full-image fallback.
3. [Experiment 03 — Retrieval configuration](03_retrieval_configuration/REPORT.md)
   holds the selected representation pipeline fixed and determines the returned
   candidate count and measured gallery-depth recommendation.

Together, the reports explain the controlled comparisons, decision rules,
supporting evidence, limitations, and runtime consequences behind the current
service defaults.
