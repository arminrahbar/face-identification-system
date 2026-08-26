from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RepositoryPackagingTests(unittest.TestCase):
    def test_runtime_dependencies_include_pinned_gunicorn(self) -> None:
        requirements = (
            REPOSITORY_ROOT / "requirements.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("gunicorn==26.1.0", requirements.splitlines())

    def test_container_runs_nonroot_single_worker_service(self) -> None:
        dockerfile = (
            REPOSITORY_ROOT / "identification_service" / "Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn("FROM python:3.11-slim-bookworm", dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cpu", dockerfile)
        self.assertIn("TORCH_HOME=/home/app/.cache/torch", dockerfile)
        self.assertIn("/home/app/.cache/torch/checkpoints", dockerfile)
        self.assertIn("USER app", dockerfile)
        self.assertIn('"--workers=1"', dockerfile)
        self.assertIn('"identification_service.app:app"', dockerfile)
        self.assertNotIn("identification_service/storage", dockerfile)

    def test_container_context_excludes_data_and_analysis(self) -> None:
        ignored = set(
            (REPOSITORY_ROOT / ".dockerignore")
            .read_text(encoding="utf-8")
            .splitlines()
        )

        self.assertIn("identification_service/storage", ignored)
        self.assertIn("experiments", ignored)
        self.assertIn("tests", ignored)
        self.assertIn("scratch", ignored)

    def test_ci_runs_tests_and_builds_the_service_image(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("python -m unittest discover -s tests -v", workflow)
        self.assertIn("python -m pip check", workflow)
        self.assertIn("https://download.pytorch.org/whl/cpu", workflow)
        self.assertIn("identification_service/Dockerfile", workflow)
        self.assertIn("gunicorn --check-config", workflow)


if __name__ == "__main__":
    unittest.main()
