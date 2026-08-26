FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/app/.cache/torch

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install \
        --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1 \
        torchvision==0.27.1 \
    && python -m pip install \
        --no-cache-dir \
        --requirement /tmp/requirements.txt

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app \
    && install -d -o app -g app /home/app/.cache/torch/checkpoints

COPY --chown=app:app identification_service/ /app/identification_service/
COPY --chown=app:app LICENSE /app/LICENSE

EXPOSE 5000

USER app

CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "--threads=4", "--timeout=180", "--access-logfile=-", "--error-logfile=-", "identification_service.app:app"]
