FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/final-model.pt

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts/final-model.pt ./artifacts/final-model.pt
RUN python -m pip install --no-cache-dir .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "finland_geoai.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
