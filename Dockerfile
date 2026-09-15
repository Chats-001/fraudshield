FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUDSHIELD_ARTIFACT_DIR=/app/artifacts/model \
    FRAUDSHIELD_AUDIT_DB=/app/runtime/predictions.sqlite3

WORKDIR /app
COPY pyproject.toml README.md ./
COPY fraudshield ./fraudshield
COPY artifacts/model ./artifacts/model
RUN pip install --no-cache-dir . && \
    addgroup --system fraudshield && \
    adduser --system --ingroup fraudshield fraudshield && \
    mkdir -p /app/runtime && chown -R fraudshield:fraudshield /app/runtime

USER fraudshield
EXPOSE 8000
CMD ["uvicorn", "fraudshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

