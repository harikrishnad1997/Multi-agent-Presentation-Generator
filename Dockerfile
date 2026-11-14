FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    PRESENTATION_OUTPUT_DIR=/app/out/api

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN mkdir -p /app/out/api && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/out
USER appuser

EXPOSE 8000

CMD ["uvicorn", "presentation_generator.api:app", "--host", "0.0.0.0", "--port", "8000"]
