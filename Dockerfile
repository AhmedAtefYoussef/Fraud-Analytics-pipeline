# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install compilation essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create wheel directory to bundle packages
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final runtime container
FROM python:3.10-slim AS runner

WORKDIR /app

# Add a non-root system user for secure isolation
RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy project source and configs
COPY src/ /app/src/
COPY config/ /app/config/
COPY requirements.txt /app/

# Ensure permissions belong to non-root user
RUN mkdir -p /app/data /app/models /app/logs /app/reports/figures && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

# Expose potential monitoring/MLflow port
EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "src/models/predict_model.py"]
