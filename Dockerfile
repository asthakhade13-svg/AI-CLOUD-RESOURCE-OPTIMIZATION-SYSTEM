# Use official slim Python runtime as a base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies needed by ML algorithms (OpenMP runtime libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application modules, data, artifacts, and frontend
COPY src/ /app/src/
COPY app/ /app/app/
COPY ml_service/ /app/ml_service/
COPY rl/ /app/rl/
COPY simulation/ /app/simulation/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/
COPY index.html /app/
COPY main.py /app/
COPY start.sh /app/

# Sanitize CRLF to LF line endings and set execution permission
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Expose public gateway port and internal ML engine port
EXPOSE 8000
EXPOSE 8050

# Run startup orchestrator
CMD ["/app/start.sh"]
