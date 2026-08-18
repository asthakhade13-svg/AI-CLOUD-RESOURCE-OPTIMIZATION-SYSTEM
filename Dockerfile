# Use official slim Python runtime as a base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source modules, serialized artifacts, and backend script
COPY src/ /app/src/
COPY app/ /app/app/
COPY artifacts/ /app/artifacts/
COPY index.html /app/
COPY main.py /app/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
