# Start with Python base image optimized for M1/M2
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/data

# Command to run the training script
CMD ["python", "train.py"]