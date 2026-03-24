# Use Python base image
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for image processing and LaTeX rendering
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip --no-cache-dir

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Установить Python и зависимости
RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    apt-get install -y libsm6 libxext6 libxrender-dev git wget curl

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
COPY . .

# Create directories for checkpoints and outputs
RUN mkdir -p checkpoints report/images

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["python", "-m", "src.train", "--config", "configs/train_config.yaml"]
