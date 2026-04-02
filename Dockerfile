# Use a slim Python image to save RAM
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# CRITICAL: Install the CPU-only version of Torch to stay under 512MB RAM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port Render expects
EXPOSE 10000

# Start the app using your specific folder structure
CMD ["uvicorn", "cnn_model.api:app", "--host", "0.0.0.0", "--port", "10000"]