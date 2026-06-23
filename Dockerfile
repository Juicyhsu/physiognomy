FROM python:3.10-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker build cache
COPY requirements.txt .

# Install dependencies (including FastAPI, Uvicorn, and LLMs)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port (defaulting to 8000, Zeabur will override this using the PORT env var)
ENV PORT=8000
EXPOSE 8000

# Start command: run uvicorn and bind to the port defined by the system
CMD ["sh", "-c", "uvicorn web_app.main:app --host 0.0.0.0 --port ${PORT}"]
