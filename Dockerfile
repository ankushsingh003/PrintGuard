FROM python:3.10-slim

WORKDIR /app

# Install only necessary system libraries
# libgomp1 is needed for torch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Optimize pip installation: only what's needed for inference
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi uvicorn Pillow numpy python-multipart

COPY . .

# Render provides the PORT env var
ENV PORT=8000
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn API.app:app --host 0.0.0.0 --port $PORT"]
