FROM python:3.10-slim

WORKDIR /app

# Selective package installation to save time and RAM
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch/torchvision first (largest layers)
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining requirements
RUN pip install --no-cache-dir fastapi uvicorn Pillow numpy python-multipart

# Copy only what is needed for the API and frontend
COPY API ./API
COPY MODEL ./MODEL
COPY DATA_PREPROCESSING ./DATA_PREPROCESSING
COPY TRAINING ./TRAINING
COPY WEB_APP ./WEB_APP
COPY requirements.txt .

ENV PORT=10000
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn API.app:app --host 0.0.0.0 --port $PORT"]
