FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi uvicorn Pillow numpy \
    opencv-python-headless python-multipart scikit-learn \
    seaborn matplotlib pandas

COPY . .

EXPOSE 8000

CMD ["uvicorn", "API.app:app", "--host", "0.0.0.0", "--port", "8000"]
