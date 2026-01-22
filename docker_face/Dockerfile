FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 wget git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for face detection
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3"]
