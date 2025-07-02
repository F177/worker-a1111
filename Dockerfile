# Use RunPod's official PyTorch base image
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROOT=/stable-diffusion-webui

WORKDIR /

# Install system dependencies including cuDNN
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    libcudnn9 \
    libcudnn9-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Stable Diffusion WebUI
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /stable-diffusion-webui
WORKDIR /stable-diffusion-webui
RUN git checkout v1.9.3

# --- CORREÇÃO: Instala a extensão Reactor PRIMEIRO ---
RUN cd extensions && \
    git clone https://github.com/Gourieff/sd-webui-reactor-sfw.git

# --- CORREÇÃO: Instala as dependências do Reactor DEPOIS de clonar ---
RUN cd extensions/sd-webui-reactor-sfw && \
    pip install --no-cache-dir -r requirements.txt

# --- CORREÇÃO: Força a reinstalação da biblioteca da GPU ---
RUN pip uninstall -y onnxruntime onnxruntime-gpu && \
    pip install --no-cache-dir \
    -r requirements_versions.txt \
    protobuf==3.20.3 \
    xformers==0.0.24 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.20.0 \
    runpod \
    boto3 \
    opencv-python \
    albumentations==1.3.1

# --- CORREÇÃO: Remove criação de diretórios desnecessários ---
RUN mkdir -p models/Stable-diffusion \
    models/Lora \
    embeddings \
    models/insightface

# Download models
WORKDIR /tmp

# Main models
RUN wget -O /stable-diffusion-webui/models/Stable-diffusion/ultimaterealismo.safetensors \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epicrealismXL_vxviiCrystalclear.safetensors"

RUN wget -O /stable-diffusion-webui/models/Stable-diffusion/sd_xl_refiner_1.0.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"

# LoRA
RUN wget -O /stable-diffusion-webui/models/Lora/epiCRealnessRC1.safetensors \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epiCRealnessRC1.safetensors"

# Negative embeddings
RUN wget -O /stable-diffusion-webui/embeddings/veryBadImageNegative_v1.3.pt \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/verybadimagenegative_v1.3.pt"

RUN wget -O /stable-diffusion-webui/embeddings/FastNegativeV2.pt \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/FastNegativeV2.pt"

# Download Reactor's face swap model
# Download Reactor's face swap model
RUN wget -O /stable-diffusion-webui/models/insightface/inswapper_128.onnx \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/inswapper_128.onnx"

# Pre-cache insightface models
RUN python3 -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0, det_size=(640, 640))"

# Pre-initialize A1111 (downloads additional dependencies)
WORKDIR /stable-diffusion-webui
RUN python3 -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy application files
WORKDIR /
COPY handler.py /handler.py
COPY start.sh /start.sh

# Make start script executable
RUN chmod +x /start.sh

# Set the default command
CMD ["/start.sh"]