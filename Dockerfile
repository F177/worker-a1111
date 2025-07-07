# ==============================================================================
# === BUILD STAGE ===
# ==============================================================================
# Use a "builder" stage to download all assets and install build-time tools.
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV ROOT=/stable-diffusion-webui
WORKDIR /

# Install only build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone repositories in a single layer
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /stable-diffusion-webui && \
    cd /stable-diffusion-webui && \
    git checkout v1.9.3 && \
    cd extensions && \
    git clone https://codeberg.org/Gourieff/sd-webui-reactor.git

# Create all model directories
RUN mkdir -p \
    /stable-diffusion-webui/models/Stable-diffusion \
    /stable-diffusion-webui/models/Lora \
    /stable-diffusion-webui/embeddings \
    /stable-diffusion-webui/models/insightface \
    /stable-diffusion-webui/models/Codeformer \
    /stable-diffusion-webui/models/GFPGAN \
    /stable-diffusion-webui/models/VAE

# Download all models and dependencies in a single layer to optimize the build.
# This is the most important optimization for reducing cold start times.
RUN \
    # Main Models
    wget -qO /stable-diffusion-webui/models/Stable-diffusion/ultimaterealismo.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/superrealismo2.safetensors" && \
    wget -qO /stable-diffusion-webui/models/Stable-diffusion/sd_xl_refiner_1.0.safetensors "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors" && \
    \
    # LoRA and Embeddings
    wget -qO /stable-diffusion-webui/models/Lora/epiCRealnessRC1.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epiCRealnessRC1.safetensors" && \
    wget -qO /stable-diffusion-webui/embeddings/veryBadImageNegative_v1.3.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/verybadimagenegative_v1.3.pt" && \
    wget -qO /stable-diffusion-webui/embeddings/FastNegativeV2.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/FastNegativeV2.pt" && \
    \
    # ReActor Faceswap Model
    wget -qO /stable-diffusion-webui/models/insightface/inswapper_128.onnx "https://huggingface.co/Fabricioi/modelorealista/resolve/main/inswapper_128.onnx" && \
    \
    # Official VAE for SDXL (prevents download of VAE-approx [cite: 242, 531])
    wget -qO /stable-diffusion-webui/models/VAE/sdxl_vae.safetensors "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors" && \
    \
    # Face restoration models
    wget -qO /stable-diffusion-webui/models/Codeformer/codeformer-v0.1.0.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" && \
    wget -qO /stable-diffusion-webui/models/GFPGAN/detection_Resnet50_Final.pth "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -qO /stable-diffusion-webui/models/GFPGAN/parsing_bisenet.pth "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth" && \
    wget -qO /stable-diffusion-webui/models/GFPGAN/parsing_parsenet.pth "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"


# ==============================================================================
# === FINAL STAGE ===
# ==============================================================================
# The final image starts from the same base but will only copy necessary artifacts.
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROOT=/stable-diffusion-webui \
    OMP_NUM_THREADS=1 \
    # Preload tcmalloc for better memory management
    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

WORKDIR /

# Install only RUNTIME system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Copy the cloned repos and all downloaded models from the builder stage
COPY --from=builder /stable-diffusion-webui /stable-diffusion-webui
WORKDIR /stable-diffusion-webui

# Uninstall default onnxruntime and install all Python dependencies in one layer.
# Versions are pinned for reproducibility.
RUN pip uninstall -y onnxruntime && \
    pip install --no-cache-dir \
    insightface==0.7.3 \
    onnxruntime-gpu==1.18.0 \
    boto3==1.34.131 \
    runpod==2.6.1 \
    xformers==0.0.24 \
    opencv-python==4.9.0.80 \
    albumentations==1.3.1 \
    protobuf==3.20.3

# Force download and cache of 'buffalo_l' models during the build [cite: 864]
RUN python3 -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

# Pre-run A1111 setup to download its own dependencies
RUN python3 -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy application files last to maximize layer caching
WORKDIR /
COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Define the default command
CMD ["/start.sh"]