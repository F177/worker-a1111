# Use a imagem base oficial do RunPod para PyTorch
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROOT=/stable-diffusion-webui
ENV OMP_NUM_THREADS=1

WORKDIR /

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
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
    && rm -rf /var/lib/apt/lists/*

# Clona o Stable Diffusion WebUI
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /stable-diffusion-webui
WORKDIR /stable-diffusion-webui
RUN git checkout v1.9.3

# Instala a extensão ReActor
RUN cd extensions && \
    git clone https://codeberg.org/Gourieff/sd-webui-reactor.git

# Instala todas as dependências do Python de uma vez com as versões corretas para GPU
RUN pip install --no-cache-dir \
    insightface==0.7.3 \
    onnxruntime-gpu==1.18.0 \
    boto3 \
    runpod \
    xformers==0.0.24 \
    opencv-python \
    albumentations==1.3.1 \
    protobuf==3.20.3

# Cria os diretórios para os modelos
RUN mkdir -p models/Stable-diffusion \
    models/Lora \
    embeddings \
    models/insightface

# Baixa os modelos principais
RUN wget -O /stable-diffusion-webui/models/Stable-diffusion/ultimaterealismo.safetensors \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epicrealismXL_vxviiCrystalclear.safetensors"

RUN wget -O /stable-diffusion-webui/models/Stable-diffusion/sd_xl_refiner_1.0.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"

# Baixa LoRA e Embeddings
RUN wget -O /stable-diffusion-webui/models/Lora/epiCRealnessRC1.safetensors \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epiCRealnessRC1.safetensors"

RUN wget -O /stable-diffusion-webui/embeddings/veryBadImageNegative_v1.3.pt \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/verybadimagenegative_v1.3.pt"

RUN wget -O /stable-diffusion-webui/embeddings/FastNegativeV2.pt \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/FastNegativeV2.pt"

# Baixa o modelo de faceswap do ReActor
RUN wget -O /stable-diffusion-webui/models/insightface/inswapper_128.onnx \
    "https://huggingface.co/Fabricioi/modelorealista/resolve/main/inswapper_128.onnx"

# ==============================================================================
# === NOVA LINHA DE OTIMIZAÇÃO ===
# Força o download dos modelos do insightface durante o build da imagem
# ==============================================================================
RUN python3 -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

# Pré-inicializa o A1111 para baixar outras dependências
WORKDIR /stable-diffusion-webui
RUN python3 -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copia os arquivos da aplicação
WORKDIR /
COPY handler.py /handler.py
COPY start.sh /start.sh

# Torna o script de início executável
RUN chmod +x /start.sh

# Define o comando padrão
CMD ["/start.sh"]