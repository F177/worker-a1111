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

# Desinstala onnxruntime padrão para garantir que a versão GPU seja usada
RUN pip uninstall -y onnxruntime

# Instala todas as dependências do Python de uma vez com as versões corretas para GPU
# --- FIX APPLIED HERE ---
RUN pip install --no-cache-dir \
    # FIX: Pin numpy to a version before 2.0 to ensure compatibility with onnxruntime
    numpy<2 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.18.0 \
    boto3==1.34.131 \
    # FIX: Use a valid, existing version of the runpod library
    runpod==1.7.12 \
    xformers==0.0.24 \
    opencv-python==4.9.0.80 \
    albumentations==1.3.1 \
    protobuf==3.20.3

# ==============================================================================
# === OTIMIZAÇÃO: DOWNLOAD DE TODOS OS MODELOS DURANTE O BUILD ===
# ==============================================================================

# Cria todos os diretórios de modelos necessários
RUN mkdir -p \
    models/Stable-diffusion \
    models/Lora \
    embeddings \
    models/insightface \
    models/GFPGAN \
    models/Codeformer \
    models/VAE

# Baixa todos os modelos e dependências em uma única camada para otimizar o build
RUN \
    # Modelos Principais
    wget -O /stable-diffusion-webui/models/Stable-diffusion/ultimaterealismo.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epicrealismXL_vxviiCrystalclear.safetensors" && \
    wget -O /stable-diffusion-webui/models/Stable-diffusion/sd_xl_refiner_1.0.safetensors "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors" && \
    \
    # LoRA e Embeddings
    wget -O /stable-diffusion-webui/models/Lora/epiCRealnessRC1.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epiCRealnessRC1.safetensors" && \
    wget -O /stable-diffusion-webui/embeddings/veryBadImageNegative_v1.3.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/verybadimagenegative_v1.3.pt" && \
    wget -O /stable-diffusion-webui/embeddings/FastNegativeV2.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/FastNegativeV2.pt" && \
    \
    # Modelo de faceswap do ReActor
    wget -O /stable-diffusion-webui/models/insightface/inswapper_128.onnx "https://huggingface.co/Fabricioi/modelorealista/resolve/main/inswapper_128.onnx" && \
    \
    # Modelo VAE oficial para SDXL (evita download do VAE-approx)
    wget -O /stable-diffusion-webui/models/VAE/sdxl_vae.safetensors "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors" && \
    \
    # Modelos de restauração de face (GFPGAN e CodeFormer)
    wget -O /stable-diffusion-webui/models/Codeformer/codeformer-v0.1.0.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" && \
    wget -O /stable-diffusion-webui/models/GFPGAN/detection_Resnet50_Final.pth "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -O /stable-diffusion-webui/models/GFPGAN/parsing_bisenet.pth "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth" && \
    wget -O /stable-diffusion-webui/models/GFPGAN/parsing_parsenet.pth "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

# Força o download e cache dos modelos do insightface ('buffalo_l') durante o build
# This command will now succeed because of the numpy fix above.
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