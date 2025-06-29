# ---------------------------------------------------------------------------- #
#                          Stage 1: Download Models                              #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget && \
    # Main Base Model
    wget -q -O /ultimaterealismo.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epicrealismXL_vxviiCrystalclear.safetensors" && \
    # <<< NEW: SDXL Refiner Model >>>
    wget -q -O /sd_xl_refiner_1.0.safetensors "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors" && \
    # LoRA Model
    wget -q -O /epiCRealnessRC1.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epiCRealnessRC1.safetensors" && \
    # Negative Embeddings
    wget -q -O /veryBadImageNegative_v1.3.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/verybadimagenegative_v1.3.pt" && \
    wget -q -O /FastNegativeV2.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/FastNegativeV2.pt"


# ---------------------------------------------------------------------------- #
#                         Stage 2: Build Final Image                           #
# ---------------------------------------------------------------------------- #
FROM python:3.10.14-slim as build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers insightface onnxruntime && \ 
    pip install xformers && \
    pip install insightface && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy all models to their correct directories
COPY --from=download /ultimaterealismo.safetensors /stable-diffusion-webui/models/Stable-diffusion/
COPY --from=download /sd_xl_refiner_1.0.safetensors /stable-diffusion-webui/models/Stable-diffusion/
COPY --from=download /epiCRealnessRC1.safetensors /stable-diffusion-webui/models/Lora/
COPY --from=download /veryBadImageNegative_v1.3.pt /stable-diffusion-webui/embeddings/
COPY --from=download /FastNegativeV2.pt /stable-diffusion-webui/embeddings/

# Install dependencies from your requirements.txt
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Add the handler and startup script from your src folder
ADD src .

RUN chmod +x /start.sh
CMD ["/start.sh"]
