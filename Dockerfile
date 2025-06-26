# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 as download

RUN apk add --no-cache wget && \
    # Main Model
    wget -q -O /model.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epicrealismXL_vxviiCrystalclear.safetensors" && \
    # LoRA Model
    wget -q -O /epicrealness.safetensors "https://huggingface.co/Fabricioi/modelorealista/resolve/main/epiCRealnessRC1.safetensors" && \
    # Negative Embeddings
    wget -q -O /veryBadImageNegative_v1.3.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/verybadimagenegative_v1.3.pt" && \
    wget -q -O /FastNegativeV2.pt "https://huggingface.co/Fabricioi/modelorealista/resolve/main/FastNegativeV2.pt"


FROM python:3.10.14-slim as build_final_image

ARG CACHE_BUSTER=1
ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "Busting cache with value: $CACHE_BUSTER" && \
    apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# Copy the main model
COPY --from=download /model.safetensors /stable-diffusion-webui/models/Stable-diffusion/epicrealismXL_vxviLastfameRealism.safetensors
# Copy the LoRA model
COPY --from=download /epicrealness.safetensors /stable-diffusion-webui/models/Lora/epicrealness.safetensors
# Copy the embeddings
COPY --from=download /veryBadImageNegative_v1.3.pt /stable-diffusion-webui/embeddings/veryBadImageNegative_v1.3.pt
COPY --from=download /FastNegativeV2.pt /stable-diffusion-webui/embeddings/FastNegativeV2.pt


# install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .

ADD src .

RUN chmod +x /start.sh
CMD /start.sh