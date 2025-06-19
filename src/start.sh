#!/usr/bin/env bash

echo "Worker Initiated"

# Start the WebUI API in the background
echo "Starting Automatic1111 API..."
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true

# Launch the webui.py script with corrected arguments
python /stable-diffusion-webui/webui.py \
    --xformers \
    --no-half-vae \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    --skip-install \
    --ckpt /model.safetensors \
    --opt-sdp-attention \
    --disable-safe-unpickle \
    --port 3000 \
    --api \
    --nowebui \
    --skip-version-check \
    --no-hashing \
    --no-download-sd-model &

# Immediately start the handler (NOT RECOMMENDED)
echo "Starting RunPod Handler without health check..."
python -u /handler.py