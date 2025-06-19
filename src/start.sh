#!/usr/bin/env bash

echo "Worker Iniciado"

# 1. Otimização de Memória (TCMalloc)
echo "Configurando TCMalloc..."
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true

# 2. Iniciar a API do Automatic1111 em segundo plano com todas as otimizações
# O caminho do modelo foi ajustado para /model.safetensors para ser compatível com seu Dockerfile.
echo "Iniciando API do Automatic1111..."
python /stable-diffusion-webui/launch.py \
    --xformers \
    --opt-sdp-attention \
    --no-half-vae \
    --api \
    --nowebui \
    --port 3000 \
    --disable-safe-unpickle \
    --no-hashing \
    --no-download-sd-model \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    --skip-version-check \
    --skip-install \
    --ckpt /model.safetensors &

# 3. Iniciar o Handler do RunPod com 'exec'
# Isto garante que o container desligue corretamente.
echo "Iniciando o handler do RunPod..."
exec python /handler.py