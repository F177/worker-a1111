#!/usr/bin/env bash

echo "Worker Initiated"

# Set environment variables for better performance
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true

# --- CORREÇÃO FINAL: Limpa os caches de scripts antes de iniciar ---
# Isso força o A1111 a carregar a versão mais recente da extensão ReActor
echo "Cleaning up old extension caches..."
rm -rf /stable-diffusion-webui/extensions/sd-webui-reactor-sfw
rm -rf /stable-diffusion-webui/extensions-builtin/sd-webui-reactor-sfw
rm -rf /stable-diffusion-webui/tmp

# The only job of this script is to start the Python handler,
# which now manages the A1111 server itself.
echo "Starting RunPod Handler"
python3 -u /handler.py