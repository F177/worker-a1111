#!/usr/bin/env bash

echo "Worker Iniciado"

# Função de limpeza agressiva que será executada na saída do script
cleanup() {
    echo "Recebido sinal de saída. Forçando o desligamento de todos os processos A1111..."
    # pkill -f "launch.py" -> Procura por qualquer processo cujo comando contenha "launch.py".
    # -9 -> Envia o sinal SIGKILL, que não pode ser ignorado (encerramento forçado).
    pkill -f -9 "launch.py"
    echo "Processos A1111 finalizados à força."
}

# O comando 'trap' garante que a função 'cleanup' seja executada quando o script terminar.
trap cleanup EXIT SIGINT SIGTERM

# Inicia o servidor A1111 em segundo plano com todas as otimizações
echo "Iniciando API do A1111 em segundo plano..."
python /stable-diffusion-webui/launch.py \
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

# Inicia o handler do Python e espera ele terminar
python /handler.py