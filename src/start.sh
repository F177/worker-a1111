#!/usr/bin/env bash

echo "Worker Iniciado"

# 1. Função de limpeza AGRESSIVA
# Esta função usa 'pkill' para forçar o encerramento de todos os processos do A1111.
cleanup() {
    echo "Recebido sinal de saída. Forçando o desligamento de todos os processos A1111..."
    # pkill -f "launch.py" -> Procura por qualquer processo cujo comando contenha "launch.py".
    # -9 -> Envia o sinal SIGKILL, que não pode ser ignorado. É a forma mais forte de encerrar um processo.
    pkill -f -9 "launch.py"
    echo "Processos A1111 finalizados à força."
}

# 2. Armar a armadilha (trap)
# A lógica do trap permanece a mesma. Ele executará a nossa nova função 'cleanup' na saída.
trap cleanup EXIT SIGINT SIGTERM

# 3. Iniciar o A1111 em segundo plano
echo "Iniciando API do A1111..."
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

# 4. Iniciar o handler.py e esperar
# Não precisamos mais capturar o PID, pois o pkill o encontrará pelo nome.
echo "API do A1111 iniciada. Iniciando o handler.py..."
python /handler.py