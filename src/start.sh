#!/usr/bin/env bash

echo "Worker Iniciado"

# Função que será executada na saída do script
cleanup() {
    echo "Recebido sinal de saída. Desligando o servidor A1111 (PID: $A1111_PID)..."
    # Envia o sinal de término (SIGTERM) para o processo do A1111
    # O 'kill 0' garante que todo o grupo de processos seja encerrado, caso o A1111 crie filhos.
    kill -s SIGTERM 0
    echo "Sinal de término enviado. Aguardando finalização..."
    wait $A1111_PID
    echo "Servidor A1111 desligado."
}

# O comando 'trap' registra a função 'cleanup' para ser executada quando o script
# receber um sinal para terminar (EXIT, SIGINT, SIGTERM).
trap cleanup EXIT SIGINT SIGTERM

# Inicia o servidor A1111 em segundo plano
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

# Captura o Process ID (PID) do servidor A1111
A1111_PID=$!
echo "API do A1111 iniciada com PID: $A1111_PID"

# Inicia o handler do Python e espera ele terminar.
# Não usamos 'exec' aqui, pois o script precisa continuar existindo para que o 'trap' funcione.
python /handler.py

# Quando o handler.py terminar, o script chegará ao fim, o que acionará o 'trap EXIT',
# chamando a função 'cleanup' e garantindo que o processo do A1111 seja encerrado.