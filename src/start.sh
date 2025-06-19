#!/usr/bin/env bash

echo "Worker Iniciado"

# 1. Definir uma função de limpeza
# Esta função será chamada quando o script estiver prestes a terminar.
cleanup() {
    echo "Recebido sinal de saída. Desligando o servidor A1111..."
    # A variável $A1111_PID foi definida quando iniciamos o servidor.
    # 'kill' envia o sinal de término para o processo.
    kill -s SIGTERM $A1111_PID
    echo "Servidor A1111 desligado."
}

# 2. Armar a armadilha (trap)
# O comando 'trap' registra a função 'cleanup' para ser executada
# quando o script receber os sinais EXIT, SIGINT, ou SIGTERM.
# EXIT é o mais importante, pois é acionado quando o script termina normalmente.
trap cleanup EXIT SIGINT SIGTERM

# 3. Configurar ambiente e iniciar o A1111 em segundo plano
echo "Configurando ambiente e iniciando a API..."
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

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

# 4. Capturar o Process ID (PID) do servidor A1111
# A variável '$!' contém o PID do último comando executado em segundo plano.
A1111_PID=$!
echo "API do A1111 iniciada com PID: $A1111_PID"

# 5. Iniciar o handler.py (sem 'exec')
# O script agora espera aqui. Quando o handler.py terminar, o script continuará,
# o que acionará a armadilha 'EXIT' e chamará a função 'cleanup'.
python /handler.py