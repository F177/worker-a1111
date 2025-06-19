#!/usr/bin/env bash

echo "Worker Iniciado"

# Configura o alocador de memória otimizado
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Executa o handler do Python, que agora controla tudo.
# O 'exec' ainda é importante para que o Python seja o processo principal.
echo "Entregando o controle para o handler.py..."
exec python /handler.py