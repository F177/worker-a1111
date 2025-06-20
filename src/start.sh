#!/usr/bin/env bash

# O único trabalho deste script é executar o handler do Python.
# 'exec' garante que o Python se torne o processo principal do container.
exec python /handler.py