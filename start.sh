#!/usr/bin/env bash

echo "Worker Initiated"

# Set environment variables for better performance
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true

# The only job of this script is to start the Python handler,
# which now manages the A1111 server itself.
echo "Starting RunPod Handler"
python -u /handler.py