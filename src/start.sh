#!/bin/bash
set -e

# Move the downloaded model to the correct directory if it exists
if [ -f "model.safetensors" ]; then
    echo "Moving model to the correct directory..."
    mv model.safetensors stable-diffusion-webui/models/Stable-diffusion/
fi

# Change to the stable-diffusion-webui directory
cd stable-diffusion-webui

# Execute the handler with its absolute path
echo "Starting the handler..."
python -u /handler.py