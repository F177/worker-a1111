# Stage 1: Começamos de uma imagem base do RunPod que já contém CUDA e PyTorch
# Isso elimina os downloads gigantes e os problemas de tamanho de camada.
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel AS build_final_image

# Definimos os argumentos e variáveis de ambiente
ARG A1111_RELEASE=v1.9.3
ENV DEBIAN_FRONTEND=noninteractive \
    ROOT=/stable-diffusion-webui

# Instala apenas as dependências de sistema que faltam
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git rsync wget libgoogle-perftools-dev && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR ${ROOT}

# Clona o A1111 e remove o .git para economizar espaço
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git . && \
    git reset --hard ${A1111_RELEASE} && \
    rm -rf .git

# Instala o xformers e os requisitos do A1111.
# Isso será muito mais rápido pois o torch já está instalado.
RUN pip install xformers==0.0.22.post7 && \
    pip install -r requirements_versions.txt

# Baixa nosso modelo de exemplo
RUN wget -q -O ${ROOT}/models/Stable-diffusion/model.safetensors https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v6.safetensors

# Voltamos para a raiz para copiar os arquivos do worker
WORKDIR /

# Copia os arquivos de dependência e da aplicação
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/handler.py .
COPY src/start.sh .
RUN chmod +x /start.sh

# Define o comando de inicialização
CMD ["/start.sh"]