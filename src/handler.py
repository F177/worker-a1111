import time
import runpod
import requests
import subprocess
import os
import signal
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

# O comando para iniciar o servidor A1111, com todas as otimizações
# que discutimos, quebrado em uma lista para o subprocess.
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/launch.py",
    "--xformers",
    "--opt-sdp-attention",
    "--no-half-vae",
    "--api",
    "--nowebui",
    "--port", "3000",
    "--disable-safe-unpickle",
    "--no-hashing",
    "--no-download-sd-model",
    "--skip-python-version-check",
    "--skip-torch-cuda-test",
    "--skip-version-check",
    "--skip-install",
    "--ckpt", "/model.safetensors"
]

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# ---------------------------------------------------------------------------- #
#                              Funções do Worker                               #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    """Verifica se o serviço está pronto para receber requisições."""
    retries = 0
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                print("Serviço ainda não está pronto. Tentando novamente...")
        except Exception as err:
            print("Erro ao esperar pelo serviço: ", err)
        time.sleep(0.2)

def run_inference(inference_request):
    """Executa a inferência."""
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()

# ---------------------------------------------------------------------------- #
#                                Handler do RunPod                             #
# ---------------------------------------------------------------------------- #
def handler(event):
    """O handler que será chamado pelo RunPod Serverless."""
    json_output = run_inference(event["input"])
    return json_output


if __name__ == "__main__":
    a1111_process = None
    try:
        # Inicia o processo do servidor A1111
        # preexec_fn=os.setsid cria uma nova sessão, tornando o processo o líder de um grupo.
        # Isso nos permite matar o processo e todos os seus filhos de uma vez.
        print("Iniciando o servidor A1111 em segundo plano...")
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid)

        # Espera o serviço ficar pronto
        print("Esperando a API do A1111 ficar pronta...")
        wait_for_service(url=f'{LOCAL_URL}/sd-models')

        # Inicia o servidor RunPod
        print("API do A1111 pronta. Iniciando o servidor RunPod...")
        runpod.serverless.start({"handler": handler})

    finally:
        # Este bloco é EXECUTADO SEMPRE que o 'try' termina, seja por sucesso ou erro.
        # É aqui que garantimos que o processo do servidor seja encerrado.
        if a1111_process:
            print("Desligando o servidor A1111...")
            # Envia um sinal de término para todo o grupo de processos
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
            a1111_process.wait() # Espera o processo realmente terminar
            print("Servidor A1111 desligado.")