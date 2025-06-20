import time
import runpod
import requests
import subprocess
import os
import signal
import sys
from requests.adapters import HTTPAdapter, Retry

# --- CONFIGURAÇÃO ---
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/launch.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-install", "--skip-version-check", "--disable-safe-unpickle",
    "--ckpt", "/model.safetensors"
]
a1111_process = None

# --- FUNÇÕES DE REDE ---
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    """Espera o serviço A1111 estar pronto."""
    while True:
        try:
            requests.get(url, timeout=30)
            return
        except requests.exceptions.RequestException:
            print("Serviço A1111 ainda não está pronto, aguardando...")
            time.sleep(2)

def run_inference(inference_request):
    """Roda a inferência."""
    response = automatic_session.post(f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
    response.raise_for_status()
    return response.json()

# --- FUNÇÃO PRINCIPAL DO HANDLER ---
def handler(event):
    """
    Processa o job, retorna o resultado e, em seguida, inicia o processo de autodestruição.
    """
    global a1111_process
    try:
        print("Job recebido. Iniciando a geração...")
        json_output = run_inference(event["input"])
        print("Geração finalizada. Retornando o resultado para a plataforma.")
        return json_output
    finally:
        # Este bloco é executado DEPOIS que o 'return' acima é concluído.
        # Ele desliga o servidor A1111 e depois o próprio script.
        print("Iniciando autodestruição do worker...")
        if a1111_process:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        # sys.exit() encerra o processo Python de forma limpa.
        # Como é o processo principal (devido ao 'exec' no start.sh), o container desliga.
        sys.exit(0)

# --- PONTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    try:
        # Inicia o servidor A1111 como um processo filho
        print("Iniciando o servidor A1111 em segundo plano...")
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid)

        # Espera o serviço ficar pronto
        wait_for_service(url=f'{LOCAL_URL}/sd-models')

        # Inicia o handler do RunPod
        print("Iniciando o handler do RunPod...")
        runpod.serverless.start({"handler": handler})

    except Exception as e:
        print(f"Um erro fatal ocorreu: {e}")
    finally:
        # Garante que o processo A1111 seja encerrado mesmo se houver um erro na inicialização.
        if a1111_process:
            print("Limpando o processo A1111 na saída...")
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)