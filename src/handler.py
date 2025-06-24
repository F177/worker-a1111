import time
import runpod
import requests
import subprocess
import os
import signal
import sys
import threading
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
shutdown_flag = threading.Event() # Usaremos um "eventjjjo" para sinalizar o desligamento

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
    """Roda a inferência com timeout aumentado."""
    response = automatic_session.post(f'{LOCAL_URL}/txt2img', json=inference_request, timeout=900)
    response.raise_for_status()
    return response.json()

# --- FUNÇÃO PRINCIPAL DO HANDLER ---
def handler(event):
    """
    Processa o job, retorna o resultado e sinaliza para o desligamento.
    """
    try:
        print("Job recebido. Iniciando a geração...")
        json_output = run_inference(event["input"])
        print("Geração finalizada. Retornando o resultado para a plataforma.")
        return json_output
    finally:
        # Em vez de sys.exit(), nós apenas ativamos a flag.
        print("Sinalizando para o desligamento do worker...")
        shutdown_flag.set()

# --- PONTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    try:
        print("Iniciando o servidor A1111 em segundo plano...")
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid)

        wait_for_service(url=f'{LOCAL_URL}/sd-models')

        print("Iniciando o handler do RunPod...")
        # A chamada para o start agora é o nosso ponto de bloqueio principal.
        runpod.serverless.start({"handler": handler})

        # O código abaixo só será executado DEPOIS que runpod.serverless.start() terminar.
        # Ele só termina quando o worker é sinalizado para desligar. A biblioteca do runpod
        # não tem um mecanismo de parada limpo, então vamos esperar a flag ser ativada.
        print("Loop do RunPod terminado. Aguardando a flag de desligamento...")
        shutdown_flag.wait() # Espera até que o handler ative a flag

    except Exception as e:
        print(f"Um erro fatal ocorreu: {e}")
    finally:
        # Garante que o processo A1111 seja encerrado na saída.
        if a1111_process:
            print("Limpando o processo A1111 na saída final...")
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        
        print("Worker encerrado.")