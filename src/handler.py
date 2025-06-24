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

# <<< CHANGE 1: REMOVED the incompatible "--ckpt" argument
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/launch.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-install", "--skip-version-check", "--disable-safe-unpickle"
]

a1111_process = None
shutdown_flag = threading.Event()

# --- SESSÃO COM REQUISIÇÕES ROBUSTAS ---
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# --- AGUARDA O A1111 FICAR PRONTO ---
def wait_for_service(url):
    while True:
        try:
            print(f"Verificando se o serviço está pronto: {url}")
            # Use the session for the health check as well
            r = automatic_session.get(url, timeout=30)
            # A1111 might return 405 Method Not Allowed if you GET a POST-only endpoint, which is fine.
            if r.status_code in [200, 405]:
                print("A1111 está pronto.")
                return
        except requests.exceptions.RequestException:
            print("Serviço A1111 ainda não está pronto, aguardando...")
        time.sleep(2)

# --- FAZ A INFERÊNCIA ---
def run_inference(inference_request):
    try:
        print("Enviando requisição para A1111...")
        response = automatic_session.post(
            f'{LOCAL_URL}/txt2img',
            json=inference_request,
            timeout=900
        )
        response.raise_for_status()
        print("Resposta recebida com sucesso.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro durante a inferência: {e}")
        return {
            "error": "Falha na geração de imagem",
            "details": str(e)
        }

# --- HANDLER PRINCIPAL ---
def handler(event):
    try:
        print("Job recebido.")
        if "input" not in event:
            return {
                "error": "Formato inválido. Esperado: { 'input': { ... } }"
            }
        json_output = run_inference(event["input"])
        return json_output
    finally:
        print("Sinalizando desligamento do worker após job.")
        shutdown_flag.set()

# --- INICIALIZAÇÃO ---
# <<< CHANGE 2: CORRECTED the typo from "_main_" to "__main__"
if __name__ == "__main__":
    try:
        print("Iniciando o servidor A1111 em segundo plano...")
        a1111_process = subprocess.Popen(
            A1111_COMMAND,
            preexec_fn=os.setsid
        )

        # Check a reliable endpoint to confirm the API is running
        wait_for_service(f"{LOCAL_URL}/progress")

        print("A1111 pronto. Iniciando o handler do RunPod...")
        runpod.serverless.start({"handler": handler})

        print("RunPod finalizado. Aguardando shutdown_flag...")
        shutdown_flag.wait()

    except Exception as e:
        print(f"Erro fatal no worker: {e}", file=sys.stderr)

    finally:
        if a1111_process:
            print("Encerrando processo A1111...")
            # Use killpg to terminate the entire process group
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        print("Worker finalizado com sucesso.")