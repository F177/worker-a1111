import os
import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

# Sessão com retry para lidar com erros intermitentes
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    """Espera o serviço A1111 estar pronto para receber requisições."""
    retries = 0
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                print("Serviço ainda não está pronto. Tentando novamente...")
        time.sleep(0.2)

def run_inference(inference_request):
    """Executa a geração de imagem via txt2img."""
    response = automatic_session.post(
        url=f'{LOCAL_URL}/txt2img',
        json=inference_request,
        timeout=600
    )
    return response.json()

def handler(event):
    """Função principal chamada pela infra do RunPod."""
    result = run_inference(event["input"])

    # Verifica se está em ambiente de teste da plataforma
    in_test = os.environ.get("RUNPOD_TEST_ENVIRONMENT", "false").lower() == "true"

    if not in_test:
        print("Job finalizado. Encerrando processo para liberar VRAM.")
        os._exit(0)
    else:
        print("Ambiente de teste detectado. Mantendo processo vivo.")

    return result

if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("API do A1111 pronta. Iniciando o servidor RunPod...")
    runpod.serverless.start({"handler": handler})
