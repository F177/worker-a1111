import os
import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import sys

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
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
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
    return response.json()

def handler(event):
    result = run_inference(event["input"])
    print("Job executado. Finalizando processo.")
    os._exit(0)  # Encerra imediatamente TODO o processo, acionando o trap no start.sh
    return result

if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("API do A1111 pronta. Iniciando o servidor RunPod...")
    runpod.serverless.start({"handler": handler})
