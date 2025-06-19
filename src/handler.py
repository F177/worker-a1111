import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

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
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
    return response.json()

def handler(event):
    """O handler que será chamado pelo RunPod Serverless."""
    json_output = run_inference(event["input"])
    return json_output


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("API do A1111 pronta. Iniciando o servidor RunPod...")
    runpod.serverless.start({"handler": handler})