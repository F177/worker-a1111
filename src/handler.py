import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
import time

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


def wait_for_service(url):
    """Verifica se o serviço A1111 está pronto."""
    while True:
        try:
            requests.get(url, timeout=120)
            print("Serviço A1111 está pronto.")
            return
        except requests.exceptions.RequestException:
            print("Serviço ainda não está pronto, tentando novamente...")
            time.sleep(2)
        except Exception as err:
            print("Erro ao esperar pelo serviço: ", err)
            time.sleep(2)


def run_inference(inference_request):
    """Roda a inferência e retorna o resultado."""
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()


def handler(event):
    """
    Função principal chamada pelo RunPod.
    Apenas executa o trabalho e retorna o resultado. Nada mais.
    """
    print("Job recebido. Iniciando a geração...")
    json_output = run_inference(event["input"])
    print("Geração finalizada. Retornando o resultado para a plataforma.")
    return json_output


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("Iniciando o handler do RunPod...")
    runpod.serverless.start({"handler": handler})