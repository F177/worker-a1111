import os
import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            time.sleep(0.5)

def run_inference(inference_request):
    response = automatic_session.post(f"{LOCAL_URL}/txt2img", json=inference_request, timeout=600)
    return response.json()

def handler(event):
    result = run_inference(event["input"])

    # Detecta o job de teste (prompt padrão do RunPod)
    if "prompt" in event.get("input", {}) and event["input"]["prompt"] == "a photo of an astronaut riding a horse on mars":
        print("Job de teste detectado. Mantendo processo vivo.")
    else:
        print("Job real detectado. Encerrando processo.")
        os._exit(0)

    return result

if __name__ == "__main__":
    wait_for_service(f"{LOCAL_URL}/sd-models")
    runpod.serverless.start({"handler": handler})
