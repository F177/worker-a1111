import time
import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    """
    Check if the service is ready to receive requests.
    """
    retries = 0
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)


def run_inference(inference_request):
    """
    Run inference on a request.
    """
    # --- Positive Prompt Modifications ---
    lora_level = inference_request.get("lora_level", 1.0)
    lora_prompt = f"<lora:epicrealness:{lora_level}>"
    if "prompt" in inference_request:
        inference_request["prompt"] = f"{inference_request['prompt']}, {lora_prompt}"
    else:
        inference_request["prompt"] = lora_prompt
        
    # --- Negative Prompt Modifications ---
    # Define the trigger words for the negative embeddings
    negative_embeddings = "veryBadImageNegative_v1.3, FastNegativeV2"

    # Get the user's negative prompt, or an empty string if not provided
    user_negative_prompt = inference_request.get("negative_prompt", "")
    
    # Combine the user's negative prompt with the embeddings
    inference_request["negative_prompt"] = f"{user_negative_prompt}, {negative_embeddings}"

    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    """
    This is the handler function that will be called by the serverless.
    """
    json = run_inference(event["input"])
    return json


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})