import time
import runpod
import requests
import subprocess
import os
import signal
import sys
import threading
from requests.adapters import HTTPAdapter, Retry

# --- CONFIGURATION ---
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

# A1111 is now started here. Note the --opt-sdp-attention flag is added
# and --ckpt is correctly removed for faster startup.
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-version-check", "--disable-safe-unpickle", "--no-hashing",
    "--opt-sdp-attention", "--no-download-sd-model"
]

a1111_process = None
shutdown_flag = threading.Event()

# --- NETWORK FUNCTIONS ---
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    """Waits for the A1111 service to be ready."""
    while not shutdown_flag.is_set():
        try:
            requests.get(url, timeout=5)
            print("A1111 service is ready.")
            return True
        except requests.exceptions.RequestException:
            print("Waiting for A1111 service...")
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected error while waiting for service: {e}")
            return False
    return False

def run_inference(inference_request):
    """Runs inference with the provided payload, adding refiner logic."""
    # 1. Apply LoRA to the positive prompt
    lora_level = inference_request.get("lora_level", 0.6)
    lora_prompt = f"<lora:epiCRealnessRC1:{lora_level}>"
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, {lora_prompt}"
    
    # 2. Apply negative embeddings
    negative_embeddings = "veryBadImageNegative_v1.3, FastNegativeV2"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, {negative_embeddings}"

    # 3. CRITICAL: Tell A1111 which base model to use and set CLIP Skip
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        
        # <<< --- FIX: Add this line to correctly handle clip_skip --- >>>
        "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)
    }
    
    # Add SDXL Refiner Logic if requested
    if inference_request.get("use_refiner", False):
        print("Refiner enabled for this request.")
        inference_request["refiner_checkpoint"] = "sd_xl_refiner_1.0.safetensors"
        inference_request["refiner_switch_at"] = inference_request.get("refiner_switch_at", 0.8)

    if "override_settings" not in inference_request:
        inference_request["override_settings"] = {}
    inference_request["override_settings"].update(override_settings)

    # 4. Send the request
    # The top-level "clip_skip" will be ignored, but the one in override_settings will work
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
    return response.json()

# --- RUNPOD HANDLER ---
def handler(event):
    """Main function called by RunPod to process a job."""
    try:
        print("Job received. Starting inference...")
        json_output = run_inference(event["input"])
        print("Inference complete.")
        return json_output
    finally:
        # This block is crucial for your optimization.
        # It ensures that after the job is done, the shutdown is signaled.
        print("Job finished. Signaling for worker shutdown.")
        shutdown_flag.set()

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    try:
        print("Starting A1111 server in the background...")
        # This script now launches and manages the A1111 process
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid)

        if wait_for_service(url=f'{LOCAL_URL}/progress'):
            print("Starting RunPod serverless handler...")
            runpod.serverless.start({"handler": handler})

        # The script will pause here until the handler sets the shutdown_flag
        shutdown_flag.wait()

    except Exception as e:
        print(f"A fatal error occurred in the main process: {e}")
        shutdown_flag.set()
    finally:
        # This block ensures the A1111 server is terminated cleanly.
        if a1111_process and a1111_process.poll() is None:
            print("Terminating A1111 process...")
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
            a1111_process.wait()
        print("Worker has shut down.")