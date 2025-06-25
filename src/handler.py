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

# The command to start the A1111 server.
# We specify the checkpoint file directly since we are using a single model.
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py",
    "--xformers",
    "--no-half-vae",
    "--api",
    "--nowebui",
    "--port", "3000",
    "--skip-version-check",
    "--disable-safe-unpickle",
    "--ckpt", "/stable-diffusion-webui/models/Stable-diffusion/Deliberate_v6.safetensors"
]

a1111_process = None
shutdown_flag = threading.Event()

# --- NETWORK FUNCTIONS ---
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    """
    Waits for the A1111 service to be ready.
    """
    while True:
        try:
            requests.get(url, timeout=30)
            print("A1111 service is ready.")
            return
        except requests.exceptions.RequestException:
            print("A1111 service not ready yet, waiting...")
            time.sleep(2)
        except Exception as e:
            print(f"An unexpected error occurred while waiting for the service: {e}")
            # If a major error occurs, stop waiting.
            sys.exit(1)

def run_inference(inference_request):
    """
    Sends the inference request to the A1111 API.
    """
    response = automatic_session.post(
        url=f'{LOCAL_URL}/txt2img',
        json=inference_request,
        timeout=600
    )
    return response.json()

# --- RUNPOD HANDLER ---
def handler(event):
    """
    The main handler function called by RunPod.
    It runs the inference and then signals for shutdown.
    """
    try:
        print("Job received. Starting inference...")
        json_output = run_inference(event["input"])
        print("Inference complete.")
        return json_output
    finally:
        # This is the key optimization: signal that the worker should shut down.
        print("Job finished. Signaling for worker shutdown.")
        shutdown_flag.set()

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    try:
        print("Starting A1111 server in the background...")
        # Start the A1111 process
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid)

        # Wait for the service to be fully ready
        wait_for_service(url=f'{LOCAL_URL}/progress')
        
        # Give the model a few extra seconds to load after the web server is up
        print("Server is up. Giving model 10 seconds to load into VRAM...")
        time.sleep(10)

        print("Worker is ready. Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})

        # The code will block at runpod.serverless.start() until a job is processed.
        # After the handler runs and sets the flag, the start() function might exit.
        # We wait for the shutdown flag to be sure.
        print("RunPod handler finished. Waiting for shutdown signal.")
        shutdown_flag.wait()

    except Exception as e:
        print(f"A fatal error occurred in the main process: {e}")

    finally:
        # This block ensures that no matter what, we try to clean up the A1111 process.
        if a1111_process:
            print("Terminating A1111 process group...")
            try:
                # Kill the entire process group to ensure all child processes are terminated
                os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                print("A1111 process already terminated.")
        print("Worker has shut down.")

