import time
import runpod
import requests
import subprocess
import os
import signal
import sys
import threading
from requests.adapters import HTTPAdapter, Retry
import cv2
import insightface
import numpy as np
import boto3
import uuid
import base64 

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

# --- S3 Client (for saving cropped faces) ---
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# --- Face Analysis Setup ---
face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))


def detect_and_save_faces(image_bytes):
    """Detects faces in an image, crops them, and uploads them to S3."""
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = face_analyzer.get(img)
        if not faces:
            return []

        detected_faces = []
        for i, face in enumerate(faces):
            # Crop the face using the bounding box
            bbox = face.bbox.astype(int)
            cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Convert cropped image back to bytes (PNG format)
            _, buffer = cv2.imencode('.png', cropped_img)
            face_bytes = buffer.tobytes()

            # Generate a unique ID and S3 key
            face_id = f"f-{uuid.uuid4()}"
            s3_key = f"faces/{face_id}.png"

            # Upload to S3
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=face_bytes, ContentType='image/png')

            detected_faces.append({"face_id": face_id, "face_index": i})

        return detected_faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

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
    """Runs inference with the provided payload, adding refiner and IP-Adapter logic."""
    # [cite_start]1. Apply LoRA to the positive prompt [cite: 1]
    lora_level = inference_request.get("lora_level", 0.6)
    lora_prompt = f"<lora:epiCRealnessRC1:{lora_level}>"
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, {lora_prompt}"
    
    # [cite_start]2. Apply negative embeddings [cite: 1]
    negative_embeddings = "veryBadImageNegative_v1.3, FastNegativeV2"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, {negative_embeddings}"

    # [cite_start]3. Set base model and CLIP Skip via override_settings [cite: 1]
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)
    }
    
    # [cite_start]4. Add SDXL Refiner Logic if requested [cite: 1]
    if inference_request.get("use_refiner", False):
        print("Refiner enabled for this request.")
        inference_request["refiner_checkpoint"] = "sd_xl_refiner_1.0.safetensors"
        inference_request["refiner_switch_at"] = inference_request.get("refiner_switch_at", 0.8)

    if "override_settings" not in inference_request:
        inference_request["override_settings"] = {}
    inference_request["override_settings"].update(override_settings)

    # 5. <<< ADD THIS SECTION TO HANDLE IP-ADAPTER >>>
    if 'ip_adapter_image_b64' in inference_request:
        print("IP-Adapter data found. Constructing script arguments.")
        ip_adapter_args = {
            "args": [
                {
                    "enabled": True,
                    "image": inference_request.get("ip_adapter_image_b64"),
                    "weight": inference_request.get("ip_adapter_weight", 0.6),
                    "model": "ip-adapter_sdxl.bin",
                }
            ]
        }
        
        if "alwayson_scripts" not in inference_request:
            inference_request["alwayson_scripts"] = {}
        # The key must be "IP-Adapter" to match the extension's name
        inference_request["alwayson_scripts"]["IP-Adapter"] = ip_adapter_args
    
    # [cite_start]6. Send the final request to A1111 [cite: 1]
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
    return response.json()

# --- RUNPOD HANDLER ---
def handler(event):
    """Main function called by RunPod to process a job."""
    try:
        print("Job received. Starting inference...")
        json_output = run_inference(event["input"])
        if "ip_adapter_image_b64" not in event["input"]:
            print("Running face detection on the new image.")
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces

        print("Inference complete.")
        return json_output
    finally:
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