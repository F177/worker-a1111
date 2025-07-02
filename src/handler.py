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
import json

# --- CONFIGURATION ---
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py", "--xformers", "--no-half-vae", 
    "--api", "--nowebui", "--port", "3000", "--skip-version-check", 
    "--disable-safe-unpickle", "--no-hashing", "--opt-sdp-attention", 
    "--no-download-sd-model", "--enable-insecure-extension-access",
    "--api-log", "--cors-allow-origins=*"
]
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

def set_rector_device(device_str):
    """
    Força a extensão ReActor a usar um dispositivo específico (CUDA)
    modificando o arquivo de configuração dela em tempo de execução.
    """
    try:
        reactor_helpers_path = "/stable-diffusion-webui/extensions/sd-webui-reactor/scripts/reactor_helpers.py"
        with open(reactor_helpers_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        in_set_device_func = False
        device_already_set = any("DEVICE = " in line for line in lines if "def set_Device" in line or in_set_device_func)

        if not device_already_set:
            for line in lines:
                new_lines.append(line)
                if "def set_Device(DEVICE):" in line:
                    new_lines.append(f"    DEVICE = '{device_str}'\n")
            
            with open(reactor_helpers_path, "w") as f:
                f.writelines(new_lines)
            
            print(f"ReActor device explicitly set to {device_str}")
        else:
            print("ReActor device override already seems to be in place.")
            
    except Exception as e:
        print(f"Could not force ReActor device: {e}")

# --- INITIALIZATION ---
s3_client = boto3.client('s3')
face_analyzer = None
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Face analyzer initialized successfully.")
except Exception as e:
    print(f"Warning: Face analyzer initialization failed: {e}")

# --- HELPER FUNCTIONS ---
def detect_and_save_faces(image_bytes):
    if not face_analyzer or not S3_BUCKET_NAME:
        return []
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = face_analyzer.get(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        detected_faces = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            padding = 20
            y1, y2 = max(0, bbox[1] - padding), min(bgr_img.shape[0], bbox[3] + padding)
            x1, x2 = max(0, bbox[0] - padding), min(bgr_img.shape[1], bbox[2] + padding)
            cropped_img = bgr_img[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.png', cropped_img)
            
            face_id = f"f-{uuid.uuid4()}"
            s3_key = f"faces/{face_id}.png"
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=buffer.tobytes(), ContentType='image/png')
            detected_faces.append({"face_id": face_id, "face_index": i, "bbox": bbox.tolist()})
            print(f"Detected and saved face {i} as {face_id}")
        return detected_faces
    except Exception as e:
        print(f"Error during face detection: {e}")
        return []

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    for _ in range(60): # Wait up to 5 minutes
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print("A1111 service is ready.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    return False

def run_inference(inference_request):
    print(f"Starting inference with keys: {list(inference_request.keys())}")
    lora_level = inference_request.get("lora_level", 0.6)
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, <lora:epiCRealnessRC1:{lora_level}>"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, veryBadImageNegative_v1.3, FastNegativeV2"
    inference_request["override_settings"] = {"sd_model_checkpoint": "ultimaterealismo.safetensors", "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)}
    
    print("Sending request to A1111...")
    try:
        response = automatic_session.post(f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
        response.raise_for_status()
        print("A1111 request completed successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling A1111 API: {e}" + (f" - {e.response.text}" if e.response else "")
        print(error_msg)
        return {"error": error_msg}

# --- RUNPOD HANDLER ---
def handler(event):
    print(f"=== RunPod Job Started === | Event keys: {list(event.keys()) if event else 'None'}")
    try:
        input_data = event["input"]
        is_faceswap_job = "alwayson_scripts" in input_data and "reactor" in input_data.get("alwayson_scripts", {})
        
        json_output = run_inference(input_data)
        if "error" in json_output:
            return json_output
        
        if not is_faceswap_job and "images" in json_output:
            print("Standard job finished. Running face detection...")
            image_bytes = base64.b64decode(json_output['images'][0])
            json_output['detected_faces'] = detect_and_save_faces(image_bytes)
        else:
            print("Faceswap job finished. Skipping face detection.")
            json_output['detected_faces'] = []
            
        print("=== RunPod Job Completed Successfully ===")
        return json_output
        
    except Exception as e:
        return {"error": f"Handler error: {e}"}

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    set_rector_device("CUDA")
    print("=== Starting RunPod Worker ===")
    a1111_process = subprocess.Popen(A1111_COMMAND, stdout=sys.stdout, stderr=sys.stderr)
    if wait_for_service(url=f'{LOCAL_URL}/progress'):
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start A1111 service. Exiting.")
        a1111_process.terminate()
        sys.exit(1)