import runpod
import requests
import subprocess
import os
import signal
import sys
import threading
import time  # <-- CRITICAL FIX: Added missing import
from requests.adapters import HTTPAdapter, Retry
import cv2
import insightface
import numpy as np
import boto3
import uuid
import base64
import traceback

# --- CONFIGURATION ---
# Centralized configuration for the A1111 API and server commands.
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/launch.py",
    "--listen",
    "--port", "3000",
    "--xformers",
    "--no-half-vae",
    "--api",
    "--nowebui",
    "--skip-version-check",
    "--disable-safe-unpickle",
    "--no-hashing",
    "--opt-sdp-attention",
    "--no-download-sd-model",
    "--enable-insecure-extension-access",
    "--api-log",
    "--cors-allow-origins=*",
    # OPTIMIZATION: Set the default checkpoint on startup to avoid loading the wrong model.
    "--ckpt", "/stable-diffusion-webui/models/Stable-diffusion/ultimaterealismo.safetensors"
]
S3_BUCKET_NAME = os.environ.get('S3_FACES_BUCKET_NAME')

# --- GLOBAL INITIALIZATIONS ---
# Initializing these objects globally is a key performance optimization.
# They are created only once per cold start and reused across multiple jobs.

# Setup S3 Client
s3_client = boto3.client('s3')

# Setup Face Analyzer
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Face analyzer initialized successfully")
except Exception as e:
    face_analyzer = None
    print(f"Warning: Face analyzer initialization failed: {e}")

# Setup a resilient requests session for communicating with the A1111 API
automatic_session = requests.Session()
retries = Retry(total=15, backoff_factor=0.5, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# --- GLOBALS FOR PROCESS MANAGEMENT ---
a1111_process = None
shutdown_flag = threading.Event()

# --- HELPER FUNCTIONS ---

def wait_for_service(url, max_wait=300):
    """Waits for the A1111 service to become ready."""
    start_time = time.time()
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = automatic_session.get(url, timeout=5)
            if response.status_code == 200:
                print("A1111 service is ready.")
                return True
        except requests.exceptions.RequestException:
            print("Waiting for A1111 service...")
            time.sleep(2)
    
    print(f"Service failed to start within {max_wait} seconds.")
    return False

def detect_and_save_faces(image_bytes):
    """Detects faces in an image, crops them, and uploads them to S3."""
    if not face_analyzer or not S3_BUCKET_NAME:
        print("Face analyzer or S3 bucket not configured, skipping face detection.")
        return []
        
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print("Error: Could not decode image.")
            return []

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)

        if not faces:
            print("No faces detected.")
            return []

        detected_faces = []
        for i, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)
                # Define padding to capture the area around the face
                padding = 20
                y1, y2 = max(0, bbox[1] - padding), min(bgr_img.shape[0], bbox[3] + padding)
                x1, x2 = max(0, bbox[0] - padding), min(bgr_img.shape[1], bbox[2] + padding)
                
                cropped_img = bgr_img[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.png', cropped_img)
                
                face_id = f"f-{uuid.uuid4()}"
                s3_key = f"faces/{face_id}.png"

                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME, 
                    Key=s3_key, 
                    Body=buffer.tobytes(), 
                    ContentType='image/png'
                )
                detected_faces.append({"face_id": face_id, "face_index": i, "bbox": bbox.tolist()})
            except Exception as e:
                print(f"Error processing face {i}: {e}")
        
        return detected_faces
    except Exception as e:
        print(f"Error during face detection process: {e}")
        traceback.print_exc()
        return []

# --- MAIN INFERENCE LOGIC ---

def run_inference(inference_request):
    """Prepares and sends the inference request to the A1111 API."""
    print(f"Starting inference with keys: {list(inference_request.keys())}")
    
    # --- Payload Preparation ---
    lora_level = inference_request.get("lora_level", 0.6)
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, <lora:epiCRealnessRC1:{lora_level}>"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, veryBadImageNegative_v1.3, FastNegativeV2"
    
    # Use override_settings to ensure the correct model is used without relying on API model switching
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)
    }
    if "override_settings" not in inference_request:
        inference_request["override_settings"] = {}
    inference_request["override_settings"].update(override_settings)
    
    print("Sending final payload to A1111 API...")
    try:
        response = automatic_session.post(
            url=f'{LOCAL_URL}/txt2img', 
            json=inference_request, 
            timeout=600  # 10-minute timeout for the API call
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        print("A1111 request completed successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling A1111 API: {e}")
        return {"error": f"Error calling A1111 API: {str(e)}", "details": traceback.format_exc()}


# --- RUNPOD HANDLER ---

def handler(event):
    """Main function called by RunPod to process a job."""
    print("=== RunPod Job Started ===")
    try:
        if not event or "input" not in event:
            return {"error": "No input provided in event"}
        
        input_data = event["input"]

        # Check if this is a ReActor (faceswap) job
        is_faceswap = "alwayson_scripts" in input_data and "reactor" in input_data.get("alwayson_scripts", {})
        if is_faceswap:
            print("Face Swap (ReActor) job detected.")
        
        json_output = run_inference(input_data)
        
        if "error" in json_output:
            return json_output
        
        # Run face detection on the output only if it's NOT a faceswap job
        if not is_faceswap and "images" in json_output and json_output.get("images"):
            print("Running face detection on generated image...")
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces
        elif is_faceswap:
            print("Skipping face detection because a face swap was performed.")
        
        print("=== RunPod Job Completed Successfully ===")
        return json_output
        
    except Exception as e:
        print("=== RunPod Job Failed ===")
        traceback.print_exc()
        return {"error": f"Error in handler: {str(e)}"}
    
    finally:
        # This block ensures the shutdown signal is always set, allowing the main loop to exit.
        print("Signaling worker shutdown...")
        shutdown_flag.set()

# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    print("=== RunPod Worker Starting ===")
    
    # Start the A1111 server as a background process
    a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid, stdout=sys.stdout, stderr=sys.stderr)
    
    if wait_for_service(url=f'{LOCAL_URL}/progress'):
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start A1111 service. Exiting.")
        if a1111_process:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        sys.exit(1)

    # Wait for the handler to finish its job and signal shutdown
    shutdown_flag.wait()
    
    # Gracefully terminate the A1111 process
    if a1111_process and a1111_process.poll() is None:
        print("Terminating A1111 process...")
        try:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
            a1111_process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            print("Graceful shutdown failed, forcing kill.")
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGKILL)
            
    print("=== RunPod Worker Shutdown Complete ===")