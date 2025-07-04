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
import traceback

# --- CONFIGURATION ---
LOCAL_URL = "http://127.0.0.1:3000" # Base URL for A1111
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-version-check", "--disable-safe-unpickle", "--no-hashing",
    "--opt-sdp-attention", "--no-download-sd-model", "--enable-insecure-extension-access",
    "--api-log", "--cors-allow-origins=*"
]

# --- S3 Client ---
s3_client = boto3.client('s3')
S3_FACES_BUCKET_NAME = os.environ.get('S3_FACES_BUCKET_NAME')

# --- Face Analysis Setup ---
face_analyzer = None
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Face analyzer initialized successfully")
except Exception as e:
    print(f"Warning: Face analyzer initialization failed: {e}")

def detect_and_save_faces(image_bytes):
    """Detects faces in an image, crops them, and uploads them to S3."""
    if not face_analyzer or not S3_FACES_BUCKET_NAME:
        print("Face analyzer or S3 bucket not configured, skipping face detection.")
        return []
        
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print("Error: could not decode image.")
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
                padding = 20
                y1 = max(0, bbox[1] - padding)
                y2 = min(bgr_img.shape[0], bbox[3] + padding)
                x1 = max(0, bbox[0] - padding)
                x2 = min(bgr_img.shape[1], bbox[2] + padding)
                
                cropped_img = bgr_img[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.png', cropped_img)
                
                face_id = f"f-{uuid.uuid4()}"
                s3_key = f"faces/{face_id}.png"

                s3_client.put_object(
                    Bucket=S3_FACES_BUCKET_NAME, 
                    Key=s3_key, 
                    Body=buffer.tobytes(), 
                    ContentType='image/png'
                )

                detected_faces.append({
                    "face_id": face_id, 
                    "face_index": i,
                    "bbox": bbox.tolist()
                })
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue
        
        print(f"Successfully detected and saved {len(detected_faces)} faces.")
        return detected_faces
    except Exception as e:
        print(f"Error during face detection process: {e}")
        return []

a1111_process = None
shutdown_flag = threading.Event()
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url, max_wait=300):
    """Waits for the A1111 service to be ready."""
    start_time = time.time()
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("A1111 service is ready.")
                return True
        except requests.exceptions.RequestException:
            print("Waiting for A1111 service...")
            time.sleep(5)
    
    print(f"Service failed to start within {max_wait} seconds")
    return False

def check_controlnet_available():
    """Check if ControlNet extension is available."""
    try:
        response = automatic_session.get(f'{LOCAL_URL}/controlnet/version', timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def get_controlnet_models():
    """Get available ControlNet models."""
    try:
        response = automatic_session.get(f'{LOCAL_URL}/controlnet/model_list', timeout=10)
        if response.status_code == 200:
            return response.json().get("model_list", [])
        return []
    except Exception:
        return []

def run_inference(inference_request):
    """Prepares and sends the inference request to the A1111 API."""
    print(f"Starting inference with keys: {list(inference_request.keys())}")
    
    # 1. Apply LoRA to the positive prompt
    lora_level = inference_request.get("lora_level", 0.6)
    lora_prompt = f"<lora:epiCRealnessRC1:{lora_level}>"
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, {lora_prompt}"
    
    # 2. Apply negative embeddings
    negative_embeddings = "veryBadImageNegative_v1.3, FastNegativeV2"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, {negative_embeddings}"

    # 3. Set base model and CLIP Skip via override_settings
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)
    }
    
    # 4. Add SDXL Refiner Logic if requested
    if inference_request.get("use_refiner", False):
        print("Refiner enabled for this request.")
        inference_request["refiner_checkpoint"] = "sd_xl_refiner_1.0.safetensors"
        inference_request["refiner_switch_at"] = inference_request.get("refiner_switch_at", 0.8)

    if "override_settings" not in inference_request:
        inference_request["override_settings"] = {}
    inference_request["override_settings"].update(override_settings)

    # 5. Handle IP-Adapter through ControlNet
    if 'ip_adapter_image_b64' in inference_request:
        print("IP-Adapter image detected. Setting up ControlNet...")
        
        if not check_controlnet_available():
            print("Warning: ControlNet not available, IP-Adapter will be skipped")
        else:
            available_models = get_controlnet_models()
            ip_adapter_model = next((model for model in available_models if 'ip-adapter' in model.lower() and 'sdxl' in model.lower()), "ip-adapter_sdxl [7d943a46]")
            
            print(f"Using IP-Adapter model: {ip_adapter_model}")
            
            controlnet_args = {
                "args": [{
                    "enabled": True, "image": inference_request['ip_adapter_image_b64'],
                    "module": "ip-adapter_clip_sdxl", "model": ip_adapter_model,
                    "weight": inference_request.get("ip_adapter_weight", 0.6),
                    "resize_mode": 1, "lowvram": False, "processor_res": 512,
                    "threshold_a": 0.5, "threshold_b": 0.5, "guidance_start": 0.0,
                    "guidance_end": 1.0, "pixel_perfect": False, "control_mode": 0
                }]
            }
            
            if "alwayson_scripts" not in inference_request:
                inference_request["alwayson_scripts"] = {}
            inference_request["alwayson_scripts"]["controlnet"] = controlnet_args
        
        # CORRECTED: 'del' statements are now safely inside the 'if' block
        del inference_request['ip_adapter_image_b64']
        if 'ip_adapter_weight' in inference_request:
            del inference_request['ip_adapter_weight']
    
    # 6. Send request to A1111
    print("Sending final payload to A1111 API...")
    try:
        response = automatic_session.post(
            url=f'{LOCAL_URL}/sdapi/v1/txt2img', 
            json=inference_request, 
            timeout=600
        )
        
        if response.status_code != 200:
            error_msg = f"A1111 API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return {"error": error_msg}
        
        result = response.json()
        print("A1111 request completed successfully.")
        return result
        
    except Exception as e:
        error_msg = f"Error calling A1111 API: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

def handler(event):
    """ Main RunPod handler function. """
    print("=== RunPod Job Started ===")
    try:
        input_data = event["input"]
        
        # Check if the job is for faceswapping by looking for the ReActor script args.
        is_faceswap = "alwayson_scripts" in input_data and "reactor" in input_data["alwayson_scripts"]

        print(f"Is faceswap request: {is_faceswap}")

        json_output = run_inference(input_data)
        
        if "error" in json_output:
            return json_output
        
        # Only run face detection if it was NOT a faceswap job and images were produced.
        if not is_faceswap and "images" in json_output and json_output["images"]:
            print("Running face detection on generated image...")
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces
        elif is_faceswap:
            print("Skipping face detection because a faceswap was performed.")
        
        print("=== RunPod Job Completed Successfully ===")
        return json_output

    except Exception as e:
        print("An error occurred in the handler.")
        traceback.print_exc()
        return {"error": f"Handler failed: {str(e)}"}
    
    finally:
        # This ensures the worker shuts down after one job, as is common for serverless endpoints.
        print("Signaling worker shutdown...")
        shutdown_flag.set()

if __name__ == "__main__":
    print("=== RunPod Worker Starting ===")
    a1111_process = subprocess.Popen(
        A1111_COMMAND, 
        preexec_fn=os.setsid,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if wait_for_service(url=f'{LOCAL_URL}/sdapi/v1/progress'):
        print("A1111 service confirmed. Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start A1111 service. Exiting.")
        if a1111_process:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        sys.exit(1)

    # Wait for the handler to signal shutdown
    shutdown_flag.wait()
    
    if a1111_process and a1111_process.poll() is None:
        print("Terminating A1111 process...")
        try:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
            a1111_process.wait(timeout=20)
        except:
            print("Force killing A1111 process...")
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGKILL)
            
    print("=== RunPod Worker Shutdown Complete ===")