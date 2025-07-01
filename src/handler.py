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
    "python", "/stable-diffusion-webui/webui.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-version-check", "--disable-safe-unpickle", "--no-hashing",
    "--opt-sdp-attention", "--no-download-sd-model", "--enable-insecure-extension-access",
    "--api-log", "--cors-allow-origins=*"
]

# --- S3 Client (for saving cropped faces) ---
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# --- Face Analysis Setup ---
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Face analyzer initialized successfully")
except Exception as e:
    print(f"Warning: Face analyzer initialization failed: {e}")
    face_analyzer = None

def detect_and_save_faces(image_bytes):
    """Detects faces in an image, crops them, and uploads them to S3."""
    if not face_analyzer:
        print("Face analyzer not available, skipping face detection")
        return []
        
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print("Error: could not decode image.")
            return []

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)
        if not faces:
            print("No faces detected in image")
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
                face_bytes = buffer.tobytes()

                face_id = f"f-{uuid.uuid4()}"
                s3_key = f"faces/{face_id}.png"

                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME, 
                    Key=s3_key, 
                    Body=face_bytes, 
                    ContentType='image/png'
                )

                detected_faces.append({
                    "face_id": face_id, 
                    "face_index": i,
                    "bbox": bbox.tolist()
                })
                print(f"Saved face {i} as {face_id}")
                
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue

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

def wait_for_service(url, max_wait=300):
    start_time = time.time()
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("A1111 service is ready.")
                return True
        except requests.exceptions.RequestException:
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error while waiting for service: {e}")
            time.sleep(5)
    
    print(f"Service failed to start within {max_wait} seconds")
    return False

def run_inference(inference_request):
    """
    Runs inference by passing the request directly to the A1111 API.
    This function correctly handles 'alwayson_scripts' like ReActor.
    """
    print(f"Starting inference with keys: {list(inference_request.keys())}")

    # Apply LoRA to the positive prompt
    lora_level = inference_request.get("lora_level", 0.6)
    lora_prompt = f"<lora:epiCRealnessRC1:{lora_level}>"
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, {lora_prompt}"
    
    # Apply negative embeddings
    negative_embeddings = "veryBadImageNegative_v1.3, FastNegativeV2"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, {negative_embeddings}"

    # Set base model and CLIP Skip via override_settings
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)
    }
    
    if "override_settings" not in inference_request:
        inference_request["override_settings"] = {}
    inference_request["override_settings"].update(override_settings)

    # Send the request to A1111
    if "alwayson_scripts" in inference_request:
        print("Faceswap/ReActor arguments detected in payload.")
    
    print("Sending request to A1111...")
    try:
        response = automatic_session.post(
            url=f'{LOCAL_URL}/txt2img', 
            json=inference_request, 
            timeout=600
        )
        response.raise_for_status() # Will raise an exception for 4xx/5xx status codes
        
        result = response.json()
        print("A1111 request completed successfully")

        # Extract and parse the info string to include it in the final output
        if 'info' in result:
            try:
                result['info'] = json.loads(result['info'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse A1111 info string: {result['info']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling A1111 API: {str(e)}"
        if e.response:
            error_msg += f" - {e.response.text}"
        print(error_msg)
        return {"error": error_msg}

# --- RUNPOD HANDLER ---
def handler(event):
    """Main function called by RunPod to process a job."""
    print(f"=== RunPod Job Started === | Event keys: {list(event.keys()) if event else 'None'}")
    
    try:
        if not event or "input" not in event:
            return {"error": "No input provided in event"}
        
        input_data = event["input"]
        print(f"Input data keys: {list(input_data.keys())}")
        
        # Check if this is a faceswap request
        is_faceswap_job = "alwayson_scripts" in input_data and "reactor" in input_data["alwayson_scripts"]
        
        # Run inference
        json_output = run_inference(input_data)
        
        if "error" in json_output:
            print(f"Inference failed: {json_output['error']}")
            return json_output
        
        # If this was NOT a faceswap job, run face detection to find new faces.
        # Otherwise, the face was already swapped, so we don't need to run detection again.
        if not is_faceswap_job and "images" in json_output:
            print("Standard generation complete. Running face detection to find usable faces...")
            try:
                image_bytes = base64.b64decode(json_output['images'][0])
                detected_faces = detect_and_save_faces(image_bytes)
                json_output['detected_faces'] = detected_faces
                print(f"Face detection completed. Found {len(detected_faces)} faces.")
            except Exception as e:
                print(f"Face detection on generated image failed: {e}")
                json_output['detected_faces'] = []
        else:
             print("Faceswap job complete. Skipping post-generation face detection.")
             json_output['detected_faces'] = [] # Ensure the key exists
        
        print("=== RunPod Job Completed Successfully ===")
        return json_output
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(f"=== RunPod Job Failed: {error_msg} ===")
        return {"error": error_msg}
    
# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    print("=== RunPod Worker Starting ===")
    
    try:
        a1111_process = subprocess.Popen(
            A1111_COMMAND, 
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        if wait_for_service(url=f'{LOCAL_URL}/progress', max_wait=300):
            print("A1111 service and extensions are ready!")
            runpod.serverless.start({"handler": handler})
        else:
            print("Failed to start A1111 service")
            if a1111_process:
                a1111_process.terminate()
            sys.exit(1)

    except Exception as e:
        print(f"Fatal error in main process: {e}")
    
    finally:
        if a1111_process and a1111_process.poll() is None:
            print("Terminating A1111 process...")
            a1111_process.terminate()
            a1111_process.wait(timeout=30)
        
        print("=== RunPod Worker Shutdown Complete ===")