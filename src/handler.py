import time
import runpod
import requests
import subprocess
import os
import signal
import sys
import threading
import base64
import uuid
from requests.adapters import HTTPAdapter, Retry
import cv2
import insightface
import numpy as np
import boto3

# --- CONFIGURATION ---
LOCAL_A1111_URL = "http://127.0.0.1:3000/sdapi/v1"
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# A1111 startup command with optimizations for a server environment
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-version-check", "--disable-safe-unpickle", "--no-hashing",
    "--opt-sdp-attention", "--no-download-sd-model", "--enable-insecure-extension-access",
    "--api-log", "--cors-allow-origins=*"
]

# --- GLOBAL VARIABLES ---
a1111_process = None
shutdown_flag = threading.Event()
face_analyzer = None

# --- INITIALIZATION ---

# Setup S3 Client for saving cropped faces
s3_client = boto3.client('s3')

# Setup Insightface Analyzer for face detection
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Insightface analyzer initialized successfully.")
except Exception as e:
    print(f"⚠️ Warning: Insightface analyzer initialization failed: {e}")

# Setup persistent requests session with retries for A1111 API
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# --- SERVICE FUNCTIONS ---

def wait_for_service(url, max_wait=300):
    """Waits for the A1111 service to become available."""
    start_time = time.time()
    print("Waiting for A1111 service to be ready...")
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("✅ A1111 service is ready.")
                return True
        except requests.exceptions.RequestException:
            time.sleep(5)
    print(f"❌ Service failed to start within {max_wait} seconds.")
    return False

def detect_and_save_faces(image_bytes):
    """
    Detects faces in a given image, crops them, and uploads them to S3.
    This is used to catalog faces from generations that did NOT use a face swap.
    """
    if not face_analyzer:
        print("Face analyzer not available, skipping face detection.")
        return []
        
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print("Error: Could not decode image for face detection.")
            return []

        # Insightface expects RGB, but OpenCV decodes as BGR
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)
        
        if not faces:
            print("No faces detected in the generated image.")
            return []

        print(f"Detected {len(faces)} faces. Cropping and saving to S3...")
        detected_faces = []
        for i, face in enumerate(faces):
            try:
                # Get bounding box and add padding
                bbox = face.bbox.astype(int)
                padding = 20
                y1, y2 = max(0, bbox[1] - padding), min(bgr_img.shape[0], bbox[3] + padding)
                x1, x2 = max(0, bbox[0] - padding), min(bgr_img.shape[1], bbox[2] + padding)
                
                cropped_img = bgr_img[y1:y2, x1:x2]
                
                # Encode back to bytes and upload
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
                print(f"  -> Saved face {i} with ID {face_id}")
                
            except Exception as e:
                print(f"Error processing face {i}: {e}")
        return detected_faces
    except Exception as e:
        print(f"An error occurred during face detection: {e}")
        return []

def run_inference(payload):
    """Prepares and sends the generation request to the A1111 API."""
    print("Processing inference request...")

    # Step 1: Apply default LoRA and negative embeddings
    lora_level = payload.get("lora_level", 0.6)
    payload["prompt"] = f"{payload.get('prompt', '')}, <lora:epiCRealnessRC1:{lora_level}>"
    payload["negative_prompt"] = f"{payload.get('negative_prompt', '')}, veryBadImageNegative_v1.3, FastNegativeV2"

    # Step 2: Set base model and CLIP Skip via override_settings
    # This ensures these settings are applied for this specific job.
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        "CLIP_stop_at_last_layers": payload.get("clip_skip", 1)
    }

    # If an existing override_settings block is in the payload, merge with it.
    payload.setdefault("override_settings", {}).update(override_settings)

    # Step 3: Remove our custom keys from payload before sending to A1111
    payload.pop("lora_level", None)
    
    # Step 4: Send the final payload to the txt2img endpoint
    print("Sending request to A1111 API...")
    try:
        response = automatic_session.post(
            url=f'{LOCAL_A1111_URL}/txt2img', 
            json=payload, 
            timeout=600  # 10-minute timeout for the generation
        )
        response.raise_for_status() # Raise exception for non-200 status codes
        print("✅ A1111 request completed successfully.")
        return response.json()
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling A1111 API: {e}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

# --- RUNPOD HANDLER ---
def handler(event):
    """Main function called by RunPod to process a job."""
    print("\n--- RunPod Job Received ---")
    
    try:
        if not event or "input" not in event:
            return {"error": "Invalid job input. 'input' field is missing."}
        
        job_input = event["input"]
        
        # Run the main image generation process
        result = run_inference(job_input)
        
        if "error" in result:
            return result
        
        # After generation, run face detection if it was NOT a face swap job.
        # Face swap jobs are identified by the presence of the 'alwayson_scripts' key.
        if "alwayson_scripts" not in job_input and "images" in result:
            print("Running face detection on generated image...")
            try:
                image_bytes = base64.b64decode(result['images'][0])
                detected_faces = detect_and_save_faces(image_bytes)
                result['detected_faces'] = detected_faces
            except Exception as e:
                print(f"⚠️ Face detection failed after generation: {e}")
                result['detected_faces'] = []
        
        print("--- RunPod Job Completed Successfully ---\n")
        return result
        
    except Exception as e:
        error_msg = f"An unexpected error occurred in the handler: {e}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    
# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    print("--- RunPod Worker Starting ---")
    
    try:
        # Start the A1111 web UI in a separate process
        print("Launching A1111 server process...")
        a1111_process = subprocess.Popen(A1111_COMMAND)
        
        # Wait for the API to be responsive before starting the handler
        if wait_for_service(url=f'{LOCAL_A1111_URL}/progress'):
            # Give extensions a moment to load after the API is up
            time.sleep(10) 
            print("Starting RunPod serverless handler...")
            runpod.serverless.start({"handler": handler})
        else:
            print("A1111 service failed to start. Exiting worker.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal error in main process: {e}")
        sys.exit(1)
    
    finally:
        # Ensure clean shutdown of the A1111 process
        if a1111_process and a1111_process.poll() is None:
            print("Terminating A1111 process...")
            try:
                a1111_process.terminate()
                a1111_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("A1111 did not terminate gracefully, killing process...")
                a1111_process.kill()
        
        print("--- RunPod Worker Shutdown Complete ---")