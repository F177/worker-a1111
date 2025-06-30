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

# Updated A1111 command with better flags for RunPod
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-version-check", "--disable-safe-unpickle", "--no-hashing",
    "--opt-sdp-attention", "--no-download-sd-model", "--enable-insecure-extension-access",
    "--api-log", "--cors-allow-origins=*"
]

# --- S3 Client (for saving cropped faces) ---
s3_client = None
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1') # Default to a common region if not set

if S3_BUCKET_NAME and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        print("S3 client initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize S3 client: {e}")
        s3_client = None
else:
    print("Warning: S3 environment variables (S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) are not fully set. Face detection uploads will be disabled.")


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
    
    if not s3_client:
        print("S3 client not configured, skipping face upload to S3.")
        return []
        
    try:
        # Decode image
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print("Error: could not decode image.")
            return []

        # Convert BGR to RGB for face analysis
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = face_analyzer.get(rgb_img)
        if not faces:
            print("No faces detected in image")
            return []

        detected_faces = []
        for i, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)
                # Add some padding to the bounding box
                padding = 20
                y1 = max(0, bbox[1] - padding)
                y2 = min(bgr_img.shape[0], bbox[3] + padding)
                x1 = max(0, bbox[0] - padding)
                x2 = min(bgr_img.shape[1], bbox[2] + padding)
                
                cropped_img = bgr_img[y1:y2, x1:x2]
                
                # Encode image
                _, buffer = cv2.imencode('.png', cropped_img)
                face_bytes = buffer.tobytes()

                # Generate unique face ID
                face_id = f"f-{uuid.uuid4()}"
                s3_key = f"faces/{face_id}.png"

                # Upload to S3
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
    """Waits for the A1111 service to be ready."""
    start_time = time.time()
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("A1111 service is ready.")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Waiting for A1111 service... ({e})")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error while waiting for service: {e}")
            time.sleep(5)
    
    print(f"Service failed to start within {max_wait} seconds")
    return False

def check_controlnet_available():
    """Check if ControlNet extension is available."""
    try:
        response = automatic_session.get(f'{LOCAL_URL}/controlnet/version', timeout=10)
        if response.status_code == 200:
            version_info = response.json()
            print(f"ControlNet version: {version_info}")
            return True
        else:
            print(f"ControlNet not available: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking ControlNet: {e}")
        return False

def get_controlnet_models():
    """Get available ControlNet models."""
    try:
        response = automatic_session.get(f'{LOCAL_URL}/controlnet/model_list', timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"Available ControlNet models: {models}")
            return models
        else:
            print(f"Failed to get ControlNet models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error getting ControlNet models: {e}")
        return []

def run_inference(inference_request):
    """Runs inference with the provided payload."""
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
        
        # Check if ControlNet is available
        if not check_controlnet_available():
            print("Warning: ControlNet not available, IP-Adapter will be skipped")
        else:
            # Get available models
            available_models = get_controlnet_models()
            
            # Find IP-Adapter model
            ip_adapter_model = None
            for model in available_models:
                if 'ip-adapter' in model.lower() and 'sdxl' in model.lower():
                    ip_adapter_model = model
                    break
            
            if not ip_adapter_model:
                print("Warning: No IP-Adapter SDXL model found. Available models:", available_models)
                ip_adapter_model = "ip-adapter_sdxl [7d943a46]"  # Default fallback
            
            print(f"Using IP-Adapter model: {ip_adapter_model}")
            
            # Set up ControlNet args for IP-Adapter
            controlnet_args = {
                "args": [
                    {
                        "enabled": True,
                        "image": inference_request['ip_adapter_image_b64'],
                        "module": "ip-adapter_clip_sdxl",
                        "model": ip_adapter_model,
                        "weight": inference_request.get("ip_adapter_weight", 0.6),
                        "resize_mode": 1,
                        "lowvram": False,
                        "processor_res": 512,
                        "threshold_a": 0.5,
                        "threshold_b": 0.5,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "pixel_perfect": False,
                        "control_mode": 0
                    }
                ]
            }
            
            if "alwayson_scripts" not in inference_request:
                inference_request["alwayson_scripts"] = {}
            
            inference_request["alwayson_scripts"]["controlnet"] = controlnet_args
        
        # Clean up the original IP-Adapter fields
        del inference_request['ip_adapter_image_b64']
        if 'ip_adapter_weight' in inference_request:
            del inference_request['ip_adapter_weight']
    
    # 6. Send request to A1111
    print("Sending request to A1111...")
    try:
        response = automatic_session.post(
            url=f'{LOCAL_URL}/txt2img', 
            json=inference_request, 
            timeout=600
        )
        
        if response.status_code != 200:
            error_msg = f"A1111 API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return {"error": error_msg}
        
        result = response.json()
        print("A1111 request completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Error calling A1111 API: {str(e)}"
        print(error_msg)
        return {"error": error_msg}

# --- RUNPOD HANDLER ---
def handler(event):
    """Main function called by RunPod to process a job."""
    print(f"=== RunPod Job Started ===")
    print(f"Event keys: {list(event.keys()) if event else 'None'}")
    
    try:
        # Validate input
        if not event or "input" not in event:
            return {"error": "No input provided in event"}
        
        input_data = event["input"]
        print(f"Input data keys: {list(input_data.keys())}")
        
        # Run inference
        print("Starting inference...")
        json_output = run_inference(input_data)
        
        # Check for errors
        if "error" in json_output:
            print(f"Inference failed: {json_output['error']}")
            return json_output
        
        # Run face detection only if we're not using IP-Adapter input
        if "ip_adapter_image_b64" not in input_data and "images" in json_output:
            print("Running face detection on generated image...")
            try:
                image_bytes = base64.b64decode(json_output['images'][0])
                detected_faces = detect_and_save_faces(image_bytes)
                json_output['detected_faces'] = detected_faces
                print(f"Face detection completed. Found {len(detected_faces)} faces.")
            except Exception as e:
                print(f"Face detection failed: {e}")
                json_output['detected_faces'] = []
        
        print("=== RunPod Job Completed Successfully ===")
        return json_output
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(f"=== RunPod Job Failed ===")
        print(error_msg)
        return {"error": error_msg}
    
    finally:
        # Signal shutdown after job completion
        print("Signaling worker shutdown...")
        shutdown_flag.set()

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    print("=== RunPod Worker Starting ===")
    
    try:
        print("Starting A1111 server...")
        a1111_process = subprocess.Popen(
            A1111_COMMAND, 
            preexec_fn=os.setsid,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        print("Waiting for A1111 service to be ready...")
        if wait_for_service(url=f'{LOCAL_URL}/progress', max_wait=300):
            print("A1111 service is ready!")
            
            # Give extensions time to load
            print("Waiting for extensions to load...")
            time.sleep(10)
            
            # Check ControlNet availability
            check_controlnet_available()
            get_controlnet_models()
            
            print("Starting RunPod serverless handler...")
            runpod.serverless.start({"handler": handler})
        else:
            print("Failed to start A1111 service")
            sys.exit(1)

        # Wait for shutdown signal
        shutdown_flag.wait()
        print("Shutdown signal received")

    except Exception as e:
        print(f"Fatal error in main process: {e}")
        shutdown_flag.set()
        sys.exit(1)
    
    finally:
        # Clean shutdown
        if a1111_process and a1111_process.poll() is None:
            print("Terminating A1111 process...")
            try:
                os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
                a1111_process.wait(timeout=30)
            except:
                print("Force killing A1111 process...")
                os.killpg(os.getpgid(a1111_process.pid), signal.SIGKILL)
        
        print("=== RunPod Worker Shutdown Complete ===")