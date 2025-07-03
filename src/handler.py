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
from safetensors.numpy import save_file

# --- CONFIGURATION ---
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/webui.py", "--xformers", "--no-half-vae", "--api",
    "--nowebui", "--port", "3000", "--skip-version-check", "--disable-safe-unpickle",
    "--no-hashing", "--opt-sdp-attention", "--no-download-sd-model",
    "--enable-insecure-extension-access", "--api-log", "--cors-allow-origins=*"
]

# --- S3 Client (for saving cropped faces) ---
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# --- Face Analysis Setup ---
face_analyzer = None
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Face analyzer initialized successfully")
except Exception as e:
    print(f"Warning: Face analyzer initialization failed: {e}")

# --- FACE MODEL SAVING ---
FACE_MODELS_PATH = "/stable-diffusion-webui/models/reactor/faces"
os.makedirs(FACE_MODELS_PATH, exist_ok=True)

def save_face_model(image_b64, job_id):
    if not face_analyzer:
        print("Face analyzer not available, cannot save face model.")
        return None
    try:
        print("Saving new face model...")
        image_bytes = base64.b64decode(image_b64)
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)
        if not faces:
            print("No faces found in the provided image for the model.")
            return None
        
        face = faces[0]
        embedding = face.normed_embedding
        face_model_data = {"embedding": embedding}
        
        model_name = f"job_{job_id}.safetensors"
        model_path = os.path.join(FACE_MODELS_PATH, model_name)
        save_file(face_model_data, model_path)
        
        print(f"Face model saved successfully as '{model_name}'")
        return model_name
    except Exception as e:
        print(f"Error saving face model: {e}")
        return None

def detect_and_save_faces(image_bytes):
    """Detects faces in an image, crops them, and uploads them to S3."""
    if not face_analyzer or not S3_BUCKET_NAME:
        print("Face analyzer or S3_BUCKET_NAME not available, skipping face detection.")
        return []
    # ... (rest of the function is the same as original)
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            return []
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)
        if not faces:
            return []
        detected_faces = []
        for i, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)
                padding = 20
                y1, y2 = max(0, bbox[1] - padding), min(bgr_img.shape[0], bbox[3] + padding)
                x1, x2 = max(0, bbox[0] - padding), min(bgr_img.shape[1], bbox[2] + padding)
                cropped_img = bgr_img[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.png', cropped_img)
                face_bytes = buffer.tobytes()
                face_id = f"f-{uuid.uuid4()}"
                s3_key = f"faces/{face_id}.png"
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=face_bytes, ContentType='image/png')
                detected_faces.append({"face_id": face_id, "face_index": i, "bbox": bbox.tolist()})
                print(f"Saved face {i} as {face_id}")
            except Exception as e:
                print(f"Error processing face {i}: {e}")
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
    # ... (function is the same as original)
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

def run_inference(inference_request, job_id):
    """Runs inference with the provided payload."""
    print(f"Starting inference with keys: {list(inference_request.keys())}")
    
    # --- Standard Payload Modifications ---
    lora_level = inference_request.get("lora_level", 0.6)
    inference_request["prompt"] = f"{inference_request.get('prompt', '')}, <lora:epiCRealnessRC1:{lora_level}>"
    inference_request["negative_prompt"] = f"{inference_request.get('negative_prompt', '')}, veryBadImageNegative_v1.3, FastNegativeV2"
    
    override_settings = {
        "sd_model_checkpoint": "ultimaterealismo.safetensors",
        "CLIP_stop_at_last_layers": inference_request.get("clip_skip", 1)
    }
    if "override_settings" not in inference_request:
        inference_request["override_settings"] = {}
    inference_request["override_settings"].update(override_settings)

    # --- ReActor Faceswap Logic ---
    if 'source_face_b64' in inference_request:
        model_name = save_face_model(inference_request["source_face_b64"], job_id)
        if model_name:
            face_strength = inference_request.get("face_strength", 0.8)
            
            # This is the full, corrected argument list based on reactor_faceswap.py
            reactor_args = [
                None, True, "0", "0", "inswapper_128.onnx", "CodeFormer", face_strength,
                False, "None", 1.0, 1.0, False, True, 2, 0, 0, False, 0.5, True,
                False, "CUDA", False, 1, model_name, None, [], False, False,
                0.5, 5, "tab_single"
            ]
            
            if "alwayson_scripts" not in inference_request:
                inference_request["alwayson_scripts"] = {}
            inference_request["alwayson_scripts"]["reactor"] = {"args": reactor_args}
            print(f"ReActor faceswap arguments added to payload, using model '{model_name}'")
        
        # Clean up keys
        del inference_request['source_face_b64']
        if 'face_strength' in inference_request:
            del inference_request['face_strength']
            
    # --- Send request to A1111 ---
    print("Sending request to A1111...")
    try:
        response = automatic_session.post(url=f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
        if response.status_code != 200:
            return {"error": f"A1111 API Error: {response.status_code} - {response.text}"}
        return response.json()
    except Exception as e:
        return {"error": f"Error calling A1111 API: {str(e)}"}

# --- RUNPOD HANDLER ---
def handler(event):
    print(f"=== RunPod Job Started: {event.get('id')} ===")
    try:
        if not event or "input" not in event:
            return {"error": "No input provided in event"}
        
        input_data = event["input"]
        job_id = event.get('id', 'temp_job')

        json_output = run_inference(input_data, job_id)
        
        if "error" in json_output:
            return json_output
        
        if "images" in json_output:
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces
        
        print(f"=== RunPod Job Completed: {job_id} ===")
        return json_output
        
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}
    finally:
        print("Signaling worker shutdown...")
        shutdown_flag.set()

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    print("=== RunPod Worker Starting ===")
    try:
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid, stdout=sys.stdout, stderr=sys.stderr)
        if wait_for_service(url=f'{LOCAL_URL}/progress', max_wait=300):
            print("A1111 service is ready! Starting RunPod handler...")
            runpod.serverless.start({"handler": handler})
        else:
            print("Failed to start A1111 service")
            sys.exit(1)

        shutdown_flag.wait()
    except Exception as e:
        print(f"Fatal error in main process: {e}")
    finally:
        if a1111_process and a1111_process.poll() is None:
            print("Terminating A1111 process...")
            try:
                os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
                a1111_process.wait(timeout=30)
            except:
                os.killpg(os.getpgid(a1111_process.pid), signal.SIGKILL)
        print("=== RunPod Worker Shutdown Complete ===")