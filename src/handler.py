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
    # This function remains the same and is used for non-faceswap jobs
    if not face_analyzer or not S3_FACES_BUCKET_NAME:
        return []
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None: return []
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)
        if not faces: return []
        detected_faces = []
        for i, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)
                padding = 20
                y1, y2 = max(0, bbox[1] - padding), min(bgr_img.shape[0], bbox[3] + padding)
                x1, x2 = max(0, bbox[0] - padding), min(bgr_img.shape[1], bbox[2] + padding)
                cropped_img = bgr_img[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.png', cropped_img)
                face_id = f"f-{uuid.uuid4()}"
                s3_key = f"faces/{face_id}.png"
                s3_client.put_object(
                    Bucket=S3_FACES_BUCKET_NAME, Key=s3_key, 
                    Body=buffer.tobytes(), ContentType='image/png'
                )
                detected_faces.append({"face_id": face_id, "face_index": i, "bbox": bbox.tolist()})
            except Exception as e:
                print(f"Error processing face {i}: {e}")
        return detected_faces
    except Exception as e:
        print(f"Error during face detection: {e}")
        return []

a1111_process = None
shutdown_flag = threading.Event()
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url, max_wait=300):
    start_time = time.time()
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("A1111 service is ready.")
                return True
        except requests.exceptions.RequestException:
            time.sleep(5)
    print(f"Service failed to start within {max_wait} seconds")
    return False

# <<< NEW, ROBUST run_inference FUNCTION >>>
def run_inference(inference_request):
    """
    Handles image generation and faceswapping using the more reliable
    External ReActor API endpoint.
    """
    is_faceswap = 'source_face_b64' in inference_request and inference_request['source_face_b64']
    
    # Step 1: Generate the base image.
    print("Step 1: Generating base image via /txt2img...")
    # Remove faceswap-specific fields before sending to txt2img
    source_face_b64 = inference_request.pop('source_face_b64', None)
    face_strength = inference_request.pop('face_strength', 0.8)

    try:
        response_txt2img = automatic_session.post(f'{LOCAL_URL}/sdapi/v1/txt2img', json=inference_request, timeout=600)
        response_txt2img.raise_for_status()
        result_txt2img = response_txt2img.json()
        print("Base image generated successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error calling /txt2img API: {e}")
        return {"error": f"Error generating base image: {str(e)}"}

    if not result_txt2img.get("images"):
        return {"error": "Base image generation failed, no images returned."}
        
    # If it's not a faceswap job, we are done.
    if not is_faceswap:
        return result_txt2img

    # Step 2: Perform the faceswap using the External ReActor API
    print("Step 2: Performing faceswap via /reactor/image...")
    target_image_b64 = result_txt2img["images"][0]
    
    reactor_payload = {
        "source_image": source_face_b64,
        "target_image": target_image_b64,
        "source_faces_index": [0],
        "face_index": [0],
        "upscaler": "None",
        "scale": 1,
        "upscale_visibility": 1,
        "face_restorer": "CodeFormer",
        "restorer_visibility": face_strength,
        "restore_first": True,
        "model": "inswapper_128.onnx",
        "device": "CPU" # Matches the device ReActor is running on
    }

    try:
        response_reactor = automatic_session.post(f'{LOCAL_URL}/reactor/image', json=reactor_payload, timeout=300)
        response_reactor.raise_for_status()
        result_reactor = response_reactor.json()
        print("Faceswap completed successfully.")
        
        # Replace the original image with the faceswapped one to maintain a consistent output format
        result_txt2img["images"] = [result_reactor.get("image")]
        return result_txt2img

    except requests.exceptions.RequestException as e:
        print(f"Error calling /reactor/image API: {e}")
        return {"error": f"Error during faceswap: {str(e)}"}

def handler(event):
    """ Main RunPod handler function. """
    print("=== RunPod Job Started ===")
    try:
        input_data = event["input"]
        is_faceswap = 'source_face_b64' in input_data and input_data.get('source_face_b64')

        print(f"Is faceswap request: {is_faceswap}")

        json_output = run_inference(input_data)
        
        if "error" in json_output:
            return json_output
        
        # We only run face detection if it was NOT a faceswap job
        if not is_faceswap and "images" in json_output:
            print("Running face detection on generated image...")
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces
        else:
            print("Skipping face detection because a faceswap was performed.")
        
        print("=== RunPod Job Completed Successfully ===")
        return json_output
    except Exception as e:
        print(f"Handler error: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        print("Signaling worker shutdown...")
        shutdown_flag.set()

if __name__ == "__main__":
    print("=== RunPod Worker Starting ===")
    a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid, stdout=sys.stdout, stderr=sys.stderr)
    
    if wait_for_service(url=f'{LOCAL_URL}/sdapi/v1/progress'):
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start A1111 service. Exiting.")
        if a1111_process:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        sys.exit(1)

    shutdown_flag.wait()
    if a1111_process and a1111_process.poll() is None:
        print("Terminating A1111 process...")
        os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
    print("=== RunPod Worker Shutdown Complete ===")