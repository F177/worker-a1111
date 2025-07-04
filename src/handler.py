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
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
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
    if not face_analyzer or not S3_FACES_BUCKET_NAME:
        print("Face analyzer or S3_FACES_BUCKET_NAME not available, skipping face detection.")
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
                    Bucket=S3_FACES_BUCKET_NAME, 
                    Key=s3_key, 
                    Body=buffer.tobytes(), 
                    ContentType='image/png'
                )
                detected_faces.append({"face_id": face_id, "face_index": i, "bbox": bbox.tolist()})
                print(f"Saved face {i} as {face_id}")
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

# <<< CORREÇÃO AQUI >>>
def check_reactor_availability():
    """Check if ReActor extension is available by checking the script list."""
    try:
        response = requests.get(f'{LOCAL_URL}/scripts', timeout=10)
        if response.status_code == 200:
            scripts_data = response.json()
            # The API returns a list of script filenames for txt2img and img2img
            txt2img_scripts = scripts_data.get('txt2img', [])
            # The script filename for ReActor is 'reactor_faceswap.py'
            reactor_available = "reactor_faceswap.py" in txt2img_scripts
            print(f"ReActor extension available: {reactor_available}")
            return reactor_available
        else:
            print(f"Failed to check for available scripts: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking ReActor availability: {e}")
        return False

def run_inference(inference_request):
    """ Correctly prepares and sends a request for faceswapping or standard generation. """
    print("Preparing inference payload...")
    if "alwayson_scripts" not in inference_request:
        inference_request["alwayson_scripts"] = {}

    # Handle faceswap via ReActor
    if 'source_face_b64' in inference_request and inference_request['source_face_b64']:
        print("Source face detected. Configuring ReActor for faceswap.")
        
        # Full argument list for ReActor based on its script definition
        reactor_args = [
            inference_request.get('source_face_b64'),  # 1. img
            True,                                      # 2. enable
            '0',                                       # 3. source_faces_index
            '0',                                       # 4. faces_index
            'inswapper_128.onnx',                      # 5. model
            'CodeFormer',                              # 6. face_restorer_name
            1,                                         # 7. face_restorer_visibility
            False,                                     # 8. restore_first
            'None',                                    # 9. upscaler_name
            1,                                         # 10. upscaler_scale
            1,                                         # 11. upscaler_visibility
            False,                                     # 12. swap_in_source
            True,                                      # 13. swap_in_generated
            1,                                         # 14. console_logging_level
            0,                                         # 15. gender_source
            0,                                         # 16. gender_target
            False,                                     # 17. save_original
            inference_request.get('face_strength', 0.8),# 18. codeformer_weight
            True,                                      # 19. source_hash_check
            False,                                     # 20. target_hash_check
            'CUDA',                                    # 21. device
            False,                                     # 22. mask_face
            0,                                         # 23. select_source
            'None',                                    # 24. face_model
            '',                                        # 25. source_folder
            None,                                      # 26. imgs
            False,                                     # 27. random_image
            False,                                     # 28. upscale_force
            0.6,                                       # 29. det_thresh
            0,                                         # 30. det_maxnum
            'tab_single',                              # 31. selected_tab
        ]
        inference_request["alwayson_scripts"]["reactor"] = {"args": reactor_args}
        
        # Clean up the original fields after adding them to the script args
        del inference_request['source_face_b64']
        if 'face_strength' in inference_request:
            del inference_request['face_strength']
    
    # Send the request to A1111
    print("Sending request to A1111 API...")
    try:
        response = automatic_session.post(f'{LOCAL_URL}/txt2img', json=inference_request, timeout=600)
        print(f"A1111 API response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"A1111 API error response: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling A1111 API: {e}")
        return {"error": str(e)}

def handler(event):
    """ Main RunPod handler function. """
    print("=== RunPod Job Started ===")
    try:
        input_data = event["input"]
        print(f"Input data keys: {list(input_data.keys())}")
        
        original_input = input_data.copy()
        is_faceswap = 'source_face_b64' in original_input and original_input['source_face_b64']
        print(f"Is faceswap request: {is_faceswap}")

        json_output = run_inference(input_data)
        
        if "error" in json_output:
            return json_output
        
        if not is_faceswap and "images" in json_output:
            print("Running face detection on generated image...")
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces
        else:
            print("Skipping face detection because a faceswap was performed or no image was generated.")
        
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
    
    if wait_for_service(url=f'{LOCAL_URL}/progress'):
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