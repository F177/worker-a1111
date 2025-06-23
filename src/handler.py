import time
import runpod
from runpod.serverless.utils import rp_upload
import requests
from requests.adapters import HTTPAdapter, Retry
import base64
import tempfile
import os

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    """
    Check if the service is ready to receive requests.
    """
    retries = 0
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            retries += 1
            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)
        time.sleep(0.2)

def run_inference(inference_request):
    """
    Run inference on a request.
    """
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()

def process_images(images, job_id):
    """
    Process generated images - either upload to S3 or return as base64.
    
    Args:
        images (list): List of base64 encoded images from A1111
        job_id (str): The job ID for S3 upload naming
    
    Returns:
        list: List of processed image data with metadata
    """
    processed_images = []
    
    if not images:
        return processed_images
    
    print(f"worker-a1111 - Processing {len(images)} generated image(s)...")
    
    for idx, base64_image in enumerate(images):
        try:
            filename = f"generated_image_{idx + 1}.png"
            
            # Check if S3 upload is configured
            if os.environ.get("BUCKET_ENDPOINT_URL"):
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(base64_image)
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file.write(image_bytes)
                        temp_file_path = temp_file.name
                    
                    print(f"worker-a1111 - Wrote image bytes to temporary file: {temp_file_path}")
                    
                    # Upload to S3
                    print(f"worker-a1111 - Uploading {filename} to S3...")
                    s3_url = rp_upload.upload_image(job_id, temp_file_path)
                    
                    # Clean up temp file
                    os.remove(temp_file_path)
                    
                    print(f"worker-a1111 - Uploaded {filename} to S3: {s3_url}")
                    
                    # Add to processed images with S3 URL
                    processed_images.append({
                        "filename": filename,
                        "type": "s3_url",
                        "data": s3_url
                    })
                    
                except Exception as e:
                    error_msg = f"Error uploading {filename} to S3: {e}"
                    print(f"worker-a1111 - {error_msg}")
                    
                    # Clean up temp file if it exists
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except OSError as rm_err:
                            print(f"worker-a1111 - Error removing temp file {temp_file_path}: {rm_err}")
                    
                    # Fallback to base64 if S3 upload fails
                    processed_images.append({
                        "filename": filename,
                        "type": "base64",
                        "data": base64_image,
                        "error": error_msg
                    })
            else:
                # Return as base64 if S3 is not configured
                processed_images.append({
                    "filename": filename,
                    "type": "base64",
                    "data": base64_image
                })
                print(f"worker-a1111 - Returning {filename} as base64")
                
        except Exception as e:
            error_msg = f"Error processing image {idx + 1}: {e}"
            print(f"worker-a1111 - {error_msg}")
            processed_images.append({
                "filename": f"error_image_{idx + 1}.png",
                "type": "error",
                "error": error_msg
            })
    
    return processed_images

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    """
    This is the handler function that will be called by the serverless.
    """
    job_id = event.get("id", "unknown_job")
    
    try:
        # Run inference
        print(f"worker-a1111 - Starting inference for job {job_id}")
        inference_result = run_inference(event["input"])
        
        # Extract images from the response
        images = inference_result.get("images", [])
        
        if not images:
            print(f"worker-a1111 - No images generated for job {job_id}")
            return {
                "status": "success_no_images",
                "images": [],
                "info": inference_result.get("info", {}),
                "parameters": inference_result.get("parameters", {})
            }
        
        # Process images (S3 upload or base64)
        processed_images = process_images(images, job_id)
        
        # Prepare response
        response = {
            "images": processed_images,
            "info": inference_result.get("info", {}),
            "parameters": inference_result.get("parameters", {})
        }
        
        # Add any errors if present
        errors = [img for img in processed_images if img.get("type") == "error"]
        if errors:
            response["errors"] = [img.get("error") for img in errors]
            print(f"worker-a1111 - Job {job_id} completed with errors")
        else:
            print(f"worker-a1111 - Job {job_id} completed successfully with {len(processed_images)} image(s)")
        
        return response
        
    except Exception as e:
        error_msg = f"Unexpected error in handler: {e}"
        print(f"worker-a1111 - {error_msg}")
        return {
            "error": error_msg,
            "job_id": job_id
        }

if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})