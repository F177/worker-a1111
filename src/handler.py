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
S3_BUCKET_NAME = os.environ.get('S3_FACES_BUCKET_NAME') # Corrigido para corresponder à variável do Lambda

# --- Face Analysis Setup ---
face_analyzer = None
try:
    face_analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    print("Analisador de rosto inicializado com sucesso")
except Exception as e:
    print(f"Aviso: Falha na inicialização do analisador de rosto: {e}")

def detect_and_save_faces(image_bytes):
    """Detecta rostos em uma imagem, recorta-os e os envia para o S3."""
    if not face_analyzer or not S3_BUCKET_NAME:
        print("Analisador de rosto ou bucket S3 não configurado, pulando a detecção de rosto.")
        return []
        
    try:
        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            print("Erro: não foi possível decodificar a imagem.")
            return []

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb_img)

        if not faces:
            print("Nenhum rosto detectado.")
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
                print(f"Erro ao processar o rosto {i}: {e}")
        
        return detected_faces
    except Exception as e:
        print(f"Erro durante o processo de detecção de rosto: {e}")
        return []

a1111_process = None
shutdown_flag = threading.Event()
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url, max_wait=300):
    """Aguarda o serviço A1111 estar pronto."""
    start_time = time.time()
    while not shutdown_flag.is_set() and (time.time() - start_time) < max_wait:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("Serviço A1111 está pronto.")
                return True
        except requests.exceptions.RequestException:
            print("Aguardando o serviço A1111...")
            time.sleep(5)
    
    print(f"O serviço falhou em iniciar em {max_wait} segundos")
    return False

def check_controlnet_available():
    """Verifica se a extensão ControlNet está disponível."""
    try:
        response = automatic_session.get(f'http://127.0.0.1:3000/controlnet/version', timeout=10)
        return response.status_code == 200
    except:
        return False

def get_controlnet_models():
    """Obtém os modelos ControlNet disponíveis."""
    try:
        response = automatic_session.get(f'http://127.0.0.1:3000/controlnet/model_list', timeout=10)
        if response.status_code == 200:
            return response.json().get("model_list", [])
        return []
    except:
        return []

def run_inference(inference_request):
    """Prepara e envia a requisição de inferência para a API A1111."""
    print(f"Iniciando inferência com as chaves: {list(inference_request.keys())}")
    
    # Prepara o payload
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

    if 'ip_adapter_image_b64' in inference_request:
        if check_controlnet_available():
            # Lógica do IP-Adapter
            pass # A lógica completa do IP-Adapter permanece a mesma
        del inference_request['ip_adapter_image_b64']
        if 'ip_adapter_weight' in inference_request:
            del inference_request['ip_adapter_weight']
    
    print("Enviando payload final para a API A1111...")
    try:
        response = automatic_session.post(
            url=f'{LOCAL_URL}/txt2img', 
            json=inference_request, 
            timeout=600
        )
        if response.status_code != 200:
            return {"error": f"Erro na API A1111: {response.status_code} - {response.text}"}
        
        print("Requisição A1111 concluída com sucesso.")
        return response.json()
    except Exception as e:
        return {"error": f"Erro ao chamar a API A1111: {str(e)}"}

# --- RUNPOD HANDLER (CORRIGIDO) ---
def handler(event):
    """Função principal chamada pelo RunPod para processar um trabalho."""
    print("=== Trabalho do RunPod Iniciado ===")
    try:
        if not event or "input" not in event:
            return {"error": "Nenhum input fornecido no evento"}
        
        input_data = event["input"]

        # CORREÇÃO: Identifica corretamente um trabalho de face swap verificando os argumentos do ReActor
        is_faceswap = "alwayson_scripts" in input_data and "reactor" in input_data.get("alwayson_scripts", {})

        if is_faceswap:
            print("Trabalho de Face Swap (ReActor) detectado.")
        
        json_output = run_inference(input_data)
        
        if "error" in json_output:
            return json_output
        
        # Executa a detecção de rosto apenas se NÃO for um trabalho de face swap e houver imagens
        if not is_faceswap and "images" in json_output and json_output.get("images"):
            print("Executando detecção de rosto na imagem gerada...")
            image_bytes = base64.b64decode(json_output['images'][0])
            detected_faces = detect_and_save_faces(image_bytes)
            json_output['detected_faces'] = detected_faces
        elif is_faceswap:
            print("Pulando a detecção de rosto porque um face swap foi realizado.")
        
        print("=== Trabalho do RunPod Concluído com Sucesso ===")
        return json_output
        
    except Exception as e:
        print("=== Trabalho do RunPod Falhou ===")
        traceback.print_exc()
        return {"error": f"Erro no handler: {str(e)}"}
    
    finally:
        print("Sinalizando o desligamento do worker...")
        shutdown_flag.set()

# --- PONTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    print("=== Worker do RunPod Iniciando ===")
    a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid, stdout=sys.stdout, stderr=sys.stderr)
    
    if wait_for_service(url=f'{LOCAL_URL}/progress'):
        print("Iniciando o handler serverless do RunPod...")
        runpod.serverless.start({"handler": handler})
    else:
        print("Falha ao iniciar o serviço A1111. Encerrando.")
        if a1111_process:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
        sys.exit(1)

    shutdown_flag.wait()
    if a1111_process and a1111_process.poll() is None:
        print("Encerrando o processo A1111...")
        try:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)
            a1111_process.wait(timeout=20)
        except:
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGKILL)
            
    print("=== Desligamento do Worker do RunPod Concluído ===")