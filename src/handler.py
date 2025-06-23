import time
import runpod
import requests
import subprocess
import os
import signal
import sys
import threading
from requests.adapters import HTTPAdapter, Retry

# --- Importações para funcionalidade S3 ---
from runpod.serverless.utils import rp_upload
import base64
from io import BytesIO
import tempfile

# --- CONFIGURAÇÃO ---
LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"
A1111_COMMAND = [
    "python", "/stable-diffusion-webui/launch.py",
    "--xformers", "--no-half-vae", "--api", "--nowebui", "--port", "3000",
    "--skip-install", "--skip-version-check", "--disable-safe-unpickle",
    "--ckpt", "/model.safetensors"
]
a1111_process = None
shutdown_flag = threading.Event() # Usaremos um "evento" para sinalizar o desligamento

# --- DEFINIÇÃO MANUAL DAS VARIÁVEIS DE AMBIENTE S3 NO CÓDIGO (NÃO RECOMENDADO) ---
# --- SUBSTITUA OS VALORES ABAIXO PELOS SEUS VALORES REAIS DO BUCKET S3 ---
os.environ["BUCKET_ENDPOINT_URL"] = "https://realismo-runpod-images.s3.us-east-2.amazonaws.com"
os.environ["BUCKET_NAME"] = "realismo-runpod-images"
os.environ["BUCKET_ACCESS_KEY_ID"] = "SUA_CHAVE_DE_ACESSO_AWS" # <-- SUBSTITUA PELA SUA CHAVE REAL
os.environ["BUCKET_SECRET_ACCESS_KEY"] = "SUA_CHAVE_SECRETA_AWS" # <-- SUBSTITUA PELA SUA CHAVE REAL
# --- FIM DA DEFINIÇÃO MANUAL ---


# --- FUNÇÕES DE REDE ---
automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    """Espera o serviço A1111 estar pronto."""
    while True:
        try:
            requests.get(url, timeout=30)
            return
        except requests.exceptions.RequestException:
            print("Serviço A1111 ainda não está pronto, aguardando...")
            time.sleep(2)

def run_inference(inference_request):
    """Roda a inferência com timeout aumentado."""
    response = automatic_session.post(f'{LOCAL_URL}/txt2img', json=inference_request, timeout=900)
    response.raise_for_status()
    return response.json()

# --- FUNÇÃO PRINCIPAL DO HANDLER ---
def handler(job):
    """
    Processa o job, retorna o resultado e sinaliza para o desligamento.
    Adiciona a funcionalidade de upload para S3.
    """
    job_input = job["input"]
    job_id = job["id"] # Obter job_id do objeto job

    output_data = []
    errors = []

    try:
        print("Job recebido. Iniciando a geração...")
        json_output = run_inference(job_input) # Use job_input aqui

        # Processar as imagens geradas
        if "images" in json_output:
            print(f"Geração finalizada. Processando {len(json_output['images'])} imagem(ns)...")
            for i, img_b64 in enumerate(json_output["images"]):
                try:
                    # A imagem já vem em base64, mas pode ter o prefixo 'data:image/png;base64,'
                    # ComfyUI handler lida com isso, vamos replicar a lógica.
                    if "," in img_b64:
                        base64_data = img_b64.split(",", 1)[1]
                    else:
                        base64_data = img_b64

                    image_bytes = base64.b64decode(base64_data)
                    filename = f"image_{job_id}_{i}.png" # Nome único para o arquivo

                    # A verificação os.environ.get("BUCKET_ENDPOINT_URL") ainda é válida,
                    # mas agora esperamos que a variável esteja sempre definida aqui.
                    if os.environ.get("BUCKET_ENDPOINT_URL"):
                        try:
                            with tempfile.NamedTemporaryFile(
                                suffix=".png", delete=False
                            ) as temp_file:
                                temp_file.write(image_bytes)
                                temp_file_path = temp_file.name
                            print(
                                f"worker-a1111 - Wrote image bytes to temporary file: {temp_file_path}"
                            )

                            print(f"worker-a1111 - Uploading {filename} to S3...")
                            s3_url = rp_upload.upload_image(job_id, temp_file_path)
                            os.remove(temp_file_path)  # Limpa o arquivo temporário
                            print(
                                f"worker-a1111 - Uploaded {filename} to S3: {s3_url}"
                            )
                            output_data.append(
                                {
                                    "filename": filename,
                                    "type": "s3_url",
                                    "data": s3_url,
                                }
                            )
                        except Exception as e:
                            error_msg = f"Error uploading {filename} to S3: {e}"
                            print(f"worker-a1111 - {error_msg}")
                            errors.append(error_msg)
                            if "temp_file_path" in locals() and os.path.exists(
                                temp_file_path
                            ):
                                try:
                                    os.remove(temp_file_path)
                                except OSError as rm_err:
                                    print(
                                        f"worker-a1111 - Error removing temp file {temp_file_path}: {rm_err}"
                                    )
                    else:
                        # Se por algum motivo BUCKET_ENDPOINT_URL não estiver definido, retorna como base64
                        output_data.append(
                            {
                                "filename": filename,
                                "type": "base64",
                                "data": base64_data, # Use o base64_data já limpo
                            }
                        )
                        print(f"worker-a1111 - Encoded {filename} as base64")

                except base64.binascii.Error as e:
                    error_msg = f"Error decoding base64 for image {i}: {e}"
                    print(f"worker-a1111 - {error_msg}")
                    errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error processing image {i}: {e}"
                    print(f"worker-a1111 - {error_msg}")
                    errors.append(error_msg)
            print("Processamento de imagens concluído.")
        else:
            print("Nenhuma imagem 'images' encontrada na resposta da inferência.")

    except requests.RequestException as e:
        print(f"worker-a1111 - HTTP Request Error during inference: {e}")
        errors.append(f"HTTP communication error with A1111: {e}")
    except Exception as e:
        print(f"worker-a1111 - Unexpected Handler Error: {e}")
        errors.append(f"An unexpected error occurred: {e}")
    finally:
        print("Sinalizando para o desligamento do worker...")
        shutdown_flag.set()

    final_result = {}
    if output_data:
        final_result["images"] = output_data

    if errors:
        final_result["errors"] = errors
        print(f"worker-a1111 - Job completed with errors/warnings: {errors}")
        # Se houver erros e nenhuma imagem de saída bem-sucedida, retorne um erro primário.
        if not output_data:
            print(f"worker-a1111 - Job failed with no output images.")
            return {
                "error": "Job processing failed",
                "details": errors,
            }
    elif not output_data:
        print(
            f"worker-a1111 - Job completed successfully, but the workflow produced no images."
        )
        final_result["status"] = "success_no_images"
        final_result["images"] = []


    print(f"worker-a1111 - Job completed. Returning {len(output_data)} image(s).")
    return final_result


# --- PONTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    try:
        print("Iniciando o servidor A1111 em segundo plano...")
        a1111_process = subprocess.Popen(A1111_COMMAND, preexec_fn=os.setsid)

        wait_for_service(url=f'{LOCAL_URL}/sd-models')

        print("Iniciando o handler do RunPod...")
        runpod.serverless.start({"handler": handler})

        print("Loop do RunPod terminado. Aguardando a flag de desligamento...")
        shutdown_flag.wait()

    except Exception as e:
        print(f"Um erro fatal ocorreu: {e}")
    finally:
        if a1111_process:
            print("Limpando o processo A1111 na saída final...")
            os.killpg(os.getpgid(a1111_process.pid), signal.SIGTERM)

        print("Worker encerrado.")