import runpod
import time
import os

# Um handler de teste que apenas imprime o job e retorna sucesso.
def handler(job):
    print("--- Test handler received a job ---")
    print(job)
    time.sleep(2) # Simula algum trabalho
    return {"message": "Test worker is alive and responding!"}

# Inicia o servidor serverless do RunPod
print(f"--- Starting a minimal test worker (PID: {os.getpid()}) ---")
runpod.serverless.start({"handler": handler})