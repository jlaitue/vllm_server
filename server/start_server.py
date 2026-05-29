import os
import subprocess

model_cache = "/scratch/ljl5178/cache/HuggingFace/"
compile_cache = "/scratch/ljl5178/cache/vllm"

env = os.environ.copy()
env["HF_HOME"] = model_cache
env["HUGGINGFACE_HUB_CACHE"] = f"{model_cache}/hub"

env["CUDA_HOME"] = "/share/apps/NYUAD5/cuda/12.2.0"
env["CUDA_PATH"] = env["CUDA_HOME"]

env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
env["LD_LIBRARY_PATH"] = (
    f"{env['CUDA_HOME']}/lib64:"
    f"{env.get('CONDA_PREFIX', '')}/lib:"
    f"{env.get('LD_LIBRARY_PATH', '')}"
)

env["VLLM_CACHE_ROOT"] = compile_cache
env["TORCHINDUCTOR_CACHE_DIR"] = f"{compile_cache}/torchinductor"
env["TRITON_CACHE_DIR"] = f"{compile_cache}/triton"

cmd = [
    "vllm",
    "serve",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--trust-remote-code",
    "--max-model-len", "8192",
]

print("CUDA_HOME:", env["CUDA_HOME"])
print("PATH:", env["PATH"].split(":")[0])
print("LD_LIBRARY_PATH:", env["LD_LIBRARY_PATH"].split(":")[0])

subprocess.run(cmd, env=env, check=True)