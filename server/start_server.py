import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
parser.add_argument("--tensor-parallel-size", default="1")
parser.add_argument("--max-model-len", default="8192")
parser.add_argument("--port", default="8000")
args, extra_args = parser.parse_known_args()

model_cache = "/scratch/ljl5178/cache/HuggingFace/"
compile_cache = "/scratch/ljl5178/cache/vllm"

env = os.environ.copy()
env["HF_HOME"] = model_cache
env["HUGGINGFACE_HUB_CACHE"] = f"{model_cache}/hub"

env["CUDA_HOME"] = "/share/apps/NYUAD5/cuda/13.0.0"
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
    args.model,
    "--host", "0.0.0.0",
    "--port", args.port,
    "--trust-remote-code",
    "--max-model-len", args.max_model_len,
    "--tensor-parallel-size", args.tensor_parallel_size,
    *extra_args,
]

print("CUDA_HOME:", env["CUDA_HOME"])
print("PATH:", env["PATH"].split(":")[0])
print("LD_LIBRARY_PATH:", env["LD_LIBRARY_PATH"].split(":")[0])

subprocess.run(cmd, env=env, check=True)