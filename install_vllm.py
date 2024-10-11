import subprocess
import sys

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')

def is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_vllm():
    if is_cuda_available():
        print("CUDA is available. Installing vLLM with CUDA support...")
        vllm_version = "0.6.1.post1"
        python_version = "310"
        install_command = f"""
        pip3 install https://github.com/vllm-project/vllm/releases/download/v{vllm_version}/vllm-{vllm_version}+cu118-cp{python_version}-cp{python_version}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
        """
    else:
        print("CUDA is not available. Installing vLLM without CUDA support...")
        install_command = "pip3 install vllm"

    output, error = run_command(install_command)
    print(output)
    if error:
        print("Error occurred during installation:")
        print(error)

if __name__ == "__main__":
    install_vllm()