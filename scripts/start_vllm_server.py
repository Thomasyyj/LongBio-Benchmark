import argparse
import torch
import os
import subprocess
from loguru import logger
from omegaconf import OmegaConf

def start_vllm_server(args):
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_params = OmegaConf.load(args.config_path)['model_params']
    model = args.model_path.split('/')[-1]
    base_port = model_params['models_port'].get(model)
    if base_port is None:
        base_port = int(args.port_name)  # ensure base_port is an integer

    n_total=torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        # Return the list of device IDs as specified in the environment variable
        available_devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
    else:
        # Otherwise, generate a list from 0 to the number of CUDA devices detected
        available_devices = [str(i) for i in range(n_total)]
    logger.info(f"Available devices: {available_devices}")

    n_parallel = args.data_parallel_num
    n_per_process = n_total // n_parallel
    assert n_total % n_parallel == 0, f"{n_total} gpus is not divisible by {n_parallel}"

    processes = []
    for i in range(n_parallel):
        start_idx = i * n_per_process
        end_idx = start_idx + n_per_process
        assigned_gpus = available_devices[start_idx:end_idx]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned_gpus)

        port = base_port + i*100  # assign a unique port to each service
        # Construct the CLI command for vLLM for this instance
        cmd = [
            "vllm", "serve", f"{args.model_path}",
            f"--port={port}",
            f"--tensor-parallel-size={args.n_gpu//n_parallel}",
            f"--max-model-len={model_params['max_model_len']}",
            f"--gpu-memory-utilization={model_params['gpu_memory_utilization']}",
            f"--dtype={model_params['dtype']}",
        ]

        if model_params.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        if model_params.get("enable_chunked_prefill"):
            cmd.append("--enable-chunked-prefill")
        if model_params.get("enforce_eager"):
            cmd.append("--enforce-eager")

        # Launch the vLLM server instance
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Optionally, wait for all launched processes to finish
    for p in processes:
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference experiments")
    parser.add_argument('-m', '--model_path', type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument('-c', '--config_path', type=str, default="scripts/config/params.yaml")
    parser.add_argument('-ng', '--n_gpu', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=2024)
    parser.add_argument('-pn', '--port_name', type=int, default=2025)
    parser.add_argument('-dp', '--data_parallel_num', type=int, default=1)
    args = parser.parse_args()
    logger.info(f"Start the vLLM server with args:{vars(args)}")
    start_vllm_server(args)
