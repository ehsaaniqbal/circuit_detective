from __future__ import annotations

import os
import platform
import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        print(f"missing_command: {exc}", flush=True)
        return
    print(f"returncode: {completed.returncode}", flush=True)
    if completed.stdout:
        print("stdout:", completed.stdout, sep="\n", flush=True)
    if completed.stderr:
        print("stderr:", completed.stderr, sep="\n", flush=True)


def main() -> None:
    print(f"python: {sys.version}", flush=True)
    print(f"platform: {platform.platform()}", flush=True)
    print(f"executable: {sys.executable}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES')}", flush=True)
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}", flush=True)
    run(["nvidia-smi"])

    import torch

    print(f"torch: {torch.__version__}", flush=True)
    print(f"torch.version.cuda: {torch.version.cuda}", flush=True)
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}", flush=True)
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"torch.cuda.current_device: {torch.cuda.current_device()}", flush=True)
        print(f"torch.cuda.device_name: {torch.cuda.get_device_name(0)}", flush=True)
        tensor = torch.randn(2, 2, device="cuda")
        print(f"cuda_tensor: {tensor.device} {tensor.sum().item():.6f}", flush=True)

        import bitsandbytes as bnb

        print(f"bitsandbytes: {bnb.__version__}", flush=True)
        parameter = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))
        optimizer = bnb.optim.AdamW8bit([parameter], lr=1e-3)
        loss = parameter.square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("bnb_optimizer_step: ok", flush=True)


if __name__ == "__main__":
    main()
