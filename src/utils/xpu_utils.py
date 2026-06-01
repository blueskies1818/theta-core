"""Intel XPU utilities for memory management and device selection."""

import os
import torch


def get_device() -> torch.device:
    """Get the best available device: XPU > CUDA > CPU."""
    if torch.xpu.is_available():
        return torch.device("xpu:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_name() -> str:
    """Get human-readable device name."""
    device = get_device()
    if device.type == "xpu":
        return torch.xpu.get_device_name(device.index or 0)
    elif device.type == "cuda":
        return torch.cuda.get_device_name(device.index or 0)
    return "CPU"


def print_device_info() -> None:
    """Print diagnostic information about the available device."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    VRAM: {props.total_mem / 1e9:.2f} GB")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {get_device()}")


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_vram_usage(
    num_params: int,
    batch_size: int = 1,
    seq_length: int = 1024,
    bytes_per_param: int = 2,  # bf16
) -> dict:
    """Estimate VRAM usage for a training step."""
    model_weight_bytes = num_params * bytes_per_param
    gradients_bytes = num_params * bytes_per_param
    optimizer_bytes = num_params * 8  # Adam: 2 * (momentum + variance) in fp32
    activation_factor = 4  # Conservative estimate
    activation_bytes = model_weight_bytes * activation_factor * (seq_length / 1024)

    total = model_weight_bytes + gradients_bytes + optimizer_bytes + activation_bytes

    return {
        "model_weights_gb": model_weight_bytes / 1e9,
        "gradients_gb": gradients_bytes / 1e9,
        "optimizer_states_gb": optimizer_bytes / 1e9,
        "activations_gb": activation_bytes / 1e9,
        "total_estimated_gb": total / 1e9,
    }
