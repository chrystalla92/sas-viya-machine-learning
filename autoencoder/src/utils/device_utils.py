"""
Device detection and CUDA utilities for PyTorch operations.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device: The selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def setup_cuda() -> dict:
    """
    Setup CUDA environment and return device information.
    
    Returns:
        dict: Information about available devices
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "memory_allocated": 0,
        "memory_reserved": 0,
    }
    
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated"] = torch.cuda.memory_allocated(0)
        info["memory_reserved"] = torch.cuda.memory_reserved(0)
        
        # Set some optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        logger.info(f"CUDA setup complete. Using {info['device_name']}")
    else:
        logger.info("CUDA not available. Using CPU.")
    
    return info


def print_device_info():
    """Print detailed information about available devices."""
    info = setup_cuda()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Device Count: {info['device_count']}")
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"Memory Allocated: {info['memory_allocated'] / 1024**2:.1f} MB")
        print(f"Memory Reserved: {info['memory_reserved'] / 1024**2:.1f} MB")
    else:
        print("Using CPU for computations")
    
    print("=" * 50)


if __name__ == "__main__":
    print_device_info()
