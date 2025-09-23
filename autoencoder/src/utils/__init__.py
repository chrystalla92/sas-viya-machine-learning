"""
Utility functions and helpers for autoencoder implementations.

This module provides common utilities for visualization, metrics,
device detection, and other helper functions.
"""

# Import utility functions
try:
    from .device_utils import get_device, setup_cuda
    from .visualization import plot_reconstruction, plot_latent_space
    from .metrics import compute_reconstruction_loss, compute_metrics
    from .config import load_config, save_config
    
    __all__ = [
        "get_device",
        "setup_cuda", 
        "plot_reconstruction",
        "plot_latent_space",
        "compute_reconstruction_loss",
        "compute_metrics",
        "load_config",
        "save_config",
    ]
except ImportError:
    # Graceful handling during development
    __all__ = []

# Module metadata
__version__ = "0.1.0"
