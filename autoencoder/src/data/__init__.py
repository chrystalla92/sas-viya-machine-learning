"""
Data processing and loading modules for autoencoders.

This module provides utilities for loading, preprocessing, and managing
datasets for autoencoder training and evaluation.
"""

# Import key data processing components
try:
    from .mnist_loader import MNISTDataLoader
    from .data_transforms import get_transforms
    from .dataset_utils import create_data_loaders
    
    __all__ = [
        "MNISTDataLoader",
        "get_transforms", 
        "create_data_loaders",
    ]
except ImportError:
    # Graceful handling during development
    __all__ = []

# Module metadata
__version__ = "0.1.0"
