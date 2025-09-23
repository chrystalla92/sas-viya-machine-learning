"""
Data processing and loading modules for autoencoders.

This module provides utilities for loading, preprocessing, and managing
datasets for autoencoder training and evaluation.
"""

# Import key data processing components
try:
    from .mnist_loader import MNISTDataLoader, create_mnist_loader
    from .preprocessing import (
        MidrangeScaler,
        validate_tensor_shape,
        validate_pixel_range,
        flatten_images,
        unflatten_images,
        compute_data_statistics,
        normalize_for_autoencoder,
        preprocess_mnist_batch,
        mnist_to_sas_format
    )
    
    __all__ = [
        # Main loader class
        "MNISTDataLoader",
        "create_mnist_loader",
        # Preprocessing utilities
        "MidrangeScaler",
        "validate_tensor_shape",
        "validate_pixel_range",
        "flatten_images",
        "unflatten_images",
        "compute_data_statistics",
        "normalize_for_autoencoder",
        "preprocess_mnist_batch",
        "mnist_to_sas_format",
    ]
except ImportError as e:
    # Graceful handling during development
    print(f"Warning: Some data modules could not be imported: {e}")
    __all__ = []

# Module metadata
__version__ = "0.1.0"
