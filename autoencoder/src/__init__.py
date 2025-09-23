"""
SAS Viya Autoencoder Python Package

This package provides PyTorch-based implementations of autoencoders
for efficient encoding/decoding, feature extraction, and model inference.
"""

__version__ = "0.1.0"
__author__ = "SAS Institute"
__email__ = "support@sas.com"

# Import main modules for easier access
try:
    from .models import *
    from .data import *
    from . import utils
    from .inference import AutoencoderInference, create_inference_pipeline, tensor_to_flat, flat_to_images
    from .training import AutoencoderTrainer, create_trainer
except ImportError:
    # Graceful handling during development/setup
    pass

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Core model components
    "Autoencoder",
    "create_autoencoder",
    # Training components
    "AutoencoderTrainer",
    "create_trainer",
    # Inference/Scoring components
    "AutoencoderInference",
    "create_inference_pipeline",
    # Utility functions
    "tensor_to_flat",
    "flat_to_images",
    # Preprocessing utilities
    "MidrangeScaler",
    "normalize_for_autoencoder",
]
