"""
MNIST Autoencoder Package

This package provides a comprehensive PyTorch implementation of an autoencoder
for MNIST data that matches SAS neural network behavior.

Modules:
- model: Core autoencoder architecture
- training: Training framework and utilities
- mnist_data: Data loading and preprocessing
- data_utils: Data manipulation utilities
- evaluation: Model evaluation and metrics
- checkpoints: Model checkpointing utilities
"""

from .model import MNISTAutoencoder, create_sas_compatible_autoencoder

__version__ = "1.0.0"
__all__ = [
    'MNISTAutoencoder',
    'create_sas_compatible_autoencoder'
]
