"""
Autoencoder model definitions and architectures.

This module contains PyTorch implementations of various autoencoder
architectures including basic autoencoders, variational autoencoders,
and convolutional autoencoders.
"""

# Import core model classes
try:
    from .base_autoencoder import BaseAutoencoder
    from .vanilla_autoencoder import VanillaAutoencoder
    from .conv_autoencoder import ConvAutoencoder
    from .variational_autoencoder import VariationalAutoencoder
    from .training import AutoencoderTrainer
    
    __all__ = [
        "BaseAutoencoder",
        "VanillaAutoencoder", 
        "ConvAutoencoder",
        "VariationalAutoencoder",
        "AutoencoderTrainer",
    ]
except ImportError:
    # Graceful handling during development
    __all__ = []

# Module metadata
__version__ = "0.1.0"
