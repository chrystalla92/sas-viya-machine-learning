"""
SAS Viya Autoencoder Python Package

This package provides PyTorch-based implementations of autoencoders
for efficient encoding/decoding and feature extraction.
"""

__version__ = "0.1.0"
__author__ = "SAS Institute"
__email__ = "support@sas.com"

# Import main modules for easier access
try:
    from .models import *
    from .data import *
    from . import utils
except ImportError:
    # Graceful handling during development/setup
    pass

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
