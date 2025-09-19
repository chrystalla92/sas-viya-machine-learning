"""
Test suite for SAS-Python autoencoder compatibility.

This package contains unit tests to verify that the Python autoencoder implementation
follows the SAS autoencoder implementation based on the comparison table in
SAS_Autoencoder_Properties.md.

Test Modules:
- test_architecture_compatibility: Tests architecture properties (layers, activations, etc.)
- test_preprocessing_compatibility: Tests data preprocessing and transforms
- test_training_compatibility: Tests training configuration and procedures

The tests are designed to pass when the Python implementation properly follows
the SAS implementation specifications.
"""

__version__ = "1.0.0"
__author__ = "SAS Autoencoder Compatibility Tests"

# Import test modules for easy access
from . import test_architecture_compatibility
from . import test_preprocessing_compatibility
from . import test_training_compatibility

__all__ = [
    "test_architecture_compatibility",
    "test_preprocessing_compatibility",
    "test_training_compatibility"
]