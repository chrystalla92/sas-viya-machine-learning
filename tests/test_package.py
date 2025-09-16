"""
Basic tests to ensure the package structure is working correctly.
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import mnist_autoencoder
    assert mnist_autoencoder.__version__ == "0.1.0"


def test_submodules_import():
    """Test that all submodules can be imported."""
    import mnist_autoencoder.data
    import mnist_autoencoder.models
    import mnist_autoencoder.training
    import mnist_autoencoder.evaluation
    
    # Test that submodules are accessible through the main package
    assert hasattr(mnist_autoencoder, 'data')
    assert hasattr(mnist_autoencoder, 'models')
    assert hasattr(mnist_autoencoder, 'training')
    assert hasattr(mnist_autoencoder, 'evaluation')


def test_package_metadata():
    """Test that package metadata is correctly set."""
    import mnist_autoencoder
    
    assert hasattr(mnist_autoencoder, '__version__')
    assert hasattr(mnist_autoencoder, '__author__')
    assert mnist_autoencoder.__author__ == "SAS Institute"


def test_package_all_attribute():
    """Test that __all__ is properly defined."""
    import mnist_autoencoder
    
    expected_all = ["data", "models", "training", "evaluation"]
    assert mnist_autoencoder.__all__ == expected_all
