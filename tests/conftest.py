"""
Pytest configuration and shared fixtures for mnist_autoencoder tests.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_mnist_data():
    """
    Generate sample MNIST-like data for testing.
    
    Returns:
        tuple: (data, labels) where data is (N, 28, 28) and labels is (N,)
    """
    batch_size = 10
    data = torch.rand(batch_size, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return data, labels


@pytest.fixture
def sample_batch():
    """
    Generate a small batch of normalized data for testing.
    
    Returns:
        torch.Tensor: Normalized batch data (N, 1, 28, 28)
    """
    batch_size = 4
    data = torch.rand(batch_size, 1, 28, 28)
    return data


@pytest.fixture(scope="session")
def device():
    """
    Get the appropriate device for testing.
    
    Returns:
        torch.device: CUDA if available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_model_path(tmp_path):
    """
    Provide a temporary path for saving/loading models during tests.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Path: Temporary file path for model storage
    """
    return tmp_path / "test_model.pth"
