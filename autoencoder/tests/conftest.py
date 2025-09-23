"""
Pytest configuration and shared fixtures for autoencoder tests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def sample_mnist_batch():
    """Create a sample MNIST-like batch for testing."""
    batch_size = 32
    return torch.randn(batch_size, 1, 28, 28)

@pytest.fixture
def sample_latent_vector():
    """Create a sample latent vector for testing."""
    batch_size = 32
    latent_dim = 128
    return torch.randn(batch_size, latent_dim)

@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = iter([
        (torch.randn(32, 1, 28, 28), torch.zeros(32, dtype=torch.long))
        for _ in range(10)
    ])
    mock_loader.__len__.return_value = 10
    return mock_loader

@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary path for saving/loading models during tests."""
    return tmp_path / "test_model.pth"
