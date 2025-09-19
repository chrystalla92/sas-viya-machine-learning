"""
Autoencoder package for MNIST data processing and neural network training.

This package provides utilities for loading MNIST binary data files,
converting them to PyTorch tensors, and creating DataLoaders for training
autoencoder models.
"""

from .data_loader import (
    read_idx3_ubyte,
    read_idx1_ubyte,
    midrange_standardize,
    load_mnist_data,
    load_mnist_training_data,
    load_mnist_test_data
)

from .datasets import (
    MNISTAutoencoderDataset,
    MNISTDatasetFromFiles,
    create_mnist_dataloaders,
    create_simple_dataloader,
    get_train_dataloader,
    get_test_dataloader
)

__version__ = "1.0.0"
__all__ = [
    # Data loading functions
    "read_idx3_ubyte",
    "read_idx1_ubyte", 
    "midrange_standardize",
    "load_mnist_data",
    "load_mnist_training_data",
    "load_mnist_test_data",
    
    # Dataset classes
    "MNISTAutoencoderDataset",
    "MNISTDatasetFromFiles",
    
    # DataLoader utilities
    "create_mnist_dataloaders",
    "create_simple_dataloader", 
    "get_train_dataloader",
    "get_test_dataloader"
]
