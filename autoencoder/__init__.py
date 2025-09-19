"""
Autoencoder package for MNIST data processing and neural network training.

This package provides utilities for loading MNIST binary data files,
converting them to PyTorch tensors, creating DataLoaders, and building
autoencoder models for training and inference.
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

from .autoencoder_model import (
    Encoder,
    Decoder,
    AutoencoderMLP,
    create_mnist_autoencoder
)

from .model_utils import (
    xavier_uniform_init,
    xavier_normal_init,
    kaiming_uniform_init,
    kaiming_normal_init,
    get_activation_function,
    get_initialization_function,
    count_parameters,
    get_model_summary,
    print_model_summary,
    validate_model_config,
    move_model_to_device
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
    "get_test_dataloader",
    
    # Model classes
    "Encoder",
    "Decoder", 
    "AutoencoderMLP",
    "create_mnist_autoencoder",
    
    # Model utilities
    "xavier_uniform_init",
    "xavier_normal_init",
    "kaiming_uniform_init",
    "kaiming_normal_init",
    "get_activation_function",
    "get_initialization_function",
    "count_parameters",
    "get_model_summary",
    "print_model_summary",
    "validate_model_config",
    "move_model_to_device"
]
