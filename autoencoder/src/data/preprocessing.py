"""
Preprocessing utilities for autoencoder data preparation.

This module provides preprocessing functions that match SAS behavior,
particularly for midrange scaling and data validation.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class MidrangeScaler:
    """
    Midrange scaler that transforms data to [-1,1] range.
    
    This matches SAS 'midrange' standardization behavior:
    - Finds min and max of the data
    - Scales to [-1,1] using: (value - min) / (max - min) * 2 - 1
    
    For MNIST pixel data (0-255), this becomes:
    - (pixel - 0) / (255 - 0) * 2 - 1 = pixel / 255 * 2 - 1
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize midrange scaler.
        
        Args:
            feature_range: Desired output range (min, max)
        """
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> 'MidrangeScaler':
        """
        Fit the scaler to data by computing min and max.
        
        Args:
            data: Input tensor to fit on
            
        Returns:
            Self for method chaining
        """
        self.data_min = data.min()
        self.data_max = data.max()
        self.fitted = True
        logger.debug(f"MidrangeScaler fitted: min={self.data_min}, max={self.data_max}")
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data to the specified range.
        
        Args:
            data: Input tensor to transform
            
        Returns:
            Transformed tensor in range [-1,1]
        """
        # For MNIST, we know the range is [0,1] after ToTensor()
        # So we can transform directly without fitting
        if not self.fitted:
            # For MNIST tensors (already normalized to [0,1])
            data_min, data_max = 0.0, 1.0
        else:
            data_min, data_max = self.data_min, self.data_max
        
        # Scale to [-1, 1]: (x - min) / (max - min) * 2 - 1
        # For [0,1] input: x * 2 - 1
        range_min, range_max = self.feature_range
        scaled = (data - data_min) / (data_max - data_min)
        transformed = scaled * (range_max - range_min) + range_min
        
        return transformed
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data back to original scale.
        
        Args:
            data: Scaled tensor to transform back
            
        Returns:
            Tensor in original scale
        """
        if not self.fitted:
            data_min, data_max = 0.0, 1.0
        else:
            data_min, data_max = self.data_min, self.data_max
            
        range_min, range_max = self.feature_range
        
        # Reverse the transformation
        normalized = (data - range_min) / (range_max - range_min)
        original = normalized * (data_max - data_min) + data_min
        
        return original


def validate_tensor_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[int, ...],
    tensor_name: str = "tensor"
) -> None:
    """
    Validate that a tensor has the expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        tensor_name: Name for error messages
        
    Raises:
        ValueError: If shape doesn't match
    """
    if tensor.shape != expected_shape:
        raise ValueError(
            f"{tensor_name} has shape {tensor.shape}, "
            f"but expected {expected_shape}"
        )


def validate_pixel_range(
    data: torch.Tensor,
    expected_range: Tuple[float, float] = (-1.0, 1.0),
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that pixel data is in the expected range.
    
    Args:
        data: Pixel data tensor
        expected_range: Expected (min, max) range
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if data is in expected range
        
    Raises:
        ValueError: If data is outside expected range
    """
    data_min = data.min().item()
    data_max = data.max().item()
    expected_min, expected_max = expected_range
    
    if data_min < expected_min - tolerance or data_max > expected_max + tolerance:
        raise ValueError(
            f"Data range [{data_min:.6f}, {data_max:.6f}] is outside "
            f"expected range [{expected_min}, {expected_max}]"
        )
    
    return True


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    """
    Flatten image tensors from (N, H, W) or (N, C, H, W) to (N, -1).
    
    Args:
        images: Image tensor to flatten
        
    Returns:
        Flattened tensor with shape (N, H*W) or (N, C*H*W)
    """
    batch_size = images.shape[0]
    return images.view(batch_size, -1)


def unflatten_images(
    flattened: torch.Tensor,
    target_shape: Tuple[int, ...] = (28, 28)
) -> torch.Tensor:
    """
    Reshape flattened vectors back to image format.
    
    Args:
        flattened: Flattened tensor of shape (N, features)
        target_shape: Target image shape (H, W) or (C, H, W)
        
    Returns:
        Reshaped tensor with image dimensions
    """
    batch_size = flattened.shape[0]
    return flattened.view(batch_size, *target_shape)


def compute_data_statistics(data: torch.Tensor) -> dict:
    """
    Compute comprehensive statistics for a data tensor.
    
    Args:
        data: Input tensor
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'shape': tuple(data.shape),
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std()),
        'median': float(data.median()),
        'q25': float(data.quantile(0.25)),
        'q75': float(data.quantile(0.75)),
    }
    
    return stats


def normalize_for_autoencoder(
    images: torch.Tensor,
    scaling_method: str = 'midrange',
    target_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for autoencoder input.
    
    Args:
        images: Input image tensor
        scaling_method: Scaling method ('midrange', 'standard', 'minmax')
        target_shape: Target shape for flattening (if None, auto-flatten)
        
    Returns:
        Preprocessed tensor ready for autoencoder
    """
    processed = images
    
    # Flatten if needed
    if len(processed.shape) > 2:
        if target_shape:
            processed = processed.view(processed.shape[0], *target_shape)
        processed = flatten_images(processed)
    
    # Apply scaling
    if scaling_method == 'midrange':
        scaler = MidrangeScaler()
        processed = scaler.transform(processed)
    elif scaling_method == 'standard':
        mean = processed.mean()
        std = processed.std()
        processed = (processed - mean) / std
    elif scaling_method == 'minmax':
        min_val = processed.min()
        max_val = processed.max()
        processed = (processed - min_val) / (max_val - min_val)
    
    return processed


def create_data_splits(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[torch.utils.data.Dataset, ...]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Input dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    torch.manual_seed(random_seed)
    
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    return torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )


# Convenience functions for common operations
def preprocess_mnist_batch(batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Preprocess a batch of MNIST data for autoencoder training.
    
    Args:
        batch: Tuple of (images, labels) from MNIST dataloader
        
    Returns:
        Preprocessed images tensor
    """
    images, _ = batch
    return normalize_for_autoencoder(images, scaling_method='midrange')


def mnist_to_sas_format(
    data: torch.Tensor,
    labels: torch.Tensor
) -> np.ndarray:
    """
    Convert MNIST tensors to SAS-compatible format.
    
    Args:
        data: Flattened image data tensor (N, 784)
        labels: Label tensor (N,)
        
    Returns:
        Numpy array with labels in first column, pixels in remaining columns
    """
    return np.column_stack([labels.numpy(), data.numpy()])
