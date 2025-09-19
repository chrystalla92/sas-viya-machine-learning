"""
MNIST Data Loader for Autoencoder

This module provides functionality to load MNIST binary files directly from filesystem
and convert them to PyTorch tensors with proper standardization.
"""

import os
import struct
import numpy as np
import torch
from typing import Tuple, Optional


def read_idx3_ubyte(filepath: str) -> np.ndarray:
    """
    Read IDX3-UBYTE format file (MNIST images).
    
    Format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000/10000      number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ....
    xxxx     unsigned byte   ??               pixel
    
    Args:
        filepath (str): Path to the IDX3-UBYTE file
        
    Returns:
        np.ndarray: Array of shape (N, 784) containing flattened 28x28 images
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            # Read header (16 bytes)
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            # Validate magic number
            if magic != 2051:
                raise ValueError(f"Invalid magic number for images: {magic}, expected 2051")
            
            # Validate dimensions
            if rows != 28 or cols != 28:
                raise ValueError(f"Expected 28x28 images, got {rows}x{cols}")
            
            # Read image data
            image_data = f.read()
            
            # Convert to numpy array and reshape
            images = np.frombuffer(image_data, dtype=np.uint8)
            expected_size = num_images * rows * cols
            
            if len(images) != expected_size:
                raise ValueError(f"Expected {expected_size} bytes, got {len(images)}")
            
            # Reshape to (N, 784) - flattened 28x28 images
            images = images.reshape(num_images, rows * cols)
            
            return images
            
    except struct.error as e:
        raise ValueError(f"Error reading image file header: {e}")
    except Exception as e:
        raise ValueError(f"Error reading image file: {e}")


def read_idx1_ubyte(filepath: str) -> np.ndarray:
    """
    Read IDX1-UBYTE format file (MNIST labels).
    
    Format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number
    0004     32 bit integer  60000/10000      number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ....
    xxxx     unsigned byte   ??               label
    
    Args:
        filepath (str): Path to the IDX1-UBYTE file
        
    Returns:
        np.ndarray: Array of shape (N,) containing labels
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Label file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            # Read header (8 bytes)
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            # Validate magic number
            if magic != 2049:
                raise ValueError(f"Invalid magic number for labels: {magic}, expected 2049")
            
            # Read label data
            label_data = f.read()
            
            # Convert to numpy array
            labels = np.frombuffer(label_data, dtype=np.uint8)
            
            if len(labels) != num_labels:
                raise ValueError(f"Expected {num_labels} labels, got {len(labels)}")
            
            return labels
            
    except struct.error as e:
        raise ValueError(f"Error reading label file header: {e}")
    except Exception as e:
        raise ValueError(f"Error reading label file: {e}")


def midrange_standardize(data: np.ndarray) -> np.ndarray:
    """
    Apply midrange standardization to the data.
    
    Formula: (x - midrange) / (max - min)
    where midrange = (max + min) / 2
    
    This is equivalent to the SAS 'standardize=midrange' option.
    
    Args:
        data (np.ndarray): Input data to standardize
        
    Returns:
        np.ndarray: Standardized data
    """
    data_min = data.min()
    data_max = data.max()
    
    # Handle case where all values are the same
    if data_max == data_min:
        return np.zeros_like(data)
    
    midrange = (data_max + data_min) / 2
    return (data - midrange) / (data_max - data_min)


def validate_data(images: np.ndarray, labels: np.ndarray) -> None:
    """
    Validate MNIST data for correct shapes and value ranges.
    
    Args:
        images (np.ndarray): Image data of shape (N, 784)
        labels (np.ndarray): Label data of shape (N,)
        
    Raises:
        ValueError: If data doesn't meet expected specifications
    """
    # Check shapes
    if len(images.shape) != 2 or images.shape[1] != 784:
        raise ValueError(f"Images must have shape (N, 784), got {images.shape}")
    
    if len(labels.shape) != 1:
        raise ValueError(f"Labels must have shape (N,), got {labels.shape}")
    
    # Check matching number of samples
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch in number of samples: {images.shape[0]} images, {labels.shape[0]} labels")
    
    # Check label ranges
    if labels.min() < 0 or labels.max() > 9:
        raise ValueError(f"Labels must be in range 0-9, got range {labels.min()}-{labels.max()}")
    
    # Check if we have the expected number of samples
    num_samples = images.shape[0]
    if num_samples not in [60000, 10000]:
        print(f"Warning: Expected 60000 (train) or 10000 (test) samples, got {num_samples}")


def load_mnist_data(images_path: str, labels_path: str, 
                   standardize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST data from binary files and convert to PyTorch tensors.
    
    Args:
        images_path (str): Path to the IDX3-UBYTE images file
        labels_path (str): Path to the IDX1-UBYTE labels file  
        standardize (bool): Whether to apply midrange standardization
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (images, labels)
            - images: Float32 tensor of shape (N, 784)
            - labels: Long tensor of shape (N,)
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If data is invalid or corrupted
    """
    # Read binary files
    images = read_idx3_ubyte(images_path)
    labels = read_idx1_ubyte(labels_path)
    
    # Validate data
    validate_data(images, labels)
    
    # Convert images to float32 for processing
    images = images.astype(np.float32)
    
    # Apply standardization if requested
    if standardize:
        images = midrange_standardize(images)
    
    # Convert to PyTorch tensors
    images_tensor = torch.from_numpy(images).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    return images_tensor, labels_tensor


def load_mnist_training_data(data_dir: str = "./data", 
                           standardize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST training data (60,000 samples).
    
    Args:
        data_dir (str): Directory containing MNIST files
        standardize (bool): Whether to apply midrange standardization
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (images, labels)
    """
    images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    
    return load_mnist_data(images_path, labels_path, standardize)


def load_mnist_test_data(data_dir: str = "./data", 
                        standardize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load MNIST test data (10,000 samples).
    
    Args:
        data_dir (str): Directory containing MNIST files
        standardize (bool): Whether to apply midrange standardization
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (images, labels)
    """
    images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")
    
    return load_mnist_data(images_path, labels_path, standardize)
