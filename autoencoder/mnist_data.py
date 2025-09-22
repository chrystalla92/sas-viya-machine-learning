"""
MNIST Data Processing Module

This module provides functionality to read MNIST IDX files and process them
to match the behavior of the SAS createData.sas implementation.

Key features:
- IDX binary format parser for MNIST dataset files
- Image flattening from 28x28 to 784-dimensional vectors
- Midrange standardization matching SAS behavior
- Memory-efficient processing for large datasets
"""

import struct
import numpy as np
import os
from typing import Tuple, Optional, Union


class MNISTReader:
    """
    MNIST IDX file reader that processes binary files to match SAS behavior.
    
    The IDX file format uses big-endian byte order:
    - Images file: magic(4), num_images(4), rows(4), cols(4), pixel_data
    - Labels file: magic(4), num_labels(4), label_data
    """
    
    def __init__(self):
        self.images = None
        self.labels = None
        self._images_raw = None
        self._standardization_params = None
    
    def read_idx_images(self, filepath: str) -> np.ndarray:
        """
        Read MNIST images from IDX3 file format.
        
        Args:
            filepath: Path to the train-images-idx3-ubyte file
            
        Returns:
            numpy array of shape (num_images, 784) containing flattened images
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Images file not found: {filepath}")
            
        try:
            with open(filepath, 'rb') as f:
                # Read header (16 bytes total)
                magic = struct.unpack('>I', f.read(4))[0]  # big-endian 32-bit int
                num_images = struct.unpack('>I', f.read(4))[0]
                rows = struct.unpack('>I', f.read(4))[0]
                cols = struct.unpack('>I', f.read(4))[0]
                
                # Validate magic number and dimensions
                if magic != 2051:
                    raise ValueError(f"Invalid magic number for images: {magic}")
                if rows != 28 or cols != 28:
                    raise ValueError(f"Expected 28x28 images, got {rows}x{cols}")
                
                # Read pixel data
                pixel_data = f.read(num_images * rows * cols)
                if len(pixel_data) != num_images * rows * cols:
                    raise ValueError("Incomplete pixel data")
                
                # Convert to numpy array and reshape to (num_images, 784)
                images = np.frombuffer(pixel_data, dtype=np.uint8)
                images = images.reshape(num_images, rows * cols)
                
                # Convert to float for processing (matching SAS behavior)
                self._images_raw = images.astype(np.float64)
                return self._images_raw
                
        except Exception as e:
            raise ValueError(f"Error reading images file: {str(e)}")
    
    def read_idx_labels(self, filepath: str) -> np.ndarray:
        """
        Read MNIST labels from IDX1 file format.
        
        Args:
            filepath: Path to the train-labels-idx1-ubyte file
            
        Returns:
            numpy array of shape (num_labels,) containing label values
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Labels file not found: {filepath}")
            
        try:
            with open(filepath, 'rb') as f:
                # Read header (8 bytes total)
                magic = struct.unpack('>I', f.read(4))[0]  # big-endian 32-bit int
                num_labels = struct.unpack('>I', f.read(4))[0]
                
                # Validate magic number
                if magic != 2049:
                    raise ValueError(f"Invalid magic number for labels: {magic}")
                
                # Read label data
                label_data = f.read(num_labels)
                if len(label_data) != num_labels:
                    raise ValueError("Incomplete label data")
                
                # Convert to numpy array
                labels = np.frombuffer(label_data, dtype=np.uint8)
                return labels.astype(np.int32)
                
        except Exception as e:
            raise ValueError(f"Error reading labels file: {str(e)}")
    
    def apply_midrange_standardization(self, images: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply midrange standardization to match SAS 'standardize=midrange' behavior.
        
        Midrange standardization formula:
        standardized_x = (x - midrange) / range
        where:
        - midrange = (max + min) / 2
        - range = max - min
        
        Args:
            images: Optional images array. If None, uses self._images_raw
            
        Returns:
            Standardized images array
        """
        if images is None:
            if self._images_raw is None:
                raise ValueError("No images loaded. Call read_idx_images first.")
            images = self._images_raw
        
        # Calculate midrange standardization parameters per pixel (column-wise)
        min_vals = np.min(images, axis=0)  # min per pixel across all images
        max_vals = np.max(images, axis=0)  # max per pixel across all images
        
        midrange = (max_vals + min_vals) / 2.0
        range_vals = max_vals - min_vals
        
        # Avoid division by zero for constant pixels
        # If range is 0, standardized value should be 0 (all pixels same value)
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        # Apply standardization
        standardized = (images - midrange) / range_vals
        
        # Store parameters for later use
        self._standardization_params = {
            'midrange': midrange,
            'range': range_vals,
            'min_vals': min_vals,
            'max_vals': max_vals
        }
        
        return standardized
    
    def load_mnist_dataset(self, images_path: str, labels_path: str, 
                          standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load complete MNIST dataset from IDX files.
        
        Args:
            images_path: Path to images IDX file
            labels_path: Path to labels IDX file  
            standardize: Whether to apply midrange standardization
            
        Returns:
            Tuple of (images, labels) where images are optionally standardized
        """
        # Read raw data
        images = self.read_idx_images(images_path)
        labels = self.read_idx_labels(labels_path)
        
        # Verify matching counts
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images, {len(labels)} labels")
        
        # Apply standardization if requested
        if standardize:
            images = self.apply_midrange_standardization(images)
        
        self.images = images
        self.labels = labels
        
        return images, labels
    
    def get_standardization_params(self) -> Optional[dict]:
        """Get the standardization parameters used."""
        return self._standardization_params
    
    def create_sas_compatible_dataset(self, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Create dataset in SAS-compatible format where:
        - Column 0: labels (var1)
        - Columns 1-784: pixel values (var2-var785)
        
        Args:
            images: Image data array of shape (n, 784)
            labels: Label data array of shape (n,)
            
        Returns:
            Combined dataset of shape (n, 785) matching SAS output format
        """
        if len(images) != len(labels):
            raise ValueError("Images and labels must have same length")
            
        if images.shape[1] != 784:
            raise ValueError("Images must be flattened to 784 pixels")
        
        # Combine labels and images: [label, pixel1, pixel2, ..., pixel784]
        dataset = np.column_stack([labels, images])
        return dataset


def load_mnist_data(images_path: str, labels_path: str, 
                   standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load MNIST data matching SAS createData.sas behavior.
    
    Args:
        images_path: Path to MNIST images IDX file (e.g., train-images-idx3-ubyte)
        labels_path: Path to MNIST labels IDX file (e.g., train-labels-idx1-ubyte)
        standardize: Whether to apply midrange standardization (default: True)
        
    Returns:
        Tuple of (images, labels) where:
        - images: shape (n, 784), standardized pixel values
        - labels: shape (n,), integer labels 0-9
        
    Example:
        >>> images, labels = load_mnist_data('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
        >>> print(f"Loaded {len(images)} training examples")
        >>> print(f"Image shape: {images.shape}, Label shape: {labels.shape}")
    """
    reader = MNISTReader()
    return reader.load_mnist_dataset(images_path, labels_path, standardize)


def create_sas_format_dataset(images_path: str, labels_path: str, 
                             standardize: bool = True) -> np.ndarray:
    """
    Create dataset in exact SAS format matching createData.sas output.
    
    Returns array where:
    - Column 0: labels (var1) 
    - Columns 1-784: standardized pixel values (var2-var785)
    
    Args:
        images_path: Path to MNIST images IDX file
        labels_path: Path to MNIST labels IDX file
        standardize: Whether to apply midrange standardization
        
    Returns:
        Dataset array of shape (n, 785) matching SAS mnist_train format
    """
    reader = MNISTReader()
    images, labels = reader.load_mnist_dataset(images_path, labels_path, standardize)
    return reader.create_sas_compatible_dataset(images, labels)
