"""
Data Utilities Module

This module provides utility functions for dataset handling, including
train/validation splits, batch processing, and data validation.

Designed to work with the mnist_data.py module for MNIST dataset processing.
"""

import numpy as np
import os
from typing import Tuple, Optional, List, Union, Dict
import warnings


def train_validation_split(images: np.ndarray, labels: np.ndarray, 
                          validation_ratio: float = 0.2, 
                          random_seed: Optional[int] = None, 
                          stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and validation sets.
    
    Args:
        images: Image data of shape (n, 784)
        labels: Label data of shape (n,)
        validation_ratio: Fraction of data to use for validation (0.0 to 1.0)
        random_seed: Random seed for reproducible splits
        stratify: Whether to maintain class distribution in splits
        
    Returns:
        Tuple of (train_images, val_images, train_labels, val_labels)
        
    Raises:
        ValueError: If validation_ratio is not between 0 and 1
    """
    if not 0 <= validation_ratio <= 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    
    if len(images) != len(labels):
        raise ValueError("Images and labels must have same length")
    
    n_samples = len(images)
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if validation_ratio == 0:
        return images, np.array([]), labels, np.array([])
    
    if stratify:
        # Stratified split to maintain class distribution
        unique_labels = np.unique(labels)
        train_indices = []
        val_indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)
            
            n_val = int(len(label_indices) * validation_ratio)
            val_indices.extend(label_indices[:n_val])
            train_indices.extend(label_indices[n_val:])
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        
    else:
        # Simple random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_val = int(n_samples * validation_ratio)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
    
    return (images[train_indices], images[val_indices], 
            labels[train_indices], labels[val_indices])


def create_batches(images: np.ndarray, labels: Optional[np.ndarray] = None, 
                  batch_size: int = 32, shuffle: bool = True, 
                  random_seed: Optional[int] = None) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Create batches from dataset for memory-efficient processing.
    
    Args:
        images: Image data of shape (n, 784)
        labels: Optional label data of shape (n,)
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before batching
        random_seed: Random seed for shuffling
        
    Returns:
        List of batches. Each batch is either:
        - np.ndarray if labels is None
        - Tuple[np.ndarray, np.ndarray] if labels provided
    """
    n_samples = len(images)
    
    if labels is not None and len(labels) != n_samples:
        raise ValueError("Images and labels must have same length")
    
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    batches = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_images = images[batch_indices]
        
        if labels is not None:
            batch_labels = labels[batch_indices]
            batches.append((batch_images, batch_labels))
        else:
            batches.append(batch_images)
    
    return batches


def validate_dataset_format(dataset: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None,
                           check_sas_format: bool = True) -> Dict[str, Union[bool, str, int]]:
    """
    Validate dataset format and return diagnostic information.
    
    Args:
        dataset: Dataset to validate
        expected_shape: Expected shape tuple (optional)
        check_sas_format: Whether to check SAS-compatible format (785 columns)
        
    Returns:
        Dictionary with validation results and diagnostic info
    """
    result = {
        'is_valid': True,
        'shape': dataset.shape,
        'dtype': str(dataset.dtype),
        'issues': []
    }
    
    # Check basic properties
    if dataset.size == 0:
        result['is_valid'] = False
        result['issues'].append('Dataset is empty')
        return result
    
    # Check expected shape
    if expected_shape is not None:
        if dataset.shape != expected_shape:
            result['is_valid'] = False
            result['issues'].append(f'Shape mismatch: expected {expected_shape}, got {dataset.shape}')
    
    # Check SAS format (785 columns: 1 label + 784 pixels)
    if check_sas_format:
        if len(dataset.shape) != 2:
            result['is_valid'] = False
            result['issues'].append('Dataset must be 2D for SAS format')
        elif dataset.shape[1] != 785:
            result['is_valid'] = False
            result['issues'].append(f'SAS format requires 785 columns, got {dataset.shape[1]}')
        else:
            # Check label column (should be integers 0-9 for MNIST)
            labels = dataset[:, 0]
            unique_labels = np.unique(labels)
            if not all(isinstance(x, (int, np.integer)) for x in unique_labels):
                result['issues'].append('Labels should be integers')
            if not all(0 <= x <= 9 for x in unique_labels):
                result['issues'].append('MNIST labels should be 0-9')
            
            result['n_samples'] = dataset.shape[0]
            result['n_features'] = dataset.shape[1] - 1  # excluding label column
            result['unique_labels'] = len(unique_labels)
            result['label_distribution'] = {int(label): int(np.sum(labels == label)) 
                                          for label in unique_labels}
    
    return result


def save_dataset_csv(dataset: np.ndarray, filepath: str, 
                     include_header: bool = False,
                     sas_compatible: bool = True) -> None:
    """
    Save dataset to CSV file in format compatible with SAS output.
    
    Args:
        dataset: Dataset array to save
        filepath: Output file path
        include_header: Whether to include column headers
        sas_compatible: Whether to format for SAS compatibility
        
    Raises:
        ValueError: If dataset format is invalid
        IOError: If file cannot be written
    """
    # Validate dataset
    validation = validate_dataset_format(dataset, check_sas_format=sas_compatible)
    if not validation['is_valid']:
        raise ValueError(f"Invalid dataset format: {validation['issues']}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare headers if requested
        if include_header:
            if sas_compatible and dataset.shape[1] == 785:
                # SAS format: var1 (label), var2-var785 (pixels)
                headers = ['var1'] + [f'var{i}' for i in range(2, 786)]
            else:
                # Generic format
                headers = [f'col_{i}' for i in range(dataset.shape[1])]
            
            # Save with headers
            np.savetxt(filepath, dataset, delimiter=',', fmt='%g', 
                      header=','.join(headers), comments='')
        else:
            # Save without headers (matching SAS putnames=no)
            np.savetxt(filepath, dataset, delimiter=',', fmt='%g')
            
    except Exception as e:
        raise IOError(f"Error writing file {filepath}: {str(e)}")


def load_dataset_csv(filepath: str, has_header: bool = False) -> np.ndarray:
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to CSV file
        has_header: Whether CSV file has header row
        
    Returns:
        Dataset array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        skip_header = 1 if has_header else 0
        dataset = np.loadtxt(filepath, delimiter=',', skiprows=skip_header)
        return dataset
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {str(e)}")


def memory_efficient_process(images_path: str, labels_path: str,
                           process_function,
                           batch_size: int = 1000,
                           **kwargs) -> np.ndarray:
    """
    Process large MNIST datasets in memory-efficient batches.
    
    This function is useful when the full dataset doesn't fit in memory
    or when processing time is long.
    
    Args:
        images_path: Path to MNIST images file
        labels_path: Path to MNIST labels file  
        process_function: Function to apply to each batch
        batch_size: Size of processing batches
        **kwargs: Additional arguments for process_function
        
    Returns:
        Processed dataset
        
    Note:
        process_function should accept (images_batch, labels_batch, **kwargs)
        and return processed batch
    """
    from .mnist_data import MNISTReader
    
    # Read dataset metadata first
    reader = MNISTReader()
    
    # For memory efficiency, we'd need to implement streaming reading
    # For now, this is a placeholder that loads full dataset
    # In production, this would read and process chunks
    
    warnings.warn("Memory-efficient processing not fully implemented. "
                  "Loading full dataset into memory.", UserWarning)
    
    images, labels = reader.load_mnist_dataset(images_path, labels_path, standardize=False)
    
    # Process in batches
    results = []
    for i in range(0, len(images), batch_size):
        end_idx = min(i + batch_size, len(images))
        batch_images = images[i:end_idx]
        batch_labels = labels[i:end_idx]
        
        batch_result = process_function(batch_images, batch_labels, **kwargs)
        results.append(batch_result)
    
    # Combine results
    if results:
        return np.vstack(results)
    else:
        return np.array([])


def dataset_statistics(images: np.ndarray, labels: np.ndarray) -> Dict[str, Union[float, int, Dict]]:
    """
    Calculate comprehensive statistics for MNIST dataset.
    
    Args:
        images: Image data of shape (n, 784)
        labels: Label data of shape (n,)
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'n_samples': len(images),
        'n_features': images.shape[1] if len(images.shape) > 1 else 0,
        'n_classes': len(np.unique(labels)),
        'class_distribution': {},
        'pixel_statistics': {},
        'data_range': {}
    }
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    stats['class_distribution'] = {int(label): int(count) 
                                 for label, count in zip(unique_labels, counts)}
    
    # Pixel statistics
    if len(images.shape) == 2:
        stats['pixel_statistics'] = {
            'mean': float(np.mean(images)),
            'std': float(np.std(images)),
            'min': float(np.min(images)),
            'max': float(np.max(images)),
            'mean_per_pixel': np.mean(images, axis=0),
            'std_per_pixel': np.std(images, axis=0)
        }
    
    # Data range
    stats['data_range'] = {
        'pixel_min': float(np.min(images)),
        'pixel_max': float(np.max(images)),
        'label_min': int(np.min(labels)),
        'label_max': int(np.max(labels))
    }
    
    return stats


def compare_datasets(dataset1: np.ndarray, dataset2: np.ndarray, 
                    tolerance: float = 1e-10) -> Dict[str, Union[bool, float, str]]:
    """
    Compare two datasets for equality (useful for validating SAS compatibility).
    
    Args:
        dataset1: First dataset
        dataset2: Second dataset  
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'are_equal': False,
        'shape_match': False,
        'max_difference': float('inf'),
        'mean_difference': float('inf'),
        'issues': []
    }
    
    # Shape comparison
    if dataset1.shape != dataset2.shape:
        comparison['issues'].append(f'Shape mismatch: {dataset1.shape} vs {dataset2.shape}')
        return comparison
    
    comparison['shape_match'] = True
    
    # Value comparison
    diff = np.abs(dataset1 - dataset2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    comparison['max_difference'] = float(max_diff)
    comparison['mean_difference'] = float(mean_diff)
    
    if max_diff <= tolerance:
        comparison['are_equal'] = True
    else:
        comparison['issues'].append(f'Values differ by up to {max_diff:.2e}')
        
        # Find problematic locations
        problem_indices = np.where(diff > tolerance)
        if len(problem_indices[0]) > 0:
            n_problems = len(problem_indices[0])
            comparison['issues'].append(f'{n_problems} values exceed tolerance')
    
    return comparison
