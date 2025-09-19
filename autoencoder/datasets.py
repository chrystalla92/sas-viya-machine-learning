"""
PyTorch Dataset and DataLoader classes for MNIST autoencoder training.

This module provides PyTorch-compatible dataset classes that work with the
binary MNIST files and support proper batching for training pipelines.
"""

import os
from typing import Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .data_loader import load_mnist_data, load_mnist_training_data, load_mnist_test_data


class MNISTAutoencoderDataset(Dataset):
    """
    PyTorch Dataset for MNIST autoencoder training.
    
    For autoencoders, both input and target are the same (the images).
    """
    
    def __init__(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Initialize the dataset.
        
        Args:
            images (torch.Tensor): Image data of shape (N, 784)
            labels (Optional[torch.Tensor]): Label data of shape (N,). 
                                           Can be None if labels not needed.
        """
        self.images = images
        self.labels = labels
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a single sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            torch.Tensor: Image tensor for autoencoder training (input = target)
            OR
            Tuple[torch.Tensor, torch.Tensor]: (image, label) if labels are provided
        """
        image = self.images[idx]
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            # For autoencoder: input and target are the same
            return image


class MNISTDatasetFromFiles(Dataset):
    """
    PyTorch Dataset that loads MNIST data directly from binary files.
    
    This provides memory-efficient loading by keeping file paths and loading 
    data on demand, though for MNIST the entire dataset typically fits in memory.
    """
    
    def __init__(self, images_path: str, labels_path: str, 
                 standardize: bool = True, include_labels: bool = False):
        """
        Initialize the dataset from file paths.
        
        Args:
            images_path (str): Path to IDX3-UBYTE images file
            labels_path (str): Path to IDX1-UBYTE labels file
            standardize (bool): Whether to apply midrange standardization
            include_labels (bool): Whether to return labels along with images
        """
        self.images_path = images_path
        self.labels_path = labels_path
        self.standardize = standardize
        self.include_labels = include_labels
        
        # Load data once to avoid repeated file I/O
        self._images, self._labels = load_mnist_data(
            images_path, labels_path, standardize
        )
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._images)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a single sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            torch.Tensor: Image tensor for autoencoder training
            OR
            Tuple[torch.Tensor, torch.Tensor]: (image, label) if include_labels=True
        """
        image = self._images[idx]
        
        if self.include_labels:
            label = self._labels[idx]
            return image, label
        else:
            return image


def create_mnist_dataloaders(data_dir: str = "./data", 
                           batch_size: int = 32,
                           train_val_split: float = 0.8,
                           standardize: bool = True,
                           num_workers: int = 0,
                           shuffle: bool = True,
                           include_labels: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for MNIST training, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing MNIST binary files
        batch_size (int): Batch size for DataLoaders
        train_val_split (float): Fraction of training data to use for training
                                (remainder used for validation)
        standardize (bool): Whether to apply midrange standardization
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the training data
        include_labels (bool): Whether to include labels in the output
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Load training data
    train_images, train_labels = load_mnist_training_data(data_dir, standardize)
    
    # Create training dataset
    if include_labels:
        full_train_dataset = MNISTAutoencoderDataset(train_images, train_labels)
    else:
        full_train_dataset = MNISTAutoencoderDataset(train_images)
    
    # Split training data into train/validation
    total_train_samples = len(full_train_dataset)
    train_samples = int(train_val_split * total_train_samples)
    val_samples = total_train_samples - train_samples
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_samples, val_samples]
    )
    
    # Load test data
    test_images, test_labels = load_mnist_test_data(data_dir, standardize)
    if include_labels:
        test_dataset = MNISTAutoencoderDataset(test_images, test_labels)
    else:
        test_dataset = MNISTAutoencoderDataset(test_images)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def create_simple_dataloader(data_dir: str = "./data",
                           dataset_type: str = "train", 
                           batch_size: int = 32,
                           standardize: bool = True,
                           shuffle: bool = True,
                           include_labels: bool = False) -> DataLoader:
    """
    Create a simple DataLoader for either training or test data.
    
    Args:
        data_dir (str): Directory containing MNIST binary files
        dataset_type (str): Either "train" or "test"
        batch_size (int): Batch size for DataLoader
        standardize (bool): Whether to apply midrange standardization
        shuffle (bool): Whether to shuffle the data
        include_labels (bool): Whether to include labels in the output
        
    Returns:
        DataLoader: Configured DataLoader
        
    Raises:
        ValueError: If dataset_type is not "train" or "test"
    """
    if dataset_type == "train":
        images, labels = load_mnist_training_data(data_dir, standardize)
    elif dataset_type == "test":
        images, labels = load_mnist_test_data(data_dir, standardize)
    else:
        raise ValueError(f"dataset_type must be 'train' or 'test', got '{dataset_type}'")
    
    if include_labels:
        dataset = MNISTAutoencoderDataset(images, labels)
    else:
        dataset = MNISTAutoencoderDataset(images)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Convenience functions for quick access
def get_train_dataloader(data_dir: str = "./data", batch_size: int = 32, **kwargs) -> DataLoader:
    """Get training DataLoader with default parameters."""
    return create_simple_dataloader(data_dir, "train", batch_size, **kwargs)


def get_test_dataloader(data_dir: str = "./data", batch_size: int = 32, **kwargs) -> DataLoader:
    """Get test DataLoader with default parameters."""  
    return create_simple_dataloader(data_dir, "test", batch_size, **kwargs)
