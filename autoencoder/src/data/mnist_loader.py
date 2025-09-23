"""
MNIST Data Loader for Autoencoder Training

This module provides a comprehensive MNIST data loading and preprocessing
implementation that matches SAS midrange scaling behavior and provides
PyTorch tensors ready for autoencoder training.
"""

import os
import logging
import pandas as pd
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np

from .preprocessing import MidrangeScaler, validate_tensor_shape

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNISTDataLoader:
    """
    MNIST Data Loader with midrange scaling and autoencoder-ready preprocessing.
    
    This class handles:
    - Automatic MNIST dataset downloading
    - Midrange scaling to [-1,1] range (matching SAS behavior)
    - Tensor flattening from (28,28) to (784,) shape
    - Data subset functionality for testing consistency
    - CSV export for SAS compatibility
    - Proper error handling and validation
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        download: bool = True,
        batch_size: int = 64,
        validation_split: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize MNIST Data Loader.
        
        Args:
            data_root: Root directory for data storage
            download: Whether to download dataset if not present
            batch_size: Batch size for data loaders
            validation_split: Fraction of training data for validation
            random_seed: Random seed for reproducibility
        """
        self.data_root = data_root
        self.download = download
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_seed = random_seed
        
        # Initialize scaler
        self.scaler = MidrangeScaler()
        
        # Storage for datasets
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        
        # Ensure data directory exists
        os.makedirs(data_root, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
    def _get_transforms(self, apply_scaling: bool = True) -> transforms.Compose:
        """
        Create transformation pipeline for MNIST data.
        
        Args:
            apply_scaling: Whether to apply midrange scaling
            
        Returns:
            Composed transformations
        """
        transform_list = [
            transforms.ToTensor(),  # Converts PIL Image to tensor [0,1]
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
        ]
        
        if apply_scaling:
            # Apply midrange scaling to [-1,1]
            transform_list.append(
                transforms.Lambda(lambda x: self.scaler.transform(x))
            )
            
        return transforms.Compose(transform_list)
    
    def load_datasets(self, apply_scaling: bool = True) -> None:
        """
        Load MNIST train and test datasets.
        
        Args:
            apply_scaling: Whether to apply midrange scaling
            
        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info("Loading MNIST datasets...")
            
            # Define transformations
            transform = self._get_transforms(apply_scaling)
            
            # Load training dataset
            self.train_dataset = datasets.MNIST(
                root=self.data_root,
                train=True,
                download=self.download,
                transform=transform
            )
            
            # Load test dataset
            self.test_dataset = datasets.MNIST(
                root=self.data_root,
                train=False,
                download=self.download,
                transform=transform
            )
            
            logger.info(f"Training samples: {len(self.train_dataset)}")
            logger.info(f"Test samples: {len(self.test_dataset)}")
            
            # Validate data shapes
            sample_data, sample_label = self.train_dataset[0]
            validate_tensor_shape(sample_data, expected_shape=(784,))
            
            logger.info("MNIST datasets loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MNIST datasets: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}")
    
    def create_data_loaders(
        self,
        shuffle_train: bool = True,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create PyTorch data loaders for training, validation, and testing.
        
        Args:
            shuffle_train: Whether to shuffle training data
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple of (train_loader, test_loader, val_loader)
            
        Raises:
            RuntimeError: If datasets are not loaded
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise RuntimeError("Datasets not loaded. Call load_datasets() first.")
        
        try:
            # Create validation split if needed
            val_loader = None
            train_loader_dataset = self.train_dataset
            
            if self.validation_split > 0:
                train_size = len(self.train_dataset)
                val_size = int(train_size * self.validation_split)
                train_size = train_size - val_size
                
                # Create indices for train/val split
                indices = torch.randperm(len(self.train_dataset))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                # Create subset datasets
                train_loader_dataset = Subset(self.train_dataset, train_indices)
                self.val_dataset = Subset(self.train_dataset, val_indices)
                
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )
            
            # Create data loaders
            train_loader = DataLoader(
                train_loader_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers
            )
            
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            
            logger.info("Data loaders created successfully")
            return train_loader, test_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise RuntimeError(f"Data loader creation failed: {e}")
    
    def get_first_n_samples(
        self,
        n: int = 10,
        from_train: bool = True,
        return_tensors: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
        """
        Extract first N samples for testing consistency with SAS.
        
        Args:
            n: Number of samples to extract
            from_train: Whether to extract from training set (vs test set)
            return_tensors: Whether to return PyTorch tensors (vs numpy arrays)
            
        Returns:
            Tuple of (data, labels) for first N samples
            
        Raises:
            RuntimeError: If datasets are not loaded
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise RuntimeError("Datasets not loaded. Call load_datasets() first.")
        
        try:
            # Select dataset
            dataset = self.train_dataset if from_train else self.test_dataset
            
            # Extract first n samples
            data_list = []
            labels_list = []
            
            for i in range(min(n, len(dataset))):
                data, label = dataset[i]
                data_list.append(data)
                labels_list.append(label)
            
            # Stack into tensors
            data_tensor = torch.stack(data_list)
            labels_tensor = torch.tensor(labels_list)
            
            # Validate shapes
            validate_tensor_shape(data_tensor, expected_shape=(n, 784))
            
            if return_tensors:
                logger.info(f"Extracted {n} samples as tensors with shape {data_tensor.shape}")
                return data_tensor, labels_tensor
            else:
                logger.info(f"Extracted {n} samples as numpy arrays with shape {data_tensor.numpy().shape}")
                return data_tensor.numpy(), labels_tensor.numpy()
                
        except Exception as e:
            logger.error(f"Failed to extract first {n} samples: {e}")
            raise RuntimeError(f"Sample extraction failed: {e}")
    
    def export_to_csv(
        self,
        n_samples: int = 10,
        output_path: str = "mnist_samples.csv",
        from_train: bool = True,
        include_labels: bool = True
    ) -> str:
        """
        Export first N samples to CSV for SAS compatibility.
        
        Args:
            n_samples: Number of samples to export
            output_path: Path for output CSV file
            from_train: Whether to export from training set
            include_labels: Whether to include labels in first column
            
        Returns:
            Path to created CSV file
            
        Raises:
            RuntimeError: If export fails
        """
        try:
            # Get samples
            data, labels = self.get_first_n_samples(
                n=n_samples,
                from_train=from_train,
                return_tensors=False
            )
            
            # Create DataFrame
            if include_labels:
                # SAS format: var1 (label), var2-var785 (pixels)
                columns = ['var1'] + [f'var{i}' for i in range(2, 786)]
                df_data = np.column_stack([labels, data])
            else:
                # Only pixel data: var1-var784
                columns = [f'var{i}' for i in range(1, 785)]
                df_data = data
            
            df = pd.DataFrame(df_data, columns=columns)
            
            # Save to CSV
            full_path = os.path.join(self.data_root, output_path)
            df.to_csv(full_path, index=False)
            
            logger.info(f"Exported {n_samples} samples to {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise RuntimeError(f"CSV export failed: {e}")
    
    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.train_dataset is None:
            return {"error": "Datasets not loaded"}
        
        # Sample some data to get statistics
        sample_data, _ = self.get_first_n_samples(n=100, return_tensors=True)
        
        stats = {
            "train_samples": len(self.train_dataset),
            "test_samples": len(self.test_dataset),
            "data_shape": tuple(sample_data.shape[1:]),
            "data_min": float(sample_data.min()),
            "data_max": float(sample_data.max()),
            "data_mean": float(sample_data.mean()),
            "data_std": float(sample_data.std()),
        }
        
        if self.val_dataset:
            stats["val_samples"] = len(self.val_dataset)
        
        return stats


def create_mnist_loader(**kwargs) -> MNISTDataLoader:
    """
    Convenience function to create and initialize MNIST data loader.
    
    Args:
        **kwargs: Arguments passed to MNISTDataLoader constructor
        
    Returns:
        Initialized MNISTDataLoader instance
    """
    loader = MNISTDataLoader(**kwargs)
    loader.load_datasets()
    return loader
