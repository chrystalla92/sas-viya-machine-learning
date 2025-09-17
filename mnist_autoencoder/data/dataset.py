"""
MNIST dataset loader with torchvision integration and IDX file support.
"""

import os
import struct
import gzip
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Dict, Callable
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from .transforms import MNISTTransforms


class IDXFileReader:
    """Utility class for reading MNIST IDX format files."""
    
    @staticmethod
    def read_idx_images(file_path: Union[str, Path]) -> np.ndarray:
        """
        Read MNIST images from IDX3 format file.
        
        Args:
            file_path: Path to IDX3 format file (e.g., train-images.idx3-ubyte)
            
        Returns:
            numpy array of shape (N, 28, 28) containing images
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"IDX images file not found: {file_path}")
            
        try:
            # Handle both gzipped and uncompressed files
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    data = f.read()
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            # Parse IDX3 header
            magic, num_images, rows, cols = struct.unpack('>IIII', data[0:16])
            
            # Validate magic number for images (2051)
            if magic != 2051:
                raise ValueError(f"Invalid magic number for images: {magic}, expected 2051")
            
            # Validate dimensions
            if rows != 28 or cols != 28:
                raise ValueError(f"Invalid image dimensions: {rows}x{cols}, expected 28x28")
            
            # Extract image data
            images = np.frombuffer(data[16:], dtype=np.uint8)
            images = images.reshape((num_images, rows, cols))
            
            return images
            
        except Exception as e:
            raise ValueError(f"Error reading IDX images file {file_path}: {e}")
    
    @staticmethod
    def read_idx_labels(file_path: Union[str, Path]) -> np.ndarray:
        """
        Read MNIST labels from IDX1 format file.
        
        Args:
            file_path: Path to IDX1 format file (e.g., train-labels.idx1-ubyte)
            
        Returns:
            numpy array of shape (N,) containing labels
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"IDX labels file not found: {file_path}")
            
        try:
            # Handle both gzipped and uncompressed files
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    data = f.read()
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            # Parse IDX1 header
            magic, num_labels = struct.unpack('>II', data[0:8])
            
            # Validate magic number for labels (2049)
            if magic != 2049:
                raise ValueError(f"Invalid magic number for labels: {magic}, expected 2049")
            
            # Extract label data
            labels = np.frombuffer(data[8:], dtype=np.uint8)
            
            return labels
            
        except Exception as e:
            raise ValueError(f"Error reading IDX labels file {file_path}: {e}")


class MNISTDataset(Dataset):
    """
    MNIST dataset with both torchvision integration and direct IDX file support.
    
    Features:
    - Automatic download via torchvision
    - Direct IDX file loading for custom locations
    - Configurable preprocessing (flatten, normalize)
    - Data validation and caching
    - Compatible with SAS createData.sas output format
    """
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        train: bool = True,
        download: bool = True,
        images_file: Optional[Union[str, Path]] = None,
        labels_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        flatten: bool = True,
        normalize: str = "01",  # "01" for [0,1], "11" for [-1,1]
        cache_data: bool = True
    ):
        """
        Initialize MNIST dataset.
        
        Args:
            root: Root directory for dataset (used with torchvision)
            train: If True, use training set, else test set
            download: If True, download dataset if not found (torchvision only)
            images_file: Path to IDX images file (for direct loading)
            labels_file: Path to IDX labels file (for direct loading)
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            flatten: If True, flatten images to 784 features
            normalize: Normalization method ("01" or "11")
            cache_data: If True, cache processed data in memory
        """
        self.root = Path(root) if root else Path.cwd() / "data"
        self.train = train
        self.download = download
        self.images_file = Path(images_file) if images_file else None
        self.labels_file = Path(labels_file) if labels_file else None
        self.transform = transform
        self.target_transform = target_transform
        self.flatten = flatten
        self.normalize = normalize
        self.cache_data = cache_data
        
        # Validate normalization method
        if normalize not in ["01", "11"]:
            raise ValueError("normalize must be '01' for [0,1] or '11' for [-1,1]")
        
        # Data cache
        self._cached_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Load data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load MNIST data from either torchvision or IDX files."""
        if self.images_file and self.labels_file:
            # Load from IDX files directly
            self._load_from_idx()
        else:
            # Load from torchvision
            self._load_from_torchvision()
            
    def _load_from_idx(self) -> None:
        """Load data from IDX format files."""
        try:
            # Read images and labels
            images = IDXFileReader.read_idx_images(self.images_file)
            labels = IDXFileReader.read_idx_labels(self.labels_file)
            
            # Validate data consistency
            if len(images) != len(labels):
                raise ValueError(
                    f"Mismatch between number of images ({len(images)}) "
                    f"and labels ({len(labels)})"
                )
            
            # Convert to tensors
            self.data = torch.from_numpy(images).float()
            self.targets = torch.from_numpy(labels).long()
            
            print(f"Loaded {len(self.data)} samples from IDX files")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from IDX files: {e}")
            
    def _load_from_torchvision(self) -> None:
        """Load data using torchvision MNIST."""
        try:
            # Create temporary transform for loading
            temp_transform = transforms.ToTensor()
            
            # Load MNIST dataset
            mnist = torchvision.datasets.MNIST(
                root=str(self.root),
                train=self.train,
                download=self.download,
                transform=temp_transform
            )
            
            # Extract data and targets
            data_loader = DataLoader(mnist, batch_size=len(mnist), shuffle=False)
            data, targets = next(iter(data_loader))
            
            # Remove channel dimension and convert to proper format
            self.data = data.squeeze(1).float()  # (N, 28, 28)
            self.targets = targets.long()
            
            print(f"Loaded {len(self.data)} samples from torchvision MNIST")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from torchvision: {e}")
    
    def _preprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to data."""
        # Flatten if requested
        if self.flatten:
            data = data.reshape(data.size(0), -1)  # (N, 784)
            
            # Validate flattened size
            if data.size(1) != 784:
                raise ValueError(f"Expected 784 features after flattening, got {data.size(1)}")
        
        # Normalize pixel values
        if self.normalize == "01":
            # Normalize to [0, 1]
            data = data / 255.0
        elif self.normalize == "11":
            # Normalize to [-1, 1]
            data = (data / 255.0) * 2.0 - 1.0
            
        return data
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample by index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get raw data
        image = self.data[idx].clone()
        target = self.targets[idx].clone()
        
        # Apply preprocessing
        image = self._preprocess_data(image.unsqueeze(0)).squeeze(0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
            
        return image, target
    
    def get_cached_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get preprocessed data as cached tensors.
        
        Returns:
            Tuple of (data, targets) tensors with preprocessing applied
        """
        if self._cached_data is None and self.cache_data:
            # Preprocess all data
            data = self._preprocess_data(self.data.clone())
            targets = self.targets.clone()
            
            self._cached_data = (data, targets)
            print(f"Cached preprocessed data: {data.shape}")
            
        return self._cached_data if self._cached_data is not None else (
            self._preprocess_data(self.data.clone()), 
            self.targets.clone()
        )
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate dataset integrity and return statistics.
        
        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Check basic properties
            num_samples = len(self.data)
            results["statistics"]["num_samples"] = num_samples
            results["statistics"]["data_shape"] = tuple(self.data.shape)
            results["statistics"]["targets_shape"] = tuple(self.targets.shape)
            
            # Validate shapes
            if len(self.data.shape) != 3:
                results["errors"].append(f"Expected 3D data tensor, got {len(self.data.shape)}D")
                results["valid"] = False
            
            if self.data.shape[1:] != (28, 28):
                results["errors"].append(f"Expected (28, 28) image size, got {self.data.shape[1:]}")
                results["valid"] = False
                
            # Check flattened size after preprocessing
            if self.flatten:
                sample_data = self._preprocess_data(self.data[:1])
                if sample_data.shape[1] != 784:
                    results["errors"].append(f"Expected 784 features after flattening, got {sample_data.shape[1]}")
                    results["valid"] = False
                    
            # Validate data ranges
            min_val, max_val = self.data.min().item(), self.data.max().item()
            results["statistics"]["data_range"] = (min_val, max_val)
            
            if min_val < 0 or max_val > 255:
                results["warnings"].append(f"Pixel values outside expected range [0, 255]: [{min_val}, {max_val}]")
            
            # Validate targets
            unique_targets = torch.unique(self.targets)
            results["statistics"]["unique_targets"] = unique_targets.tolist()
            
            if len(unique_targets) > 10:
                results["warnings"].append(f"More than 10 unique targets found: {len(unique_targets)}")
            
            # Check for NaN or infinite values
            if torch.isnan(self.data).any():
                results["errors"].append("NaN values found in data")
                results["valid"] = False
                
            if torch.isinf(self.data).any():
                results["errors"].append("Infinite values found in data")
                results["valid"] = False
                
            # Memory usage
            data_memory = self.data.element_size() * self.data.nelement()
            results["statistics"]["memory_usage_mb"] = data_memory / (1024 * 1024)
            
        except Exception as e:
            results["errors"].append(f"Validation error: {e}")
            results["valid"] = False
        
        return results


class MNISTDataLoader:
    """
    DataLoader wrapper with MNIST-specific optimizations and caching.
    """
    
    def __init__(
        self,
        dataset: MNISTDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        use_cache: bool = True
    ):
        """
        Initialize MNIST DataLoader.
        
        Args:
            dataset: MNISTDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop last incomplete batch
            use_cache: Whether to use cached preprocessed data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_cache = use_cache
        
        # Create DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
    def __iter__(self):
        """Iterator interface."""
        return iter(self.dataloader)
        
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.dataloader)
    
    def get_full_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get entire dataset as tensors (useful for small datasets).
        
        Returns:
            Tuple of (data, targets) tensors
        """
        if self.use_cache:
            return self.dataset.get_cached_data()
        else:
            # Load all data at once
            full_loader = DataLoader(
                self.dataset, 
                batch_size=len(self.dataset), 
                shuffle=False
            )
            return next(iter(full_loader))
    
    def create_train_test_split(
        self, 
        test_ratio: float = 0.2, 
        random_seed: int = 42
    ) -> Tuple["MNISTDataLoader", "MNISTDataLoader"]:
        """
        Create train/test split from current dataset.
        
        Args:
            test_ratio: Proportion of data for test set
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        from torch.utils.data import random_split
        
        # Calculate split sizes
        dataset_size = len(self.dataset)
        test_size = int(test_ratio * dataset_size)
        train_size = dataset_size - test_size
        
        # Create split
        torch.manual_seed(random_seed)
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        
        # Create new loaders
        train_loader = MNISTDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            use_cache=False  # Subset datasets don't support caching
        )
        
        test_loader = MNISTDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            use_cache=False
        )
        
        return train_loader, test_loader
