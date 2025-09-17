"""
Comprehensive tests for MNIST data loading and preprocessing.
"""

import os
import tempfile
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from mnist_autoencoder.data.dataset import MNISTDataset, MNISTDataLoader, IDXFileReader
from mnist_autoencoder.data.transforms import (
    MNISTTransforms, GaussianNoise, Flatten, Normalize, 
    ValidateShape, ToFloat32, create_training_transforms, 
    create_evaluation_transforms, create_sas_compatible_transforms
)


class TestIDXFileReader:
    """Test IDX file reading functionality."""
    
    @pytest.fixture
    def sample_idx_images(self):
        """Create sample IDX images file."""
        # Create fake MNIST image data
        num_images, rows, cols = 10, 28, 28
        magic = 2051  # Magic number for images
        
        # Header + image data
        header = struct.pack('>IIII', magic, num_images, rows, cols)
        images = np.random.randint(0, 256, (num_images, rows, cols), dtype=np.uint8)
        data = header + images.tobytes()
        
        return data, images
    
    @pytest.fixture
    def sample_idx_labels(self):
        """Create sample IDX labels file."""
        # Create fake MNIST label data
        num_labels = 10
        magic = 2049  # Magic number for labels
        
        # Header + label data
        header = struct.pack('>II', magic, num_labels)
        labels = np.random.randint(0, 10, num_labels, dtype=np.uint8)
        data = header + labels.tobytes()
        
        return data, labels
    
    def test_read_idx_images_success(self, sample_idx_images):
        """Test successful IDX images file reading."""
        data, expected_images = sample_idx_images
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            f.flush()
            
            try:
                images = IDXFileReader.read_idx_images(f.name)
                np.testing.assert_array_equal(images, expected_images)
                assert images.shape == (10, 28, 28)
            finally:
                os.unlink(f.name)
    
    def test_read_idx_labels_success(self, sample_idx_labels):
        """Test successful IDX labels file reading."""
        data, expected_labels = sample_idx_labels
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            f.flush()
            
            try:
                labels = IDXFileReader.read_idx_labels(f.name)
                np.testing.assert_array_equal(labels, expected_labels)
                assert labels.shape == (10,)
            finally:
                os.unlink(f.name)
    
    def test_read_idx_images_file_not_found(self):
        """Test error when images file not found."""
        with pytest.raises(FileNotFoundError):
            IDXFileReader.read_idx_images("nonexistent_file.idx")
    
    def test_read_idx_labels_file_not_found(self):
        """Test error when labels file not found."""
        with pytest.raises(FileNotFoundError):
            IDXFileReader.read_idx_labels("nonexistent_file.idx")
    
    def test_read_idx_images_invalid_magic(self):
        """Test error with invalid magic number for images."""
        # Create file with wrong magic number
        wrong_magic = 1234
        header = struct.pack('>IIII', wrong_magic, 10, 28, 28)
        data = header + b'\x00' * (10 * 28 * 28)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="Invalid magic number"):
                    IDXFileReader.read_idx_images(f.name)
            finally:
                os.unlink(f.name)
    
    def test_read_idx_labels_invalid_magic(self):
        """Test error with invalid magic number for labels."""
        # Create file with wrong magic number
        wrong_magic = 1234
        header = struct.pack('>II', wrong_magic, 10)
        data = header + b'\x00' * 10
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="Invalid magic number"):
                    IDXFileReader.read_idx_labels(f.name)
            finally:
                os.unlink(f.name)
    
    def test_read_idx_images_invalid_dimensions(self):
        """Test error with invalid image dimensions."""
        # Create file with wrong dimensions
        magic = 2051
        header = struct.pack('>IIII', magic, 10, 32, 32)  # Wrong dimensions
        data = header + b'\x00' * (10 * 32 * 32)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="Invalid image dimensions"):
                    IDXFileReader.read_idx_images(f.name)
            finally:
                os.unlink(f.name)


class TestMNISTDataset:
    """Test MNIST dataset functionality."""
    
    @pytest.fixture
    def mock_torchvision_mnist(self):
        """Mock torchvision MNIST dataset."""
        with patch('torchvision.datasets.MNIST') as mock_mnist:
            # Create fake data
            fake_data = torch.randint(0, 256, (100, 1, 28, 28), dtype=torch.uint8)
            fake_targets = torch.randint(0, 10, (100,), dtype=torch.long)
            
            # Mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_mnist.return_value = mock_dataset
            
            # Mock DataLoader behavior
            with patch('torch.utils.data.DataLoader') as mock_dataloader:
                mock_dataloader.return_value.__iter__ = lambda: iter([(fake_data, fake_targets)])
                yield mock_mnist, fake_data, fake_targets
    
    def test_mnist_dataset_torchvision(self, mock_torchvision_mnist):
        """Test MNIST dataset with torchvision backend."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        dataset = MNISTDataset(train=True, download=True)
        
        # Check that data was loaded correctly
        assert len(dataset) == 100
        assert dataset.data.shape == (100, 28, 28)
        assert dataset.targets.shape == (100,)
    
    def test_mnist_dataset_idx_files(self, sample_idx_images, sample_idx_labels):
        """Test MNIST dataset with IDX files."""
        img_data, expected_images = sample_idx_images
        lbl_data, expected_labels = sample_idx_labels
        
        with tempfile.NamedTemporaryFile(delete=False) as img_file, \
             tempfile.NamedTemporaryFile(delete=False) as lbl_file:
            
            img_file.write(img_data)
            lbl_file.write(lbl_data)
            img_file.flush()
            lbl_file.flush()
            
            try:
                dataset = MNISTDataset(
                    images_file=img_file.name,
                    labels_file=lbl_file.name
                )
                
                assert len(dataset) == 10
                assert dataset.data.shape == (10, 28, 28)
                assert dataset.targets.shape == (10,)
                
                # Check data content
                np.testing.assert_array_equal(
                    dataset.data.numpy(), expected_images.astype(np.float32)
                )
                np.testing.assert_array_equal(
                    dataset.targets.numpy(), expected_labels
                )
                
            finally:
                os.unlink(img_file.name)
                os.unlink(lbl_file.name)
    
    def test_mnist_dataset_getitem_flattened(self, mock_torchvision_mnist):
        """Test dataset __getitem__ with flattening."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        dataset = MNISTDataset(train=True, flatten=True, normalize="01")
        
        # Get first sample
        image, target = dataset[0]
        
        # Check flattened shape
        assert image.shape == (784,)
        assert 0 <= image.min() <= image.max() <= 1  # Normalized to [0, 1]
        assert isinstance(target.item(), int)
    
    def test_mnist_dataset_getitem_not_flattened(self, mock_torchvision_mnist):
        """Test dataset __getitem__ without flattening."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        dataset = MNISTDataset(train=True, flatten=False, normalize="11")
        
        # Get first sample
        image, target = dataset[0]
        
        # Check original shape
        assert image.shape == (28, 28)
        assert -1 <= image.min() <= image.max() <= 1  # Normalized to [-1, 1]
    
    def test_mnist_dataset_normalization_methods(self, mock_torchvision_mnist):
        """Test different normalization methods."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        # Test [0, 1] normalization
        dataset_01 = MNISTDataset(normalize="01", flatten=True)
        image_01, _ = dataset_01[0]
        assert 0 <= image_01.min() <= image_01.max() <= 1
        
        # Test [-1, 1] normalization
        dataset_11 = MNISTDataset(normalize="11", flatten=True)
        image_11, _ = dataset_11[0]
        assert -1 <= image_11.min() <= image_11.max() <= 1
    
    def test_mnist_dataset_invalid_normalization(self):
        """Test error with invalid normalization method."""
        with pytest.raises(ValueError, match="normalize must be"):
            MNISTDataset(normalize="invalid")
    
    def test_mnist_dataset_cached_data(self, mock_torchvision_mnist):
        """Test cached data functionality."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        dataset = MNISTDataset(train=True, cache_data=True, flatten=True)
        
        # Get cached data
        data, targets = dataset.get_cached_data()
        
        assert data.shape == (100, 784)
        assert targets.shape == (100,)
        assert dataset._cached_data is not None
    
    def test_mnist_dataset_validation(self, mock_torchvision_mnist):
        """Test data validation functionality."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        dataset = MNISTDataset(train=True)
        results = dataset.validate_data()
        
        assert results["valid"] is True
        assert "num_samples" in results["statistics"]
        assert "data_shape" in results["statistics"]
        assert results["statistics"]["num_samples"] == 100
    
    def test_mnist_dataset_mismatched_data_labels(self, sample_idx_images, sample_idx_labels):
        """Test error when images and labels have different counts."""
        img_data, _ = sample_idx_images
        
        # Create labels with different count
        wrong_num_labels = 5  # Different from 10 images
        magic = 2049
        header = struct.pack('>II', magic, wrong_num_labels)
        labels = np.random.randint(0, 10, wrong_num_labels, dtype=np.uint8)
        lbl_data = header + labels.tobytes()
        
        with tempfile.NamedTemporaryFile(delete=False) as img_file, \
             tempfile.NamedTemporaryFile(delete=False) as lbl_file:
            
            img_file.write(img_data)
            lbl_file.write(lbl_data)
            img_file.flush()
            lbl_file.flush()
            
            try:
                with pytest.raises(ValueError, match="Mismatch between number of images"):
                    MNISTDataset(
                        images_file=img_file.name,
                        labels_file=lbl_file.name
                    )
            finally:
                os.unlink(img_file.name)
                os.unlink(lbl_file.name)


class TestMNISTDataLoader:
    """Test MNIST DataLoader functionality."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock MNIST dataset."""
        with patch('mnist_autoencoder.data.dataset.MNISTDataset') as mock_cls:
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset.get_cached_data = MagicMock(return_value=(
                torch.randn(100, 784), torch.randint(0, 10, (100,))
            ))
            mock_cls.return_value = mock_dataset
            yield mock_dataset
    
    def test_mnist_dataloader_initialization(self, mock_dataset):
        """Test DataLoader initialization."""
        loader = MNISTDataLoader(
            mock_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        assert loader.batch_size == 32
        assert loader.shuffle is True
        assert loader.num_workers == 0
    
    def test_mnist_dataloader_full_dataset(self, mock_dataset):
        """Test getting full dataset."""
        loader = MNISTDataLoader(mock_dataset, use_cache=True)
        
        data, targets = loader.get_full_dataset()
        
        assert data.shape == (100, 784)
        assert targets.shape == (100,)
        mock_dataset.get_cached_data.assert_called_once()
    
    def test_mnist_dataloader_train_test_split(self, mock_dataset):
        """Test train/test split functionality."""
        loader = MNISTDataLoader(mock_dataset, batch_size=16)
        
        train_loader, test_loader = loader.create_train_test_split(
            test_ratio=0.2, random_seed=42
        )
        
        # Check that loaders were created
        assert isinstance(train_loader, MNISTDataLoader)
        assert isinstance(test_loader, MNISTDataLoader)
        assert train_loader.batch_size == 16
        assert test_loader.batch_size == 16


class TestTransforms:
    """Test transform functionality."""
    
    def test_gaussian_noise(self):
        """Test Gaussian noise transform."""
        noise_transform = GaussianNoise(noise_factor=0.1)
        
        data = torch.zeros(10, 784)
        noisy_data = noise_transform(data)
        
        # Data should be different due to noise
        assert not torch.equal(data, noisy_data)
        assert noisy_data.shape == data.shape
    
    def test_flatten_transform(self):
        """Test flatten transform."""
        flatten = Flatten()
        
        data = torch.randn(5, 28, 28)
        flattened = flatten(data)
        
        assert flattened.shape == (5 * 28 * 28,)
    
    def test_normalize_01(self):
        """Test [0, 1] normalization."""
        normalize = Normalize("01")
        
        # Test with [0, 255] range data
        data = torch.randint(0, 256, (10, 784), dtype=torch.float)
        normalized = normalize(data)
        
        assert 0 <= normalized.min() <= normalized.max() <= 1
    
    def test_normalize_11(self):
        """Test [-1, 1] normalization."""
        normalize = Normalize("11")
        
        # Test with [0, 255] range data
        data = torch.randint(0, 256, (10, 784), dtype=torch.float)
        normalized = normalize(data)
        
        assert -1 <= normalized.min() <= normalized.max() <= 1
    
    def test_normalize_invalid_method(self):
        """Test error with invalid normalization method."""
        with pytest.raises(ValueError):
            Normalize("invalid")
    
    def test_validate_shape_success(self):
        """Test successful shape validation."""
        validator = ValidateShape(784)
        
        # Test with correct 1D shape
        data_1d = torch.randn(784)
        validated = validator(data_1d)
        assert torch.equal(data_1d, validated)
        
        # Test with correct 2D shape
        data_2d = torch.randn(10, 784)
        validated = validator(data_2d)
        assert torch.equal(data_2d, validated)
        
        # Test with correct 3D shape
        data_3d = torch.randn(10, 28, 28)
        validated = validator(data_3d)
        assert torch.equal(data_3d, validated)
    
    def test_validate_shape_failure(self):
        """Test shape validation failure."""
        validator = ValidateShape(784)
        
        # Test with wrong 1D shape
        with pytest.raises(ValueError):
            validator(torch.randn(512))  # Wrong number of features
        
        # Test with wrong 2D shape
        with pytest.raises(ValueError):
            validator(torch.randn(10, 512))  # Wrong number of features
        
        # Test with wrong 3D shape
        with pytest.raises(ValueError):
            validator(torch.randn(10, 32, 32))  # Wrong image dimensions
    
    def test_to_float32(self):
        """Test float32 conversion."""
        converter = ToFloat32()
        
        # Test with int tensor
        int_data = torch.randint(0, 256, (10, 784), dtype=torch.int32)
        float_data = converter(int_data)
        
        assert float_data.dtype == torch.float32
    
    def test_mnist_transforms_autoencoder(self):
        """Test MNISTTransforms autoencoder pipeline."""
        transform = MNISTTransforms.get_autoencoder_transform(
            normalize_method="01",
            add_noise=True,
            noise_factor=0.1
        )
        
        # Test with tensor data
        data = torch.randint(0, 256, (28, 28), dtype=torch.float)
        transformed = transform(data)
        
        assert transformed.shape == (784,)
        assert 0 <= transformed.min() <= transformed.max() <= 1.1  # Allow for noise
    
    def test_mnist_transforms_validation(self):
        """Test MNISTTransforms validation pipeline."""
        transform = MNISTTransforms.get_validation_transform("11")
        
        data = torch.randint(0, 256, (28, 28), dtype=torch.float)
        transformed = transform(data)
        
        assert transformed.shape == (784,)
        assert -1 <= transformed.min() <= transformed.max() <= 1
    
    def test_mnist_transforms_sas_compatible(self):
        """Test SAS-compatible transforms."""
        transform = MNISTTransforms.get_sas_compatible_transform()
        
        # Input as [0, 1] range (like from ToTensor())
        data = torch.rand(28, 28)
        transformed = transform(data)
        
        assert transformed.shape == (784,)
        assert 0 <= transformed.min() <= transformed.max() <= 255
    
    def test_create_training_transforms(self):
        """Test convenience function for training transforms."""
        transform = create_training_transforms(
            normalize="01",
            add_noise=True,
            noise_factor=0.05,
            validate_shape=True
        )
        
        data = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
        transformed = transform(data)
        
        assert transformed.shape == (784,)
        assert transformed.dtype == torch.float32
    
    def test_create_evaluation_transforms(self):
        """Test convenience function for evaluation transforms."""
        transform = create_evaluation_transforms("11", validate_shape=True)
        
        data = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
        transformed = transform(data)
        
        assert transformed.shape == (784,)
        assert -1 <= transformed.min() <= transformed.max() <= 1
    
    def test_create_sas_compatible_transforms(self):
        """Test convenience function for SAS-compatible transforms."""
        transform = create_sas_compatible_transforms()
        
        data = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
        transformed = transform(data)
        
        assert transformed.shape == (784,)
        assert transformed.dtype == torch.float32


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complete_pipeline(self, mock_torchvision_mnist):
        """Test complete data loading and preprocessing pipeline."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        # Create dataset with custom transforms
        custom_transform = create_training_transforms(
            normalize="01",
            add_noise=False,
            validate_shape=True
        )
        
        dataset = MNISTDataset(
            train=True,
            flatten=False,  # Let transform handle flattening
            transform=custom_transform,
            cache_data=True
        )
        
        # Create DataLoader
        loader = MNISTDataLoader(
            dataset,
            batch_size=16,
            shuffle=True
        )
        
        # Test data loading
        for batch_idx, (data, targets) in enumerate(loader):
            assert data.shape[0] <= 16  # Batch size
            assert data.shape[1] == 784  # Flattened features
            assert 0 <= data.min() <= data.max() <= 1  # Normalized
            
            if batch_idx >= 2:  # Test a few batches
                break
    
    def test_sas_format_compatibility(self, mock_torchvision_mnist):
        """Test that output format matches SAS createData.sas expectations."""
        mock_mnist, fake_data, fake_targets = mock_torchvision_mnist
        
        # Use SAS-compatible transforms
        dataset = MNISTDataset(
            train=True,
            flatten=True,
            normalize="01"  # Will be overridden by transform
        )
        
        # Get data in format that matches SAS output
        data, targets = dataset.get_cached_data()
        
        # Check format expectations based on SAS code analysis
        assert data.shape[1] == 784  # 784 features (pixels)
        assert len(data) == len(targets)  # Same number of samples
        assert targets.dtype in [torch.long, torch.int64]  # Integer labels
        
        # Data should be in proper range for autoencoder training
        assert 0 <= data.min() <= data.max() <= 1


# Fixtures for sample data
@pytest.fixture
def sample_idx_images():
    """Create sample IDX images file data."""
    # Create fake MNIST image data
    num_images, rows, cols = 10, 28, 28
    magic = 2051  # Magic number for images
    
    # Header + image data
    header = struct.pack('>IIII', magic, num_images, rows, cols)
    images = np.random.randint(0, 256, (num_images, rows, cols), dtype=np.uint8)
    data = header + images.tobytes()
    
    return data, images


@pytest.fixture
def sample_idx_labels():
    """Create sample IDX labels file data."""
    # Create fake MNIST label data
    num_labels = 10
    magic = 2049  # Magic number for labels
    
    # Header + label data
    header = struct.pack('>II', magic, num_labels)
    labels = np.random.randint(0, 10, num_labels, dtype=np.uint8)
    data = header + labels.tobytes()
    
    return data, labels


if __name__ == "__main__":
    pytest.main([__file__])
