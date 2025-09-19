"""
Test script to verify MNIST data loading functionality.

This script demonstrates how to use the data loading utilities and validates
that the implementation meets the specified requirements.
"""

import os
import torch
from data_loader import (
    load_mnist_training_data, 
    load_mnist_test_data,
    midrange_standardize
)
from datasets import (
    create_mnist_dataloaders,
    get_train_dataloader,
    get_test_dataloader
)


def test_basic_loading():
    """Test basic data loading functionality."""
    print("=" * 60)
    print("Testing Basic Data Loading")
    print("=" * 60)
    
    # Test data directory (adjust path as needed)
    data_dir = "./data"  # or "../data" or wherever MNIST files are located
    
    try:
        # Test training data loading
        print("Loading training data...")
        train_images, train_labels = load_mnist_training_data(data_dir)
        print(f"✓ Training data loaded successfully")
        print(f"  - Images shape: {train_images.shape}")
        print(f"  - Labels shape: {train_labels.shape}")
        print(f"  - Images dtype: {train_images.dtype}")
        print(f"  - Labels dtype: {train_labels.dtype}")
        print(f"  - Images range: [{train_images.min():.4f}, {train_images.max():.4f}]")
        print(f"  - Labels range: [{train_labels.min()}, {train_labels.max()}]")
        
        # Test test data loading
        print("\nLoading test data...")
        test_images, test_labels = load_mnist_test_data(data_dir)
        print(f"✓ Test data loaded successfully")
        print(f"  - Images shape: {test_images.shape}")
        print(f"  - Labels shape: {test_labels.shape}")
        print(f"  - Images dtype: {test_images.dtype}")
        print(f"  - Labels dtype: {test_labels.dtype}")
        print(f"  - Images range: [{test_images.min():.4f}, {test_images.max():.4f}]")
        print(f"  - Labels range: [{test_labels.min()}, {test_labels.max()}]")
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("  Please ensure MNIST binary files are in the specified directory:")
        print("  - train-images.idx3-ubyte")
        print("  - train-labels.idx1-ubyte") 
        print("  - t10k-images.idx3-ubyte")
        print("  - t10k-labels.idx1-ubyte")
        return False
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    return True


def test_standardization():
    """Test midrange standardization functionality."""
    print("\n" + "=" * 60)
    print("Testing Midrange Standardization")
    print("=" * 60)
    
    # Create test data
    test_data = torch.tensor([0, 50, 100, 150, 200, 255], dtype=torch.float32)
    print(f"Original data: {test_data}")
    
    # Apply standardization
    standardized = midrange_standardize(test_data.numpy())
    print(f"Standardized data: {standardized}")
    
    # Verify properties
    print(f"Min: {standardized.min():.4f}")
    print(f"Max: {standardized.max():.4f}")
    print(f"Mean: {standardized.mean():.4f}")
    
    # For range [0, 255], midrange should be 127.5
    # So standardized range should be [-0.5, 0.5]
    expected_min = -0.5
    expected_max = 0.5
    
    if abs(standardized.min() - expected_min) < 1e-6 and abs(standardized.max() - expected_max) < 1e-6:
        print("✓ Standardization working correctly")
    else:
        print(f"✗ Standardization incorrect. Expected range [{expected_min}, {expected_max}]")


def test_dataloaders():
    """Test PyTorch DataLoader functionality."""
    print("\n" + "=" * 60)
    print("Testing DataLoaders")
    print("=" * 60)
    
    data_dir = "./data"
    batch_size = 64
    
    try:
        # Test simple dataloader creation
        print("Creating simple train dataloader...")
        train_loader = get_train_dataloader(data_dir, batch_size)
        print(f"✓ Train DataLoader created with {len(train_loader)} batches")
        
        print("Creating simple test dataloader...")
        test_loader = get_test_dataloader(data_dir, batch_size)
        print(f"✓ Test DataLoader created with {len(test_loader)} batches")
        
        # Test batch iteration
        print("\nTesting batch iteration...")
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx == 0:  # Just test the first batch
                if isinstance(batch_data, tuple):
                    images, labels = batch_data
                    print(f"✓ First batch (with labels):")
                    print(f"  - Images shape: {images.shape}")
                    print(f"  - Labels shape: {labels.shape}")
                else:
                    images = batch_data
                    print(f"✓ First batch (autoencoder mode):")
                    print(f"  - Images shape: {images.shape}")
                break
        
        # Test train/val/test split
        print("\nTesting train/validation/test split...")
        train_loader, val_loader, test_loader = create_mnist_dataloaders(
            data_dir, batch_size=32, train_val_split=0.8
        )
        print(f"✓ Split created:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
    except Exception as e:
        print(f"✗ Error with DataLoaders: {e}")
        return False
    
    return True


def test_memory_efficiency():
    """Test memory usage and efficiency."""
    print("\n" + "=" * 60)
    print("Testing Memory Efficiency")
    print("=" * 60)
    
    data_dir = "./data"
    
    try:
        # Monitor memory usage during loading
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load data
        train_images, train_labels = load_mnist_training_data(data_dir)
        test_images, test_labels = load_mnist_test_data(data_dir)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"Memory usage:")
        print(f"  - Before loading: {mem_before:.1f} MB")
        print(f"  - After loading: {mem_after:.1f} MB") 
        print(f"  - Memory used: {mem_used:.1f} MB")
        
        # Calculate expected memory usage
        # 70,000 images * 784 pixels * 4 bytes (float32) = ~215 MB for images
        # 70,000 labels * 8 bytes (int64) = ~0.5 MB for labels
        expected_mb = (70000 * 784 * 4 + 70000 * 8) / 1024 / 1024
        print(f"  - Expected: ~{expected_mb:.1f} MB")
        
        if mem_used < expected_mb * 2:  # Allow 2x overhead for reasonable usage
            print("✓ Memory usage is reasonable")
        else:
            print("⚠ Memory usage seems high")
            
    except ImportError:
        print("psutil not available, skipping memory test")
    except Exception as e:
        print(f"✗ Error in memory test: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("MNIST Data Loading Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_loading,
        test_standardization,
        test_dataloaders,
        test_memory_efficiency
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✓ All tests passed! Implementation is ready for use.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
