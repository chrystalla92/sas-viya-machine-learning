#!/usr/bin/env python3
"""
Test script for MNIST data loading implementation.

This script verifies that the MNIST data loading and preprocessing
implementation works correctly and produces consistent results
with the first 10 samples for SAS compatibility testing.
"""

import sys
import os
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import MNISTDataLoader, create_mnist_loader


def test_basic_functionality():
    """Test basic MNIST loading functionality."""
    print("=" * 60)
    print("Testing Basic MNIST Loading Functionality")
    print("=" * 60)
    
    try:
        # Create loader
        loader = MNISTDataLoader(data_root="./data", batch_size=32)
        
        # Load datasets
        loader.load_datasets()
        print("✓ Datasets loaded successfully")
        
        # Get statistics
        stats = loader.get_data_stats()
        print(f"✓ Training samples: {stats['train_samples']}")
        print(f"✓ Test samples: {stats['test_samples']}")
        print(f"✓ Data shape: {stats['data_shape']}")
        print(f"✓ Data range: [{stats['data_min']:.3f}, {stats['data_max']:.3f}]")
        print(f"✓ Data mean: {stats['data_mean']:.3f}, std: {stats['data_std']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality: {e}")
        return False


def test_first_10_samples():
    """Test extraction of first 10 samples for SAS compatibility."""
    print("\n" + "=" * 60)
    print("Testing First 10 Samples Extraction")
    print("=" * 60)
    
    try:
        # Create loader
        loader = create_mnist_loader(data_root="./data")
        
        # Get first 10 samples
        data, labels = loader.get_first_n_samples(n=10, from_train=True)
        
        print(f"✓ Extracted 10 samples with data shape: {data.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"✓ Labels: {labels.tolist()}")
        
        # Validate expected shape (10, 784)
        assert data.shape == (10, 784), f"Expected (10, 784), got {data.shape}"
        print("✓ Shape validation passed")
        
        # Validate data is in [-1, 1] range (midrange scaling)
        assert data.min() >= -1.001 and data.max() <= 1.001, \
            f"Data should be in [-1,1], got [{data.min():.3f}, {data.max():.3f}]"
        print("✓ Midrange scaling validation passed")
        
        # Print sample statistics for each image
        print("\nSample statistics for first 10 images:")
        for i in range(10):
            sample = data[i]
            print(f"  Sample {i} (label {labels[i]}): "
                  f"min={sample.min():.3f}, max={sample.max():.3f}, "
                  f"mean={sample.mean():.3f}, std={sample.std():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in first 10 samples test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_export():
    """Test CSV export functionality for SAS compatibility."""
    print("\n" + "=" * 60)
    print("Testing CSV Export for SAS Compatibility")
    print("=" * 60)
    
    try:
        # Create loader
        loader = create_mnist_loader(data_root="./data")
        
        # Export first 10 samples to CSV
        csv_path = loader.export_to_csv(
            n_samples=10,
            output_path="mnist_train_10_test.csv",
            include_labels=True
        )
        
        print(f"✓ CSV exported to: {csv_path}")
        
        # Read back and validate
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print(f"✓ CSV shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns[:5])}...{list(df.columns[-3:])}")
        
        # Should have 785 columns (1 label + 784 pixels) and 10 rows
        assert df.shape == (10, 785), f"Expected (10, 785), got {df.shape}"
        print("✓ CSV format validation passed")
        
        # Check data ranges
        pixel_data = df.iloc[:, 1:].values  # Skip first column (labels)
        print(f"✓ Pixel data range: [{pixel_data.min():.3f}, {pixel_data.max():.3f}]")
        
        # Print first few rows
        print("\nFirst 3 rows (label + first 5 pixels):")
        print(df.iloc[:3, :6])
        
        return True
        
    except Exception as e:
        print(f"✗ Error in CSV export test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loaders():
    """Test PyTorch data loader creation."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Data Loaders")
    print("=" * 60)
    
    try:
        # Create loader
        loader = create_mnist_loader(data_root="./data", batch_size=16, validation_split=0.1)
        
        # Create data loaders
        train_loader, test_loader, val_loader = loader.create_data_loaders()
        
        print(f"✓ Training batches: {len(train_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")
        print(f"✓ Validation batches: {len(val_loader) if val_loader else 'None'}")
        
        # Test a batch
        batch_data, batch_labels = next(iter(train_loader))
        print(f"✓ Batch data shape: {batch_data.shape}")
        print(f"✓ Batch labels shape: {batch_labels.shape}")
        print(f"✓ Batch data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data loaders test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consistency_check():
    """Compare results with expected SAS-like behavior."""
    print("\n" + "=" * 60)
    print("Testing Consistency with SAS Behavior")
    print("=" * 60)
    
    try:
        # Create loader
        loader = create_mnist_loader(data_root="./data")
        
        # Get raw data (no scaling)
        loader_raw = MNISTDataLoader(data_root="./data")
        loader_raw.load_datasets(apply_scaling=False)
        data_raw, labels_raw = loader_raw.get_first_n_samples(n=10, return_tensors=True)
        
        # Get scaled data
        data_scaled, labels_scaled = loader.get_first_n_samples(n=10, return_tensors=True)
        
        print(f"✓ Raw data range: [{data_raw.min():.3f}, {data_raw.max():.3f}]")
        print(f"✓ Scaled data range: [{data_scaled.min():.3f}, {data_scaled.max():.3f}]")
        
        # Verify midrange scaling: (x - 0) / (1 - 0) * 2 - 1 = x * 2 - 1
        expected_scaled = data_raw * 2.0 - 1.0
        diff = torch.abs(data_scaled - expected_scaled).max()
        print(f"✓ Scaling accuracy (max diff): {diff:.6f}")
        
        assert diff < 1e-6, f"Scaling not accurate, max diff: {diff}"
        print("✓ Midrange scaling matches expected formula")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in consistency check: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("MNIST Data Loading Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_first_10_samples,
        test_csv_export,
        test_data_loaders,
        test_consistency_check,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
            print("✓ PASSED\n")
        else:
            print("✗ FAILED\n")
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Implementation is ready.")
        print("\nThe MNIST data loading implementation:")
        print("- Automatically downloads MNIST dataset")
        print("- Applies midrange scaling to [-1,1] range")
        print("- Flattens images to 784-dimensional vectors")
        print("- Provides first N samples extraction")
        print("- Exports to CSV for SAS compatibility")
        print("- Includes proper validation and error handling")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
