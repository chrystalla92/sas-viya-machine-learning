"""
Example usage of MNIST data loading and preprocessing.

This script demonstrates how to use the MNIST dataset loader with both
torchvision integration and direct IDX file reading.
"""

import torch
from mnist_autoencoder.data import MNISTDataset, MNISTDataLoader, MNISTTransforms


def main():
    """Demonstrate MNIST data loading and preprocessing."""
    
    print("MNIST Data Loading Examples")
    print("=" * 40)
    
    # Example 1: Load from torchvision (automatic download)
    print("\n1. Loading MNIST from torchvision...")
    dataset = MNISTDataset(
        train=True,
        download=True,
        flatten=True,
        normalize="01",
        cache_data=True
    )
    
    print(f"Loaded {len(dataset)} training samples")
    print(f"Data shape: {dataset.data.shape}")
    print(f"Targets shape: {dataset.targets.shape}")
    
    # Validate data
    validation_results = dataset.validate_data()
    print(f"Data validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
    print(f"Memory usage: {validation_results['statistics']['memory_usage_mb']:.2f} MB")
    
    # Example 2: Get a sample
    print("\n2. Sample data inspection...")
    image, label = dataset[0]
    print(f"Sample shape: {image.shape}")
    print(f"Sample label: {label.item()}")
    print(f"Pixel value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Example 3: Create DataLoader with batching
    print("\n3. Creating DataLoader...")
    dataloader = MNISTDataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a batch
    batch_data, batch_targets = next(iter(dataloader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")
    
    # Example 4: Create train/test split
    print("\n4. Creating train/test split...")
    train_loader, test_loader = dataloader.create_train_test_split(
        test_ratio=0.2, 
        random_seed=42
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Example 5: SAS-compatible format
    print("\n5. SAS-compatible data format...")
    sas_transform = MNISTTransforms.get_sas_compatible_transform()
    
    # Apply to a sample
    raw_image = dataset.data[0]  # Get raw image data
    sas_format = sas_transform(raw_image)
    
    print(f"SAS format shape: {sas_format.shape}")
    print(f"SAS format range: [{sas_format.min():.1f}, {sas_format.max():.1f}]")
    
    # Example 6: Custom transforms
    print("\n6. Custom preprocessing pipeline...")
    from mnist_autoencoder.data.transforms import create_training_transforms
    
    custom_transform = create_training_transforms(
        normalize="11",
        add_noise=True,
        noise_factor=0.05,
        validate_shape=True
    )
    
    custom_dataset = MNISTDataset(
        train=True,
        transform=custom_transform,
        flatten=False  # Let transform handle it
    )
    
    sample_image, sample_label = custom_dataset[0]
    print(f"Custom transform output shape: {sample_image.shape}")
    print(f"Custom transform range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    
    # Example 7: Performance comparison
    print("\n7. Performance comparison...")
    
    # Cached data access
    import time
    
    start_time = time.time()
    cached_data, cached_targets = dataset.get_cached_data()
    cached_time = time.time() - start_time
    
    print(f"Cached data access time: {cached_time:.4f} seconds")
    print(f"Cached data shape: {cached_data.shape}")
    
    print("\nAll examples completed successfully!")


def demonstrate_idx_loading():
    """
    Demonstrate loading from IDX files (requires actual IDX files).
    
    This function shows how to load MNIST data directly from IDX format files,
    which is useful when you have the raw MNIST dataset files.
    """
    print("\nIDX File Loading Example")
    print("=" * 30)
    
    # Note: This requires actual IDX files to be present
    # You would typically download them from http://yann.lecun.com/exdb/mnist/
    
    try:
        dataset = MNISTDataset(
            images_file="path/to/train-images.idx3-ubyte",
            labels_file="path/to/train-labels.idx1-ubyte",
            flatten=True,
            normalize="01"
        )
        
        print(f"Loaded {len(dataset)} samples from IDX files")
        
        # Validate the loaded data
        validation_results = dataset.validate_data()
        if validation_results['valid']:
            print("IDX data validation: PASSED")
        else:
            print("IDX data validation: FAILED")
            for error in validation_results['errors']:
                print(f"  Error: {error}")
                
    except FileNotFoundError:
        print("IDX files not found. To use this feature:")
        print("1. Download MNIST IDX files from http://yann.lecun.com/exdb/mnist/")
        print("2. Update the file paths in this function")
    except Exception as e:
        print(f"Error loading IDX files: {e}")


if __name__ == "__main__":
    main()
    demonstrate_idx_loading()
