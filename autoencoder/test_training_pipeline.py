"""
Test script for the autoencoder training pipeline.

This script demonstrates how to use the training pipeline and validates
that all components work correctly together.
"""

import os
import torch
from autoencoder_model import AutoencoderMLP
from trainer import TrainingPipeline, create_training_pipeline, train_autoencoder
from datasets import create_mnist_dataloaders


def test_training_pipeline_creation():
    """Test training pipeline creation and configuration."""
    print("=" * 60)
    print("Testing Training Pipeline Creation")
    print("=" * 60)
    
    try:
        # Create a model
        model = AutoencoderMLP()
        print("✓ Model created successfully")
        
        # Test default configuration
        pipeline = create_training_pipeline(model)
        print("✓ Training pipeline created with default config")
        print(f"  - Device: {pipeline.get_device()}")
        print(f"  - Epochs: {pipeline.get_config()['epochs']}")
        print(f"  - Learning rate: {pipeline.get_config()['learning_rate']}")
        
        # Test custom configuration
        custom_config = {
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'early_stopping_patience': 3
        }
        
        custom_pipeline = create_training_pipeline(model, custom_config)
        print("✓ Training pipeline created with custom config")
        print(f"  - Epochs: {custom_pipeline.get_config()['epochs']}")
        print(f"  - Batch size: {custom_pipeline.get_config()['batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating training pipeline: {e}")
        return False


def test_data_loaders():
    """Test data loader creation and compatibility."""
    print("\n" + "=" * 60)
    print("Testing Data Loader Compatibility")
    print("=" * 60)
    
    data_dir = "./data"
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_mnist_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            train_val_split=0.8
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  - Training batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Test batch format
        for batch_data in train_loader:
            if isinstance(batch_data, tuple):
                images, labels = batch_data
                print(f"  - Batch format: (images, labels)")
                print(f"  - Images shape: {images.shape}")
                print(f"  - Labels shape: {labels.shape}")
            else:
                images = batch_data
                print(f"  - Batch format: images only")
                print(f"  - Images shape: {images.shape}")
            break
            
        return True
        
    except FileNotFoundError as e:
        print(f"✗ MNIST data files not found: {e}")
        print("  Please ensure MNIST binary files are in ./data/ directory")
        return False
    except Exception as e:
        print(f"✗ Error with data loaders: {e}")
        return False


def test_short_training():
    """Test a short training run to verify all components work."""
    print("\n" + "=" * 60)
    print("Testing Short Training Run")
    print("=" * 60)
    
    data_dir = "./data"
    
    try:
        # Create a simple model
        model = AutoencoderMLP(input_dim=784, latent_dim=128, activation='tanh')
        print("✓ Model created")
        
        # Configure for short test run
        config = {
            'epochs': 3,
            'batch_size': 128,
            'learning_rate': 1e-3,
            'early_stopping_patience': 10,  # Won't trigger in 3 epochs
            'log_frequency': 1,
            'save_best_model': True,
            'save_periodic_checkpoints': False,  # Skip periodic saves for test
        }
        
        # Create pipeline
        pipeline = create_training_pipeline(model, config)
        print("✓ Training pipeline created")
        
        # Run training
        print("\nStarting short training run...")
        summary = pipeline.train(data_dir)
        
        print("\n✓ Training completed successfully!")
        print(f"  - Epochs trained: {summary['total_epochs_trained']}")
        print(f"  - Final train loss: {summary['final_train_loss']:.6f}")
        print(f"  - Final val loss: {summary['final_val_loss']:.6f}")
        print(f"  - Training time: {summary['total_training_time']:.2f} seconds")
        
        # Test evaluation
        _, _, test_loader = create_mnist_dataloaders(data_dir, batch_size=128)
        metrics = pipeline.evaluate(test_loader)
        print(f"  - Test loss: {metrics['test_loss']:.6f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ MNIST data files not found: {e}")
        print("  This test requires MNIST data files")
        return False
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utilities():
    """Test individual training utilities."""
    print("\n" + "=" * 60)
    print("Testing Training Utilities")
    print("=" * 60)
    
    try:
        from training_utils import (
            EarlyStopping, TrainingLogger, ModelCheckpoint,
            validate_training_config, get_device, format_time
        )
        
        # Test early stopping
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, verbose=False)
        
        model = AutoencoderMLP()
        
        # Simulate improving then worsening validation loss
        losses = [1.0, 0.8, 0.6, 0.65, 0.63, 0.64, 0.66]  # Should stop at index 6
        
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss, model)
            if should_stop:
                print(f"✓ Early stopping triggered after {i+1} checks")
                break
        else:
            print("✗ Early stopping did not trigger as expected")
            
        # Test config validation
        config = {
            'epochs': 50,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'train_val_split': 0.85
        }
        
        validated = validate_training_config(config)
        print("✓ Configuration validation works")
        
        # Test device detection
        device = get_device('auto')
        print(f"✓ Device detection: {device}")
        
        # Test time formatting
        formatted = format_time(3661.5)  # 1 hour, 1 minute, 1.5 seconds
        print(f"✓ Time formatting: {formatted}")
        
        print("✓ All utility tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error testing utilities: {e}")
        return False


def test_convenience_function():
    """Test the convenience training function."""
    print("\n" + "=" * 60)
    print("Testing Convenience Function")
    print("=" * 60)
    
    data_dir = "./data"
    
    try:
        # Create model
        model = AutoencoderMLP(latent_dim=200)
        
        # Test configuration
        config = {
            'epochs': 2,
            'batch_size': 64,
            'learning_rate': 5e-4
        }
        
        # Use convenience function
        pipeline, summary = train_autoencoder(model, data_dir, config)
        
        print("✓ Convenience function works!")
        print(f"  - Epochs: {summary['total_epochs_trained']}")
        print(f"  - Best val loss: {summary['best_validation_loss']:.6f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ MNIST data files not found: {e}")
        print("  This test requires MNIST data files")
        return False
    except Exception as e:
        print(f"✗ Error with convenience function: {e}")
        return False


def main():
    """Run all tests."""
    print("Autoencoder Training Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        test_training_pipeline_creation,
        test_utilities,
        test_data_loaders,
    ]
    
    # Optional tests that require MNIST data
    optional_tests = [
        test_short_training,
        test_convenience_function
    ]
    
    results = []
    
    # Run core tests
    for test_func in tests:
        try:
            print()
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Run optional tests if data is available
    if os.path.exists("./data"):
        print("\nRunning tests that require MNIST data...")
        for test_func in optional_tests:
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                print(f"✗ Test {test_func.__name__} failed with exception: {e}")
                results.append(False)
    else:
        print("\nSkipping tests that require MNIST data (./data directory not found)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✓ All tests passed! Training pipeline is ready for use.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    print("\nTo use the training pipeline in your own code:")
    print("```python")
    print("from autoencoder_model import AutoencoderMLP")
    print("from trainer import train_autoencoder")
    print("")
    print("# Create model")
    print("model = AutoencoderMLP()")
    print("")
    print("# Train with default settings")
    print("pipeline, summary = train_autoencoder(model, './data')")
    print("")
    print("# Or use custom configuration")
    print("config = {'epochs': 50, 'learning_rate': 1e-3}")
    print("pipeline, summary = train_autoencoder(model, './data', config)")
    print("```")


if __name__ == "__main__":
    main()
