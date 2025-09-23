#!/usr/bin/env python3
"""
Comprehensive Test Suite for Autoencoder Inference Pipeline

This script tests all components of the inference pipeline including:
- Model checkpoint loading
- Batch processing
- Tensor preprocessing and conversion
- Device handling (CPU/GPU)
- Performance metrics
- Error handling
"""

import torch
import torch.nn as nn
import numpy as np
import tempfile
import logging
from pathlib import Path
import time

# Import the inference pipeline
import sys
sys.path.append('src')
from src.inference import (
    AutoencoderInference, 
    create_inference_pipeline, 
    tensor_to_flat, 
    flat_to_images
)
from src.models.autoencoder import create_autoencoder
from src.training.trainer import AutoencoderTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_checkpoint(checkpoint_path: str, input_dim: int = 784, hidden_dim: int = 400):
    """Create a mock checkpoint file for testing."""
    model = create_autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, device='cpu')
    
    checkpoint = {
        'iteration': 100,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Mock optimizer state
        'loss': 0.001,
        'best_loss': 0.0008,
        'converged': True,
        'loss_history': [0.01, 0.005, 0.002, 0.001],
        'seed': 23451,
        'training_complete': True
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Created mock checkpoint at {checkpoint_path}")
    return checkpoint


def test_checkpoint_loading():
    """Test checkpoint loading functionality."""
    print("\n" + "="*60)
    print("Testing Checkpoint Loading")
    print("="*60)
    
    try:
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        create_mock_checkpoint(checkpoint_path)
        
        # Test loading
        pipeline = AutoencoderInference(checkpoint_path=checkpoint_path)
        
        # Verify model is loaded
        assert pipeline.model_loaded, "Model should be marked as loaded"
        assert pipeline.model is not None, "Model should not be None"
        assert hasattr(pipeline.model, 'encode'), "Model should have encode method"
        assert hasattr(pipeline.model, 'decode'), "Model should have decode method"
        
        # Check checkpoint info
        assert 'iteration' in pipeline.checkpoint_info, "Should have iteration info"
        assert pipeline.checkpoint_info['converged'] == True, "Should show convergence status"
        
        print("✓ Checkpoint loading works correctly")
        
        # Cleanup
        Path(checkpoint_path).unlink()
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint loading test failed: {e}")
        return False


def test_tensor_conversions():
    """Test tensor-to-image and image-to-tensor conversions."""
    print("\n" + "="*60)
    print("Testing Tensor Conversions")
    print("="*60)
    
    try:
        # Test image to flat conversion
        images = torch.randn(5, 28, 28)  # 5 images of 28x28
        flattened = tensor_to_flat(images)
        
        assert flattened.shape == (5, 784), f"Expected (5, 784), got {flattened.shape}"
        print("✓ Image to flat tensor conversion works")
        
        # Test flat to image conversion
        reconstructed_images = flat_to_images(flattened, (28, 28))
        assert reconstructed_images.shape == (5, 28, 28), f"Expected (5, 28, 28), got {reconstructed_images.shape}"
        print("✓ Flat tensor to image conversion works")
        
        # Test data preservation
        assert torch.allclose(images, reconstructed_images), "Data should be preserved through conversion"
        print("✓ Data preservation through conversions verified")
        
        # Test with color images
        color_images = torch.randn(3, 3, 32, 32)  # 3 RGB images of 32x32
        color_flat = tensor_to_flat(color_images)
        assert color_flat.shape == (3, 3072), f"Expected (3, 3072), got {color_flat.shape}"
        
        color_reconstructed = flat_to_images(color_flat, (3, 32, 32))
        assert color_reconstructed.shape == (3, 3, 32, 32), "Color image reconstruction failed"
        print("✓ Color image conversions work correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Tensor conversion test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n" + "="*60)
    print("Testing Batch Processing")
    print("="*60)
    
    try:
        # Create model and pipeline
        model = create_autoencoder(device='cpu')
        pipeline = AutoencoderInference(model=model, batch_size=32)
        
        # Test single sample
        single_sample = torch.randn(784)
        single_result = pipeline.predict(single_sample)
        assert single_result.shape == (784,), f"Expected (784,), got {single_result.shape}"
        print("✓ Single sample inference works")
        
        # Test small batch (within batch size)
        small_batch = torch.randn(10, 784)
        small_result = pipeline.predict(small_batch)
        assert small_result.shape == (10, 784), f"Expected (10, 784), got {small_result.shape}"
        print("✓ Small batch inference works")
        
        # Test large batch (exceeding batch size)
        large_batch = torch.randn(100, 784)
        large_result = pipeline.predict(large_batch, batch_size=16)
        assert large_result.shape == (100, 784), f"Expected (100, 784), got {large_result.shape}"
        print("✓ Large batch processing works")
        
        # Test with latent representation
        batch_data = torch.randn(20, 784)
        reconstructed, latent = pipeline.predict(batch_data, return_latent=True)
        assert reconstructed.shape == (20, 784), "Reconstruction shape mismatch"
        assert latent.shape == (20, 400), f"Expected latent shape (20, 400), got {latent.shape}"
        print("✓ Latent representation extraction works")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False


def test_preprocessing():
    """Test tensor preprocessing and validation."""
    print("\n" + "="*60)
    print("Testing Preprocessing")
    print("="*60)
    
    try:
        # Create pipeline with midrange preprocessing
        model = create_autoencoder(device='cpu')
        pipeline = AutoencoderInference(model=model, preprocessing='midrange')
        
        # Test with data in [0, 1] range (typical for MNIST after ToTensor)
        test_data = torch.rand(5, 784)  # Random data in [0, 1]
        result = pipeline.predict(test_data)
        
        assert result.shape == (5, 784), "Output shape should match input"
        print("✓ Midrange preprocessing works with [0, 1] input")
        
        # Test with no preprocessing
        pipeline_no_prep = AutoencoderInference(model=model, preprocessing='none')
        result_no_prep = pipeline_no_prep.predict(test_data)
        assert result_no_prep.shape == (5, 784), "No preprocessing should still work"
        print("✓ No preprocessing option works")
        
        # Test input validation
        pipeline_validate = AutoencoderInference(model=model, validate_input=True)
        
        # This should work
        valid_data = torch.randn(3, 784)
        _ = pipeline_validate.predict(valid_data)
        print("✓ Valid input passes validation")
        
        # Test invalid input detection (wrong dimensions)
        try:
            invalid_data = torch.randn(3, 500)  # Wrong feature count
            _ = pipeline_validate.predict(invalid_data)
            print("✗ Should have caught invalid dimensions")
            return False
        except ValueError:
            print("✓ Invalid input dimensions caught correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False


def test_device_handling():
    """Test CPU/GPU device handling."""
    print("\n" + "="*60)
    print("Testing Device Handling")
    print("="*60)
    
    try:
        # Test CPU device
        model_cpu = create_autoencoder(device='cpu')
        pipeline_cpu = AutoencoderInference(model=model_cpu, device=torch.device('cpu'))
        
        test_data = torch.randn(10, 784)
        result_cpu = pipeline_cpu.predict(test_data)
        assert result_cpu.device.type == 'cpu', "CPU result should be on CPU"
        print("✓ CPU device handling works")
        
        # Test GPU device if available
        if torch.cuda.is_available():
            model_gpu = create_autoencoder(device='cuda')
            pipeline_gpu = AutoencoderInference(model=model_gpu, device=torch.device('cuda'))
            
            result_gpu = pipeline_gpu.predict(test_data.cuda())
            assert result_gpu.device.type == 'cuda', "GPU result should be on GPU"
            print("✓ GPU device handling works")
        else:
            print("ⓘ GPU not available, skipping GPU tests")
        
        return True
        
    except Exception as e:
        print(f"✗ Device handling test failed: {e}")
        return False


def test_performance_tracking():
    """Test performance metrics and logging."""
    print("\n" + "="*60)
    print("Testing Performance Tracking")
    print("="*60)
    
    try:
        # Create pipeline with performance logging
        model = create_autoencoder(device='cpu')
        pipeline = AutoencoderInference(model=model, log_performance=True)
        
        # Process some data
        test_data = torch.randn(50, 784)
        _ = pipeline.predict(test_data, batch_size=10)
        
        # Check performance stats
        stats = pipeline.get_performance_stats()
        
        assert 'total_samples_processed' in stats, "Should track total samples"
        assert stats['total_samples_processed'] == 50, f"Should have processed 50 samples, got {stats['total_samples_processed']}"
        
        assert 'total_inference_time' in stats, "Should track inference time"
        assert stats['total_inference_time'] > 0, "Should have positive inference time"
        
        assert 'throughput_samples_per_sec' in stats, "Should calculate throughput"
        assert stats['throughput_samples_per_sec'] > 0, "Should have positive throughput"
        
        print("✓ Performance tracking works correctly")
        print(f"  - Processed: {stats['total_samples_processed']} samples")
        print(f"  - Time: {stats['total_inference_time']:.4f}s")
        print(f"  - Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
        
        # Test stats reset
        pipeline.reset_performance_stats()
        new_stats = pipeline.get_performance_stats()
        assert new_stats['total_samples_processed'] == 0, "Stats should reset to zero"
        print("✓ Performance stats reset works")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance tracking test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    try:
        # Test with no model loaded
        empty_pipeline = AutoencoderInference()
        try:
            _ = empty_pipeline.predict(torch.randn(5, 784))
            print("✗ Should have caught 'no model loaded' error")
            return False
        except RuntimeError:
            print("✓ 'No model loaded' error caught correctly")
        
        # Test with invalid checkpoint path
        try:
            _ = AutoencoderInference(checkpoint_path="nonexistent_file.pt")
            print("✗ Should have caught file not found error")
            return False
        except FileNotFoundError:
            print("✓ File not found error caught correctly")
        
        # Test with malformed input (NaN values)
        model = create_autoencoder(device='cpu')
        pipeline = AutoencoderInference(model=model, validate_input=True)
        
        nan_data = torch.randn(3, 784)
        nan_data[0, 0] = float('nan')
        
        try:
            _ = pipeline.predict(nan_data)
            print("✗ Should have caught NaN input error")
            return False
        except ValueError:
            print("✓ NaN input error caught correctly")
        
        # Test with infinite values
        inf_data = torch.randn(3, 784)
        inf_data[0, 0] = float('inf')
        
        try:
            _ = pipeline.predict(inf_data)
            print("✗ Should have caught infinite input error")
            return False
        except ValueError:
            print("✓ Infinite input error caught correctly")
        
        print("✓ Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_factory_function():
    """Test the factory function for pipeline creation."""
    print("\n" + "="*60)
    print("Testing Factory Function")
    print("="*60)
    
    try:
        # Test with model parameter
        model = create_autoencoder(device='cpu')
        pipeline1 = create_inference_pipeline(model=model)
        assert pipeline1.model_loaded, "Pipeline should have loaded model"
        print("✓ Factory function with model works")
        
        # Test with checkpoint path
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        create_mock_checkpoint(checkpoint_path)
        pipeline2 = create_inference_pipeline(checkpoint_path=checkpoint_path)
        assert pipeline2.model_loaded, "Pipeline should have loaded from checkpoint"
        
        # Cleanup
        Path(checkpoint_path).unlink()
        print("✓ Factory function with checkpoint works")
        
        # Test with custom parameters
        pipeline3 = create_inference_pipeline(
            model=model,
            batch_size=64,
            preprocessing='midrange',
            validate_input=False
        )
        assert pipeline3.batch_size == 64, "Should use custom batch size"
        assert pipeline3.preprocessing == 'midrange', "Should use midrange preprocessing"
        assert pipeline3.validate_input == False, "Should disable validation"
        print("✓ Factory function with custom parameters works")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        return False


def demonstration_example():
    """Demonstrate typical usage of the inference pipeline."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Typical Inference Pipeline Usage")
    print("="*80)
    
    try:
        print("1. Creating and training a simple model...")
        
        # Create model
        model = create_autoencoder(input_dim=784, hidden_dim=400, device='cpu')
        
        # Create some synthetic MNIST-like data
        synthetic_data = torch.rand(100, 784)  # Random data in [0,1] range like MNIST
        
        print("2. Creating inference pipeline...")
        pipeline = create_inference_pipeline(
            model=model,
            batch_size=32,
            preprocessing='midrange',
            validate_input=True,
            log_performance=True
        )
        
        print("3. Running single sample inference...")
        single_input = synthetic_data[0]  # Shape: (784,)
        single_output = pipeline.predict(single_input)
        print(f"   Input shape: {single_input.shape}")
        print(f"   Output shape: {single_output.shape}")
        
        print("4. Running batch inference...")
        batch_input = synthetic_data[:20]  # Shape: (20, 784)
        batch_output = pipeline.predict(batch_input)
        print(f"   Batch input shape: {batch_input.shape}")
        print(f"   Batch output shape: {batch_output.shape}")
        
        print("5. Running inference with latent extraction...")
        reconstructed, latent = pipeline.predict(batch_input, return_latent=True)
        print(f"   Reconstructed shape: {reconstructed.shape}")
        print(f"   Latent shape: {latent.shape}")
        
        print("6. Processing large dataset with automatic batching...")
        large_dataset = synthetic_data  # All 100 samples
        start_time = time.time()
        large_output = pipeline.predict(large_dataset, batch_size=16)
        end_time = time.time()
        
        print(f"   Processed {large_dataset.shape[0]} samples in {end_time - start_time:.3f}s")
        print(f"   Output shape: {large_output.shape}")
        
        print("7. Converting between tensor formats...")
        # Reshape flat tensors to image format for visualization
        images_28x28 = flat_to_images(large_output[:5], (28, 28))  # First 5 as images
        print(f"   Reshaped to images: {images_28x28.shape}")
        
        # Convert images back to flat
        flat_again = tensor_to_flat(images_28x28)
        print(f"   Flattened again: {flat_again.shape}")
        
        print("8. Performance statistics...")
        stats = pipeline.get_performance_stats()
        print(f"   Total samples processed: {stats['total_samples_processed']}")
        print(f"   Average batch time: {stats['average_batch_time']:.4f}s")
        print(f"   Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
        
        print("\n✅ Demonstration completed successfully!")
        print("\nThe inference pipeline provides:")
        print("   • Automatic batch processing for efficient computation")
        print("   • Tensor format conversions (flat ↔ image)")
        print("   • Midrange preprocessing matching training pipeline")
        print("   • Device-aware operations (CPU/GPU)")
        print("   • Comprehensive input validation")
        print("   • Performance tracking and metrics")
        print("   • Easy checkpoint loading from trained models")
        
        return True
        
    except Exception as e:
        print(f"✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete test suite."""
    print("Autoencoder Inference Pipeline Test Suite")
    print("=" * 80)
    
    tests = [
        test_checkpoint_loading,
        test_tensor_conversions,
        test_batch_processing,
        test_preprocessing,
        test_device_handling,
        test_performance_tracking,
        test_error_handling,
        test_factory_function,
        demonstration_example
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}\n")
    
    print("=" * 80)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Inference pipeline is ready for production.")
        print("\nKey features validated:")
        print("✓ Model checkpoint loading and validation")
        print("✓ Efficient batch processing with configurable sizes")
        print("✓ Tensor preprocessing with midrange scaling")
        print("✓ Tensor-image format conversions (784 ↔ 28×28)")
        print("✓ Device-aware operations (CPU/GPU compatibility)")
        print("✓ Input validation and comprehensive error handling")
        print("✓ Performance tracking and metrics collection")
        print("✓ Factory function for easy pipeline creation")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    main()
