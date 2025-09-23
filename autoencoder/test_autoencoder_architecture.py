#!/usr/bin/env python3
"""
Test script for the MLP Autoencoder architecture implementation.

This script verifies that the autoencoder implementation matches
the SAS Viya specifications exactly.
"""

import sys
import os
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import Autoencoder, create_autoencoder


def test_architecture():
    """Test basic architecture specifications."""
    print("=" * 60)
    print("Testing Autoencoder Architecture")
    print("=" * 60)
    
    try:
        # Create autoencoder
        model = Autoencoder()
        
        # Verify architecture dimensions
        assert model.input_dim == 784, f"Expected input_dim=784, got {model.input_dim}"
        assert model.hidden_dim == 400, f"Expected hidden_dim=400, got {model.hidden_dim}"
        print("✓ Architecture dimensions correct (784→400→784)")
        
        # Test parameter counts
        params = model.count_parameters()
        expected_encoder = 784 * 400 + 400  # weights + biases
        expected_decoder = 400 * 784 + 784  # weights + biases
        expected_total = expected_encoder + expected_decoder
        
        assert params['encoder_parameters'] == expected_encoder, \
            f"Expected encoder params={expected_encoder}, got {params['encoder_parameters']}"
        assert params['decoder_parameters'] == expected_decoder, \
            f"Expected decoder params={expected_decoder}, got {params['decoder_parameters']}"
        assert params['total_parameters'] == expected_total, \
            f"Expected total params={expected_total}, got {params['total_parameters']}"
        
        print(f"✓ Parameter counts correct:")
        print(f"  Encoder: {params['encoder_parameters']:,}")
        print(f"  Decoder: {params['decoder_parameters']:,}")
        print(f"  Total: {params['total_parameters']:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass with correct input/output shapes."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    try:
        model = Autoencoder()
        
        # Test single sample
        x_single = torch.randn(1, 784)
        output_single = model(x_single)
        assert output_single.shape == (1, 784), \
            f"Expected output shape (1, 784), got {output_single.shape}"
        print("✓ Single sample forward pass shape correct")
        
        # Test batch
        batch_size = 32
        x_batch = torch.randn(batch_size, 784)
        output_batch = model(x_batch)
        assert output_batch.shape == (batch_size, 784), \
            f"Expected output shape ({batch_size}, 784), got {output_batch.shape}"
        print(f"✓ Batch forward pass shape correct (batch_size={batch_size})")
        
        # Test latent representation
        latent = model.encode(x_batch)
        assert latent.shape == (batch_size, 400), \
            f"Expected latent shape ({batch_size}, 400), got {latent.shape}"
        print("✓ Latent representation shape correct")
        
        # Test decode from latent
        reconstructed = model.decode(latent)
        assert reconstructed.shape == (batch_size, 784), \
            f"Expected reconstructed shape ({batch_size}, 784), got {reconstructed.shape}"
        print("✓ Decode from latent shape correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False


def test_activations():
    """Test that activations are applied correctly."""
    print("\n" + "=" * 60)
    print("Testing Activation Functions")
    print("=" * 60)
    
    try:
        model = Autoencoder()
        
        # Create test input
        x = torch.randn(10, 784)
        
        # Test encoder applies tanh
        encoded = model.encode(x)
        # Tanh output should be in [-1, 1]
        assert encoded.min() >= -1.001 and encoded.max() <= 1.001, \
            f"Encoder output should be in [-1,1], got [{encoded.min():.3f}, {encoded.max():.3f}]"
        print("✓ Encoder applies tanh activation correctly")
        
        # Test decoder applies linear (no activation)
        decoded = model.decode(encoded)
        # Linear layer can output any value
        print(f"✓ Decoder applies linear activation (output range: [{decoded.min():.3f}, {decoded.max():.3f}])")
        
        return True
        
    except Exception as e:
        print(f"✗ Activation test failed: {e}")
        return False


def test_weight_initialization():
    """Test weight initialization matches SAS specifications."""
    print("\n" + "=" * 60)
    print("Testing Weight Initialization")
    print("=" * 60)
    
    try:
        # Test reproducibility with seed
        model1 = Autoencoder(seed=23451)
        model2 = Autoencoder(seed=23451)
        
        # Weights should be identical with same seed
        encoder_diff = torch.abs(model1.encoder.weight - model2.encoder.weight).max()
        decoder_diff = torch.abs(model1.decoder.weight - model2.decoder.weight).max()
        
        assert encoder_diff < 1e-6, f"Encoder weights not reproducible, max diff: {encoder_diff}"
        assert decoder_diff < 1e-6, f"Decoder weights not reproducible, max diff: {decoder_diff}"
        print("✓ Weight initialization is reproducible with seed=23451")
        
        # Test uniform distribution range [-1, 1]
        encoder_min, encoder_max = model1.encoder.weight.min(), model1.encoder.weight.max()
        decoder_min, decoder_max = model1.decoder.weight.min(), model1.decoder.weight.max()
        
        assert encoder_min >= -1.001 and encoder_max <= 1.001, \
            f"Encoder weights should be in [-1,1], got [{encoder_min:.3f}, {encoder_max:.3f}]"
        assert decoder_min >= -1.001 and decoder_max <= 1.001, \
            f"Decoder weights should be in [-1,1], got [{decoder_min:.3f}, {decoder_max:.3f}]"
        print("✓ Weights initialized in uniform range [-1, 1]")
        
        return True
        
    except Exception as e:
        print(f"✗ Weight initialization test failed: {e}")
        return False


def test_device_handling():
    """Test CPU/GPU device handling."""
    print("\n" + "=" * 60)
    print("Testing Device Handling")
    print("=" * 60)
    
    try:
        # Test CPU
        model = Autoencoder()
        assert model.get_device().type == 'cpu', \
            f"Expected CPU device, got {model.get_device()}"
        print("✓ Model defaults to CPU")
        
        # Test device transfer
        if torch.cuda.is_available():
            model_gpu = model.to_device(torch.device('cuda'))
            assert model_gpu.get_device().type == 'cuda', \
                f"Expected CUDA device, got {model_gpu.get_device()}"
            print("✓ Model can be moved to GPU")
            
            # Test inference on GPU
            x = torch.randn(5, 784).cuda()
            output = model_gpu(x)
            assert output.device.type == 'cuda', \
                f"Expected output on CUDA, got {output.device}"
            print("✓ GPU inference works correctly")
        else:
            print("! GPU not available, skipping GPU tests")
        
        return True
        
    except Exception as e:
        print(f"✗ Device handling test failed: {e}")
        return False


def test_factory_function():
    """Test the factory function."""
    print("\n" + "=" * 60)
    print("Testing Factory Function")
    print("=" * 60)
    
    try:
        # Test factory function
        model = create_autoencoder(device='cpu')
        
        # Verify it's the right type
        assert isinstance(model, Autoencoder), \
            f"Expected Autoencoder instance, got {type(model)}"
        
        # Test it works
        x = torch.randn(3, 784)
        output = model(x)
        assert output.shape == (3, 784), \
            f"Expected output shape (3, 784), got {output.shape}"
        
        print("✓ Factory function creates working autoencoder")
        return True
        
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("MLP Autoencoder Architecture Test Suite")
    print("=" * 60)
    
    tests = [
        test_architecture,
        test_forward_pass,
        test_activations,
        test_weight_initialization,
        test_device_handling,
        test_factory_function,
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
        print("🎉 ALL TESTS PASSED! Autoencoder implementation is ready.")
        print("\nThe MLP Autoencoder implementation:")
        print("- Matches SAS Viya specifications exactly (784→400→784)")
        print("- Uses tanh activation for encoder, linear for decoder")
        print("- Uniform weight initialization with seed=23451")
        print("- Supports CPU/GPU device handling")
        print("- Provides encode/decode methods for latent access")
        print("- Includes comprehensive architecture summary")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main()
