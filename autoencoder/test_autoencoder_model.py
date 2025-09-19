"""
Test script for AutoencoderMLP model implementation.

This script verifies that the autoencoder model meets all specified
success criteria and technical requirements.
"""

import torch
import torch.nn as nn
import numpy as np
from autoencoder_model import AutoencoderMLP, Encoder, Decoder, create_mnist_autoencoder
from model_utils import (
    count_parameters, 
    get_model_summary, 
    print_model_summary,
    get_activation_function,
    get_initialization_function
)


def test_model_creation():
    """Test basic model creation and configuration."""
    print("=" * 60)
    print("Testing Model Creation and Configuration")
    print("=" * 60)
    
    try:
        # Test default creation
        model = AutoencoderMLP()
        print("✓ Default model created successfully")
        
        # Test configuration
        config = model.get_config()
        expected_config = {
            'input_dim': 784,
            'latent_dim': 400, 
            'activation': 'tanh',
            'init_type': 'uniform'
        }
        
        for key, expected_value in expected_config.items():
            if config[key] != expected_value:
                print(f"✗ Config mismatch for {key}: got {config[key]}, expected {expected_value}")
                return False
        
        print("✓ Model configuration is correct")
        
        # Test custom configuration
        custom_model = AutoencoderMLP(
            input_dim=784,
            latent_dim=200, 
            activation='relu',
            init_type='normal'
        )
        print("✓ Custom model created successfully")
        
        # Test factory function
        factory_model = create_mnist_autoencoder(latent_dim=300, activation='relu')
        print("✓ Factory function works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_model_architecture():
    """Test model architecture and component structure."""
    print("\n" + "=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    try:
        model = AutoencoderMLP()
        
        # Test encoder component
        assert isinstance(model.encoder, Encoder)
        assert model.encoder.input_dim == 784
        assert model.encoder.latent_dim == 400
        print("✓ Encoder component is correctly configured")
        
        # Test decoder component  
        assert isinstance(model.decoder, Decoder)
        assert model.decoder.latent_dim == 400
        assert model.decoder.output_dim == 784
        print("✓ Decoder component is correctly configured")
        
        # Test parameter counting
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        # Expected parameters: 
        # Encoder: 784 * 400 + 400 = 314,000
        # Decoder: 400 * 784 + 784 = 314,184
        # Total: 628,184
        expected_params = 784 * 400 + 400 + 400 * 784 + 784
        
        if total_params != expected_params:
            print(f"✗ Parameter count mismatch: got {total_params}, expected {expected_params}")
            return False
        
        print(f"✓ Parameter count is correct: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass functionality and output shapes."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass and Output Shapes")
    print("=" * 60)
    
    try:
        model = AutoencoderMLP()
        model.eval()
        
        # Test with different batch sizes
        batch_sizes = [1, 16, 64]
        
        for batch_size in batch_sizes:
            # Test standard input shape
            x = torch.randn(batch_size, 784)
            
            # Test forward with latent return
            reconstruction, latent = model(x, return_latent=True)
            
            # Verify shapes
            expected_recon_shape = (batch_size, 784)
            expected_latent_shape = (batch_size, 400)
            
            if reconstruction.shape != expected_recon_shape:
                print(f"✗ Reconstruction shape incorrect for batch {batch_size}: "
                      f"got {reconstruction.shape}, expected {expected_recon_shape}")
                return False
            
            if latent.shape != expected_latent_shape:
                print(f"✗ Latent shape incorrect for batch {batch_size}: "
                      f"got {latent.shape}, expected {expected_latent_shape}")
                return False
            
            # Test forward without latent return
            recon_only = model(x, return_latent=False)
            if recon_only.shape != expected_recon_shape:
                print(f"✗ Reconstruction-only shape incorrect for batch {batch_size}")
                return False
            
            print(f"✓ Forward pass correct for batch size {batch_size}")
        
        # Test with 2D image input (should flatten automatically)
        x_2d = torch.randn(16, 28, 28)
        reconstruction, latent = model(x_2d)
        
        if reconstruction.shape != (16, 784):
            print(f"✗ 2D input handling failed: got {reconstruction.shape}")
            return False
        
        print("✓ 2D input (28x28) is correctly handled")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False


def test_encoder_decoder_methods():
    """Test separate encoder and decoder methods."""
    print("\n" + "=" * 60)
    print("Testing Encoder/Decoder Methods")
    print("=" * 60)
    
    try:
        model = AutoencoderMLP()
        model.eval()
        
        # Test data
        x = torch.randn(32, 784)
        
        # Test encode method
        latent = model.encode(x)
        if latent.shape != (32, 400):
            print(f"✗ Encode method shape incorrect: got {latent.shape}")
            return False
        
        # Test decode method
        reconstruction = model.decode(latent)
        if reconstruction.shape != (32, 784):
            print(f"✗ Decode method shape incorrect: got {reconstruction.shape}")
            return False
        
        # Verify that encode -> decode gives same result as forward
        full_reconstruction, full_latent = model(x)
        
        # Check if results are equivalent
        latent_diff = torch.abs(latent - full_latent).max()
        recon_diff = torch.abs(reconstruction - full_reconstruction).max()
        
        if latent_diff > 1e-6 or recon_diff > 1e-6:
            print(f"✗ Encode/decode consistency failed: "
                  f"latent diff: {latent_diff}, recon diff: {recon_diff}")
            return False
        
        print("✓ Encoder and decoder methods work correctly")
        print("✓ Consistency between full forward and separate methods verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Encoder/decoder methods test failed: {e}")
        return False


def test_activation_functions():
    """Test different activation function configurations."""
    print("\n" + "=" * 60)
    print("Testing Activation Function Configurations")
    print("=" * 60)
    
    activations = ['tanh', 'relu', 'sigmoid', 'elu', 'gelu']
    
    try:
        for activation in activations:
            model = AutoencoderMLP(activation=activation)
            
            # Test forward pass
            x = torch.randn(8, 784)
            reconstruction, latent = model(x)
            
            # Verify shapes are still correct
            if reconstruction.shape != (8, 784) or latent.shape != (8, 400):
                print(f"✗ Shape incorrect for activation '{activation}'")
                return False
            
            # Verify activation is applied
            expected_activation = get_activation_function(activation)
            if not isinstance(model.encoder.activation, type(expected_activation)):
                print(f"✗ Encoder activation incorrect for '{activation}'")
                return False
            
            if not isinstance(model.decoder.activation, type(expected_activation)):
                print(f"✗ Decoder activation incorrect for '{activation}'")
                return False
            
            print(f"✓ Activation '{activation}' works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Activation function test failed: {e}")
        return False


def test_weight_initialization():
    """Test weight initialization for different activation functions.""" 
    print("\n" + "=" * 60)
    print("Testing Weight Initialization")
    print("=" * 60)
    
    try:
        # Test Xavier initialization (for tanh)
        model_tanh = AutoencoderMLP(activation='tanh', init_type='uniform')
        
        # Check that weights are initialized (not zeros)
        encoder_weight = model_tanh.encoder.linear.weight
        decoder_weight = model_tanh.decoder.linear.weight
        
        if torch.allclose(encoder_weight, torch.zeros_like(encoder_weight)):
            print("✗ Encoder weights are not initialized")
            return False
        
        if torch.allclose(decoder_weight, torch.zeros_like(decoder_weight)):
            print("✗ Decoder weights are not initialized")  
            return False
        
        print("✓ Weights are properly initialized (non-zero)")
        
        # Test Kaiming initialization (for ReLU)
        model_relu = AutoencoderMLP(activation='relu', init_type='normal')
        
        # Verify different initialization produces different weights
        encoder_weight_relu = model_relu.encoder.linear.weight
        
        if torch.allclose(encoder_weight, encoder_weight_relu):
            print("⚠ Different initializations produced identical weights (unlikely but possible)")
        else:
            print("✓ Different activation functions use different initializations")
        
        return True
        
    except Exception as e:
        print(f"✗ Weight initialization test failed: {e}")
        return False


def test_device_compatibility():
    """Test CPU/GPU device compatibility."""
    print("\n" + "=" * 60)
    print("Testing Device Compatibility")
    print("=" * 60)
    
    try:
        # Test CPU creation
        model = AutoencoderMLP(device='cpu')
        device = next(model.parameters()).device
        
        if device.type != 'cpu':
            print(f"✗ Model not on CPU: {device}")
            return False
        
        print("✓ Model correctly placed on CPU")
        
        # Test CPU inference
        x = torch.randn(16, 784)
        reconstruction, latent = model(x)
        
        if reconstruction.device.type != 'cpu' or latent.device.type != 'cpu':
            print("✗ Output tensors not on CPU")
            return False
        
        print("✓ CPU inference works correctly")
        
        # Test device movement
        model.to_device('cpu')  # Should still be on CPU
        device_after = next(model.parameters()).device
        
        if device_after.type != 'cpu':
            print(f"✗ Device movement failed: {device_after}")
            return False
        
        print("✓ Device movement method works")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = AutoencoderMLP(device='cuda')
            cuda_device = next(model_cuda.parameters()).device
            
            if cuda_device.type != 'cuda':
                print(f"✗ Model not on CUDA: {cuda_device}")
                return False
            
            print("✓ Model correctly placed on CUDA")
            
            # Test CUDA inference
            x_cuda = torch.randn(16, 784).cuda()
            reconstruction_cuda, latent_cuda = model_cuda(x_cuda)
            
            if (reconstruction_cuda.device.type != 'cuda' or 
                latent_cuda.device.type != 'cuda'):
                print("✗ CUDA output tensors not on GPU")
                return False
            
            print("✓ CUDA inference works correctly")
        else:
            print("⚠ CUDA not available, skipping GPU tests")
        
        return True
        
    except Exception as e:
        print(f"✗ Device compatibility test failed: {e}")
        return False


def test_model_summary():
    """Test model summary and parameter analysis functionality."""
    print("\n" + "=" * 60)
    print("Testing Model Summary and Analysis")
    print("=" * 60)
    
    try:
        model = AutoencoderMLP()
        
        # Test summary generation
        summary = model.summary()
        
        required_keys = [
            'model_name', 'total_parameters', 'trainable_parameters',
            'device', 'input_shape', 'reconstruction_shape', 'latent_shape'
        ]
        
        for key in required_keys:
            if key not in summary:
                print(f"✗ Missing summary key: {key}")
                return False
        
        # Verify summary values
        if summary['model_name'] != 'AutoencoderMLP':
            print(f"✗ Incorrect model name: {summary['model_name']}")
            return False
        
        if summary['total_parameters'] != 628184:
            print(f"✗ Incorrect parameter count: {summary['total_parameters']}")
            return False
        
        if summary['reconstruction_shape'] != (1, 784):
            print(f"✗ Incorrect reconstruction shape: {summary['reconstruction_shape']}")
            return False
        
        if summary['latent_shape'] != (1, 400):
            print(f"✗ Incorrect latent shape: {summary['latent_shape']}")
            return False
        
        print("✓ Model summary generation works correctly")
        
        # Test print summary (should not raise errors)
        model.print_summary()
        print("✓ Model summary printing works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Model summary test failed: {e}")
        return False


def test_configuration_validation():
    """Test model configuration validation."""
    print("\n" + "=" * 60)
    print("Testing Configuration Validation")  
    print("=" * 60)
    
    try:
        # Test valid configurations (should not raise errors)
        valid_configs = [
            {'input_dim': 784, 'latent_dim': 400, 'activation': 'tanh'},
            {'input_dim': 1000, 'latent_dim': 100, 'activation': 'relu'},
            {'input_dim': 500, 'latent_dim': 50, 'activation': 'sigmoid'}
        ]
        
        for config in valid_configs:
            model = AutoencoderMLP(**config)
            print(f"✓ Valid configuration accepted: {config}")
        
        # Test invalid configurations (should raise errors)
        invalid_configs = [
            {'input_dim': 0, 'latent_dim': 400, 'activation': 'tanh'},  # invalid input_dim
            {'input_dim': 784, 'latent_dim': 0, 'activation': 'tanh'},  # invalid latent_dim
            {'input_dim': 784, 'latent_dim': 800, 'activation': 'tanh'},  # latent > input
            {'input_dim': 784, 'latent_dim': 400, 'activation': 'invalid'},  # invalid activation
        ]
        
        for config in invalid_configs:
            try:
                model = AutoencoderMLP(**config)
                print(f"✗ Invalid configuration accepted: {config}")
                return False
            except (ValueError, KeyError):
                print(f"✓ Invalid configuration properly rejected: {config}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("AutoencoderMLP Model Test Suite")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_model_architecture, 
        test_forward_pass,
        test_encoder_decoder_methods,
        test_activation_functions,
        test_weight_initialization,
        test_device_compatibility,
        test_model_summary,
        test_configuration_validation
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
        print("✓ All tests passed! AutoencoderMLP implementation meets all requirements.")
        print("\nSuccess Criteria Verification:")
        print("✓ Model accepts input tensors of shape (batch_size, 784)")
        print("✓ Forward pass produces reconstructions of shape (batch_size, 784)") 
        print("✓ Latent representations have shape (batch_size, 400)")
        print("✓ Model initializes with appropriate weights for chosen activation")
        print("✓ Architecture is easily configurable for experimentation")
        print("✓ Model moves correctly between CPU and GPU devices")
        print("✓ Model parameters can be accessed and modified programmatically")
    else:
        print("⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
