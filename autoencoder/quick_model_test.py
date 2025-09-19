#!/usr/bin/env python3
"""Quick test to verify autoencoder model functionality."""

import torch
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from autoencoder_model import AutoencoderMLP, create_mnist_autoencoder
    from model_utils import count_parameters, print_model_summary
    
    print("Testing AutoencoderMLP Implementation")
    print("=" * 50)
    
    # Test 1: Basic model creation
    print("1. Creating model...")
    model = AutoencoderMLP()
    print(f"   ✓ Model created: {model.__class__.__name__}")
    
    # Test 2: Check architecture
    print("2. Checking architecture...")
    config = model.get_config()
    print(f"   ✓ Input dim: {config['input_dim']}")
    print(f"   ✓ Latent dim: {config['latent_dim']}")
    print(f"   ✓ Activation: {config['activation']}")
    
    # Test 3: Parameter count
    print("3. Counting parameters...")
    params = count_parameters(model)
    expected = 784 * 400 + 400 + 400 * 784 + 784  # 628,184
    print(f"   ✓ Total parameters: {params:,}")
    print(f"   ✓ Expected: {expected:,}")
    assert params == expected, f"Parameter mismatch: {params} != {expected}"
    
    # Test 4: Forward pass shapes
    print("4. Testing forward pass...")
    batch_size = 16
    x = torch.randn(batch_size, 784)
    
    # Test with latent return
    reconstruction, latent = model(x, return_latent=True)
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Reconstruction shape: {reconstruction.shape}")
    print(f"   ✓ Latent shape: {latent.shape}")
    
    # Verify shapes
    assert reconstruction.shape == (batch_size, 784), f"Reconstruction shape wrong: {reconstruction.shape}"
    assert latent.shape == (batch_size, 400), f"Latent shape wrong: {latent.shape}"
    
    # Test without latent return
    recon_only = model(x, return_latent=False)
    assert recon_only.shape == (batch_size, 784), f"Reconstruction-only shape wrong: {recon_only.shape}"
    print("   ✓ Forward pass shapes correct")
    
    # Test 5: 2D input handling
    print("5. Testing 2D input...")
    x_2d = torch.randn(batch_size, 28, 28)
    reconstruction_2d, latent_2d = model(x_2d)
    assert reconstruction_2d.shape == (batch_size, 784), f"2D input reconstruction wrong: {reconstruction_2d.shape}"
    assert latent_2d.shape == (batch_size, 400), f"2D input latent wrong: {latent_2d.shape}"
    print("   ✓ 2D input handling works")
    
    # Test 6: Encoder/Decoder methods
    print("6. Testing encoder/decoder methods...")
    latent_enc = model.encode(x)
    reconstruction_dec = model.decode(latent_enc)
    assert latent_enc.shape == (batch_size, 400), f"Encode shape wrong: {latent_enc.shape}"
    assert reconstruction_dec.shape == (batch_size, 784), f"Decode shape wrong: {reconstruction_dec.shape}"
    print("   ✓ Encoder/decoder methods work")
    
    # Test 7: Different activations
    print("7. Testing different activations...")
    for activation in ['tanh', 'relu', 'sigmoid']:
        test_model = AutoencoderMLP(activation=activation)
        test_out, test_latent = test_model(x)
        assert test_out.shape == (batch_size, 784), f"{activation} output shape wrong"
        assert test_latent.shape == (batch_size, 400), f"{activation} latent shape wrong"
        print(f"   ✓ {activation} activation works")
    
    # Test 8: Factory function
    print("8. Testing factory function...")
    factory_model = create_mnist_autoencoder(latent_dim=200)
    assert factory_model.get_latent_dim() == 200, "Factory model latent dim wrong"
    print("   ✓ Factory function works")
    
    # Test 9: Device compatibility
    print("9. Testing device compatibility...")
    model_cpu = AutoencoderMLP(device='cpu')
    device = next(model_cpu.parameters()).device
    assert device.type == 'cpu', f"Device not CPU: {device}"
    print("   ✓ CPU device works")
    
    if torch.cuda.is_available():
        model_gpu = AutoencoderMLP(device='cuda')
        gpu_device = next(model_gpu.parameters()).device
        assert gpu_device.type == 'cuda', f"Device not CUDA: {gpu_device}"
        print("   ✓ CUDA device works")
    else:
        print("   ⚠ CUDA not available, skipping GPU test")
    
    print("\n" + "=" * 50)
    print("SUCCESS: All core functionality tests passed!")
    print("=" * 50)
    
    # Print model summary
    print("\nModel Summary:")
    model.print_summary()
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the autoencoder directory")
    sys.exit(1)
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
