#!/usr/bin/env python3
"""
Test script to validate the demo components work correctly.
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    try:
        import numpy as np
        print("  ✓ NumPy")
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        print("  ✓ PyTorch")
        
        import matplotlib.pyplot as plt
        print("  ✓ Matplotlib")
        
        from synthetic_demo_data import SyntheticPatternGenerator
        print("  ✓ SyntheticPatternGenerator")
        
        from model import MNISTAutoencoder
        print("  ✓ MNISTAutoencoder")
        
        from data_utils import train_validation_split
        print("  ✓ data_utils")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    try:
        from synthetic_demo_data import SyntheticPatternGenerator
        
        generator = SyntheticPatternGenerator(image_size=16, seed=42)
        
        # Test individual patterns
        circle = generator.generate_circle()
        assert circle.shape == (16, 16), f"Circle shape wrong: {circle.shape}"
        
        rect = generator.generate_rectangle()
        assert rect.shape == (16, 16), f"Rectangle shape wrong: {rect.shape}"
        
        # Test dataset generation
        images, labels = generator.generate_dataset(num_samples=100)
        assert images.shape == (100, 256), f"Images shape wrong: {images.shape}"
        assert labels.shape == (100,), f"Labels shape wrong: {labels.shape}"
        
        print("  ✓ Pattern generation works")
        return True
    except Exception as e:
        print(f"  ✗ Synthetic data error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test autoencoder model creation."""
    print("\nTesting model creation...")
    try:
        from model import MNISTAutoencoder
        
        model = MNISTAutoencoder(input_dim=256, hidden_dim=64, seed=42)
        
        # Test forward pass
        import torch
        test_input = torch.randn(10, 256)
        output = model(test_input)
        
        assert output.shape == (10, 256), f"Output shape wrong: {output.shape}"
        
        # Test architecture info
        info = model.get_architecture_info()
        assert info['input_dim'] == 256
        assert info['hidden_dim'] == 64
        
        print("  ✓ Model creation works")
        return True
    except Exception as e:
        print(f"  ✗ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_data_utils():
    """Test data utilities."""
    print("\nTesting data utilities...")
    try:
        from data_utils import train_validation_split
        import numpy as np
        
        # Generate test data
        images = np.random.randn(100, 256)
        labels = np.random.randint(0, 5, 100)
        
        # Test split
        train_imgs, val_imgs, train_labels, val_labels = train_validation_split(
            images, labels, validation_ratio=0.2, random_seed=42
        )
        
        assert len(train_imgs) == 80, f"Train size wrong: {len(train_imgs)}"
        assert len(val_imgs) == 20, f"Val size wrong: {len(val_imgs)}"
        
        print("  ✓ Data utilities work")
        return True
    except Exception as e:
        print(f"  ✗ Data utilities error: {e}")
        traceback.print_exc()
        return False

def test_minimal_training():
    """Test minimal training loop."""
    print("\nTesting minimal training...")
    try:
        import torch
        import torch.optim as optim
        import numpy as np
        from model import MNISTAutoencoder
        from synthetic_demo_data import SyntheticPatternGenerator
        
        # Generate small dataset
        generator = SyntheticPatternGenerator(image_size=8, seed=42)
        images, labels = generator.generate_dataset(num_samples=50)
        
        # Create model
        model = MNISTAutoencoder(input_dim=64, hidden_dim=32, seed=42)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(images)
        
        # Train for a few steps
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            reconstructed = model(data_tensor)
            loss = model.reconstruction_loss(data_tensor, reconstructed)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
            elif epoch == 2:
                final_loss = loss.item()
        
        # Check that loss decreased
        assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"
        
        print(f"  ✓ Minimal training works (loss: {initial_loss:.6f} -> {final_loss:.6f})")
        return True
    except Exception as e:
        print(f"  ✗ Minimal training error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("TESTING DEMO COMPONENTS")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Synthetic Data", test_synthetic_data),
        ("Model Creation", test_model_creation),
        ("Data Utils", test_data_utils),
        ("Minimal Training", test_minimal_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Demo should work correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before running demo")
        return 1

if __name__ == "__main__":
    sys.exit(main())
