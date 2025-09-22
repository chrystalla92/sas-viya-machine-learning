#!/usr/bin/env python3
"""Debug test runner to isolate specific test failures."""

import sys
import os
import traceback

def test_data_utils():
    """Test data utilities."""
    print("Testing data utilities...")
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

def test_visualization_imports():
    """Test visualization module imports."""
    print("Testing visualization imports...")
    try:
        import visualization
        print("  ✓ Main visualization package imported")
        
        from visualization import check_available_modules
        status = check_available_modules(verbose=False)
        all_available = all(status.values())
        if all_available:
            print("  ✓ All visualization modules available")
        else:
            print("  ⚠ Some visualization modules not available")
            print(f"  Status: {status}")
        
        # Test specific imports that the test is looking for
        try:
            from visualization.plots import plot_mnist_grid, plot_reconstruction_comparison
            print("  ✓ Core plots functions imported")
        except Exception as e:
            print(f"  ✗ Could not import plots functions: {e}")
            return False
            
        try:
            from visualization.training_plots import plot_training_curves, plot_loss_convergence
            print("  ✓ Training plots functions imported")
        except Exception as e:
            print(f"  ✗ Could not import training plots functions: {e}")
            return False
            
        try:
            from visualization.analysis import plot_latent_pca, plot_weight_distributions
            print("  ✓ Analysis functions imported")
        except Exception as e:
            print(f"  ✗ Could not import analysis functions: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Visualization import error: {e}")
        traceback.print_exc()
        return False

def test_utils_imports():
    """Test utils module imports."""
    print("Testing utils imports...")
    try:
        from utils.plot_utils import setup_publication_style, COLORS
        print("  ✓ Utils functions imported")
        
        # Test the functions work
        setup_publication_style()
        print("  ✓ setup_publication_style works")
        
        primary_color = COLORS['primary']
        print(f"  ✓ Color scheme accessible (primary: {primary_color})")
        
        return True
    except Exception as e:
        print(f"  ✗ Utils import error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    try:
        from model import MNISTAutoencoder, create_sas_compatible_autoencoder
        
        model = MNISTAutoencoder(input_dim=256, hidden_dim=64, seed=42)
        
        # Test forward pass
        import torch
        test_input = torch.randn(10, 256)
        output = model(test_input)
        
        assert output.shape == (10, 256), f"Output shape wrong: {output.shape}"
        
        # Test SAS compatible model
        sas_model = create_sas_compatible_autoencoder()
        assert sas_model is not None
        
        print("  ✓ Model creation works")
        return True
    except Exception as e:
        print(f"  ✗ Model creation error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("DEBUG TEST RUNNER")
    print("="*50)
    
    tests = [
        ("Data Utils", test_data_utils),
        ("Model Creation", test_model_creation),
        ("Utils Imports", test_utils_imports),
        ("Visualization Imports", test_visualization_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    sys.exit(0 if all_passed else 1)
