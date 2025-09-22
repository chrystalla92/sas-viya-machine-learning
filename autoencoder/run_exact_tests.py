#!/usr/bin/env python3
"""
Run the exact tests that were failing to isolate the issues.
"""

import sys
import os
import traceback
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_utils():
    """Test data utilities - from test_demo.py"""
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

def test_plots_module():
    """Test the plots module - from test_visualization.py"""
    print("Testing plots module...")
    
    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 25
    images = np.random.rand(n_samples, 784)
    labels = np.random.randint(0, 10, n_samples)
    reconstructed = images + np.random.normal(0, 0.1, images.shape)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    try:
        from visualization.plots import plot_mnist_grid, plot_reconstruction_comparison
        
        # Test MNIST grid
        fig1 = plot_mnist_grid(
            images[:25], 
            labels[:25],
            grid_size=(5, 5)
        )
        plt.close(fig1)
        print("  ✓ plot_mnist_grid works")
        
        # Test reconstruction comparison
        fig2 = plot_reconstruction_comparison(
            images[:10],
            reconstructed[:10],
            labels[:10]
        )
        plt.close(fig2)
        print("  ✓ plot_reconstruction_comparison works")
        
        return True
    except Exception as e:
        print(f"  ✗ plots module test failed: {e}")
        traceback.print_exc()
        return False

def test_analysis_module():
    """Test the analysis module - from test_visualization.py"""
    print("Testing analysis module...")
    
    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 30
    images = np.random.rand(n_samples, 784)
    labels = np.random.randint(0, 10, n_samples)
    
    # Create simple model
    try:
        from model import create_sas_compatible_autoencoder
        model = create_sas_compatible_autoencoder()
        
        # Generate hidden representations
        images_tensor = torch.FloatTensor(images)
        with torch.no_grad():
            hidden_repr = model.encode(images_tensor)
        
        from visualization.analysis import plot_latent_pca, plot_weight_distributions
        
        # Test PCA visualization (with small sample to avoid long computation)
        fig1 = plot_latent_pca(
            hidden_repr[:30],
            labels[:30]
        )
        plt.close(fig1)
        print("  ✓ plot_latent_pca works")
        
        # Test weight distributions
        fig2 = plot_weight_distributions(model)
        plt.close(fig2)
        print("  ✓ plot_weight_distributions works")
        
        return True
    except Exception as e:
        print(f"  ✗ analysis module test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("EXACT FAILING TESTS REPRODUCTION")
    print("="*60)
    
    tests = [
        ("test_data_utils", test_data_utils),
        ("test_plots_module", test_plots_module),
        ("test_analysis_module", test_analysis_module),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The fixes work!")
    else:
        print("\n❌ Some tests still failing.")
    
    sys.exit(0 if all_passed else 1)
