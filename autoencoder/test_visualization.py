"""
Test Script for Visualization Framework

This script tests the basic functionality of the visualization framework
to ensure all modules are working correctly.
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all visualization modules can be imported."""
    print("Testing imports...")
    
    try:
        import visualization
        print("✓ Main visualization package imported")
    except ImportError as e:
        print(f"✗ Failed to import visualization package: {e}")
        return False
    
    try:
        from visualization import check_available_modules
        status = check_available_modules()
        all_available = all(status.values())
        if all_available:
            print("✓ All visualization modules available")
        else:
            print("⚠ Some visualization modules not available")
        return all_available
    except Exception as e:
        print(f"✗ Error checking module availability: {e}")
        return False

@pytest.fixture
def test_data():
    """Create simple test data."""
    print("Creating test data...")
    
    # Create synthetic MNIST-like data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 50
    images = np.random.rand(n_samples, 784)
    labels = np.random.randint(0, 10, n_samples)
    reconstructed = images + np.random.normal(0, 0.1, images.shape)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Create simple model
    from model import create_sas_compatible_autoencoder
    model = create_sas_compatible_autoencoder()
    
    # Generate hidden representations
    images_tensor = torch.FloatTensor(images)
    with torch.no_grad():
        hidden_repr = model.encode(images_tensor)
    
    # Create training metrics
    epochs = list(range(1, 21))
    train_losses = [0.5 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.01) 
                   for epoch in epochs]
    val_losses = [0.5 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.01) 
                 for epoch in epochs]
    
    training_metrics = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': [0.001] * len(epochs)
    }
    
    print("✓ Test data created successfully")
    return {
        'images': images,
        'labels': labels, 
        'reconstructed': reconstructed,
        'model': model,
        'hidden_repr': hidden_repr,
        'training_metrics': training_metrics
    }

def test_plots_module(test_data):
    """Test the plots module."""
    print("\nTesting plots module...")
    
    try:
        from visualization.plots import plot_mnist_grid, plot_reconstruction_comparison
        
        # Test MNIST grid
        fig1 = plot_mnist_grid(
            test_data['images'][:25], 
            test_data['labels'][:25],
            grid_size=(5, 5)
        )
        plt.close(fig1)
        print("✓ plot_mnist_grid works")
        
        # Test reconstruction comparison
        fig2 = plot_reconstruction_comparison(
            test_data['images'][:10],
            test_data['reconstructed'][:10],
            test_data['labels'][:10]
        )
        plt.close(fig2)
        print("✓ plot_reconstruction_comparison works")
        
        return True
        
    except Exception as e:
        print(f"✗ plots module test failed: {e}")
        return False

def test_training_plots_module(test_data):
    """Test the training_plots module."""
    print("\nTesting training_plots module...")
    
    try:
        from visualization.training_plots import plot_training_curves, plot_loss_convergence
        
        # Test training curves
        fig1 = plot_training_curves(
            test_data['training_metrics']['train_losses'],
            test_data['training_metrics']['val_losses'],
            test_data['training_metrics']['epochs']
        )
        plt.close(fig1)
        print("✓ plot_training_curves works")
        
        # Test convergence analysis
        fig2 = plot_loss_convergence(
            test_data['training_metrics']['train_losses']
        )
        plt.close(fig2)
        print("✓ plot_loss_convergence works")
        
        return True
        
    except Exception as e:
        print(f"✗ training_plots module test failed: {e}")
        return False

def test_analysis_module(test_data):
    """Test the analysis module.""" 
    print("\nTesting analysis module...")
    
    try:
        from visualization.analysis import plot_latent_pca, plot_weight_distributions
        
        # Test PCA visualization (with small sample to avoid long computation)
        fig1 = plot_latent_pca(
            test_data['hidden_repr'][:30],
            test_data['labels'][:30]
        )
        plt.close(fig1)
        print("✓ plot_latent_pca works")
        
        # Test weight distributions
        fig2 = plot_weight_distributions(test_data['model'])
        plt.close(fig2)
        print("✓ plot_weight_distributions works")
        
        return True
        
    except Exception as e:
        print(f"✗ analysis module test failed: {e}")
        return False

def test_utils_module():
    """Test the utils module."""
    print("\nTesting utils module...")
    
    try:
        from utils.plot_utils import setup_publication_style, COLORS
        
        # Test style setup
        setup_publication_style()
        print("✓ setup_publication_style works")
        
        # Test color scheme access
        primary_color = COLORS['primary']
        print(f"✓ Color scheme accessible (primary: {primary_color})")
        
        return True
        
    except Exception as e:
        print(f"✗ utils module test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MNIST Autoencoder Visualization Framework Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    if not imports_ok:
        print("\n✗ Import tests failed. Cannot continue.")
        return False
    
    # Create test data
    try:
        test_data = create_test_data()
    except Exception as e:
        print(f"✗ Failed to create test data: {e}")
        return False
    
    # Run module tests
    tests = [
        test_utils_module,
        lambda: test_plots_module(test_data),
        lambda: test_training_plots_module(test_data),
        lambda: test_analysis_module(test_data)
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    test_names = ["Utils", "Plots", "Training Plots", "Analysis"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<15} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Visualization framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_visualization.py")
        print("2. Check the visualization README for full documentation")
        return True
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
