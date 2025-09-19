"""
Quick test script for the visualization system.

Tests basic functionality without requiring full training or real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Test imports
try:
    from visualization import (
        plot_image_comparison, 
        plot_training_curves,
        plot_latent_space,
        plot_reconstruction_errors,
        VisualizationManager
    )
    from plot_utils import (
        validate_image_data,
        get_color_palette,
        InteractivePlotter
    )
    print("✓ All visualization imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


def test_synthetic_data():
    """Test visualization with synthetic data."""
    print("\nTesting with synthetic data...")
    
    # Create synthetic MNIST-like data
    np.random.seed(42)
    n_samples = 50
    
    # Generate fake images (28x28)
    originals = np.random.rand(n_samples, 28, 28)
    reconstructions = originals + np.random.normal(0, 0.1, originals.shape)  # Add noise
    latent_reps = np.random.rand(n_samples, 32)  # 32-dim latent space
    labels = np.random.randint(0, 10, n_samples)
    
    # Fake training data
    epochs = list(range(1, 21))
    train_losses = [1.0 * np.exp(-0.1 * i) + 0.1 + np.random.normal(0, 0.02) for i in epochs]
    val_losses = [1.0 * np.exp(-0.08 * i) + 0.15 + np.random.normal(0, 0.02) for i in epochs]
    learning_rates = [0.001 * (0.9 ** (i // 5)) for i in epochs]
    
    # Create test output directory
    test_dir = Path("./test_visualizations")
    test_dir.mkdir(exist_ok=True)
    
    print("1. Testing image comparison...")
    try:
        fig = plot_image_comparison(
            originals[:16], reconstructions[:16], labels[:16],
            title="Test Image Comparison",
            save_path=str(test_dir / "test_image_comparison"),
            show=False
        )
        print("   ✓ Image comparison test passed")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Image comparison test failed: {e}")
    
    print("2. Testing training curves...")
    try:
        fig = plot_training_curves(
            train_losses, val_losses, epochs, learning_rates,
            title="Test Training Curves",
            save_path=str(test_dir / "test_training_curves"),
            show=False
        )
        print("   ✓ Training curves test passed")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Training curves test failed: {e}")
    
    print("3. Testing latent space visualization...")
    try:
        fig = plot_latent_space(
            latent_reps, labels,
            method='pca', n_components=2,
            title="Test Latent Space",
            save_path=str(test_dir / "test_latent_space"),
            show=False
        )
        print("   ✓ Latent space test passed")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Latent space test failed: {e}")
    
    print("4. Testing reconstruction errors...")
    try:
        errors = np.mean((originals - reconstructions) ** 2, axis=(1, 2))
        fig = plot_reconstruction_errors(
            errors, labels,
            title="Test Reconstruction Errors",
            save_path=str(test_dir / "test_recon_errors"),
            show=False
        )
        print("   ✓ Reconstruction errors test passed")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Reconstruction errors test failed: {e}")
    
    print("5. Testing visualization manager...")
    try:
        manager = VisualizationManager(
            output_dir=str(test_dir / "manager_test"),
            default_formats=['png'],
            default_dpi=150
        )
        
        saved_plots = manager.create_comprehensive_report(
            originals=originals[:25],
            reconstructions=reconstructions[:25],
            latent_representations=latent_reps[:25],
            train_losses=train_losses,
            val_losses=val_losses,
            labels=labels[:25],
            epochs=epochs,
            learning_rates=learning_rates,
            report_name="test_report",
            show_plots=False
        )
        
        print(f"   ✓ Visualization manager test passed ({len(saved_plots)} plots)")
    except Exception as e:
        print(f"   ✗ Visualization manager test failed: {e}")
    
    print("6. Testing interactive plotter...")
    try:
        plotter = InteractivePlotter(
            originals=originals[:20],
            reconstructions=reconstructions[:20],
            latent_representations=latent_reps[:20],
            labels=labels[:20]
        )
        
        fig = plotter.create_sample_browser()
        if fig:
            fig.savefig(str(test_dir / "test_sample_browser.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("   ✓ Interactive plotter test passed")
        else:
            print("   ! Interactive plotter returned None")
    except Exception as e:
        print(f"   ✗ Interactive plotter test failed: {e}")
    
    print(f"\nTest completed! Check outputs in: {test_dir}")


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    # Test data validation
    try:
        test_data1 = np.random.rand(10, 784)
        test_data2 = np.random.rand(10, 784)
        validated_data1, validated_data2 = validate_image_data(test_data1, test_data2)
        print("   ✓ Data validation test passed")
    except Exception as e:
        print(f"   ✗ Data validation test failed: {e}")
    
    # Test color palette
    try:
        colors = get_color_palette(10)
        assert len(colors) == 10
        print("   ✓ Color palette test passed")
    except Exception as e:
        print(f"   ✗ Color palette test failed: {e}")


def main():
    """Run all tests."""
    print("Visualization System Test Suite")
    print("=" * 50)
    
    # Test basic utilities
    test_utilities()
    
    # Test with synthetic data
    test_synthetic_data()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("If all tests passed, the visualization system is working correctly.")
    print("Check ./test_visualizations/ for output examples.")


if __name__ == "__main__":
    main()
