"""
Visualization Package for MNIST Autoencoder

This package provides comprehensive visualization capabilities for the MNIST autoencoder
including training metrics, image reconstructions, latent space analysis, and diagnostic plots.

Key modules:
- plots: Main plotting functions for image visualization and reconstruction comparisons
- training_plots: Training-specific visualizations (loss curves, convergence plots)  
- analysis: Latent space analysis and diagnostic visualizations

Features:
- Full migration of SAS python_plot.sas functionality
- Publication-ready matplotlib/seaborn plots
- Training progress monitoring and analysis
- Latent space visualization (PCA, t-SNE)
- Reconstruction error analysis
- Weight distribution diagnostics
- Interactive and batch plotting modes
"""

# Import all main functions for easy access
try:
    from .plots import (
        plot_mnist_grid,
        plot_reconstruction_comparison,
        plot_reconstruction_grid,
        save_reconstruction_samples,
        load_and_plot_from_csv
    )
    _plots_available = True
except ImportError as e:
    print(f"Warning: Could not import plots module: {e}")
    import traceback
    traceback.print_exc()
    _plots_available = False

try:
    from .training_plots import (
        plot_training_curves,
        plot_loss_convergence,
        plot_training_progress,
        plot_learning_rate_schedule,
        load_and_plot_training_metrics,
        plot_batch_training_progress
    )
    _training_plots_available = True
except ImportError as e:
    print(f"Warning: Could not import training_plots module: {e}")
    import traceback
    traceback.print_exc()
    _training_plots_available = False

try:
    from .analysis import (
        plot_latent_space_2d,
        plot_latent_tsne,
        plot_latent_pca,
        plot_reconstruction_errors,
        plot_weight_distributions,
        plot_activation_heatmap,
        analyze_hidden_representations
    )
    _analysis_available = True
except ImportError as e:
    print(f"Warning: Could not import analysis module: {e}")
    import traceback
    traceback.print_exc()
    _analysis_available = False

# Define __all__ based on what's available
__all__ = []

if _plots_available:
    __all__.extend([
        'plot_mnist_grid',
        'plot_reconstruction_comparison', 
        'plot_reconstruction_grid',
        'save_reconstruction_samples',
        'load_and_plot_from_csv'
    ])

if _training_plots_available:
    __all__.extend([
        'plot_training_curves',
        'plot_loss_convergence',
        'plot_training_progress', 
        'plot_learning_rate_schedule',
        'load_and_plot_training_metrics',
        'plot_batch_training_progress'
    ])

if _analysis_available:
    __all__.extend([
        'plot_latent_space_2d',
        'plot_latent_tsne',
        'plot_latent_pca',
        'plot_reconstruction_errors',
        'plot_weight_distributions',
        'plot_activation_heatmap',
        'analyze_hidden_representations'
    ])

# Version and package info
__version__ = "1.0.0"
__author__ = "MNIST Autoencoder Project"
__description__ = "Comprehensive visualization toolkit for MNIST autoencoder analysis"

# Convenience function to check what's available
def check_available_modules(verbose=True):
    """Check which visualization modules are available."""
    status = {
        'plots': _plots_available,
        'training_plots': _training_plots_available,
        'analysis': _analysis_available
    }
    
    if verbose:
        print("Visualization Module Status:")
        for module, available in status.items():
            status_str = "✓ Available" if available else "✗ Not Available"
            print(f"  {module}: {status_str}")
    
    return status

# Quick start function
def quick_demo():
    """Run a quick demonstration of available visualization capabilities."""
    print("MNIST Autoencoder Visualization Package")
    print("=" * 40)
    
    status = check_available_modules()
    
    if not any(status.values()):
        print("\nNo visualization modules available. Please check dependencies:")
        print("- matplotlib>=3.5.0")
        print("- seaborn>=0.11.0") 
        print("- numpy>=1.21.0")
        print("- scikit-learn>=1.0.0")
        return
    
    print(f"\nTotal available functions: {len(__all__)}")
    print("\nFor a complete demo, run:")
    print("  python example_visualization.py")
    
    print("\nQuick usage examples:")
    if _plots_available:
        print("\n# Image visualization (SAS migration)")
        print("from visualization import plot_mnist_grid")
        print("fig = plot_mnist_grid(images, labels)")
    
    if _training_plots_available:
        print("\n# Training progress")
        print("from visualization import plot_training_curves")
        print("fig = plot_training_curves(train_losses, val_losses)")
    
    if _analysis_available:
        print("\n# Latent space analysis")
        print("from visualization import plot_latent_pca")
        print("fig = plot_latent_pca(hidden_representations, labels)")
