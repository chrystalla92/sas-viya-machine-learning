"""
Example usage of the autoencoder visualization system.

This script demonstrates how to use the visualization functions with
trained autoencoder models and create comprehensive reports.
"""

import torch
import numpy as np
from pathlib import Path

from autoencoder_model import AutoencoderMLP
from trainer import TrainingPipeline, train_autoencoder
from datasets import create_mnist_dataloaders
from visualization import (
    plot_image_comparison,
    plot_training_curves,
    plot_latent_space,
    plot_reconstruction_errors,
    plot_error_heatmap,
    create_comprehensive_report,
    VisualizationManager
)
from plot_utils import InteractivePlotter


def basic_visualization_example():
    """Basic example of using individual visualization functions."""
    print("=" * 60)
    print("Basic Visualization Example")
    print("=" * 60)
    
    # Create and train a simple model
    model = AutoencoderMLP(input_dim=784, latent_dim=128, activation='tanh')
    
    config = {
        'epochs': 5,  # Quick training for demo
        'batch_size': 64,
        'learning_rate': 1e-3
    }
    
    # Train the model
    pipeline, summary = train_autoencoder(model, './data', config)
    
    # Get test data for visualization
    _, _, test_loader = create_mnist_dataloaders('./data', batch_size=64)
    
    # Generate some test data
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, tuple):
                test_inputs, test_labels = batch_data[0], batch_data[1]
            else:
                test_inputs, test_labels = batch_data, None
            break  # Just use first batch
    
    # Get reconstructions and latent representations
    reconstructions, latent_reps = model(test_inputs, return_latent=True)
    
    # Convert to numpy
    originals = test_inputs.numpy()
    recons = reconstructions.numpy()
    latents = latent_reps.numpy()
    labels = test_labels.numpy() if test_labels is not None else None
    
    # Create output directory
    Path("./demo_visualizations").mkdir(exist_ok=True)
    
    print("\n1. Creating image comparison plot...")
    plot_image_comparison(
        originals[:16], recons[:16], labels[:16] if labels is not None else None,
        title="MNIST Reconstruction Comparison",
        save_path="./demo_visualizations/image_comparison",
        show=False
    )
    
    print("2. Creating training curves...")
    logs = pipeline.logger.logs
    plot_training_curves(
        logs['train_losses'], logs['val_losses'], logs['epochs'],
        learning_rates=logs['learning_rates'],
        title="Training Progress - Basic Model",
        save_path="./demo_visualizations/training_curves",
        show=False
    )
    
    print("3. Creating latent space visualization...")
    plot_latent_space(
        latents, labels,
        method='pca', n_components=2,
        title="Latent Space Visualization (PCA)",
        save_path="./demo_visualizations/latent_space",
        show=False
    )
    
    print("4. Creating reconstruction error analysis...")
    errors = np.mean((originals - recons) ** 2, axis=(1, 2, 3) if originals.ndim == 4 else (1, 2))
    plot_reconstruction_errors(
        errors, labels,
        title="Reconstruction Error Analysis",
        save_path="./demo_visualizations/reconstruction_errors",
        show=False
    )
    
    print("5. Creating error heatmap...")
    plot_error_heatmap(
        originals[:16], recons[:16], labels[:16] if labels is not None else None,
        title="Pixel-wise Error Heatmap",
        save_path="./demo_visualizations/error_heatmap",
        show=False
    )
    
    print("Basic visualization example completed!")
    return pipeline


def comprehensive_report_example():
    """Example of creating a comprehensive visualization report."""
    print("\n" + "=" * 60)
    print("Comprehensive Report Example")
    print("=" * 60)
    
    # Create a more interesting model
    model = AutoencoderMLP(input_dim=784, latent_dim=64, activation='relu')
    
    config = {
        'epochs': 10,
        'batch_size': 128,
        'learning_rate': 2e-3,
        'early_stopping_patience': 5
    }
    
    # Train the model
    pipeline, summary = train_autoencoder(model, './data', config)
    
    # Get test data
    _, _, test_loader = create_mnist_dataloaders('./data', batch_size=128)
    
    # Create comprehensive report using the training pipeline
    print("\nGenerating comprehensive visualization report...")
    saved_plots = pipeline.create_visualization_report(
        test_loader=test_loader,
        n_samples=200,
        save_dir="./comprehensive_report",
        report_name="comprehensive_example",
        show_plots=False
    )
    
    print(f"Report saved with {len(saved_plots)} plot types:")
    for plot_type, path in saved_plots.items():
        print(f"  - {plot_type}: {path}")
    
    return pipeline


def visualization_manager_example():
    """Example using the VisualizationManager class."""
    print("\n" + "=" * 60)
    print("VisualizationManager Example")
    print("=" * 60)
    
    # Create model and get some data
    model = AutoencoderMLP(input_dim=784, latent_dim=32, activation='elu')
    
    # Quick training
    config = {'epochs': 3, 'batch_size': 64}
    pipeline, _ = train_autoencoder(model, './data', config)
    
    # Get test data
    _, _, test_loader = create_mnist_dataloaders('./data', batch_size=64)
    
    # Prepare data for visualization
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, tuple):
                inputs, labels = batch_data[0], batch_data[1]
            else:
                inputs, labels = batch_data, None
            
            recons, latents = model(inputs, return_latent=True)
            break
    
    # Use VisualizationManager
    manager = VisualizationManager(
        output_dir="./manager_example",
        default_formats=['png', 'pdf'],
        default_dpi=200
    )
    
    # Create report
    logs = pipeline.logger.logs
    saved_plots = manager.create_comprehensive_report(
        originals=inputs.numpy(),
        reconstructions=recons.numpy(), 
        latent_representations=latents.numpy(),
        train_losses=logs['train_losses'],
        val_losses=logs['val_losses'],
        labels=labels.numpy() if labels is not None else None,
        epochs=logs['epochs'],
        learning_rates=logs['learning_rates'],
        report_name="manager_demo",
        show_plots=False
    )
    
    # Show registry
    registry = manager.get_plot_registry()
    print("\nVisualization registry:")
    for report_name, plots in registry.items():
        print(f"  {report_name}: {list(plots.keys())}")
    
    return manager


def interactive_plotting_example():
    """Example of interactive plotting capabilities."""
    print("\n" + "=" * 60)
    print("Interactive Plotting Example")
    print("=" * 60)
    
    # Get some data
    train_loader, _, _ = create_mnist_dataloaders('./data', batch_size=64)
    
    # Create a simple model for demo
    model = AutoencoderMLP(input_dim=784, latent_dim=16)
    
    with torch.no_grad():
        for batch_data in train_loader:
            if isinstance(batch_data, tuple):
                inputs, labels = batch_data[0], batch_data[1] 
            else:
                inputs, labels = batch_data, None
            
            # Simple forward pass (model not trained, just for demo)
            recons, latents = model(inputs, return_latent=True)
            break
    
    # Create interactive plotter
    plotter = InteractivePlotter(
        originals=inputs.numpy(),
        reconstructions=recons.numpy(),
        latent_representations=latents.numpy(),
        labels=labels.numpy() if labels is not None else None
    )
    
    print("Creating interactive visualizations...")
    
    # Sample browser
    fig1 = plotter.create_sample_browser()
    if fig1:
        fig1.savefig("./demo_visualizations/sample_browser.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
    
    # Error distribution explorer
    fig2 = plotter.create_error_distribution_explorer()
    if fig2:
        fig2.savefig("./demo_visualizations/error_explorer.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)
    
    # Latent space explorer
    fig3 = plotter.create_latent_space_explorer()
    if fig3:
        fig3.savefig("./demo_visualizations/latent_explorer.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
    
    print("Interactive plotting examples saved!")
    return plotter


def comparison_with_sas_example():
    """Example comparing output with the original SAS python_plot reference."""
    print("\n" + "=" * 60)
    print("SAS python_plot.sas Comparison Example")
    print("=" * 60)
    
    # Recreate the functionality from python_plot.sas but with our enhanced system
    
    # Create synthetic data similar to what would be in the SAS output CSV
    np.random.seed(42)
    n_samples = 25
    
    # Generate fake MNIST-like data (labels + 784 pixel values)
    fake_labels = np.random.randint(0, 10, n_samples)
    fake_images = np.random.rand(n_samples, 784)
    
    # Reshape to 28x28 for visualization
    fake_images_2d = fake_images.reshape(n_samples, 28, 28)
    
    print("Creating SAS-style visualization with enhanced features...")
    
    # Create a plot similar to the original SAS script but enhanced
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(15, 15))
    
    for i in range(min(25, n_samples)):
        plt.subplot(5, 5, i + 1)
        plt.subplots_adjust(hspace=0.5)
        plt.title(f"Label is {fake_labels[i]}", fontsize=10)
        plt.imshow(fake_images_2d[i], cmap='Greys', interpolation='None')
        plt.axis('off')
    
    plt.suptitle("Enhanced SAS python_plot.sas Style Visualization", fontsize=16, fontweight='bold')
    plt.savefig("./demo_visualizations/sas_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now show our enhanced version
    plot_image_comparison(
        fake_images_2d, fake_images_2d * 0.9,  # Simulate reconstruction
        fake_labels,
        n_samples=25,
        grid_size=(5, 5),
        title="Enhanced Autoencoder Visualization\n(Inspired by SAS python_plot.sas)",
        save_path="./demo_visualizations/enhanced_sas_style",
        show=False
    )
    
    print("SAS comparison completed!")
    print("- Original SAS style: ./demo_visualizations/sas_comparison.png")
    print("- Enhanced version: ./demo_visualizations/enhanced_sas_style.png")


def main():
    """Run all visualization examples."""
    print("Autoencoder Visualization System Examples")
    print("=" * 60)
    
    # Create output directories
    Path("./demo_visualizations").mkdir(exist_ok=True)
    Path("./comprehensive_report").mkdir(exist_ok=True)
    Path("./manager_example").mkdir(exist_ok=True)
    
    # Check if MNIST data is available
    if not Path('./data').exists():
        print("\n⚠ MNIST data not found in './data' directory")
        print("Please ensure MNIST binary files are available for full functionality")
        print("Running limited examples with synthetic data...")
        
        # Run only examples that don't require real MNIST data
        comparison_with_sas_example()
        return
    
    try:
        # Run all examples
        examples = [
            basic_visualization_example,
            comprehensive_report_example, 
            visualization_manager_example,
            interactive_plotting_example,
            comparison_with_sas_example
        ]
        
        for example in examples:
            try:
                example()
                print()  # Add spacing between examples
            except Exception as e:
                print(f"Error in {example.__name__}: {e}")
                print("Continuing with next example...\n")
        
        print("=" * 60)
        print("All visualization examples completed!")
        print(f"Check the following directories for outputs:")
        print("  - ./demo_visualizations/")
        print("  - ./comprehensive_report/")
        print("  - ./manager_example/")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check your data setup and dependencies")


if __name__ == "__main__":
    main()
