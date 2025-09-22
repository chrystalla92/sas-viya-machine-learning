"""
Comprehensive Visualization Examples for MNIST Autoencoder

This script demonstrates how to use the visualization package to create
publication-ready plots and analysis of the MNIST autoencoder training
and performance.

Usage:
    python example_visualization.py [--data_path DATA_PATH] [--model_path MODEL_PATH] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import autoencoder modules
from model import MNISTAutoencoder, create_sas_compatible_autoencoder
from mnist_data import load_mnist_data, MNISTReader
from training import AutoencoderTrainer, TrainingConfig
from evaluation import AutoencoderEvaluator

# Import visualization modules
from visualization.plots import (
    plot_mnist_grid, plot_reconstruction_comparison, 
    plot_reconstruction_grid, save_reconstruction_samples,
    load_and_plot_from_csv
)
from visualization.training_plots import (
    plot_training_curves, plot_loss_convergence, 
    plot_training_progress, load_and_plot_training_metrics
)
from visualization.analysis import (
    plot_latent_pca, plot_latent_tsne, plot_reconstruction_errors,
    plot_weight_distributions, plot_activation_heatmap,
    analyze_hidden_representations
)
from utils.plot_utils import setup_publication_style, enable_interactive_mode


def create_sample_data(n_samples=100):
    """
    Create sample MNIST-like data for demonstration purposes.
    
    Returns:
        Tuple of (images, labels, reconstructed_images)
    """
    print("Creating sample data for demonstration...")
    
    # Generate synthetic MNIST-like data
    np.random.seed(42)
    images = np.random.rand(n_samples, 784)
    
    # Add some structure to make it more MNIST-like
    for i in range(n_samples):
        # Create a simple pattern
        img = images[i].reshape(28, 28)
        center = (14, 14)
        radius = np.random.randint(5, 12)
        
        # Create circular patterns
        y, x = np.ogrid[:28, :28]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = np.random.rand() * 0.8 + 0.2
        
        images[i] = img.flatten()
    
    # Generate labels (0-9)
    labels = np.random.randint(0, 10, n_samples)
    
    # Generate "reconstructed" images (add some noise)
    reconstructed = images + np.random.normal(0, 0.1, images.shape)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    return images, labels, reconstructed


def create_sample_training_metrics():
    """
    Create sample training metrics for demonstration.
    
    Returns:
        Dictionary with training metrics
    """
    print("Creating sample training metrics...")
    
    # Generate synthetic training curves
    epochs = list(range(1, 101))
    
    # Create realistic loss curves
    train_losses = []
    val_losses = []
    
    initial_loss = 0.5
    for epoch in epochs:
        # Exponential decay with some noise
        train_loss = initial_loss * np.exp(-epoch * 0.02) + np.random.normal(0, 0.001)
        val_loss = initial_loss * np.exp(-epoch * 0.018) + np.random.normal(0, 0.001)
        
        train_losses.append(max(train_loss, 0.001))  # Ensure positive
        val_losses.append(max(val_loss, 0.001))
    
    # Create convergence metrics
    convergence_metrics = [abs(train_losses[i] - train_losses[i-1]) if i > 0 else 0.1 
                          for i in range(len(train_losses))]
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': [0.001] * len(epochs),  # Constant learning rate
        'convergence_metrics': convergence_metrics,
        'summary': {
            'total_epochs': len(epochs),
            'best_loss': min(train_losses),
            'best_epoch': train_losses.index(min(train_losses)) + 1,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'training_time': 450.0
        }
    }


def demonstrate_image_visualization(images, labels, reconstructed, output_dir):
    """Demonstrate image visualization capabilities."""
    print("\n=== Image Visualization Examples ===")
    
    # 1. Basic MNIST grid (replicating SAS functionality)
    print("Creating MNIST grid visualization...")
    fig1 = plot_mnist_grid(
        images[:25], labels[:25],
        title="Original MNIST-like Images (5x5 Grid)",
        save_path=str(output_dir / "mnist_grid_demo")
    )
    plt.show()
    
    # 2. Reconstruction comparison
    print("Creating reconstruction comparison...")
    fig2 = plot_reconstruction_comparison(
        images[:10], reconstructed[:10], labels[:10],
        title="Original vs Reconstructed Comparison",
        save_path=str(output_dir / "reconstruction_comparison_demo")
    )
    plt.show()
    
    # 3. Reconstruction grid with errors
    print("Creating comprehensive reconstruction grid...")
    fig3 = plot_reconstruction_grid(
        images[:20], reconstructed[:20],
        grid_size=(4, 5),
        title="Comprehensive Reconstruction Analysis",
        save_path=str(output_dir / "reconstruction_grid_demo")
    )
    plt.show()
    
    # 4. Save individual reconstruction samples
    print("Saving individual reconstruction samples...")
    save_reconstruction_samples(
        images[:10], reconstructed[:10], labels[:10],
        output_dir=str(output_dir / "individual_samples"),
        n_samples=5
    )


def demonstrate_training_visualization(training_metrics, output_dir):
    """Demonstrate training visualization capabilities."""
    print("\n=== Training Visualization Examples ===")
    
    # 1. Training curves
    print("Creating training curves...")
    fig1 = plot_training_curves(
        train_losses=training_metrics['train_losses'],
        val_losses=training_metrics['val_losses'],
        epochs=training_metrics['epochs'],
        title="Training and Validation Loss Curves",
        save_path=str(output_dir / "training_curves_demo")
    )
    plt.show()
    
    # 2. Convergence analysis
    print("Creating convergence analysis...")
    fig2 = plot_loss_convergence(
        losses=training_metrics['train_losses'],
        convergence_threshold=1e-4,
        title="Loss Convergence Analysis",
        save_path=str(output_dir / "convergence_analysis_demo")
    )
    plt.show()
    
    # 3. Comprehensive training progress
    print("Creating training progress overview...")
    fig3 = plot_training_progress(
        metrics_data=training_metrics,
        title="Comprehensive Training Progress",
        save_path=str(output_dir / "training_progress_demo")
    )
    plt.show()


def demonstrate_analysis_visualization(model, images, labels, output_dir):
    """Demonstrate analysis and diagnostic visualization capabilities."""
    print("\n=== Analysis and Diagnostic Visualization Examples ===")
    
    # Convert to tensors
    images_tensor = torch.FloatTensor(images)
    labels_tensor = torch.LongTensor(labels) if labels is not None else None
    
    # Get hidden representations
    model.eval()
    with torch.no_grad():
        hidden_repr = model.encode(images_tensor)
        reconstructed = model(images_tensor)
    
    # 1. PCA visualization of latent space
    print("Creating PCA latent space visualization...")
    fig1 = plot_latent_pca(
        hidden_repr, labels_tensor,
        title="PCA of Hidden Layer Representations",
        save_path=str(output_dir / "latent_pca_demo")
    )
    plt.show()
    
    # 2. t-SNE visualization of latent space
    print("Creating t-SNE latent space visualization...")
    fig2 = plot_latent_tsne(
        hidden_repr[:200], labels_tensor[:200] if labels_tensor is not None else None,  # Limit for t-SNE
        title="t-SNE of Hidden Layer Representations",
        save_path=str(output_dir / "latent_tsne_demo")
    )
    plt.show()
    
    # 3. Reconstruction error analysis
    print("Creating reconstruction error analysis...")
    fig3 = plot_reconstruction_errors(
        images_tensor, reconstructed, labels_tensor,
        error_type="mse",
        title="Reconstruction Error Analysis (MSE)",
        save_path=str(output_dir / "reconstruction_errors_demo")
    )
    plt.show()
    
    # 4. Weight distribution analysis
    print("Creating weight distribution analysis...")
    fig4 = plot_weight_distributions(
        model,
        title="Model Weight and Bias Distributions",
        save_path=str(output_dir / "weight_distributions_demo")
    )
    plt.show()
    
    # 5. Activation heatmap
    print("Creating activation heatmap...")
    fig5 = plot_activation_heatmap(
        hidden_repr[:50], labels_tensor[:50] if labels_tensor is not None else None,
        title="Hidden Layer Activation Patterns",
        save_path=str(output_dir / "activation_heatmap_demo")
    )
    plt.show()


def run_comprehensive_analysis(model, images, labels, output_dir):
    """Run comprehensive latent space analysis."""
    print("\n=== Comprehensive Latent Space Analysis ===")
    
    figures = analyze_hidden_representations(
        model=model,
        data=images,
        labels=labels,
        output_dir=str(output_dir / "comprehensive_analysis"),
        n_samples=min(500, len(images))
    )
    
    print(f"Comprehensive analysis complete. Generated {len(figures)} plots.")
    
    # Display one of the figures
    if 'pca' in figures:
        plt.figure(figures['pca'].number)
        plt.show()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="MNIST Autoencoder Visualization Demo")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to MNIST data (will use synthetic data if not provided)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model (will create new model if not provided)")
    parser.add_argument("--output_dir", type=str, default="./visualization_demo",
                       help="Output directory for saved plots")
    parser.add_argument("--interactive", action="store_true",
                       help="Enable interactive plotting mode")
    parser.add_argument("--no_display", action="store_true",
                       help="Don't display plots interactively (just save)")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure plotting
    setup_publication_style()
    if args.interactive:
        enable_interactive_mode()
    
    if args.no_display:
        # Use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
    
    print("MNIST Autoencoder Visualization Demonstration")
    print("=" * 50)
    
    # Load or create data
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}")
        try:
            # Try to load real MNIST data
            train_data, val_data, test_data = load_mnist_data(args.data_path)
            images = train_data[:1000]  # Use first 1000 samples
            labels = np.arange(len(images)) % 10  # Synthetic labels for demo
            
            # Create a simple model and generate reconstructions
            model = create_sas_compatible_autoencoder()
            model.eval()
            with torch.no_grad():
                images_tensor = torch.FloatTensor(images)
                reconstructed_tensor = model(images_tensor)
                reconstructed = reconstructed_tensor.numpy()
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using synthetic data instead...")
            images, labels, reconstructed = create_sample_data(1000)
            model = create_sas_compatible_autoencoder()
    else:
        print("Using synthetic data for demonstration...")
        images, labels, reconstructed = create_sample_data(1000)
        model = create_sas_compatible_autoencoder()
    
    # Load or create model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        try:
            model, _ = MNISTAutoencoder.load_model_state(args.model_path)
            # Generate reconstructions with loaded model
            model.eval()
            with torch.no_grad():
                images_tensor = torch.FloatTensor(images)
                reconstructed_tensor = model(images_tensor)
                reconstructed = reconstructed_tensor.numpy()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default model...")
            model = create_sas_compatible_autoencoder()
    
    # Create sample training metrics
    training_metrics = create_sample_training_metrics()
    
    # Save sample metrics to JSON for loading demo
    metrics_path = output_dir / "sample_training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"Data shape: {images.shape}")
    print(f"Model architecture: {model.get_architecture_info()}")
    print(f"Output directory: {output_dir}")
    
    # Run demonstrations
    try:
        # 1. Image visualization
        demonstrate_image_visualization(images, labels, reconstructed, output_dir)
        
        # 2. Training visualization
        demonstrate_training_visualization(training_metrics, output_dir)
        
        # 3. Analysis visualization
        demonstrate_analysis_visualization(model, images, labels, output_dir)
        
        # 4. Comprehensive analysis
        run_comprehensive_analysis(model, images, labels, output_dir)
        
        # 5. Demonstrate loading from JSON
        print("\n=== Loading and Plotting from JSON ===")
        fig_from_json = load_and_plot_training_metrics(
            str(metrics_path),
            plot_type="progress",
            save_path=str(output_dir / "loaded_from_json")
        )
        if not args.no_display:
            plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Visualization demonstration complete!")
    print(f"All plots saved to: {output_dir}")
    print("\nKey features demonstrated:")
    print("✓ MNIST image grid visualization (SAS migration)")
    print("✓ Original vs reconstructed image comparisons")
    print("✓ Training loss curves and convergence analysis")
    print("✓ Latent space visualization (PCA, t-SNE)")
    print("✓ Reconstruction error analysis")
    print("✓ Weight distribution analysis")
    print("✓ Activation heatmaps")
    print("✓ Publication-ready formatting")
    print("✓ Batch processing and file export")
    
    if not args.no_display:
        input("\nPress Enter to close all plots and exit...")
        plt.close('all')


if __name__ == "__main__":
    main()
