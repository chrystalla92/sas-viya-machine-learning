#!/usr/bin/env python3
"""
Demo script showing comprehensive evaluation capabilities for MLP Autoencoder.

This script demonstrates:
- Model evaluation with comprehensive metrics
- Latent space analysis and visualization
- Side-by-side image comparisons
- Error heatmap analysis
- Training curve visualization
- Publication-ready plot generation

Usage:
    python evaluate_model.py [--model_path MODEL_PATH] [--data_path DATA_PATH]
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# Import our modules
from src.autoencoder.evaluation import (
    evaluate_model_comprehensive, 
    latent_analysis,
    comprehensive_metrics,
    compare_models
)
from src.autoencoder.visualization import (
    plot_original_vs_reconstructed,
    plot_latent_space,
    plot_error_heatmap,
    plot_training_curves,
    plot_image_grid,
    plot_metrics_comparison,
    create_evaluation_report,
    setup_publication_style
)

# Import the model (assuming mnist_autoencoder is in the path)
sys.path.append(str(Path(__file__).parent.parent / 'mnist_autoencoder'))
from models.autoencoder import MLPAutoencoder


def load_mnist_data(batch_size: int = 64, num_samples: Optional[int] = None) -> DataLoader:
    """Load MNIST test data for evaluation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='../mnist_autoencoder/data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    if num_samples is not None:
        # Limit dataset size for faster demo
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_or_create_model(model_path: Optional[str] = None) -> MLPAutoencoder:
    """Load a trained model or create a new one with random weights for demo."""
    model = MLPAutoencoder()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Model loaded successfully!")
    else:
        print("No trained model found. Using randomly initialized model for demonstration.")
        print("Note: Results will not be meaningful without a trained model.")
        print("Train a model first using: python examples/train_autoencoder.py")
    
    return model


def generate_sample_training_curves() -> tuple:
    """Generate sample training curves for visualization demo."""
    # Simulate training curves
    epochs = list(range(1, 51))
    
    # Simulated training loss (decreasing with some noise)
    train_losses = [0.5 * np.exp(-0.05 * e) + 0.01 + 0.005 * np.random.randn() for e in epochs]
    
    # Simulated validation loss (similar but slightly higher)
    val_losses = [0.55 * np.exp(-0.04 * e) + 0.015 + 0.008 * np.random.randn() for e in epochs]
    
    return train_losses, val_losses, epochs


def demonstrate_evaluation_capabilities(model: MLPAutoencoder, test_loader: DataLoader,
                                      save_plots: bool = True) -> Dict[str, Any]:
    """
    Demonstrate comprehensive evaluation capabilities.
    
    Args:
        model: Trained (or untrained) autoencoder model
        test_loader: Test data loader
        save_plots: Whether to save generated plots
        
    Returns:
        Dictionary containing all evaluation results
    """
    print("=" * 60)
    print("MLP AUTOENCODER EVALUATION DEMONSTRATION")
    print("=" * 60)
    
    # Set up publication-ready plotting style
    setup_publication_style()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    # 1. Comprehensive model evaluation
    print("\n1. Performing comprehensive model evaluation...")
    eval_results = evaluate_model_comprehensive(
        model, test_loader, device, max_batches=10
    )
    results['evaluation'] = eval_results
    
    print(f"   Average MSE: {eval_results['metrics']['avg_mse']:.6f}")
    print(f"   Average PSNR: {eval_results['metrics']['avg_psnr']:.2f} dB")
    print(f"   Average SSIM: {eval_results['metrics']['avg_ssim']:.4f}")
    print(f"   Overall Quality Score: {eval_results['overall_quality']:.4f}")
    print(f"   Evaluated on {eval_results['total_samples']} samples")
    
    # 2. Latent space analysis
    print("\n2. Performing latent space analysis...")
    latent_results = latent_analysis(
        model, test_loader, device, max_samples=500
    )
    results['latent_analysis'] = latent_results
    
    print(f"   Latent dimension: {latent_results['latent_dim']}")
    print(f"   Analyzed {latent_results['num_samples']} samples")
    print(f"   PCA explained variance: {latent_results['pca_explained_variance'].sum():.1%}")
    
    # 3. Get sample data for visualizations
    print("\n3. Preparing sample data for visualizations...")
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_data, sample_labels = sample_batch
        else:
            sample_data = sample_batch
            sample_labels = None
            
        sample_data = sample_data.to(device)[:16]  # Take first 16 samples
        if sample_labels is not None:
            sample_labels = sample_labels[:16]
        
        # Get reconstructions
        reconstructed = model(sample_data)
        
        # Calculate per-sample metrics
        sample_metrics = []
        for i in range(sample_data.size(0)):
            metrics = comprehensive_metrics(
                sample_data[i:i+1], 
                reconstructed[i:i+1]
            )
            sample_metrics.append(metrics)
    
    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    
    # 4a. Original vs Reconstructed comparison
    print("   Creating original vs reconstructed comparison...")
    fig1 = plot_original_vs_reconstructed(
        sample_data, reconstructed, 
        num_samples=8, 
        labels=sample_labels,
        save_path='original_vs_reconstructed.png' if save_plots else None
    )
    if not save_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # 4b. Latent space visualization
    print("   Creating latent space visualization...")
    fig2 = plot_latent_space(
        latent_results, 
        method='both',
        save_path='latent_space_analysis.png' if save_plots else None
    )
    if not save_plots:
        plt.show()
    else:
        plt.close(fig2)
    
    # 4c. Error heatmaps
    print("   Creating error heatmaps...")
    fig3 = plot_error_heatmap(
        sample_data, reconstructed,
        num_samples=6,
        save_path='reconstruction_errors.png' if save_plots else None
    )
    if not save_plots:
        plt.show()
    else:
        plt.close(fig3)
    
    # 4d. Training curves (simulated)
    print("   Creating training curve visualization...")
    train_losses, val_losses, epochs = generate_sample_training_curves()
    fig4 = plot_training_curves(
        train_losses, val_losses, epochs,
        save_path='training_curves.png' if save_plots else None
    )
    if not save_plots:
        plt.show()
    else:
        plt.close(fig4)
    
    # 4e. Image grid
    print("   Creating image grid...")
    fig5 = plot_image_grid(
        sample_data[:16],
        labels=sample_labels[:16] if sample_labels is not None else None,
        title="Sample MNIST Images",
        save_path='image_grid.png' if save_plots else None
    )
    if not save_plots:
        plt.show()
    else:
        plt.close(fig5)
    
    # 5. Generate comprehensive evaluation report
    if save_plots:
        print("\n5. Generating comprehensive evaluation report...")
        sample_images = {
            'original': sample_data,
            'reconstructed': reconstructed
        }
        
        report_files = create_evaluation_report(
            eval_results, 
            latent_results,
            sample_images,
            save_dir='evaluation_report'
        )
        
        print("   Evaluation report saved to 'evaluation_report/' directory")
        print("   Generated files:")
        for plot_type, file_path in report_files.items():
            print(f"     {plot_type}: {file_path}")
    
    # 6. Model comparison demo (create second model for comparison)
    print("\n6. Demonstrating model comparison...")
    model2 = MLPAutoencoder(hidden_size=200)  # Different architecture
    model2 = model2.to(device)
    
    comparison_results = compare_models(
        [model, model2], test_loader, device
    )
    
    print("   Comparison Results:")
    for model_name, model_results in comparison_results['comparison_summary'].items():
        print(f"     {model_name}:")
        print(f"       Quality Score: {model_results['quality_score']:.4f}")
        print(f"       MSE: {model_results['avg_mse']:.6f}")
        print(f"       PSNR: {model_results['avg_psnr']:.2f} dB")
        print(f"       SSIM: {model_results['avg_ssim']:.4f}")
    
    print(f"\n   Best model: {comparison_results['best_model']}")
    
    results['comparison'] = comparison_results
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"✓ Comprehensive metrics computed for {eval_results['total_samples']} samples")
    print(f"✓ Latent space analyzed for {latent_results['num_samples']} samples")
    print(f"✓ Multiple visualization plots generated")
    print(f"✓ Model comparison performed")
    
    if save_plots:
        print(f"✓ All plots saved to current directory and 'evaluation_report/'")
    
    print("\nKey Findings:")
    print(f"- Model reconstruction quality: {eval_results['overall_quality']:.1%}")
    print(f"- Latent space captures {latent_results['pca_explained_variance'].sum():.1%} of variance in 2D")
    print(f"- Average reconstruction error (MSE): {eval_results['metrics']['avg_mse']:.6f}")
    
    return results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Demonstrate MLP Autoencoder evaluation capabilities"
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data_samples', 
        type=int, 
        default=1000,
        help='Number of test samples to evaluate (default: 1000)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--no_save', 
        action='store_true',
        help='Do not save plots (display instead)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading MNIST test data...")
    test_loader = load_mnist_data(
        batch_size=args.batch_size, 
        num_samples=args.data_samples
    )
    
    # Load or create model
    model = load_or_create_model(args.model_path)
    
    # Run evaluation demonstration
    results = demonstrate_evaluation_capabilities(
        model, test_loader, save_plots=not args.no_save
    )
    
    print("\nDemo completed successfully!")
    
    if not args.no_save:
        print("\nTo view the generated plots, check the current directory and 'evaluation_report/' folder.")
    
    return results


if __name__ == "__main__":
    # Check for required dependencies
    try:
        import sklearn
        import skimage
        import seaborn
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install scikit-learn scikit-image seaborn")
        sys.exit(1)
    
    main()
