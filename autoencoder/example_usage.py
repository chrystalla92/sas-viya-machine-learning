"""
Example Usage of MNIST Autoencoder Training Framework

This script demonstrates how to use the complete training and evaluation
framework for the MNIST autoencoder, showing the end-to-end workflow from
data loading to model evaluation.

Usage:
    python example_usage.py [--data-path DATA_PATH] [--epochs EPOCHS]
"""

import os
import argparse
import numpy as np
from typing import Optional

# Import our framework modules
from model import create_sas_compatible_autoencoder
from training import TrainingConfig, train_mnist_autoencoder
from evaluation import AutoencoderEvaluator
from checkpoints import CheckpointManager
from mnist_data import load_mnist_data, create_sas_format_dataset


def example_with_real_mnist_data(images_path: str, labels_path: str, 
                                max_epochs: int = 100) -> None:
    """
    Example training with real MNIST data files.
    
    Args:
        images_path: Path to MNIST images IDX file
        labels_path: Path to MNIST labels IDX file
        max_epochs: Maximum training epochs
    """
    print("=== Training MNIST Autoencoder with Real Data ===")
    
    # Create training configuration
    config = TrainingConfig()
    config.max_epochs = max_epochs
    config.validation_ratio = 0.2
    config.early_stopping_patience = 20
    config.log_interval = 5
    config.save_interval = 25
    config.checkpoint_dir = './mnist_checkpoints'
    config.log_dir = './mnist_logs'
    
    print("Training Configuration:")
    print(f"  Max Epochs: {config.max_epochs}")
    print(f"  Validation Ratio: {config.validation_ratio}")
    print(f"  Optimizer: {config.optimizer_type} (max_iters={config.max_iters})")
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")
    
    # Train the model
    model, metrics = train_mnist_autoencoder(images_path, labels_path, config)
    
    # Print training summary
    summary = metrics.get_summary()
    print("\n=== Training Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Evaluate the trained model
    print("\n=== Model Evaluation ===")
    images, labels = load_mnist_data(images_path, labels_path, standardize=True)
    
    evaluator = AutoencoderEvaluator(model)
    results = evaluator.evaluate_dataset(images[:1000])  # Evaluate on first 1000 samples
    
    print("Evaluation Results:")
    print(f"  MSE Loss: {results['mse_loss']:.8f}")
    print(f"  RMSE Loss: {results['rmse_loss']:.8f}")
    print(f"  Pixel Accuracy: {results['pixel_accuracy']:.2f}%")
    print(f"  Structural Similarity: {results['structural_similarity']:.4f}")
    
    # Generate reconstruction samples
    samples = evaluator.generate_reconstruction_samples(
        images[:100].clone().detach() if hasattr(images, 'clone') else images[:100], 
        n_samples=5
    )
    
    # Save plots and reports
    evaluator.plot_learning_curves(
        metrics.__dict__, 
        save_path='./learning_curves.png'
    )
    
    evaluator.plot_reconstruction_samples(
        samples, 
        save_path='./reconstruction_samples.png'
    )
    
    evaluator.plot_error_analysis(
        results, 
        save_path='./error_analysis.png'
    )
    
    # Generate comprehensive report
    evaluator.generate_evaluation_report(results, './evaluation_results')
    
    print(f"\nResults saved to ./evaluation_results/")
    print(f"Plots saved: learning_curves.png, reconstruction_samples.png, error_analysis.png")


def example_with_mock_data(max_epochs: int = 20) -> None:
    """
    Example training with mock data (for testing when MNIST files not available).
    
    Args:
        max_epochs: Maximum training epochs
    """
    print("=== Training MNIST Autoencoder with Mock Data ===")
    
    # Generate mock MNIST-like data
    np.random.seed(23451)
    n_samples = 5000
    
    # Create structured mock images (easier to learn patterns)
    mock_images = np.zeros((n_samples, 784))
    for i in range(n_samples):
        # Create simple patterns
        pattern_type = i % 4
        if pattern_type == 0:  # Horizontal lines
            for row in range(7, 21, 3):
                start_idx = row * 28 + 7
                end_idx = row * 28 + 21
                mock_images[i, start_idx:end_idx] = 0.8
        elif pattern_type == 1:  # Vertical lines
            for col in range(7, 21, 3):
                for row in range(7, 21):
                    idx = row * 28 + col
                    mock_images[i, idx] = 0.8
        elif pattern_type == 2:  # Diagonal
            for d in range(14):
                idx = (7 + d) * 28 + (7 + d)
                mock_images[i, idx] = 0.8
        else:  # Random dots
            dot_indices = np.random.choice(784, size=50, replace=False)
            mock_images[i, dot_indices] = 0.6
    
    # Add noise
    mock_images += np.random.normal(0, 0.1, mock_images.shape)
    mock_images = np.clip(mock_images, 0, 1)
    
    mock_labels = np.random.randint(0, 10, n_samples)
    
    print(f"Generated {n_samples} mock samples")
    
    # Create configuration
    config = TrainingConfig()
    config.max_epochs = max_epochs
    config.validation_ratio = 0.2
    config.early_stopping_patience = 10
    config.log_interval = 2
    config.save_interval = 10
    config.checkpoint_dir = './mock_checkpoints'
    config.log_dir = './mock_logs'
    
    # Train using the framework
    from training import AutoencoderTrainer
    from data_utils import train_validation_split
    
    # Split data
    train_images, val_images, train_labels, val_labels = train_validation_split(
        mock_images, mock_labels, 
        validation_ratio=config.validation_ratio,
        random_seed=config.seed
    )
    
    # Initialize and run training
    trainer = AutoencoderTrainer(config)
    metrics = trainer.train(train_images, val_images)
    
    # Evaluation
    print("\n=== Model Evaluation ===")
    evaluator = AutoencoderEvaluator(trainer.model)
    results = evaluator.evaluate_dataset(val_images)
    
    print("Evaluation Results:")
    print(f"  MSE Loss: {results['mse_loss']:.8f}")
    print(f"  Pixel Accuracy: {results['pixel_accuracy']:.2f}%")
    print(f"  Structural Similarity: {results['structural_similarity']:.4f}")
    
    # Test checkpointing
    print("\n=== Testing Checkpointing ===")
    manager = CheckpointManager(config.checkpoint_dir)
    checkpoints = manager.list_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")
    
    if manager.get_best_checkpoint_info():
        best_info = manager.get_best_checkpoint_info()
        print(f"Best checkpoint: epoch {best_info['epoch']}, loss {best_info['loss']:.6f}")
    
    print("Mock data example completed successfully!")


def compare_with_sas_expectations() -> None:
    """
    Demonstrate comparison with SAS implementation expectations.
    """
    print("=== SAS Implementation Comparison ===")
    
    # Create model with exact SAS configuration
    model = create_sas_compatible_autoencoder(seed=23451)
    arch_info = model.get_architecture_info()
    
    print("Model Architecture (should match SAS):")
    print(f"  Input Dimension: {arch_info['input_dim']} (var2-var785)")
    print(f"  Hidden Dimension: {arch_info['hidden_dim']} neurons")
    print(f"  Output Dimension: {arch_info['output_dim']} (reconstruction)")
    print(f"  Activation: {arch_info['activation']}")
    print(f"  Total Parameters: {arch_info['total_parameters']}")
    
    # Show SAS equivalent configuration
    print("\nSAS Configuration Equivalent:")
    print("  proc nnet data=mnist_train standardize=midrange;")
    print("    input var2-var785;")
    print("    architecture MLP;")
    print("    hidden 400 / act=tanh;")
    print("    train outmodel=nnetModel seed=23451;")
    print("    optimization algorithm=LBFGS maxiters=500;")
    print("  run;")
    
    print("\nExpected Behavior:")
    print("  ✓ Deterministic results with seed=23451")
    print("  ✓ Fast convergence with L-BFGS optimizer")
    print("  ✓ Low reconstruction loss on training data")
    print("  ✓ Stable training without divergence")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='MNIST Autoencoder Training Framework Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example_usage.py --mock-data --epochs 20
  python example_usage.py --data-path ./data/ --epochs 100
  python example_usage.py --compare-sas
        """
    )
    
    parser.add_argument('--data-path', type=str, 
                       help='Path to directory containing MNIST IDX files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum training epochs (default: 50)')
    parser.add_argument('--mock-data', action='store_true',
                       help='Use mock data instead of real MNIST')
    parser.add_argument('--compare-sas', action='store_true',
                       help='Show SAS implementation comparison')
    
    args = parser.parse_args()
    
    if args.compare_sas:
        compare_with_sas_expectations()
    elif args.mock_data:
        example_with_mock_data(max_epochs=args.epochs)
    elif args.data_path:
        images_path = os.path.join(args.data_path, 'train-images-idx3-ubyte')
        labels_path = os.path.join(args.data_path, 'train-labels-idx1-ubyte')
        
        if os.path.exists(images_path) and os.path.exists(labels_path):
            example_with_real_mnist_data(images_path, labels_path, max_epochs=args.epochs)
        else:
            print(f"MNIST files not found in {args.data_path}")
            print("Expected files:")
            print(f"  {images_path}")
            print(f"  {labels_path}")
            print("\nFalling back to mock data...")
            example_with_mock_data(max_epochs=args.epochs)
    else:
        print("Please specify --data-path, --mock-data, or --compare-sas")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
