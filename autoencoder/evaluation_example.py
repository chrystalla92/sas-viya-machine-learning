#!/usr/bin/env python3
"""
Comprehensive evaluation example for autoencoder models.

This script demonstrates how to use the evaluation and inference functionality
including metrics calculation, batch processing, performance benchmarking,
and SAS-compatible output generation.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoencoder import (
    # Model and data
    AutoencoderMLP,
    load_mnist_test_data,
    create_mnist_dataloaders,
    
    # Evaluation components
    ModelEvaluator,
    BatchInferenceProcessor,
    ModelComparator,
    PerformanceBenchmark,
    
    # Metrics and analysis
    calculate_reconstruction_errors,
    compute_latent_statistics,
    prepare_latent_visualization,
    
    # Model I/O
    ModelSaver,
    ModelLoader,
    create_sas_compatible_outputs
)


def create_example_model_and_data():
    """Create an example model and test data for evaluation."""
    print("Creating example model and loading test data...")
    
    # Create a simple model
    model = AutoencoderMLP(
        input_dim=784,
        latent_dim=400,
        activation='tanh',
        device='cpu'  # Use CPU for example
    )
    
    # Load test data
    try:
        test_images, test_labels = load_mnist_test_data("./data", standardize=True)
        # Use a subset for faster example execution
        test_images = test_images[:1000]
        test_labels = test_labels[:1000]
        
        print(f"Loaded {len(test_images)} test samples")
        return model, test_images.numpy(), test_labels.numpy()
        
    except FileNotFoundError:
        print("MNIST data not found. Creating synthetic data for demonstration...")
        # Create synthetic data
        n_samples = 1000
        synthetic_images = np.random.randn(n_samples, 784) * 0.1
        synthetic_labels = np.random.randint(0, 10, n_samples)
        return model, synthetic_images, synthetic_labels


def example_basic_evaluation():
    """Demonstrate basic model evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Model Evaluation")
    print("="*60)
    
    # Create model and data
    model, test_data, test_labels = create_example_model_and_data()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        batch_size=128,
        output_dir="./evaluation_results"
    )
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_dataset(
        dataset=test_data,
        labels=test_labels,
        save_results=True,
        include_visualization=True,
        results_prefix="basic_evaluation"
    )
    
    # Print key metrics
    print("\nReconstruction Metrics:")
    recon_metrics = results['reconstruction_metrics']['aggregate']
    for metric, value in recon_metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")
    
    print(f"\nLatent Space Statistics:")
    latent_stats = results['latent_statistics']
    print(f"  Effective Dimensionality: {latent_stats['effective_dimensionality']:.2f}")
    print(f"  Mean Activation: {latent_stats['mean']:.4f}")
    print(f"  Standard Deviation: {latent_stats['std']:.4f}")
    
    # Generate evaluation report
    report_path = evaluator.generate_evaluation_report()
    print(f"\nDetailed report saved to: {report_path}")
    
    return model, test_data, test_labels


def example_performance_benchmarking():
    """Demonstrate performance benchmarking."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Performance Benchmarking")
    print("="*60)
    
    # Create model and data
    model, test_data, test_labels = create_example_model_and_data()
    
    # Create evaluator
    evaluator = ModelEvaluator(model, output_dir="./evaluation_results")
    
    # Benchmark inference speed across different batch sizes
    benchmark_results = evaluator.benchmark_inference_speed(
        data=test_data,
        batch_sizes=[32, 64, 128, 256],
        n_runs=3
    )
    
    print("\nBenchmark Results:")
    for batch_size, metrics in benchmark_results['benchmark_results'].items():
        print(f"  Batch Size {batch_size:3d}: "
              f"{metrics['avg_throughput']:.1f} ± {metrics['std_throughput']:.1f} samples/sec")
    
    print(f"\nOptimal batch size: {benchmark_results['optimal_batch_size']} "
          f"({benchmark_results['optimal_throughput']:.1f} samples/sec)")


def example_model_comparison():
    """Demonstrate model comparison functionality."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Model Comparison")
    print("="*60)
    
    # Create two different models for comparison
    model1 = AutoencoderMLP(input_dim=784, latent_dim=400, activation='tanh')
    model2 = AutoencoderMLP(input_dim=784, latent_dim=200, activation='relu')
    
    # Create test data
    _, test_data, _ = create_example_model_and_data()
    test_data = test_data[:500]  # Smaller subset for faster comparison
    
    # Compare models
    comparator = ModelComparator()
    comparison_results = comparator.compare_models(
        model1, model2, test_data,
        "Large Latent (400)", "Small Latent (200)"
    )
    
    print("\nModel Comparison Results:")
    summary = comparison_results['comparison_summary']
    print(f"  Better MSE: {summary['better_mse']}")
    print(f"  Better MAE: {summary['better_mae']}")
    print(f"  Faster Inference: {summary['faster_inference']}")
    print(f"  MSE Difference: {summary['mse_improvement']:.6f}")


def example_batch_processing():
    """Demonstrate batch inference processing."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Inference Processing")
    print("="*60)
    
    # Create model and data
    model, test_data, test_labels = create_example_model_and_data()
    
    # Create batch processor
    processor = BatchInferenceProcessor(
        model=model,
        batch_size=256,
        device='cpu'
    )
    
    # Process data in batches
    original, reconstructed, latent = processor.process_data_arrays(
        test_data,
        return_latent=True,
        track_performance=True
    )
    
    # Get performance summary
    perf_summary = processor.get_performance_summary()
    
    print(f"Processed {perf_summary['total_samples']} samples in "
          f"{perf_summary['total_time']:.2f} seconds")
    print(f"Average throughput: {perf_summary['total_throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Average memory usage: {perf_summary['avg_memory_usage_mb']:.1f} MB")
    
    # Calculate metrics
    reconstruction_errors = calculate_reconstruction_errors(
        torch.from_numpy(original), torch.from_numpy(reconstructed)
    )
    
    print(f"\nReconstruction Quality:")
    print(f"  MSE: {reconstruction_errors['mse']:.6f}")
    print(f"  MAE: {reconstruction_errors['mae']:.6f}")


def example_sas_output_generation():
    """Demonstrate SAS-compatible output generation."""
    print("\n" + "="*60)
    print("EXAMPLE 5: SAS-Compatible Output Generation")
    print("="*60)
    
    # Create model and data
    model, test_data, test_labels = create_example_model_and_data()
    
    # Process through model
    processor = BatchInferenceProcessor(model)
    original, reconstructed, latent = processor.process_data_arrays(test_data)
    
    # Create SAS-compatible outputs
    output_paths = create_sas_compatible_outputs(
        original=original,
        reconstructed=reconstructed,
        latent=latent,
        output_dir="./sas_outputs",
        base_filename="mnist_autoencoder_results"
    )
    
    print("SAS-compatible outputs created:")
    for format_type, path in output_paths.items():
        print(f"  {format_type.upper()}: {path}")
    
    print(f"\nOutput includes {len(original)} samples with:")
    print(f"  - {original.shape[1]} original features")
    print(f"  - {reconstructed.shape[1]} reconstructed features")
    print(f"  - {latent.shape[1]} latent features")
    print(f"  - Per-sample reconstruction errors")


def example_model_save_load():
    """Demonstrate model saving and loading with evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Model Save/Load with Evaluation")
    print("="*60)
    
    # Create and train a simple model
    model, test_data, test_labels = create_example_model_and_data()
    
    # Save model
    saver = ModelSaver(base_dir="./models")
    model_path = saver.save_model(
        model=model,
        metadata={'description': 'Example MNIST autoencoder'},
        filepath="example_model.pth"
    )
    
    print(f"Model saved to: {model_path}")
    
    # Create evaluator and load model for evaluation
    evaluator = ModelEvaluator(model, output_dir="./evaluation_results")
    
    # Evaluate loaded model
    results = evaluator.load_and_evaluate(
        model_path=model_path,
        dataset=test_data,
        save_results=False,
        results_prefix="loaded_model_eval"
    )
    
    print(f"\nLoaded model evaluation:")
    print(f"  Loading info: {results['model_loading_info']['save_timestamp']}")
    print(f"  MSE: {results['reconstruction_metrics']['aggregate']['mse']:.6f}")


def example_latent_space_analysis():
    """Demonstrate latent space analysis and visualization preparation."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Latent Space Analysis")
    print("="*60)
    
    # Create model and data
    model, test_data, test_labels = create_example_model_and_data()
    
    # Extract latent representations
    processor = BatchInferenceProcessor(model)
    _, _, latent = processor.process_data_arrays(test_data[:500])  # Smaller subset
    
    # Analyze latent space
    latent_stats = compute_latent_statistics(torch.from_numpy(latent))
    
    print("Latent Space Statistics:")
    print(f"  Dimensions: {len(latent_stats['mean_per_dim'])}")
    print(f"  Effective Dimensionality: {latent_stats['effective_dimensionality']:.2f}")
    print(f"  Mean per dimension range: {latent_stats['mean_per_dim'].min():.3f} to {latent_stats['mean_per_dim'].max():.3f}")
    print(f"  Std per dimension range: {latent_stats['std_per_dim'].min():.3f} to {latent_stats['std_per_dim'].max():.3f}")
    
    # Prepare visualization data
    viz_data = prepare_latent_visualization(
        torch.from_numpy(latent),
        labels=torch.from_numpy(test_labels[:500]),
        use_pca=True,
        use_tsne=True
    )
    
    print(f"\nVisualization Data Prepared:")
    print(f"  Original latent shape: {viz_data['original'].shape}")
    if 'pca_data' in viz_data:
        print(f"  PCA reduced shape: {viz_data['pca_data'].shape}")
        print(f"  PCA explained variance (first 5 components): {viz_data['pca_explained_variance_ratio'][:5]}")
    if 'tsne_data' in viz_data:
        print(f"  t-SNE reduced shape: {viz_data['tsne_data'].shape}")


def main():
    """Run all evaluation examples."""
    print("Autoencoder Evaluation and Inference Examples")
    print("=" * 80)
    
    # Create output directories
    os.makedirs("evaluation_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("sas_outputs", exist_ok=True)
    
    # Run examples
    try:
        example_basic_evaluation()
        example_performance_benchmarking()
        example_model_comparison()
        example_batch_processing()
        example_sas_output_generation()
        example_model_save_load()
        example_latent_space_analysis()
        
        print("\n" + "="*80)
        print("All evaluation examples completed successfully!")
        print("Check the following directories for outputs:")
        print("  - ./evaluation_results/ - Evaluation reports and results")
        print("  - ./models/ - Saved models")
        print("  - ./sas_outputs/ - SAS-compatible data files")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during evaluation examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
