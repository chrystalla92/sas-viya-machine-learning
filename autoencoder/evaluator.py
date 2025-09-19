"""
Model evaluation and inference pipeline for autoencoder models.

This module provides the main ModelEvaluator class with comprehensive
evaluation capabilities, batch processing, and performance benchmarking.
"""

import time
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Iterator
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
import warnings
import gc

from .autoencoder_model import AutoencoderMLP
from .metrics import (
    calculate_reconstruction_errors,
    calculate_per_sample_errors,
    calculate_aggregate_errors,
    compute_latent_statistics,
    prepare_latent_visualization
)
from .model_io import (
    ModelLoader,
    SASOutputFormatter,
    create_sas_compatible_outputs
)


__all__ = [
    'ModelEvaluator',
    'BatchInferenceProcessor',
    'PerformanceBenchmark',
    'ModelComparator'
]


class PerformanceBenchmark:
    """
    Performance benchmarking utilities for model evaluation.
    """
    
    def __init__(self):
        self.reset_measurements()
    
    def reset_measurements(self):
        """Reset all performance measurements."""
        self.measurements = {
            'inference_times': [],
            'memory_usage': [],
            'batch_sizes': [],
            'throughput': [],  # samples per second
        }
    
    def start_measurement(self):
        """Start a performance measurement."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
    def end_measurement(self, batch_size: int):
        """
        End a performance measurement and record results.
        
        Args:
            batch_size (int): Size of the processed batch
        """
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        inference_time = end_time - self.start_time
        memory_used = max(0, end_memory - self.start_memory)
        throughput = batch_size / inference_time if inference_time > 0 else 0
        
        self.measurements['inference_times'].append(inference_time)
        self.measurements['memory_usage'].append(memory_used)
        self.measurements['batch_sizes'].append(batch_size)
        self.measurements['throughput'].append(throughput)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.measurements['inference_times']:
            return {'status': 'No measurements recorded'}
        
        times = np.array(self.measurements['inference_times'])
        memory = np.array(self.measurements['memory_usage'])
        throughput = np.array(self.measurements['throughput'])
        batch_sizes = np.array(self.measurements['batch_sizes'])
        
        summary = {
            'total_batches': len(times),
            'total_samples': int(np.sum(batch_sizes)),
            'total_time': float(np.sum(times)),
            'avg_batch_time': float(np.mean(times)),
            'std_batch_time': float(np.std(times)),
            'min_batch_time': float(np.min(times)),
            'max_batch_time': float(np.max(times)),
            'avg_memory_usage_mb': float(np.mean(memory)),
            'max_memory_usage_mb': float(np.max(memory)),
            'avg_throughput_samples_per_sec': float(np.mean(throughput)),
            'total_throughput_samples_per_sec': float(np.sum(batch_sizes) / np.sum(times))
        }
        
        return summary


class BatchInferenceProcessor:
    """
    Efficient batch processing for large-scale model evaluation.
    """
    
    def __init__(self, model: AutoencoderMLP, 
                 batch_size: int = 256,
                 device: Optional[Union[str, torch.device]] = None,
                 num_workers: int = 0):
        """
        Initialize batch inference processor.
        
        Args:
            model (AutoencoderMLP): Model for inference
            batch_size (int): Batch size for processing
            device (Optional[Union[str, torch.device]]): Device for computation
            num_workers (int): Number of workers for data loading
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device or next(model.parameters()).device
        self.num_workers = num_workers
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Performance tracking
        self.benchmark = PerformanceBenchmark()
    
    def process_dataset(self, dataset: Dataset, 
                       return_latent: bool = True,
                       track_performance: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Process entire dataset through model.
        
        Args:
            dataset (Dataset): Dataset to process
            return_latent (bool): Whether to return latent representations
            track_performance (bool): Whether to track performance metrics
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
                (original_data, reconstructions, latent_representations)
        """
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        originals = []
        reconstructions = []
        latents = [] if return_latent else None
        
        if track_performance:
            self.benchmark.reset_measurements()
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, tuple):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                inputs = inputs.to(self.device)
                
                if track_performance:
                    self.benchmark.start_measurement()
                
                # Forward pass
                if return_latent:
                    recon, latent = self.model(inputs, return_latent=True)
                    latents.append(latent.cpu().numpy())
                else:
                    recon = self.model(inputs, return_latent=False)
                
                # Store results
                originals.append(inputs.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())
                
                if track_performance:
                    self.benchmark.end_measurement(inputs.size(0))
                
                # Memory cleanup
                del inputs, recon
                if return_latent:
                    del latent
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate results
        originals = np.vstack(originals)
        reconstructions = np.vstack(reconstructions)
        latents = np.vstack(latents) if return_latent else None
        
        return originals, reconstructions, latents
    
    def process_data_arrays(self, data: np.ndarray,
                           return_latent: bool = True,
                           track_performance: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Process numpy arrays through model.
        
        Args:
            data (np.ndarray): Input data array
            return_latent (bool): Whether to return latent representations
            track_performance (bool): Whether to track performance metrics
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
                (original_data, reconstructions, latent_representations)
        """
        # Create tensor dataset
        tensor_data = torch.from_numpy(data).float()
        dataset = TensorDataset(tensor_data)
        
        return self.process_dataset(dataset, return_latent, track_performance)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance benchmark summary."""
        return self.benchmark.get_summary()


class ModelComparator:
    """
    Utilities for comparing different model states and configurations.
    """
    
    @staticmethod
    def compare_models(model1: AutoencoderMLP, model2: AutoencoderMLP,
                      test_data: np.ndarray,
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Compare two models on the same test data.
        
        Args:
            model1 (AutoencoderMLP): First model
            model2 (AutoencoderMLP): Second model
            test_data (np.ndarray): Test data
            model1_name (str): Name for first model
            model2_name (str): Name for second model
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        # Process data with both models
        processor1 = BatchInferenceProcessor(model1)
        processor2 = BatchInferenceProcessor(model2)
        
        orig1, recon1, latent1 = processor1.process_data_arrays(test_data)
        orig2, recon2, latent2 = processor2.process_data_arrays(test_data)
        
        # Calculate metrics for both models
        metrics1 = calculate_reconstruction_errors(
            torch.from_numpy(orig1), torch.from_numpy(recon1)
        )
        metrics2 = calculate_reconstruction_errors(
            torch.from_numpy(orig2), torch.from_numpy(recon2)
        )
        
        # Performance comparison
        perf1 = processor1.get_performance_summary()
        perf2 = processor2.get_performance_summary()
        
        # Latent space comparison
        latent_stats1 = compute_latent_statistics(torch.from_numpy(latent1))
        latent_stats2 = compute_latent_statistics(torch.from_numpy(latent2))
        
        comparison = {
            'models': {
                model1_name: {
                    'config': model1.get_config(),
                    'reconstruction_metrics': metrics1,
                    'performance_metrics': perf1,
                    'latent_statistics': latent_stats1
                },
                model2_name: {
                    'config': model2.get_config(),
                    'reconstruction_metrics': metrics2,
                    'performance_metrics': perf2,
                    'latent_statistics': latent_stats2
                }
            },
            'comparison_summary': {
                'better_mse': model1_name if metrics1['mse'] < metrics2['mse'] else model2_name,
                'better_mae': model1_name if metrics1['mae'] < metrics2['mae'] else model2_name,
                'faster_inference': model1_name if perf1.get('avg_throughput_samples_per_sec', 0) > perf2.get('avg_throughput_samples_per_sec', 0) else model2_name,
                'mse_improvement': abs(metrics1['mse'] - metrics2['mse']),
                'mae_improvement': abs(metrics1['mae'] - metrics2['mae']),
            }
        }
        
        return comparison


class ModelEvaluator:
    """
    Comprehensive model evaluation and inference pipeline.
    
    Provides unified interface for model evaluation, batch inference,
    performance benchmarking, and result formatting.
    """
    
    def __init__(self, model: AutoencoderMLP,
                 batch_size: int = 256,
                 device: Optional[Union[str, torch.device]] = None,
                 output_dir: str = "./evaluation_results"):
        """
        Initialize model evaluator.
        
        Args:
            model (AutoencoderMLP): Model to evaluate
            batch_size (int): Batch size for evaluation
            device (Optional[Union[str, torch.device]]): Device for computation
            output_dir (str): Directory for saving results
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device or next(model.parameters()).device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize batch processor
        self.processor = BatchInferenceProcessor(model, batch_size, self.device)
        
        # Initialize formatter
        self.formatter = SASOutputFormatter()
        
        # Results storage
        self.last_evaluation = None
    
    def evaluate_dataset(self, dataset: Union[Dataset, DataLoader, np.ndarray],
                        labels: Optional[np.ndarray] = None,
                        save_results: bool = True,
                        include_visualization: bool = True,
                        results_prefix: str = "evaluation") -> Dict[str, Any]:
        """
        Comprehensive evaluation of model on dataset.
        
        Args:
            dataset (Union[Dataset, DataLoader, np.ndarray]): Data to evaluate
            labels (Optional[np.ndarray]): Optional labels for analysis
            save_results (bool): Whether to save results to files
            include_visualization (bool): Whether to prepare visualization data
            results_prefix (str): Prefix for result files
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        print("Starting comprehensive model evaluation...")
        
        # Process data based on input type
        if isinstance(dataset, np.ndarray):
            original, reconstructed, latent = self.processor.process_data_arrays(dataset)
        elif isinstance(dataset, (Dataset, DataLoader)):
            if isinstance(dataset, DataLoader):
                # Convert DataLoader back to dataset for consistent processing
                all_data = []
                for batch in dataset:
                    if isinstance(batch, tuple):
                        all_data.append(batch[0])
                    else:
                        all_data.append(batch)
                dataset_array = torch.cat(all_data, dim=0).numpy()
                original, reconstructed, latent = self.processor.process_data_arrays(dataset_array)
            else:
                original, reconstructed, latent = self.processor.process_dataset(dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
        
        print(f"Processed {len(original)} samples")
        
        # Calculate reconstruction metrics
        print("Calculating reconstruction metrics...")
        reconstruction_errors = calculate_reconstruction_errors(
            torch.from_numpy(original), torch.from_numpy(reconstructed)
        )
        
        per_sample_errors = calculate_per_sample_errors(
            torch.from_numpy(original), torch.from_numpy(reconstructed)
        )
        
        aggregate_errors = calculate_aggregate_errors(per_sample_errors)
        
        # Compute latent space statistics
        print("Analyzing latent space...")
        latent_stats = compute_latent_statistics(torch.from_numpy(latent))
        
        # Performance metrics
        performance_metrics = self.processor.get_performance_summary()
        
        # Prepare visualization data if requested
        viz_data = None
        if include_visualization:
            print("Preparing visualization data...")
            viz_data = prepare_latent_visualization(
                torch.from_numpy(latent),
                labels=torch.from_numpy(labels) if labels is not None else None
            )
        
        # Compile results
        evaluation_results = {
            'evaluation_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_config': self.model.get_config(),
                'n_samples': len(original),
                'n_features': original.shape[1],
                'batch_size': self.batch_size,
                'device': str(self.device)
            },
            'reconstruction_metrics': {
                'aggregate': reconstruction_errors,
                'per_sample_stats': aggregate_errors
            },
            'latent_statistics': latent_stats,
            'performance_metrics': performance_metrics,
            'raw_data': {
                'original': original,
                'reconstructed': reconstructed,
                'latent': latent,
                'per_sample_errors': {k: v.numpy() for k, v in per_sample_errors.items()}
            }
        }
        
        if labels is not None:
            evaluation_results['raw_data']['labels'] = labels
        
        if viz_data is not None:
            evaluation_results['visualization_data'] = viz_data
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(evaluation_results, results_prefix)
        
        self.last_evaluation = evaluation_results
        print("Evaluation complete!")
        
        return evaluation_results
    
    def benchmark_inference_speed(self, data: np.ndarray,
                                 batch_sizes: List[int] = [32, 64, 128, 256, 512],
                                 n_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark inference speed across different batch sizes.
        
        Args:
            data (np.ndarray): Data for benchmarking
            batch_sizes (List[int]): Batch sizes to test
            n_runs (int): Number of runs per batch size
            
        Returns:
            Dict[str, Any]: Benchmarking results
        """
        print("Starting inference speed benchmark...")
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")
            times = []
            throughputs = []
            
            for run in range(n_runs):
                # Create processor with specific batch size
                processor = BatchInferenceProcessor(
                    self.model, batch_size=batch_size, device=self.device
                )
                
                start_time = time.time()
                _, _, _ = processor.process_data_arrays(data, track_performance=True)
                end_time = time.time()
                
                total_time = end_time - start_time
                throughput = len(data) / total_time
                
                times.append(total_time)
                throughputs.append(throughput)
            
            benchmark_results[batch_size] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_throughput': np.mean(throughputs),
                'std_throughput': np.std(throughputs)
            }
        
        # Find optimal batch size
        optimal_batch_size = max(batch_sizes, 
                                key=lambda bs: benchmark_results[bs]['avg_throughput'])
        
        benchmark_summary = {
            'benchmark_results': benchmark_results,
            'optimal_batch_size': optimal_batch_size,
            'optimal_throughput': benchmark_results[optimal_batch_size]['avg_throughput'],
            'benchmark_info': {
                'data_size': len(data),
                'n_runs_per_batch_size': n_runs,
                'device': str(self.device)
            }
        }
        
        print(f"Benchmark complete! Optimal batch size: {optimal_batch_size}")
        return benchmark_summary
    
    def _save_evaluation_results(self, results: Dict[str, Any], prefix: str):
        """Save evaluation results to various formats."""
        print("Saving evaluation results...")
        
        # Save SAS-compatible outputs
        sas_paths = create_sas_compatible_outputs(
            results['raw_data']['original'],
            results['raw_data']['reconstructed'],
            results['raw_data']['latent'],
            str(self.output_dir),
            f"{prefix}_sas_output"
        )
        
        # Save comprehensive results as JSON
        results_copy = results.copy()
        # Remove raw data for JSON (too large)
        results_copy.pop('raw_data', None)
        results_copy.pop('visualization_data', None)
        
        json_path = self.output_dir / f"{prefix}_summary.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        # Save raw data separately
        np.savez(
            self.output_dir / f"{prefix}_raw_data.npz",
            original=results['raw_data']['original'],
            reconstructed=results['raw_data']['reconstructed'],
            latent=results['raw_data']['latent'],
            **results['raw_data']['per_sample_errors']
        )
        
        print(f"Results saved to {self.output_dir}")
        print(f"SAS-compatible CSV: {sas_paths['csv']}")
        print(f"Summary JSON: {json_path}")
    
    def load_and_evaluate(self, model_path: str, dataset: Union[Dataset, np.ndarray],
                         **evaluation_kwargs) -> Dict[str, Any]:
        """
        Load model from checkpoint and evaluate.
        
        Args:
            model_path (str): Path to model checkpoint
            dataset (Union[Dataset, np.ndarray]): Data to evaluate
            **evaluation_kwargs: Additional arguments for evaluate_dataset
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Load model
        loaded_model, metadata = ModelLoader.load_model(model_path, self.device)
        
        # Update evaluator with loaded model
        self.model = loaded_model
        self.processor = BatchInferenceProcessor(loaded_model, self.batch_size, self.device)
        
        # Evaluate
        results = self.evaluate_dataset(dataset, **evaluation_kwargs)
        
        # Add loading metadata to results
        results['model_loading_info'] = metadata
        
        return results
    
    def compare_with_checkpoint(self, checkpoint_path: str, dataset: np.ndarray) -> Dict[str, Any]:
        """
        Compare current model with a checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint for comparison
            dataset (np.ndarray): Data for comparison
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        # Load checkpoint model
        checkpoint_model, _ = ModelLoader.load_model(checkpoint_path, self.device)
        
        # Compare models
        comparator = ModelComparator()
        comparison = comparator.compare_models(
            self.model, checkpoint_model, dataset,
            "Current Model", "Checkpoint Model"
        )
        
        return comparison
    
    def generate_evaluation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path (Optional[str]): Path for report file
            
        Returns:
            str: Path to generated report
        """
        if self.last_evaluation is None:
            raise ValueError("No evaluation results available. Run evaluate_dataset first.")
        
        if output_path is None:
            output_path = self.output_dir / "evaluation_report.txt"
        
        results = self.last_evaluation
        
        # Generate report text
        report_lines = [
            "=" * 80,
            "AUTOENCODER MODEL EVALUATION REPORT",
            "=" * 80,
            "",
            f"Evaluation Date: {results['evaluation_info']['timestamp']}",
            f"Model Configuration: {results['evaluation_info']['model_config']}",
            f"Dataset Size: {results['evaluation_info']['n_samples']} samples",
            f"Input Features: {results['evaluation_info']['n_features']}",
            f"Device Used: {results['evaluation_info']['device']}",
            "",
            "RECONSTRUCTION PERFORMANCE",
            "-" * 40,
        ]
        
        # Add reconstruction metrics
        recon_metrics = results['reconstruction_metrics']['aggregate']
        report_lines.extend([
            f"Mean Squared Error (MSE): {recon_metrics['mse']:.6f}",
            f"Mean Absolute Error (MAE): {recon_metrics['mae']:.6f}",
            f"Root Mean Squared Error (RMSE): {recon_metrics['rmse']:.6f}",
            f"SSIM Approximation: {recon_metrics['ssim_approx']:.4f}",
            ""
        ])
        
        # Add latent space analysis
        latent_stats = results['latent_statistics']
        report_lines.extend([
            "LATENT SPACE ANALYSIS",
            "-" * 40,
            f"Latent Dimension: {len(latent_stats['mean_per_dim'])}",
            f"Mean Activation: {latent_stats['mean']:.4f}",
            f"Standard Deviation: {latent_stats['std']:.4f}",
            f"Effective Dimensionality: {latent_stats['effective_dimensionality']:.2f}",
            ""
        ])
        
        # Add performance metrics
        perf_metrics = results['performance_metrics']
        if 'total_samples' in perf_metrics:
            report_lines.extend([
                "COMPUTATIONAL PERFORMANCE",
                "-" * 40,
                f"Total Samples: {perf_metrics['total_samples']}",
                f"Total Processing Time: {perf_metrics['total_time']:.2f} seconds",
                f"Average Throughput: {perf_metrics['total_throughput_samples_per_sec']:.1f} samples/sec",
                f"Average Memory Usage: {perf_metrics['avg_memory_usage_mb']:.1f} MB",
                ""
            ])
        
        report_lines.extend([
            "=" * 80,
            "End of Report"
        ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation report saved to: {output_path}")
        return str(output_path)
