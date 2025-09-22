"""
Model Evaluation and Metrics

This module provides comprehensive evaluation functionality for the MNIST autoencoder,
including validation loss computation, reconstruction quality metrics, and diagnostic
visualizations for training analysis.

Key features:
- Validation loss computation and analysis
- Reconstruction quality metrics (MSE, SSIM, etc.)  
- Learning curve plotting and diagnostic visualization
- Comparison with SAS implementation benchmarks
- Error analysis and diagnostic reporting
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import warnings

from ..model import MNISTAutoencoder
from ..mnist_data import MNISTReader


class EvaluationMetrics:
    """Class to compute and track various evaluation metrics."""
    
    @staticmethod
    def mse_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Mean Squared Error loss."""
        return F.mse_loss(reconstructed, original, reduction='mean').item()
    
    @staticmethod
    def mae_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Mean Absolute Error loss."""
        return F.l1_loss(reconstructed, original, reduction='mean').item()
    
    @staticmethod
    def rmse_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Root Mean Squared Error loss."""
        mse = F.mse_loss(reconstructed, original, reduction='mean')
        return torch.sqrt(mse).item()
    
    @staticmethod
    def pixel_accuracy(original: torch.Tensor, reconstructed: torch.Tensor, 
                      threshold: float = 0.1) -> float:
        """
        Compute pixel-level accuracy within threshold.
        
        Args:
            original: Original images tensor
            reconstructed: Reconstructed images tensor  
            threshold: Tolerance threshold for pixel values
            
        Returns:
            Pixel accuracy as percentage
        """
        diff = torch.abs(original - reconstructed)
        accurate_pixels = (diff <= threshold).float().mean()
        return accurate_pixels.item() * 100.0
    
    @staticmethod
    def structural_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        Compute simplified structural similarity metric.
        
        This is a simplified version of SSIM for batch evaluation.
        """
        # Convert to numpy for easier computation
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()
        
        # Compute means and variances
        mu1 = np.mean(orig_np, axis=1, keepdims=True)
        mu2 = np.mean(recon_np, axis=1, keepdims=True)
        
        sigma1_sq = np.var(orig_np, axis=1, keepdims=True)
        sigma2_sq = np.var(recon_np, axis=1, keepdims=True)
        sigma12 = np.mean((orig_np - mu1) * (recon_np - mu2), axis=1, keepdims=True)
        
        # SSIM constants
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim = numerator / denominator
        return np.mean(ssim)
    
    @staticmethod
    def reconstruction_error_distribution(original: torch.Tensor, 
                                        reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Analyze distribution of reconstruction errors.
        
        Returns:
            Dictionary with error distribution statistics
        """
        errors = torch.abs(original - reconstructed).flatten()
        errors_np = errors.detach().cpu().numpy()
        
        return {
            'mean_error': float(np.mean(errors_np)),
            'std_error': float(np.std(errors_np)),
            'min_error': float(np.min(errors_np)),
            'max_error': float(np.max(errors_np)),
            'median_error': float(np.median(errors_np)),
            'q25_error': float(np.percentile(errors_np, 25)),
            'q75_error': float(np.percentile(errors_np, 75))
        }


class AutoencoderEvaluator:
    """
    Comprehensive evaluator for MNIST autoencoder model.
    """
    
    def __init__(self, model: MNISTAutoencoder, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained autoencoder model
            device: Device to run evaluation on (auto-detect if None)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.metrics = EvaluationMetrics()
        self.evaluation_history = []
    
    def evaluate_dataset(self, data: torch.Tensor, 
                        batch_size: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive evaluation on dataset.
        
        Args:
            data: Input data tensor
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"Evaluating on {len(data)} samples...")
        
        all_losses = []
        all_mae = []
        all_rmse = []
        all_pixel_acc = []
        all_ssim = []
        all_originals = []
        all_reconstructed = []
        
        self.model.eval()
        with torch.no_grad():
            for start_idx in range(0, len(data), batch_size):
                end_idx = min(start_idx + batch_size, len(data))
                batch = data[start_idx:end_idx].to(self.device)
                
                # Forward pass
                reconstructed = self.model(batch)
                
                # Store for later analysis
                all_originals.append(batch.cpu())
                all_reconstructed.append(reconstructed.cpu())
                
                # Compute metrics
                mse = self.metrics.mse_loss(batch, reconstructed)
                mae = self.metrics.mae_loss(batch, reconstructed)
                rmse = self.metrics.rmse_loss(batch, reconstructed)
                pixel_acc = self.metrics.pixel_accuracy(batch, reconstructed)
                ssim = self.metrics.structural_similarity(batch, reconstructed)
                
                all_losses.append(mse)
                all_mae.append(mae)
                all_rmse.append(rmse)
                all_pixel_acc.append(pixel_acc)
                all_ssim.append(ssim)
        
        # Combine all data
        all_originals_tensor = torch.cat(all_originals, dim=0)
        all_reconstructed_tensor = torch.cat(all_reconstructed, dim=0)
        
        # Compute overall metrics
        overall_mse = self.metrics.mse_loss(all_originals_tensor, all_reconstructed_tensor)
        error_dist = self.metrics.reconstruction_error_distribution(
            all_originals_tensor, all_reconstructed_tensor
        )
        
        # Compile results
        results = {
            'dataset_size': len(data),
            'mse_loss': overall_mse,
            'mae_loss': float(np.mean(all_mae)),
            'rmse_loss': float(np.mean(all_rmse)),
            'pixel_accuracy': float(np.mean(all_pixel_acc)),
            'structural_similarity': float(np.mean(all_ssim)),
            'error_distribution': error_dist,
            'batch_metrics': {
                'mse_batches': all_losses,
                'mae_batches': all_mae,
                'rmse_batches': all_rmse,
                'pixel_acc_batches': all_pixel_acc,
                'ssim_batches': all_ssim
            }
        }
        
        # Store evaluation
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        self.evaluation_history.append(evaluation_record)
        
        return results
    
    def compare_with_sas_benchmarks(self, results: Dict[str, Any], 
                                   sas_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare results with SAS implementation benchmarks.
        
        Args:
            results: Evaluation results from evaluate_dataset
            sas_loss: SAS implementation loss for comparison
            
        Returns:
            Comparison analysis
        """
        comparison = {
            'pytorch_mse': results['mse_loss'],
            'pytorch_rmse': results['rmse_loss'],
            'pytorch_pixel_accuracy': results['pixel_accuracy'],
        }
        
        if sas_loss is not None:
            comparison['sas_mse'] = sas_loss
            comparison['loss_difference'] = results['mse_loss'] - sas_loss
            comparison['loss_ratio'] = results['mse_loss'] / sas_loss if sas_loss > 0 else float('inf')
            comparison['relative_difference'] = (results['mse_loss'] - sas_loss) / sas_loss * 100 if sas_loss > 0 else float('inf')
        
        return comparison
    
    def generate_reconstruction_samples(self, data: torch.Tensor, 
                                      n_samples: int = 10, 
                                      indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Generate reconstruction samples for visualization.
        
        Args:
            data: Input data tensor
            n_samples: Number of samples to generate
            indices: Specific indices to use (random if None)
            
        Returns:
            Dictionary with original and reconstructed samples
        """
        if indices is None:
            indices = torch.randperm(len(data))[:n_samples].tolist()
        else:
            n_samples = len(indices)
        
        samples = data[indices].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(samples)
        
        return {
            'original': samples.cpu(),
            'reconstructed': reconstructed.cpu(),
            'indices': indices
        }
    
    def plot_learning_curves(self, training_metrics: Dict, save_path: Optional[str] = None):
        """
        Plot learning curves and training diagnostics.
        
        Args:
            training_metrics: Training metrics from trainer.py
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Diagnostics', fontsize=16)
        
        epochs = training_metrics.get('epochs', [])
        train_losses = training_metrics.get('train_losses', [])
        val_losses = training_metrics.get('val_losses', [])
        convergence_metrics = training_metrics.get('convergence_metrics', [])
        
        # Training and validation loss
        axes[0, 0].plot(epochs, train_losses, label='Training Loss', color='blue')
        if val_losses:
            axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Loss convergence (log scale)
        if len(train_losses) > 1:
            axes[0, 1].semilogy(epochs, train_losses, label='Training Loss', color='blue')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss (log scale)')
            axes[0, 1].set_title('Loss Convergence')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Convergence metrics (improvement rate)
        if convergence_metrics:
            axes[1, 0].plot(epochs[:len(convergence_metrics)], convergence_metrics, 
                           color='green', marker='o', markersize=3)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Relative Improvement')
            axes[1, 0].set_title('Convergence Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
            axes[1, 0].legend()
        
        # Loss distribution histogram
        if len(train_losses) > 10:
            recent_losses = train_losses[-20:]  # Last 20 epochs
            axes[1, 1].hist(recent_losses, bins=10, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Recent Loss Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to: {save_path}")
        else:
            plt.show()
    
    def plot_reconstruction_samples(self, samples: Dict[str, torch.Tensor], 
                                   save_path: Optional[str] = None):
        """
        Plot original vs reconstructed image samples.
        
        Args:
            samples: Dictionary with original and reconstructed samples
            save_path: Path to save plot (optional)
        """
        original = samples['original']
        reconstructed = samples['reconstructed']
        n_samples = len(original)
        
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Original vs Reconstructed Images', fontsize=14)
        
        for i in range(n_samples):
            # Original image
            orig_img = original[i].reshape(28, 28)
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed image
            recon_img = reconstructed[i].reshape(28, 28)
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reconstruction samples saved to: {save_path}")
        else:
            plt.show()
    
    def plot_error_analysis(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot reconstruction error analysis.
        
        Args:
            results: Evaluation results
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Reconstruction Error Analysis', fontsize=14)
        
        # Error distribution statistics
        error_dist = results['error_distribution']
        stats_names = ['Mean', 'Median', 'Std', 'Q25', 'Q75']
        stats_values = [error_dist['mean_error'], error_dist['median_error'], 
                       error_dist['std_error'], error_dist['q25_error'], error_dist['q75_error']]
        
        axes[0].bar(stats_names, stats_values, color='skyblue', alpha=0.7)
        axes[0].set_title('Error Distribution Statistics')
        axes[0].set_ylabel('Error Value')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Batch metrics variation
        batch_metrics = results['batch_metrics']
        batch_mse = batch_metrics['mse_batches']
        batch_mae = batch_metrics['mae_batches']
        
        axes[1].plot(batch_mse, label='MSE', marker='o', markersize=3)
        axes[1].plot(batch_mae, label='MAE', marker='s', markersize=3)
        axes[1].set_xlabel('Batch Index')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Batch-wise Error Variation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Overall metrics comparison
        metrics_names = ['MSE', 'MAE', 'RMSE', 'Pixel Acc.', 'SSIM']
        metrics_values = [results['mse_loss'], results['mae_loss'], 
                         results['rmse_loss'], results['pixel_accuracy']/100, 
                         results['structural_similarity']]
        
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        bars = axes[2].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        axes[2].set_title('Overall Performance Metrics')
        axes[2].set_ylabel('Metric Value')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error analysis saved to: {save_path}")
        else:
            plt.show()
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 output_dir: str = './evaluation_results') -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save report and plots
            
        Returns:
            Path to generated report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report content
        report_content = f"""
# MNIST Autoencoder Evaluation Report
Generated: {datetime.now().isoformat()}

## Model Architecture
- Input Dimension: {self.model.input_dim}
- Hidden Dimension: {self.model.hidden_dim}
- Output Dimension: {self.model.input_dim}
- Total Parameters: {self.model.get_architecture_info()['total_parameters']}

## Dataset Evaluation
- Dataset Size: {results['dataset_size']} samples
- Device: {self.device}

## Reconstruction Quality Metrics

### Loss Metrics
- **MSE Loss**: {results['mse_loss']:.8f}
- **MAE Loss**: {results['mae_loss']:.8f}
- **RMSE Loss**: {results['rmse_loss']:.8f}

### Quality Metrics
- **Pixel Accuracy**: {results['pixel_accuracy']:.2f}%
- **Structural Similarity**: {results['structural_similarity']:.4f}

### Error Distribution
- Mean Error: {results['error_distribution']['mean_error']:.6f}
- Std Error: {results['error_distribution']['std_error']:.6f}
- Median Error: {results['error_distribution']['median_error']:.6f}
- Min Error: {results['error_distribution']['min_error']:.6f}
- Max Error: {results['error_distribution']['max_error']:.6f}

## Performance Analysis
The autoencoder shows {'good' if results['mse_loss'] < 0.1 else 'moderate' if results['mse_loss'] < 0.5 else 'poor'} reconstruction quality with an MSE loss of {results['mse_loss']:.6f}.

Pixel accuracy of {results['pixel_accuracy']:.1f}% indicates that most pixels are reconstructed within acceptable tolerance.

## Recommendations
{'The model achieves excellent reconstruction quality suitable for production use.' if results['mse_loss'] < 0.05 else 'Consider additional training or hyperparameter tuning to improve reconstruction quality.' if results['mse_loss'] > 0.1 else 'The model shows good reconstruction quality for most applications.'}
        """
        
        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report_content.strip())
        
        # Save results as JSON
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation report generated: {report_path}")
        print(f"Results saved to: {results_path}")
        
        return report_path


def evaluate_model_checkpoint(checkpoint_path: str, test_data: torch.Tensor,
                            output_dir: str = './evaluation_results') -> Dict[str, Any]:
    """
    Evaluate model from checkpoint file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data: Test dataset tensor
        output_dir: Directory to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    try:
        from ..utils.checkpointing import CheckpointManager
        manager = CheckpointManager()
        checkpoint_data, model = manager.load_checkpoint(checkpoint_path)
        print(f"Loaded model from epoch {checkpoint_data['epoch']} with loss {checkpoint_data['loss']:.8f}")
    except ImportError:
        # Fallback to basic model loading
        model, state_info = MNISTAutoencoder.load_model_state(checkpoint_path)
        print(f"Loaded model from epoch {state_info['epoch']} with loss {state_info['loss']:.8f}")
    
    # Initialize evaluator
    evaluator = AutoencoderEvaluator(model)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data)
    
    # Generate comprehensive report
    report_path = evaluator.generate_evaluation_report(results, output_dir)
    
    return results


if __name__ == "__main__":
    print("Evaluator module loaded successfully!")
