"""
Training and Evaluation Metrics

This module provides comprehensive metrics tracking and analysis for training
and evaluation of the MNIST autoencoder, including loss tracking, convergence
analysis, and performance monitoring.

Key features:
- Training metrics tracking (loss, epochs, convergence)
- Evaluation metrics computation (reconstruction quality)
- Convergence analysis and early stopping support
- Metrics persistence and loading
- Training diagnostics and analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class TrainingMetrics:
    """Class to track and manage training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        self.convergence_metrics = []
        self.iteration_counts = []
        self.epoch_times = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.start_time = None
        self.total_training_time = 0.0
        self.optimizer_info = {}
        
    def update(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
               lr: float = 0.0, convergence_metric: Optional[float] = None,
               iteration_count: Optional[int] = None, epoch_time: Optional[float] = None):
        """Update metrics with new values."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.learning_rates.append(lr)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            # Track best validation loss
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
        else:
            # Track best training loss if no validation
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_epoch = epoch
                
        if convergence_metric is not None:
            self.convergence_metrics.append(convergence_metric)
            
        if iteration_count is not None:
            self.iteration_counts.append(iteration_count)
            
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics values."""
        if not self.epochs:
            return {}
            
        return {
            'current_epoch': self.epochs[-1],
            'current_train_loss': self.train_losses[-1],
            'current_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.epochs),
            'convergence_metric': self.convergence_metrics[-1] if self.convergence_metrics else None
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        return {
            'total_epochs': len(self.epochs),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'training_time': self.total_training_time,
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'total_iterations': sum(self.iteration_counts) if self.iteration_counts else 0,
            'average_iterations_per_epoch': np.mean(self.iteration_counts) if self.iteration_counts else 0,
            'convergence_achieved': self.convergence_metrics[-1] < 1e-6 if self.convergence_metrics else False
        }
    
    def compute_convergence_metrics(self, window_size: int = 10) -> Dict[str, float]:
        """
        Compute convergence analysis metrics.
        
        Args:
            window_size: Window size for moving average analysis
            
        Returns:
            Dictionary with convergence metrics
        """
        if len(self.train_losses) < window_size:
            return {
                'moving_average_improvement': 0.0,
                'loss_stability': 0.0,
                'convergence_rate': 0.0,
                'epochs_to_convergence': -1
            }
        
        recent_losses = self.train_losses[-window_size:]
        older_losses = self.train_losses[-2*window_size:-window_size] if len(self.train_losses) >= 2*window_size else self.train_losses[:window_size]
        
        # Moving average improvement
        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)
        moving_avg_improvement = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0.0
        
        # Loss stability (coefficient of variation)
        loss_stability = np.std(recent_losses) / np.mean(recent_losses) if np.mean(recent_losses) > 0 else float('inf')
        
        # Convergence rate (exponential decay fit)
        if len(self.train_losses) > 10:
            x = np.arange(len(self.train_losses))
            y = np.log(np.array(self.train_losses) + 1e-10)
            convergence_rate = np.polyfit(x, y, 1)[0]  # Negative slope indicates convergence
        else:
            convergence_rate = 0.0
        
        # Estimate epochs to convergence (based on current rate)
        if convergence_rate < 0 and recent_avg > 1e-6:
            epochs_to_convergence = int(np.log(1e-6 / recent_avg) / convergence_rate)
        else:
            epochs_to_convergence = -1
        
        return {
            'moving_average_improvement': moving_avg_improvement,
            'loss_stability': loss_stability,
            'convergence_rate': convergence_rate,
            'epochs_to_convergence': epochs_to_convergence
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        metrics_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'convergence_metrics': self.convergence_metrics,
            'iteration_counts': self.iteration_counts,
            'epoch_times': self.epoch_times,
            'optimizer_info': self.optimizer_info,
            'summary': self.get_summary(),
            'convergence_analysis': self.compute_convergence_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_data = convert_numpy_types(metrics_data)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_metrics(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.epochs = data.get('epochs', [])
        self.train_losses = data.get('train_losses', [])
        self.val_losses = data.get('val_losses', [])
        self.learning_rates = data.get('learning_rates', [])
        self.convergence_metrics = data.get('convergence_metrics', [])
        self.iteration_counts = data.get('iteration_counts', [])
        self.epoch_times = data.get('epoch_times', [])
        self.optimizer_info = data.get('optimizer_info', {})
        
        # Recompute derived metrics
        if self.val_losses:
            self.best_loss = min(self.val_losses)
            self.best_epoch = self.epochs[self.val_losses.index(self.best_loss)]
        elif self.train_losses:
            self.best_loss = min(self.train_losses)
            self.best_epoch = self.epochs[self.train_losses.index(self.best_loss)]


class LBFGSMetrics:
    """Specialized metrics for L-BFGS optimizer tracking."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset L-BFGS specific metrics."""
        self.function_evaluations = []
        self.gradient_evaluations = []
        self.line_search_steps = []
        self.convergence_criteria = []
        self.tolerance_reached = []
        
    def update_lbfgs_metrics(self, func_evals: int, grad_evals: int,
                           line_search: int, convergence: bool, tolerance_met: bool):
        """Update L-BFGS specific metrics."""
        self.function_evaluations.append(func_evals)
        self.gradient_evaluations.append(grad_evals)
        self.line_search_steps.append(line_search)
        self.convergence_criteria.append(convergence)
        self.tolerance_reached.append(tolerance_met)
    
    def get_lbfgs_summary(self) -> Dict[str, Any]:
        """Get summary of L-BFGS metrics."""
        if not self.function_evaluations:
            return {}
        
        return {
            'total_function_evaluations': sum(self.function_evaluations),
            'total_gradient_evaluations': sum(self.gradient_evaluations),
            'total_line_search_steps': sum(self.line_search_steps),
            'average_func_evals_per_epoch': np.mean(self.function_evaluations),
            'convergence_rate': np.mean(self.convergence_criteria),
            'tolerance_achievement_rate': np.mean(self.tolerance_reached)
        }


class EvaluationMetrics:
    """Class for comprehensive evaluation metrics computation."""
    
    @staticmethod
    def reconstruction_loss_metrics(original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive reconstruction loss metrics.
        
        Args:
            original: Original images tensor
            reconstructed: Reconstructed images tensor
            
        Returns:
            Dictionary with loss metrics
        """
        with torch.no_grad():
            # Basic loss metrics
            mse = F.mse_loss(reconstructed, original, reduction='mean').item()
            mae = F.l1_loss(reconstructed, original, reduction='mean').item()
            rmse = torch.sqrt(F.mse_loss(reconstructed, original, reduction='mean')).item()
            
            # Normalized losses
            original_norm = torch.norm(original)
            reconstruction_error_norm = torch.norm(original - reconstructed)
            normalized_mse = (reconstruction_error_norm / original_norm).item() if original_norm > 0 else float('inf')
            
            # Element-wise statistics
            diff = torch.abs(original - reconstructed)
            max_error = torch.max(diff).item()
            min_error = torch.min(diff).item()
            median_error = torch.median(diff).item()
            
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'normalized_mse': normalized_mse,
            'max_error': max_error,
            'min_error': min_error,
            'median_error': median_error
        }
    
    @staticmethod
    def reconstruction_quality_metrics(original: torch.Tensor, 
                                     reconstructed: torch.Tensor,
                                     threshold: float = 0.1) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.
        
        Args:
            original: Original images tensor
            reconstructed: Reconstructed images tensor
            threshold: Threshold for pixel accuracy
            
        Returns:
            Dictionary with quality metrics
        """
        with torch.no_grad():
            # Pixel accuracy
            diff = torch.abs(original - reconstructed)
            pixel_accuracy = (diff <= threshold).float().mean().item() * 100.0
            
            # Structural similarity (simplified)
            orig_flat = original.view(original.size(0), -1)
            recon_flat = reconstructed.view(reconstructed.size(0), -1)
            
            # Correlation coefficient
            orig_centered = orig_flat - orig_flat.mean(dim=1, keepdim=True)
            recon_centered = recon_flat - recon_flat.mean(dim=1, keepdim=True)
            
            numerator = (orig_centered * recon_centered).sum(dim=1)
            denominator = torch.sqrt((orig_centered ** 2).sum(dim=1) * (recon_centered ** 2).sum(dim=1))
            correlation = (numerator / (denominator + 1e-8)).mean().item()
            
            # Signal-to-noise ratio
            signal_power = (original ** 2).mean()
            noise_power = ((original - reconstructed) ** 2).mean()
            snr = (10 * torch.log10(signal_power / (noise_power + 1e-8))).item()
            
        return {
            'pixel_accuracy': pixel_accuracy,
            'correlation': correlation,
            'snr_db': snr
        }
    
    @staticmethod
    def compare_with_baseline(results: Dict[str, float], 
                            baseline_results: Dict[str, float]) -> Dict[str, float]:
        """
        Compare current results with baseline (e.g., SAS implementation).
        
        Args:
            results: Current model results
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        for metric in results:
            if metric in baseline_results:
                current = results[metric]
                baseline = baseline_results[metric]
                
                if baseline != 0:
                    relative_diff = ((current - baseline) / baseline) * 100
                    ratio = current / baseline
                else:
                    relative_diff = float('inf') if current != 0 else 0
                    ratio = float('inf') if current != 0 else 1
                
                comparison[f'{metric}_baseline'] = baseline
                comparison[f'{metric}_current'] = current
                comparison[f'{metric}_relative_diff_pct'] = relative_diff
                comparison[f'{metric}_ratio'] = ratio
                comparison[f'{metric}_improved'] = current < baseline if 'loss' in metric or 'error' in metric else current > baseline
        
        return comparison


class MetricsLogger:
    """Comprehensive metrics logging and analysis."""
    
    def __init__(self, log_dir: str = './metrics_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.training_metrics = TrainingMetrics()
        self.lbfgs_metrics = LBFGSMetrics()
        self.log_file = os.path.join(log_dir, 'training_log.txt')
        
    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch metrics to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format log message
        log_msg = f"[{timestamp}] Epoch {epoch:3d}: "
        
        # Add key metrics
        if 'train_loss' in metrics:
            log_msg += f"Train Loss = {metrics['train_loss']:.8f}"
        if 'val_loss' in metrics:
            log_msg += f", Val Loss = {metrics['val_loss']:.8f}"
        if 'convergence_metric' in metrics:
            log_msg += f", Improvement = {metrics['convergence_metric']:.2e}"
        if 'epoch_time' in metrics:
            log_msg += f", Time = {metrics['epoch_time']:.2f}s"
        if 'iterations' in metrics:
            log_msg += f", Iterations = {metrics['iterations']}"
        
        # Print and log to file
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def log_training_summary(self, final_metrics: Dict[str, Any]):
        """Log training summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary_msg = f"""
[{timestamp}] Training Summary:
- Total Epochs: {final_metrics.get('total_epochs', 0)}
- Best Loss: {final_metrics.get('best_loss', 0):.8f} (Epoch {final_metrics.get('best_epoch', 0)})
- Final Train Loss: {final_metrics.get('final_train_loss', 0):.8f}
- Training Time: {final_metrics.get('training_time', 0):.2f}s
- Average Epoch Time: {final_metrics.get('average_epoch_time', 0):.2f}s
- Convergence Achieved: {final_metrics.get('convergence_achieved', False)}
"""
        
        print(summary_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(summary_msg + '\n')
    
    def save_comprehensive_metrics(self, filename: str = 'comprehensive_metrics.json'):
        """Save all metrics to comprehensive JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        
        all_metrics = {
            'training_metrics': {
                'epochs': self.training_metrics.epochs,
                'train_losses': self.training_metrics.train_losses,
                'val_losses': self.training_metrics.val_losses,
                'convergence_metrics': self.training_metrics.convergence_metrics,
                'iteration_counts': self.training_metrics.iteration_counts,
                'epoch_times': self.training_metrics.epoch_times,
                'summary': self.training_metrics.get_summary(),
                'convergence_analysis': self.training_metrics.compute_convergence_metrics()
            },
            'lbfgs_metrics': self.lbfgs_metrics.get_lbfgs_summary(),
            'timestamp': datetime.now().isoformat(),
            'log_directory': self.log_dir
        }
        
        # Convert numpy types to native Python types for JSON serialization
        all_metrics = convert_numpy_types(all_metrics)
        
        with open(filepath, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"Comprehensive metrics saved to: {filepath}")


def analyze_training_convergence(metrics: TrainingMetrics, 
                               window_size: int = 10,
                               threshold: float = 1e-6) -> Dict[str, Any]:
    """
    Analyze training convergence characteristics.
    
    Args:
        metrics: Training metrics object
        window_size: Window for convergence analysis
        threshold: Convergence threshold
        
    Returns:
        Dictionary with convergence analysis
    """
    if len(metrics.train_losses) < window_size:
        return {'converged': False, 'analysis': 'Insufficient data'}
    
    # Recent losses for analysis
    recent_losses = metrics.train_losses[-window_size:]
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]  # Negative means decreasing
    
    # Stability analysis
    loss_std = np.std(recent_losses)
    loss_mean = np.mean(recent_losses)
    stability_coeff = loss_std / loss_mean if loss_mean > 0 else float('inf')
    
    # Convergence criteria
    mean_improvement = -loss_trend  # Positive means improving
    is_stable = stability_coeff < 0.1  # Less than 10% variation
    is_below_threshold = loss_mean < threshold
    
    converged = is_stable and (is_below_threshold or mean_improvement < 1e-8)
    
    return {
        'converged': converged,
        'loss_trend': loss_trend,
        'mean_recent_loss': loss_mean,
        'loss_stability': stability_coeff,
        'mean_improvement': mean_improvement,
        'below_threshold': is_below_threshold,
        'is_stable': is_stable,
        'analysis': 'Converged' if converged else 'Still training'
    }


if __name__ == "__main__":
    print("Metrics utilities loaded successfully!")
    
    # Demo usage
    metrics = TrainingMetrics()
    
    # Simulate some training metrics
    for epoch in range(1, 11):
        loss = 1.0 / epoch  # Decreasing loss
        metrics.update(epoch, loss, None, 0.001)
    
    print("Demo metrics summary:")
    print(json.dumps(metrics.get_summary(), indent=2))
    
    print("Convergence analysis:")
    convergence = analyze_training_convergence(metrics)
    print(json.dumps(convergence, indent=2))
