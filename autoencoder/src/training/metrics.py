"""
Training Metrics and Monitoring for Autoencoder Training

This module provides comprehensive metrics tracking, logging, and analysis
for autoencoder training with L-BFGS optimization, including convergence
monitoring and performance statistics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainingSnapshot:
    """Single training iteration snapshot."""
    iteration: int
    timestamp: float
    loss: float
    convergence_metric: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    iteration_time: Optional[float] = None


class TrainingMetrics:
    """
    Comprehensive training metrics tracking for autoencoder training.
    
    This class tracks and manages:
    - Loss progression and convergence metrics
    - Training performance statistics
    - Iteration timing and throughput
    - Convergence analysis and early stopping signals
    - Model performance evaluation metrics
    """
    
    def __init__(self):
        """Initialize training metrics tracker."""
        self.snapshots: List[TrainingSnapshot] = []
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Running statistics
        self.best_loss = float('inf')
        self.best_iteration = 0
        self.convergence_count = 0
        self.total_iterations = 0
        
        # Performance metrics
        self.iteration_times: List[float] = []
        self.loss_improvements: List[float] = []
        
        logger.info("TrainingMetrics initialized")
    
    def update(
        self,
        iteration: int,
        loss: float,
        learning_rate: Optional[float] = None,
        convergence_check: bool = False,
        gradient_norm: Optional[float] = None
    ):
        """
        Update metrics with new training iteration data.
        
        Args:
            iteration: Current iteration number
            loss: Current loss value
            learning_rate: Current learning rate (if applicable)
            convergence_check: Whether convergence criteria are met
            gradient_norm: Current gradient norm (if available)
        """
        current_time = time.time()
        iteration_time = current_time - self.last_update_time
        
        # Calculate convergence metric (relative loss change)
        convergence_metric = None
        if len(self.snapshots) > 0:
            prev_loss = self.snapshots[-1].loss
            if prev_loss != 0:
                convergence_metric = abs((loss - prev_loss) / prev_loss)
        
        # Create snapshot
        snapshot = TrainingSnapshot(
            iteration=iteration,
            timestamp=current_time,
            loss=loss,
            convergence_metric=convergence_metric,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            iteration_time=iteration_time
        )
        
        self.snapshots.append(snapshot)
        
        # Update running statistics
        if loss < self.best_loss:
            loss_improvement = self.best_loss - loss
            self.best_loss = loss
            self.best_iteration = iteration
            self.loss_improvements.append(loss_improvement)
        
        if convergence_check:
            self.convergence_count += 1
        
        self.total_iterations = iteration
        self.iteration_times.append(iteration_time)
        self.last_update_time = current_time
    
    def get_loss_history(self) -> List[float]:
        """Get list of all loss values."""
        return [snapshot.loss for snapshot in self.snapshots]
    
    def get_convergence_history(self) -> List[Optional[float]]:
        """Get list of convergence metrics."""
        return [snapshot.convergence_metric for snapshot in self.snapshots]
    
    def get_iteration_times(self) -> List[float]:
        """Get list of iteration times."""
        return [snapshot.iteration_time or 0.0 for snapshot in self.snapshots]
    
    def calculate_loss_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive loss statistics.
        
        Returns:
            Dictionary with loss statistics
        """
        if not self.snapshots:
            return {}
        
        losses = self.get_loss_history()
        
        stats = {
            'final_loss': losses[-1],
            'best_loss': self.best_loss,
            'initial_loss': losses[0],
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'median_loss': np.median(losses),
            'loss_reduction': losses[0] - losses[-1] if len(losses) > 1 else 0.0,
            'relative_improvement': ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0.0,
            'best_iteration': self.best_iteration,
            'total_improvements': len(self.loss_improvements)
        }
        
        return stats
    
    def calculate_convergence_statistics(self) -> Dict[str, Any]:
        """
        Calculate convergence-related statistics.
        
        Returns:
            Dictionary with convergence statistics
        """
        convergence_metrics = [
            cm for cm in self.get_convergence_history() if cm is not None
        ]
        
        if not convergence_metrics:
            return {'convergence_metrics_available': False}
        
        stats = {
            'convergence_metrics_available': True,
            'mean_convergence_rate': np.mean(convergence_metrics),
            'std_convergence_rate': np.std(convergence_metrics),
            'min_convergence_rate': np.min(convergence_metrics),
            'max_convergence_rate': np.max(convergence_metrics),
            'convergence_trend': self._calculate_convergence_trend(convergence_metrics),
            'convergence_count': self.convergence_count,
            'convergence_ratio': self.convergence_count / len(self.snapshots) if self.snapshots else 0.0
        }
        
        return stats
    
    def calculate_performance_statistics(self) -> Dict[str, float]:
        """
        Calculate training performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.iteration_times:
            return {}
        
        total_time = time.time() - self.start_time
        
        stats = {
            'total_training_time': total_time,
            'mean_iteration_time': np.mean(self.iteration_times),
            'std_iteration_time': np.std(self.iteration_times),
            'min_iteration_time': np.min(self.iteration_times),
            'max_iteration_time': np.max(self.iteration_times),
            'iterations_per_second': len(self.iteration_times) / total_time if total_time > 0 else 0.0,
            'estimated_time_per_100_iterations': np.mean(self.iteration_times) * 100,
            'total_iterations': self.total_iterations
        }
        
        return stats
    
    def _calculate_convergence_trend(self, convergence_metrics: List[float]) -> str:
        """
        Calculate convergence trend (improving/stable/worsening).
        
        Args:
            convergence_metrics: List of convergence metric values
            
        Returns:
            Trend description string
        """
        if len(convergence_metrics) < 2:
            return "insufficient_data"
        
        # Calculate trend using linear regression
        x = np.arange(len(convergence_metrics))
        y = np.array(convergence_metrics)
        
        # Simple linear fit
        slope = np.polyfit(x, y, 1)[0]
        
        if slope < -1e-6:  # Decreasing (improving convergence)
            return "improving"
        elif slope > 1e-6:   # Increasing (worsening convergence)
            return "worsening"
        else:
            return "stable"
    
    def get_recent_performance(self, window_size: int = 10) -> Dict[str, float]:
        """
        Get performance metrics for recent iterations.
        
        Args:
            window_size: Number of recent iterations to analyze
            
        Returns:
            Dictionary with recent performance metrics
        """
        if len(self.snapshots) < window_size:
            window_size = len(self.snapshots)
        
        if window_size == 0:
            return {}
        
        recent_snapshots = self.snapshots[-window_size:]
        recent_losses = [s.loss for s in recent_snapshots]
        recent_times = [s.iteration_time or 0.0 for s in recent_snapshots]
        
        stats = {
            'recent_mean_loss': np.mean(recent_losses),
            'recent_std_loss': np.std(recent_losses),
            'recent_mean_time': np.mean(recent_times),
            'recent_loss_trend': self._calculate_loss_trend(recent_losses),
            'window_size': window_size
        }
        
        return stats
    
    def _calculate_loss_trend(self, losses: List[float]) -> str:
        """Calculate loss trend over a sequence."""
        if len(losses) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = np.mean(losses[:len(losses)//2])
        second_half = np.mean(losses[len(losses)//2:])
        
        relative_change = (second_half - first_half) / first_half if first_half != 0 else 0.0
        
        if relative_change < -0.01:  # Decreasing by more than 1%
            return "improving"
        elif relative_change > 0.01:   # Increasing by more than 1%
            return "worsening"
        else:
            return "stable"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training metrics summary.
        
        Returns:
            Dictionary with all training metrics
        """
        summary = {
            'loss_statistics': self.calculate_loss_statistics(),
            'convergence_statistics': self.calculate_convergence_statistics(),
            'performance_statistics': self.calculate_performance_statistics(),
            'recent_performance': self.get_recent_performance(),
            'snapshot_count': len(self.snapshots),
            'training_duration': time.time() - self.start_time
        }
        
        return summary
    
    def save_metrics(self, filepath: str):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics file
        """
        summary = self.get_summary()
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        # Recursively convert numpy types
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        summary = deep_convert(summary)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """
        Load metrics from JSON file.
        
        Args:
            filepath: Path to load metrics file from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Note: This loads summary data, not the full snapshot history
        logger.info(f"Training metrics summary loaded from {filepath}")
        return data
    
    def plot_training_progress(
        self,
        save_path: Optional[str] = None,
        show_convergence: bool = True,
        show_times: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive training progress plots.
        
        Args:
            save_path: Path to save the plot (optional)
            show_convergence: Whether to show convergence metrics
            show_times: Whether to show iteration times
            
        Returns:
            matplotlib Figure object
        """
        if not self.snapshots:
            logger.warning("No snapshots available for plotting")
            return None
        
        # Prepare data
        iterations = [s.iteration for s in self.snapshots]
        losses = [s.loss for s in self.snapshots]
        
        # Create subplots
        n_plots = 1 + int(show_convergence) + int(show_times)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Loss plot
        axes[plot_idx].plot(iterations, losses, 'b-', linewidth=2)
        axes[plot_idx].set_xlabel('Iteration')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Training Loss Progression')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')  # Log scale for loss
        
        # Mark best loss
        axes[plot_idx].axhline(y=self.best_loss, color='r', linestyle='--', 
                              label=f'Best Loss: {self.best_loss:.6e}')
        axes[plot_idx].legend()
        
        plot_idx += 1
        
        # Convergence plot
        if show_convergence:
            convergence_metrics = self.get_convergence_history()
            valid_convergence = [(i, cm) for i, cm in zip(iterations, convergence_metrics) if cm is not None]
            
            if valid_convergence:
                conv_iter, conv_vals = zip(*valid_convergence)
                axes[plot_idx].plot(conv_iter, conv_vals, 'g-', linewidth=2)
                axes[plot_idx].set_xlabel('Iteration')
                axes[plot_idx].set_ylabel('Convergence Metric')
                axes[plot_idx].set_title('Convergence Rate (Relative Loss Change)')
                axes[plot_idx].set_yscale('log')
                axes[plot_idx].grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Iteration times plot
        if show_times:
            times = self.get_iteration_times()
            if times:
                axes[plot_idx].plot(iterations, times, 'orange', linewidth=2)
                axes[plot_idx].set_xlabel('Iteration')
                axes[plot_idx].set_ylabel('Time (seconds)')
                axes[plot_idx].set_title('Iteration Times')
                axes[plot_idx].grid(True, alpha=0.3)
                
                # Add moving average
                if len(times) > 5:
                    window_size = min(20, len(times) // 4)
                    moving_avg = np.convolve(times, np.ones(window_size)/window_size, mode='valid')
                    moving_avg_iter = iterations[window_size-1:]
                    axes[plot_idx].plot(moving_avg_iter, moving_avg, 'r--', 
                                       label=f'Moving Average (window={window_size})')
                    axes[plot_idx].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        
        return fig
    
    def print_summary(self):
        """Print a comprehensive training summary."""
        summary = self.get_summary()
        
        print("=" * 80)
        print("TRAINING METRICS SUMMARY")
        print("=" * 80)
        
        # Loss statistics
        loss_stats = summary['loss_statistics']
        if loss_stats:
            print(f"Loss Statistics:")
            print(f"  Initial Loss:     {loss_stats['initial_loss']:.6e}")
            print(f"  Final Loss:       {loss_stats['final_loss']:.6e}")
            print(f"  Best Loss:        {loss_stats['best_loss']:.6e}")
            print(f"  Loss Reduction:   {loss_stats['loss_reduction']:.6e}")
            print(f"  Improvement:      {loss_stats['relative_improvement']:.2f}%")
            print(f"  Best Iteration:   {loss_stats['best_iteration']}")
        
        # Performance statistics
        perf_stats = summary['performance_statistics']
        if perf_stats:
            print(f"\nPerformance Statistics:")
            print(f"  Total Time:       {perf_stats['total_training_time']:.2f}s")
            print(f"  Mean Iter Time:   {perf_stats['mean_iteration_time']:.4f}s")
            print(f"  Iterations/sec:   {perf_stats['iterations_per_second']:.2f}")
            print(f"  Total Iterations: {perf_stats['total_iterations']}")
        
        # Convergence statistics
        conv_stats = summary['convergence_statistics']
        if conv_stats.get('convergence_metrics_available', False):
            print(f"\nConvergence Statistics:")
            print(f"  Convergence Count: {conv_stats['convergence_count']}")
            print(f"  Convergence Ratio: {conv_stats['convergence_ratio']:.2%}")
            print(f"  Trend:            {conv_stats['convergence_trend']}")
        
        print("=" * 80)


def calculate_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate reconstruction quality metrics.
    
    Args:
        original: Original input tensor
        reconstructed: Reconstructed output tensor
        
    Returns:
        Dictionary with reconstruction metrics
    """
    # Ensure tensors are on CPU for numpy operations
    if original.device != torch.device('cpu'):
        original = original.cpu()
    if reconstructed.device != torch.device('cpu'):
        reconstructed = reconstructed.cpu()
    
    # Convert to numpy for metric calculations
    orig_np = original.detach().numpy()
    recon_np = reconstructed.detach().numpy()
    
    # Mean Squared Error
    mse = np.mean((orig_np - recon_np) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(orig_np - recon_np))
    
    # Peak Signal-to-Noise Ratio (assuming data range [-1, 1])
    data_range = 2.0  # For [-1, 1] range
    if mse > 0:
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Structural Similarity (simplified version)
    # Correlation coefficient between original and reconstructed
    orig_flat = orig_np.flatten()
    recon_flat = recon_np.flatten()
    
    if np.std(orig_flat) > 0 and np.std(recon_flat) > 0:
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
    else:
        correlation = 0.0
    
    # Cosine similarity
    dot_product = np.sum(orig_flat * recon_flat)
    orig_norm = np.linalg.norm(orig_flat)
    recon_norm = np.linalg.norm(recon_flat)
    
    if orig_norm > 0 and recon_norm > 0:
        cosine_similarity = dot_product / (orig_norm * recon_norm)
    else:
        cosine_similarity = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'psnr': float(psnr),
        'correlation': float(correlation),
        'cosine_similarity': float(cosine_similarity)
    }


def compare_training_runs(
    metrics_list: List[TrainingMetrics],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare multiple training runs.
    
    Args:
        metrics_list: List of TrainingMetrics objects
        labels: Optional labels for each run
        
    Returns:
        Comparison analysis results
    """
    if not metrics_list:
        return {}
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(metrics_list))]
    
    comparison = {
        'num_runs': len(metrics_list),
        'run_labels': labels,
        'summaries': [],
        'best_run': None,
        'performance_comparison': {}
    }
    
    # Collect summaries
    best_final_loss = float('inf')
    best_run_idx = 0
    
    for i, metrics in enumerate(metrics_list):
        summary = metrics.get_summary()
        summary['label'] = labels[i]
        comparison['summaries'].append(summary)
        
        # Track best run by final loss
        final_loss = summary['loss_statistics'].get('final_loss', float('inf'))
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_run_idx = i
    
    comparison['best_run'] = {
        'index': best_run_idx,
        'label': labels[best_run_idx],
        'final_loss': best_final_loss
    }
    
    # Performance comparison
    final_losses = []
    training_times = []
    total_iterations = []
    
    for summary in comparison['summaries']:
        final_losses.append(summary['loss_statistics'].get('final_loss', float('inf')))
        training_times.append(summary['performance_statistics'].get('total_training_time', 0))
        total_iterations.append(summary['performance_statistics'].get('total_iterations', 0))
    
    comparison['performance_comparison'] = {
        'final_loss_stats': {
            'mean': np.mean(final_losses),
            'std': np.std(final_losses),
            'min': np.min(final_losses),
            'max': np.max(final_losses)
        },
        'training_time_stats': {
            'mean': np.mean(training_times),
            'std': np.std(training_times),
            'min': np.min(training_times),
            'max': np.max(training_times)
        },
        'iteration_stats': {
            'mean': np.mean(total_iterations),
            'std': np.std(total_iterations),
            'min': np.min(total_iterations),
            'max': np.max(total_iterations)
        }
    }
    
    return comparison
