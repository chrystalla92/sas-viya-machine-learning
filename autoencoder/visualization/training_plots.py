"""
Training-Specific Visualizations for MNIST Autoencoder

This module provides comprehensive visualization tools for monitoring and analyzing
the training process of the MNIST autoencoder, including:
- Loss curves and convergence plots
- Training progress monitoring
- Learning rate schedules
- Epoch-by-epoch analysis
- Training diagnostics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import utilities with fallback  
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from utils.plot_utils import (
        create_figure_and_axes, format_axis, save_figure, tensor_to_numpy,
        COLORS, PlotManager, apply_tight_layout
    )
except ImportError:
    # Fallback with relative import
    from ..utils.plot_utils import (
        create_figure_and_axes, format_axis, save_figure, tensor_to_numpy,
        COLORS, PlotManager, apply_tight_layout
    )


def plot_training_curves(train_losses: List[float],
                        val_losses: Optional[List[float]] = None,
                        epochs: Optional[List[int]] = None,
                        title: str = "Training Loss Curves",
                        save_path: Optional[str] = None,
                        log_scale: bool = False,
                        show_best: bool = True,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        epochs: Optional list of epoch numbers (auto-generated if None)
        title: Plot title
        save_path: Path to save the figure
        log_scale: Whether to use log scale for y-axis
        show_best: Whether to mark the best loss point
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Generate epoch numbers if not provided
    if epochs is None:
        epochs = list(range(1, len(train_losses) + 1))
    
    # Create figure
    fig, ax = create_figure_and_axes(figsize=figsize)
    ax = ax[0] if isinstance(ax, np.ndarray) else ax
    
    # Plot training loss
    ax.plot(epochs, train_losses, 
           color=COLORS['primary'], linewidth=2.5, 
           label='Training Loss', marker='o', markersize=3, alpha=0.8)
    
    # Plot validation loss if available
    if val_losses is not None:
        ax.plot(epochs, val_losses,
               color=COLORS['secondary'], linewidth=2.5,
               label='Validation Loss', marker='s', markersize=3, alpha=0.8)
    
    # Mark best points if requested
    if show_best:
        best_train_idx = np.argmin(train_losses)
        ax.scatter(epochs[best_train_idx], train_losses[best_train_idx],
                  color=COLORS['accent'], s=100, marker='*', 
                  label=f'Best Train Loss: {train_losses[best_train_idx]:.6f}',
                  zorder=5)
        
        if val_losses is not None:
            best_val_idx = np.argmin(val_losses)
            ax.scatter(epochs[best_val_idx], val_losses[best_val_idx],
                      color=COLORS['success'], s=100, marker='*',
                      label=f'Best Val Loss: {val_losses[best_val_idx]:.6f}',
                      zorder=5)
    
    # Configure axes
    if log_scale:
        ax.set_yscale('log')
        ylabel = 'Loss (log scale)'
    else:
        ylabel = 'Loss'
    
    ax = format_axis(ax, title=title, xlabel='Epoch', ylabel=ylabel, 
                    grid=True, legend=True)
    
    # Add loss statistics as text
    stats_text = f"Final Train Loss: {train_losses[-1]:.6f}\n"
    stats_text += f"Min Train Loss: {min(train_losses):.6f}"
    if val_losses is not None:
        stats_text += f"\nFinal Val Loss: {val_losses[-1]:.6f}\n"
        stats_text += f"Min Val Loss: {min(val_losses):.6f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_loss_convergence(losses: List[float],
                         convergence_threshold: float = 1e-6,
                         window_size: int = 10,
                         title: str = "Loss Convergence Analysis",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot loss convergence with moving average and convergence detection.
    
    Args:
        losses: List of loss values
        convergence_threshold: Threshold for detecting convergence
        window_size: Window size for moving average
        title: Plot title  
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    epochs = list(range(1, len(losses) + 1))
    
    # Calculate moving average
    moving_avg = []
    for i in range(len(losses)):
        start_idx = max(0, i - window_size + 1)
        avg = np.mean(losses[start_idx:i+1])
        moving_avg.append(avg)
    
    # Calculate loss differences (convergence indicator)
    loss_diffs = []
    for i in range(1, len(losses)):
        diff = abs(losses[i] - losses[i-1])
        loss_diffs.append(diff)
    
    # Create subplots
    fig, axes = create_figure_and_axes(figsize=figsize, nrows=2, ncols=1)
    
    # Top plot: Loss values and moving average
    ax1 = axes[0]
    ax1.plot(epochs, losses, color=COLORS['primary'], alpha=0.6, 
            linewidth=1, label='Raw Loss')
    ax1.plot(epochs, moving_avg, color=COLORS['secondary'], linewidth=2,
            label=f'Moving Average (window={window_size})')
    
    # Mark convergence point if applicable
    converged_epoch = None
    for i, diff in enumerate(loss_diffs):
        if diff < convergence_threshold:
            converged_epoch = i + 2  # +2 because diff starts from epoch 2
            break
    
    if converged_epoch:
        ax1.axvline(x=converged_epoch, color=COLORS['accent'], linestyle='--',
                   linewidth=2, label=f'Convergence at epoch {converged_epoch}')
    
    ax1 = format_axis(ax1, title="Loss Values and Moving Average",
                     xlabel="", ylabel="Loss", grid=True, legend=True)
    
    # Bottom plot: Loss differences (convergence indicator)
    ax2 = axes[1]
    diff_epochs = list(range(2, len(losses) + 1))
    ax2.plot(diff_epochs, loss_diffs, color=COLORS['success'], linewidth=1.5,
            label='Loss Difference (|L_t - L_{t-1}|)')
    ax2.axhline(y=convergence_threshold, color=COLORS['accent'], 
               linestyle='--', alpha=0.8, 
               label=f'Convergence Threshold ({convergence_threshold:.0e})')
    
    if converged_epoch:
        ax2.axvline(x=converged_epoch, color=COLORS['accent'], linestyle='--',
                   linewidth=2, alpha=0.8)
    
    ax2.set_yscale('log')
    ax2 = format_axis(ax2, title="Loss Convergence Indicator",
                     xlabel="Epoch", ylabel="Loss Difference (log scale)",
                     grid=True, legend=True)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Add convergence statistics
    stats_text = f"Total Epochs: {len(losses)}\n"
    stats_text += f"Final Loss: {losses[-1]:.6f}\n"
    stats_text += f"Min Loss: {min(losses):.6f}\n"
    if converged_epoch:
        stats_text += f"Converged: Epoch {converged_epoch}"
    else:
        stats_text += "Not Converged"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_training_progress(metrics_data: Dict[str, Any],
                          title: str = "Training Progress Overview",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot comprehensive training progress overview.
    
    Args:
        metrics_data: Dictionary containing training metrics
                     (should have 'epochs', 'train_losses', etc.)
        title: Plot title
        save_path: Path to save the figure  
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Extract data
    epochs = metrics_data.get('epochs', [])
    train_losses = metrics_data.get('train_losses', [])
    val_losses = metrics_data.get('val_losses', [])
    learning_rates = metrics_data.get('learning_rates', [])
    convergence_metrics = metrics_data.get('convergence_metrics', [])
    
    # Create 2x2 subplot layout
    fig, axes = create_figure_and_axes(figsize=figsize, nrows=2, ncols=2)
    
    # Plot 1: Training curves
    ax1 = axes[0, 0]
    if train_losses:
        ax1.plot(epochs, train_losses, color=COLORS['primary'], 
                linewidth=2, label='Training Loss', marker='o', markersize=2)
    if val_losses:
        ax1.plot(epochs, val_losses, color=COLORS['secondary'],
                linewidth=2, label='Validation Loss', marker='s', markersize=2)
    ax1 = format_axis(ax1, title="Loss Curves", xlabel="Epoch", ylabel="Loss",
                     grid=True, legend=True)
    
    # Plot 2: Loss distribution/histogram
    ax2 = axes[0, 1]
    if train_losses:
        ax2.hist(train_losses, bins=30, alpha=0.7, color=COLORS['primary'],
                edgecolor='black', label='Train Loss Distribution')
        ax2.axvline(np.mean(train_losses), color=COLORS['accent'], 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(train_losses):.6f}')
    ax2 = format_axis(ax2, title="Loss Distribution", xlabel="Loss Value", 
                     ylabel="Frequency", grid=True, legend=True)
    
    # Plot 3: Learning rate schedule (if available)
    ax3 = axes[1, 0]
    if learning_rates and len(learning_rates) > 1:
        ax3.plot(epochs, learning_rates, color=COLORS['success'],
                linewidth=2, label='Learning Rate')
        ax3.set_yscale('log')
        ylabel = 'Learning Rate (log scale)'
    else:
        # If no learning rate data, show loss improvement
        if len(train_losses) > 1:
            loss_improvements = [0]  # First epoch has no improvement
            for i in range(1, len(train_losses)):
                improvement = train_losses[i-1] - train_losses[i]
                loss_improvements.append(improvement)
            ax3.plot(epochs, loss_improvements, color=COLORS['success'],
                    linewidth=2, label='Loss Improvement per Epoch')
            ylabel = 'Loss Improvement'
    ax3 = format_axis(ax3, title="Learning Rate / Improvement", xlabel="Epoch", 
                     ylabel=ylabel, grid=True, legend=True)
    
    # Plot 4: Convergence metrics (if available)
    ax4 = axes[1, 1]
    if convergence_metrics:
        ax4.plot(epochs, convergence_metrics, color=COLORS['accent'],
                linewidth=2, label='Convergence Metric')
        ax4.set_yscale('log')
        ylabel = 'Convergence Metric (log scale)'
        title_4 = "Convergence Analysis"
    else:
        # Show training time per epoch or cumulative loss reduction
        if len(train_losses) > 0:
            cumulative_reduction = []
            initial_loss = train_losses[0] if train_losses else 1.0
            for loss in train_losses:
                reduction = (initial_loss - loss) / initial_loss * 100
                cumulative_reduction.append(max(0, reduction))
            ax4.plot(epochs, cumulative_reduction, color=COLORS['accent'],
                    linewidth=2, label='Loss Reduction (%)')
            ylabel = 'Loss Reduction (%)'
            title_4 = "Cumulative Loss Reduction"
    ax4 = format_axis(ax4, title=title_4, xlabel="Epoch", ylabel=ylabel,
                     grid=True, legend=True)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Add summary statistics
    if 'summary' in metrics_data:
        summary = metrics_data['summary']
        stats_text = f"Total Epochs: {summary.get('total_epochs', 'N/A')}\n"
        stats_text += f"Best Loss: {summary.get('best_loss', 'N/A'):.6f}\n"
        stats_text += f"Training Time: {summary.get('training_time', 'N/A'):.1f}s"
        
        # Add text box to one of the plots
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_learning_rate_schedule(learning_rates: List[float],
                               epochs: Optional[List[int]] = None,
                               title: str = "Learning Rate Schedule",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot learning rate schedule over training epochs.
    
    Args:
        learning_rates: List of learning rates per epoch
        epochs: Optional list of epoch numbers
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    if epochs is None:
        epochs = list(range(1, len(learning_rates) + 1))
    
    # Create figure
    fig, ax = create_figure_and_axes(figsize=figsize)
    ax = ax[0] if isinstance(ax, np.ndarray) else ax
    
    # Plot learning rate
    ax.plot(epochs, learning_rates, color=COLORS['success'], linewidth=2.5,
           marker='o', markersize=3, label='Learning Rate')
    
    # Add horizontal lines for key values
    if learning_rates:
        initial_lr = learning_rates[0]
        final_lr = learning_rates[-1]
        min_lr = min(learning_rates)
        max_lr = max(learning_rates)
        
        ax.axhline(y=initial_lr, color=COLORS['neutral'], linestyle='--', 
                  alpha=0.6, label=f'Initial LR: {initial_lr:.6f}')
        ax.axhline(y=final_lr, color=COLORS['accent'], linestyle='--',
                  alpha=0.6, label=f'Final LR: {final_lr:.6f}')
    
    # Use log scale if range is large
    lr_range = max_lr / min_lr if min_lr > 0 else 1
    if lr_range > 100:  # Use log scale if range spans more than 2 orders of magnitude
        ax.set_yscale('log')
        ylabel = 'Learning Rate (log scale)'
    else:
        ylabel = 'Learning Rate'
    
    # Configure axis
    ax = format_axis(ax, title=title, xlabel='Epoch', ylabel=ylabel,
                    grid=True, legend=True)
    
    # Add statistics
    stats_text = f"Initial: {initial_lr:.6f}\n"
    stats_text += f"Final: {final_lr:.6f}\n"
    stats_text += f"Min: {min_lr:.6f}\n"
    stats_text += f"Max: {max_lr:.6f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def load_and_plot_training_metrics(metrics_file: str,
                                  plot_type: str = "all",
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Load training metrics from JSON file and create plots.
    
    Args:
        metrics_file: Path to JSON file containing training metrics
        plot_type: Type of plot to create ('curves', 'convergence', 'progress', 'all')
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    try:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        if plot_type == "curves":
            return plot_training_curves(
                train_losses=metrics_data.get('train_losses', []),
                val_losses=metrics_data.get('val_losses', []),
                epochs=metrics_data.get('epochs', []),
                save_path=save_path
            )
        elif plot_type == "convergence":
            return plot_loss_convergence(
                losses=metrics_data.get('train_losses', []),
                save_path=save_path
            )
        elif plot_type == "progress":
            return plot_training_progress(
                metrics_data=metrics_data,
                save_path=save_path
            )
        else:  # plot_type == "all"
            return plot_training_progress(
                metrics_data=metrics_data,
                title="Complete Training Analysis",
                save_path=save_path
            )
    
    except Exception as e:
        print(f"Error loading metrics file {metrics_file}: {e}")
        # Return error plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error loading {metrics_file}\n{str(e)}", 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle="round", facecolor='lightcoral'))
        ax.set_title("Error Loading Training Metrics", fontweight='bold')
        return fig


def plot_batch_training_progress(batch_losses: List[List[float]],
                               epoch_labels: Optional[List[int]] = None,
                               title: str = "Batch-wise Training Progress",
                               save_path: Optional[str] = None,
                               max_epochs_to_show: int = 10,
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot batch-wise training progress showing loss within epochs.
    
    Args:
        batch_losses: List of lists, where each inner list contains batch losses for an epoch
        epoch_labels: Optional list of epoch numbers
        title: Plot title
        save_path: Path to save the figure
        max_epochs_to_show: Maximum number of epochs to display
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    if not batch_losses:
        # Return empty plot if no data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No batch loss data available", 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Limit number of epochs to display
    n_epochs = min(len(batch_losses), max_epochs_to_show)
    batch_losses = batch_losses[:n_epochs]
    
    if epoch_labels is None:
        epoch_labels = list(range(1, n_epochs + 1))
    else:
        epoch_labels = epoch_labels[:n_epochs]
    
    # Create figure
    fig, ax = create_figure_and_axes(figsize=figsize)
    ax = ax[0] if isinstance(ax, np.ndarray) else ax
    
    # Create color map for epochs
    colors = plt.cm.viridis(np.linspace(0, 1, n_epochs))
    
    # Plot each epoch's batch losses
    for i, (epoch_losses, epoch_num) in enumerate(zip(batch_losses, epoch_labels)):
        batch_nums = list(range(1, len(epoch_losses) + 1))
        ax.plot(batch_nums, epoch_losses, color=colors[i], 
               linewidth=1.5, alpha=0.8, label=f'Epoch {epoch_num}')
    
    # Configure axis
    ax = format_axis(ax, title=title, xlabel='Batch Number', ylabel='Loss',
                    grid=True, legend=True)
    
    # Adjust legend if too many epochs
    if n_epochs > 8:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add summary statistics
    final_epoch_losses = batch_losses[-1] if batch_losses else []
    if final_epoch_losses:
        final_epoch_mean = np.mean(final_epoch_losses)
        final_epoch_std = np.std(final_epoch_losses)
        stats_text = f"Final Epoch Stats:\n"
        stats_text += f"Mean Loss: {final_epoch_mean:.6f}\n"
        stats_text += f"Std Loss: {final_epoch_std:.6f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig
