"""
Visualization utilities for MLP Autoencoder evaluation and analysis.

This module provides comprehensive visualization capabilities including:
- Side-by-side original vs reconstructed image comparisons
- Training/validation loss curve plotting with epoch tracking  
- Latent space visualization using PCA and t-SNE projections
- Image grid displays for batch visualization
- Error heatmaps for reconstruction analysis
- Publication-ready plot formatting and styling
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

# Configure matplotlib for publication-ready plots
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)


def setup_publication_style():
    """Configure matplotlib with publication-ready styling."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })


def plot_original_vs_reconstructed(original: torch.Tensor, reconstructed: torch.Tensor,
                                 indices: Optional[List[int]] = None,
                                 num_samples: int = 8,
                                 labels: Optional[torch.Tensor] = None,
                                 metrics: Optional[Dict[str, List[float]]] = None,
                                 figsize: Tuple[int, int] = (15, 8),
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create side-by-side comparison of original and reconstructed images.
    
    Args:
        original: Original images tensor of shape (batch_size, 784)
        reconstructed: Reconstructed images tensor of same shape
        indices: Specific indices to plot (if None, randomly select)
        num_samples: Number of samples to display
        labels: Optional labels for images
        metrics: Optional per-image metrics dictionary
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    batch_size = original.shape[0]
    
    if indices is None:
        indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
    else:
        indices = indices[:num_samples]
    
    # Reshape to 28x28 for MNIST
    img_size = int(math.sqrt(original.shape[1]))
    original_imgs = original[indices].view(-1, img_size, img_size).cpu().numpy()
    reconstructed_imgs = reconstructed[indices].view(-1, img_size, img_size).cpu().numpy()
    
    n_samples = len(indices)
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Original vs Reconstructed Images', fontsize=18, fontweight='bold')
    
    for i, idx in enumerate(indices):
        # Original image
        axes[0, i].imshow(original_imgs[i], cmap='gray', interpolation='nearest')
        axes[0, i].set_title(f'Original' + (f' (Label: {labels[idx]})' if labels is not None else ''))
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(reconstructed_imgs[i], cmap='gray', interpolation='nearest')
        
        # Add metrics to title if provided
        if metrics is not None:
            title = f'Reconstructed\n'
            for metric_name, values in metrics.items():
                if len(values) > idx:
                    title += f'{metric_name.upper()}: {values[idx]:.3f} '
        else:
            title = 'Reconstructed'
        
        axes[1, i].set_title(title)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        epochs: Optional[List[int]] = None,
                        title: str = "Training and Validation Loss",
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation loss curves with epoch tracking.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values
        epochs: Optional list of epoch numbers (default: sequential)
        title: Plot title
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    if epochs is None:
        epochs = list(range(1, len(train_losses) + 1))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add best validation loss annotation
    best_val_idx = np.argmin(val_losses)
    best_val_loss = val_losses[best_val_idx]
    best_epoch = epochs[best_val_idx]
    
    ax.annotate(f'Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}',
                xy=(best_epoch, best_val_loss),
                xytext=(0.7, 0.8),
                textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_latent_space(latent_analysis: Dict[str, Any], 
                     method: str = 'both',
                     figsize: Tuple[int, int] = (15, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize latent space using PCA and/or t-SNE projections.
    
    Args:
        latent_analysis: Dictionary from latent_analysis function
        method: Visualization method ('pca', 'tsne', or 'both')
        figsize: Figure size tuple  
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    pca_proj = latent_analysis['pca_projection']
    tsne_proj = latent_analysis['tsne_projection']
    labels = latent_analysis['labels']
    explained_var = latent_analysis['pca_explained_variance']
    
    if method == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
        projections = [pca_proj, tsne_proj]
        titles = [f'PCA Projection\n(Explained Variance: {explained_var.sum():.1%})', 
                 't-SNE Projection']
    elif method == 'pca':
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax1]
        projections = [pca_proj]
        titles = [f'PCA Projection\n(Explained Variance: {explained_var.sum():.1%})']
    else:  # tsne
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax1]
        projections = [tsne_proj]
        titles = ['t-SNE Projection']
    
    fig.suptitle('Latent Space Visualization', fontsize=18, fontweight='bold')
    
    for ax, proj, title in zip(axes, projections, titles):
        if labels is not None:
            # Color by digit label
            scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', 
                               alpha=0.7, s=20)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Digit Label')
        else:
            ax.scatter(proj[:, 0], proj[:, 1], alpha=0.7, s=20)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_image_grid(images: torch.Tensor, 
                   labels: Optional[torch.Tensor] = None,
                   predictions: Optional[torch.Tensor] = None,
                   grid_size: Optional[Tuple[int, int]] = None,
                   figsize: Tuple[int, int] = (12, 12),
                   title: str = "Image Grid",
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Display multiple images in a grid layout for batch visualization.
    
    Args:
        images: Tensor of images (batch_size, 784) for MNIST
        labels: Optional true labels
        predictions: Optional predicted labels
        grid_size: Grid dimensions (rows, cols). If None, auto-calculate square grid
        figsize: Figure size tuple
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    batch_size = images.shape[0]
    img_size = int(math.sqrt(images.shape[1]))
    images_np = images.view(-1, img_size, img_size).cpu().numpy()
    
    if grid_size is None:
        grid_rows = int(math.sqrt(batch_size))
        grid_cols = math.ceil(batch_size / grid_rows)
    else:
        grid_rows, grid_cols = grid_size
    
    # Limit to grid size
    n_images = min(batch_size, grid_rows * grid_cols)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Flatten axes for easier indexing
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i in range(grid_rows * grid_cols):
        ax = axes[i]
        
        if i < n_images:
            ax.imshow(images_np[i], cmap='gray', interpolation='nearest')
            
            # Create title with labels and predictions
            title_parts = []
            if labels is not None:
                title_parts.append(f'True: {labels[i].item()}')
            if predictions is not None:
                title_parts.append(f'Pred: {predictions[i].item()}')
            
            if title_parts:
                ax.set_title(' | '.join(title_parts))
        else:
            ax.set_visible(False)
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_error_heatmap(original: torch.Tensor, reconstructed: torch.Tensor,
                      indices: Optional[List[int]] = None,
                      num_samples: int = 6,
                      figsize: Tuple[int, int] = (15, 10),
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create error heatmaps showing pixel-wise reconstruction errors.
    
    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor
        indices: Specific indices to analyze
        num_samples: Number of samples to display
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    batch_size = original.shape[0]
    
    if indices is None:
        indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)
    else:
        indices = indices[:num_samples]
    
    img_size = int(math.sqrt(original.shape[1]))
    original_imgs = original[indices].view(-1, img_size, img_size).cpu().numpy()
    reconstructed_imgs = reconstructed[indices].view(-1, img_size, img_size).cpu().numpy()
    error_maps = np.abs(original_imgs - reconstructed_imgs)
    
    n_samples = len(indices)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, n_samples, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Reconstruction Error Analysis', fontsize=18, fontweight='bold')
    
    for i, idx in enumerate(indices):
        # Original image
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(original_imgs[i], cmap='gray', interpolation='nearest')
        ax_orig.set_title(f'Original {idx}')
        ax_orig.axis('off')
        
        # Reconstructed image
        ax_recon = fig.add_subplot(gs[1, i])
        ax_recon.imshow(reconstructed_imgs[i], cmap='gray', interpolation='nearest')
        ax_recon.set_title('Reconstructed')
        ax_recon.axis('off')
        
        # Error heatmap
        ax_error = fig.add_subplot(gs[2, i])
        im = ax_error.imshow(error_maps[i], cmap='hot', interpolation='nearest')
        ax_error.set_title(f'Error (MSE: {np.mean(error_maps[i]**2):.4f})')
        ax_error.axis('off')
        
        # Add colorbar for error map
        if i == n_samples - 1:  # Only add colorbar to last plot
            cbar = plt.colorbar(im, ax=ax_error, fraction=0.046, pad=0.04)
            cbar.set_label('Absolute Error')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare metrics across different models or conditions.
    
    Args:
        metrics_dict: Dictionary mapping condition names to metrics
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    # Extract metric names and values
    conditions = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
    if len(metric_names) == 1:
        axes = [axes]
    
    fig.suptitle('Metrics Comparison', fontsize=18, fontweight='bold')
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[cond][metric] for cond in conditions]
        
        bars = axes[i].bar(conditions, values, alpha=0.7, color=sns.color_palette("husl", len(conditions)))
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_reconstruction_quality_distribution(metrics_list: List[Dict[str, float]],
                                           figsize: Tuple[int, int] = (15, 5),
                                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of reconstruction quality metrics across samples.
    
    Args:
        metrics_list: List of metric dictionaries for each sample
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    setup_publication_style()
    
    # Extract metrics into arrays
    mse_values = [m['mse'] for m in metrics_list]
    psnr_values = [m['psnr'] for m in metrics_list]
    ssim_values = [m['ssim'] for m in metrics_list]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Reconstruction Quality Distribution', fontsize=18, fontweight='bold')
    
    # MSE distribution
    ax1.hist(mse_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'MSE Distribution\nMean: {np.mean(mse_values):.4f}')
    ax1.grid(True, alpha=0.3)
    
    # PSNR distribution
    ax2.hist(psnr_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('PSNR (dB)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'PSNR Distribution\nMean: {np.mean(psnr_values):.2f} dB')
    ax2.grid(True, alpha=0.3)
    
    # SSIM distribution
    ax3.hist(ssim_values, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('SSIM')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'SSIM Distribution\nMean: {np.mean(ssim_values):.4f}')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def create_evaluation_report(model_results: Dict[str, Any], 
                           latent_analysis: Dict[str, Any],
                           sample_images: Dict[str, torch.Tensor],
                           save_dir: str = './evaluation_report') -> Dict[str, str]:
    """
    Generate a complete evaluation report with all visualizations.
    
    Args:
        model_results: Results from evaluate_model_comprehensive
        latent_analysis: Results from latent_analysis
        sample_images: Dictionary with 'original' and 'reconstructed' tensors
        save_dir: Directory to save all plots
        
    Returns:
        Dictionary mapping plot types to saved file paths
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    saved_files = {}
    
    # 1. Original vs Reconstructed comparison
    if 'original' in sample_images and 'reconstructed' in sample_images:
        fig = plot_original_vs_reconstructed(
            sample_images['original'],
            sample_images['reconstructed'],
            save_path=os.path.join(save_dir, 'original_vs_reconstructed.png')
        )
        plt.close(fig)
        saved_files['comparison'] = os.path.join(save_dir, 'original_vs_reconstructed.png')
    
    # 2. Error heatmaps
    if 'original' in sample_images and 'reconstructed' in sample_images:
        fig = plot_error_heatmap(
            sample_images['original'],
            sample_images['reconstructed'],
            save_path=os.path.join(save_dir, 'error_heatmaps.png')
        )
        plt.close(fig)
        saved_files['error_maps'] = os.path.join(save_dir, 'error_heatmaps.png')
    
    # 3. Latent space visualization
    fig = plot_latent_space(
        latent_analysis,
        save_path=os.path.join(save_dir, 'latent_space.png')
    )
    plt.close(fig)
    saved_files['latent_space'] = os.path.join(save_dir, 'latent_space.png')
    
    # 4. Metrics distribution
    if 'detailed_metrics' in model_results:
        fig = plot_reconstruction_quality_distribution(
            model_results['detailed_metrics'],
            save_path=os.path.join(save_dir, 'metrics_distribution.png')
        )
        plt.close(fig)
        saved_files['metrics_dist'] = os.path.join(save_dir, 'metrics_distribution.png')
    
    return saved_files
