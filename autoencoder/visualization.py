"""
Comprehensive visualization system for autoencoder models.

This module provides publication-ready visualizations including image comparison grids,
training curves, latent space plots, and reconstruction error analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os

from .plot_utils import (
    setup_publication_style, 
    save_plot_multiple_formats,
    validate_image_data,
    get_color_palette,
    create_subplot_grid,
    format_plot_labels,
    handle_plotting_errors
)


__all__ = [
    'plot_image_comparison',
    'plot_training_curves', 
    'plot_latent_space',
    'plot_reconstruction_errors',
    'plot_error_heatmap',
    'create_comprehensive_report',
    'VisualizationManager'
]


def plot_image_comparison(originals: np.ndarray, 
                         reconstructions: np.ndarray,
                         labels: Optional[np.ndarray] = None,
                         n_samples: int = 25,
                         grid_size: Optional[Tuple[int, int]] = None,
                         title: str = "Original vs Reconstructed Images",
                         save_path: Optional[str] = None,
                         save_formats: List[str] = ['png'],
                         dpi: int = 300,
                         show: bool = True,
                         figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Create side-by-side comparison of original and reconstructed images.
    
    Args:
        originals (np.ndarray): Original images, shape (n_samples, height, width) or (n_samples, pixels)
        reconstructions (np.ndarray): Reconstructed images, same shape as originals
        labels (Optional[np.ndarray]): Class labels for images
        n_samples (int): Number of samples to display (default: 25)
        grid_size (Optional[Tuple[int, int]]): Grid dimensions (rows, cols). Auto-calculated if None
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot (without extension)
        save_formats (List[str]): Formats to save ['png', 'pdf', 'svg']
        dpi (int): Resolution for saved plots
        show (bool): Whether to display the plot
        figsize (Optional[Tuple[float, float]]): Figure size, auto-calculated if None
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Validate and preprocess inputs
    originals, reconstructions = validate_image_data(originals, reconstructions)
    n_samples = min(n_samples, len(originals))
    
    # Calculate grid dimensions
    if grid_size is None:
        grid_rows = int(np.sqrt(n_samples))
        grid_cols = int(np.ceil(n_samples / grid_rows))
    else:
        grid_rows, grid_cols = grid_size
        n_samples = min(n_samples, grid_rows * grid_cols)
    
    # Calculate figure size
    if figsize is None:
        figsize = (grid_cols * 4, grid_rows * 2)
    
    # Setup publication style
    with setup_publication_style():
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(grid_rows, grid_cols * 2, hspace=0.3, wspace=0.1)
        
        for i in range(n_samples):
            if i >= grid_rows * grid_cols:
                break
                
            row = i // grid_cols
            col = i % grid_cols
            
            # Original image
            ax_orig = fig.add_subplot(gs[row, col * 2])
            img_orig = originals[i].reshape(28, 28) if originals[i].ndim == 1 else originals[i]
            ax_orig.imshow(img_orig, cmap='Greys_r', interpolation='nearest')
            ax_orig.axis('off')
            
            # Add label if available
            if labels is not None:
                ax_orig.set_title(f'Original\nLabel: {labels[i]}', fontsize=8, pad=2)
            else:
                ax_orig.set_title('Original', fontsize=8, pad=2)
            
            # Reconstructed image
            ax_recon = fig.add_subplot(gs[row, col * 2 + 1])
            img_recon = reconstructions[i].reshape(28, 28) if reconstructions[i].ndim == 1 else reconstructions[i]
            ax_recon.imshow(img_recon, cmap='Greys_r', interpolation='nearest')
            ax_recon.axis('off')
            ax_recon.set_title('Reconstructed', fontsize=8, pad=2)
            
            # Add reconstruction error text
            mse_error = np.mean((img_orig - img_recon) ** 2)
            ax_recon.text(0.5, -0.15, f'MSE: {mse_error:.4f}', 
                         transform=ax_recon.transAxes, ha='center', fontsize=6)
        
        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Handle saving and display
        if save_path:
            save_plot_multiple_formats(fig, save_path, save_formats, dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        epochs: Optional[List[int]] = None,
                        learning_rates: Optional[List[float]] = None,
                        title: str = "Training Progress",
                        save_path: Optional[str] = None,
                        save_formats: List[str] = ['png'],
                        dpi: int = 300,
                        show: bool = True,
                        figsize: Tuple[float, float] = (12, 5)) -> plt.Figure:
    """
    Plot training and validation loss curves with optional learning rate schedule.
    
    Args:
        train_losses (List[float]): Training loss values per epoch
        val_losses (List[float]): Validation loss values per epoch  
        epochs (Optional[List[int]]): Epoch numbers, auto-generated if None
        learning_rates (Optional[List[float]]): Learning rate values per epoch
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot (without extension)
        save_formats (List[str]): Formats to save ['png', 'pdf', 'svg']
        dpi (int): Resolution for saved plots
        show (bool): Whether to display the plot
        figsize (Tuple[float, float]): Figure size
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    if len(train_losses) != len(val_losses):
        raise ValueError("Training and validation losses must have the same length")
    
    if epochs is None:
        epochs = list(range(1, len(train_losses) + 1))
    
    with setup_publication_style():
        # Determine subplot configuration
        n_subplots = 2 if learning_rates is not None else 1
        fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
        if n_subplots == 1:
            axes = [axes]
        
        # Loss curves
        ax_loss = axes[0]
        colors = get_color_palette()
        
        ax_loss.plot(epochs, train_losses, color=colors[0], linewidth=2, 
                    label='Training Loss', marker='o', markersize=3)
        ax_loss.plot(epochs, val_losses, color=colors[1], linewidth=2, 
                    label='Validation Loss', marker='s', markersize=3)
        
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Loss (MSE)', fontsize=12)
        ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax_loss.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax_loss.grid(True, alpha=0.3)
        
        # Add best validation loss annotation
        best_val_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax_loss.annotate(f'Best: Epoch {best_val_epoch}\nLoss: {best_val_loss:.6f}',
                        xy=(best_val_epoch, best_val_loss), 
                        xytext=(best_val_epoch + len(epochs) * 0.1, best_val_loss + max(val_losses) * 0.05),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                        fontsize=10)
        
        # Learning rate schedule (if provided)
        if learning_rates is not None and len(learning_rates) == len(epochs):
            ax_lr = axes[1]
            ax_lr.plot(epochs, learning_rates, color=colors[2], linewidth=2, 
                      marker='d', markersize=3, label='Learning Rate')
            ax_lr.set_xlabel('Epoch', fontsize=12)
            ax_lr.set_ylabel('Learning Rate', fontsize=12)
            ax_lr.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax_lr.set_yscale('log')
            ax_lr.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax_lr.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Handle saving and display
        if save_path:
            save_plot_multiple_formats(fig, save_path, save_formats, dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


def plot_latent_space(latent_representations: np.ndarray,
                     labels: Optional[np.ndarray] = None,
                     method: str = 'pca',
                     n_components: int = 2,
                     title: str = "Latent Space Visualization", 
                     save_path: Optional[str] = None,
                     save_formats: List[str] = ['png'],
                     dpi: int = 300,
                     show: bool = True,
                     figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
    """
    Visualize latent space using dimensionality reduction techniques.
    
    Args:
        latent_representations (np.ndarray): Latent vectors, shape (n_samples, latent_dim)
        labels (Optional[np.ndarray]): Class labels for coloring points
        method (str): Dimensionality reduction method ('pca', 'tsne')
        n_components (int): Number of components for visualization (2 or 3)
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot (without extension)
        save_formats (List[str]): Formats to save ['png', 'pdf', 'svg']
        dpi (int): Resolution for saved plots
        show (bool): Whether to display the plot
        figsize (Tuple[float, float]): Figure size
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    if latent_representations.shape[1] < n_components:
        raise ValueError(f"Latent dimension {latent_representations.shape[1]} is smaller than n_components {n_components}")
    
    with setup_publication_style():
        # Apply dimensionality reduction if needed
        if latent_representations.shape[1] == n_components:
            # Already the right dimensionality
            reduced_data = latent_representations
            reduction_info = f"Original {n_components}D"
        else:
            if method.lower() == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
                reduced_data = reducer.fit_transform(latent_representations)
                variance_explained = reducer.explained_variance_ratio_
                reduction_info = f"PCA ({variance_explained.sum():.2%} variance)"
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=n_components, random_state=42, 
                              perplexity=min(30, latent_representations.shape[0] - 1))
                reduced_data = reducer.fit_transform(latent_representations)
                reduction_info = "t-SNE"
            else:
                raise ValueError(f"Unknown reduction method: {method}")
        
        # Create figure
        if n_components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if n_components == 3:
                    ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2],
                             c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)
                else:
                    ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                             c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        else:
            if n_components == 3:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                          alpha=0.7, s=30, c=get_color_palette()[0])
            else:
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                          alpha=0.7, s=30, c=get_color_palette()[0])
        
        # Format labels
        if n_components == 3:
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12) 
            ax.set_zlabel('Component 3', fontsize=12)
        else:
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
        
        ax.set_title(f'{title}\n({reduction_info})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Handle saving and display
        if save_path:
            save_plot_multiple_formats(fig, save_path, save_formats, dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


def plot_reconstruction_errors(errors: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              error_type: str = 'mse',
                              title: str = "Reconstruction Error Analysis",
                              save_path: Optional[str] = None,
                              save_formats: List[str] = ['png'],
                              dpi: int = 300,
                              show: bool = True,
                              figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Visualize reconstruction errors with distribution and per-class analysis.
    
    Args:
        errors (np.ndarray): Reconstruction errors per sample
        labels (Optional[np.ndarray]): Class labels for per-class analysis
        error_type (str): Type of error metric ('mse', 'mae', 'rmse')
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot (without extension)
        save_formats (List[str]): Formats to save ['png', 'pdf', 'svg']
        dpi (int): Resolution for saved plots
        show (bool): Whether to display the plot
        figsize (Tuple[float, float]): Figure size
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    with setup_publication_style():
        if labels is not None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]/2))
        
        colors = get_color_palette()
        
        # Overall error distribution
        ax_hist = axes[0]
        ax_hist.hist(errors, bins=50, alpha=0.7, color=colors[0], edgecolor='black', linewidth=0.5)
        ax_hist.axvline(np.mean(errors), color=colors[1], linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax_hist.axvline(np.median(errors), color=colors[2], linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
        ax_hist.set_xlabel(f'{error_type.upper()} Error', fontsize=12)
        ax_hist.set_ylabel('Frequency', fontsize=12)
        ax_hist.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Error statistics box plot
        ax_box = axes[1] 
        ax_box.boxplot(errors, patch_artist=True, boxprops=dict(facecolor=colors[0], alpha=0.7))
        ax_box.set_ylabel(f'{error_type.upper()} Error', fontsize=12)
        ax_box.set_title('Error Statistics', fontsize=14, fontweight='bold')
        ax_box.grid(True, alpha=0.3)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            
            # Per-class error distribution
            ax_class_hist = axes[2]
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax_class_hist.hist(errors[mask], bins=20, alpha=0.6, 
                                 label=f'Class {label}', color=plt.cm.tab10(i))
            ax_class_hist.set_xlabel(f'{error_type.upper()} Error', fontsize=12)
            ax_class_hist.set_ylabel('Frequency', fontsize=12)
            ax_class_hist.set_title('Error Distribution by Class', fontsize=14, fontweight='bold')
            ax_class_hist.legend()
            ax_class_hist.grid(True, alpha=0.3)
            
            # Per-class error statistics
            ax_class_stats = axes[3]
            class_errors = [errors[labels == label] for label in unique_labels]
            bp = ax_class_stats.boxplot(class_errors, patch_artist=True, labels=unique_labels)
            for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax_class_stats.set_xlabel('Class', fontsize=12)
            ax_class_stats.set_ylabel(f'{error_type.upper()} Error', fontsize=12)
            ax_class_stats.set_title('Error Statistics by Class', fontsize=14, fontweight='bold')
            ax_class_stats.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Handle saving and display
        if save_path:
            save_plot_multiple_formats(fig, save_path, save_formats, dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


def plot_error_heatmap(originals: np.ndarray,
                      reconstructions: np.ndarray,
                      labels: Optional[np.ndarray] = None,
                      n_samples: int = 16,
                      title: str = "Reconstruction Error Heatmap",
                      save_path: Optional[str] = None,
                      save_formats: List[str] = ['png'],
                      dpi: int = 300,
                      show: bool = True,
                      figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
    """
    Create heatmap visualization of pixel-wise reconstruction errors.
    
    Args:
        originals (np.ndarray): Original images
        reconstructions (np.ndarray): Reconstructed images
        labels (Optional[np.ndarray]): Class labels
        n_samples (int): Number of samples to display
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot (without extension)
        save_formats (List[str]): Formats to save ['png', 'pdf', 'svg']
        dpi (int): Resolution for saved plots
        show (bool): Whether to display the plot
        figsize (Optional[Tuple[float, float]]): Figure size
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Validate and preprocess inputs
    originals, reconstructions = validate_image_data(originals, reconstructions)
    n_samples = min(n_samples, len(originals))
    
    # Calculate grid dimensions
    grid_size = int(np.sqrt(n_samples))
    n_samples = grid_size * grid_size
    
    if figsize is None:
        figsize = (grid_size * 3, grid_size * 3)
    
    with setup_publication_style():
        fig, axes = plt.subplots(grid_size, grid_size * 3, figsize=figsize)
        if grid_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // grid_size
            col = i % grid_size
            
            # Reshape images
            orig_img = originals[i].reshape(28, 28) if originals[i].ndim == 1 else originals[i]
            recon_img = reconstructions[i].reshape(28, 28) if reconstructions[i].ndim == 1 else reconstructions[i]
            error_img = np.abs(orig_img - recon_img)
            
            # Original
            ax_orig = axes[row, col * 3]
            ax_orig.imshow(orig_img, cmap='Greys_r', interpolation='nearest')
            ax_orig.axis('off')
            if i < grid_size:  # Only add title to top row
                ax_orig.set_title('Original', fontsize=10, pad=2)
            if labels is not None:
                ax_orig.text(0.5, -0.1, f'Label: {labels[i]}', 
                           transform=ax_orig.transAxes, ha='center', fontsize=8)
            
            # Reconstruction
            ax_recon = axes[row, col * 3 + 1] 
            ax_recon.imshow(recon_img, cmap='Greys_r', interpolation='nearest')
            ax_recon.axis('off')
            if i < grid_size:
                ax_recon.set_title('Reconstructed', fontsize=10, pad=2)
            
            # Error heatmap
            ax_error = axes[row, col * 3 + 2]
            im = ax_error.imshow(error_img, cmap='Reds', interpolation='nearest')
            ax_error.axis('off')
            if i < grid_size:
                ax_error.set_title('Error', fontsize=10, pad=2)
            
            # Add colorbar for the first error plot only
            if i == 0:
                cbar = plt.colorbar(im, ax=ax_error, shrink=0.8)
                cbar.set_label('Absolute Error', fontsize=8)
        
        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Handle saving and display
        if save_path:
            save_plot_multiple_formats(fig, save_path, save_formats, dpi)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


class VisualizationManager:
    """
    Comprehensive manager for all autoencoder visualizations.
    
    This class provides a unified interface for creating, managing, and 
    organizing all visualization outputs.
    """
    
    def __init__(self, output_dir: str = "./visualizations", 
                 default_formats: List[str] = ['png', 'pdf'],
                 default_dpi: int = 300):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir (str): Directory for saving visualizations
            default_formats (List[str]): Default export formats
            default_dpi (int): Default DPI for saved plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_formats = default_formats
        self.default_dpi = default_dpi
        self.plot_registry = {}
    
    def create_comprehensive_report(self, 
                                   originals: np.ndarray,
                                   reconstructions: np.ndarray,
                                   latent_representations: np.ndarray,
                                   train_losses: List[float],
                                   val_losses: List[float],
                                   labels: Optional[np.ndarray] = None,
                                   epochs: Optional[List[int]] = None,
                                   learning_rates: Optional[List[float]] = None,
                                   report_name: str = "autoencoder_report",
                                   show_plots: bool = False) -> Dict[str, str]:
        """
        Create a comprehensive visualization report with all plot types.
        
        Args:
            originals (np.ndarray): Original images
            reconstructions (np.ndarray): Reconstructed images
            latent_representations (np.ndarray): Latent space representations
            train_losses (List[float]): Training losses
            val_losses (List[float]): Validation losses
            labels (Optional[np.ndarray]): Class labels
            epochs (Optional[List[int]]): Epoch numbers
            learning_rates (Optional[List[float]]): Learning rates
            report_name (str): Base name for the report files
            show_plots (bool): Whether to display plots
            
        Returns:
            Dict[str, str]: Mapping of plot types to saved file paths
        """
        saved_plots = {}
        
        # Image comparison
        img_path = self.output_dir / f"{report_name}_image_comparison"
        plot_image_comparison(
            originals, reconstructions, labels,
            save_path=str(img_path), save_formats=self.default_formats,
            dpi=self.default_dpi, show=show_plots
        )
        saved_plots['image_comparison'] = str(img_path)
        
        # Training curves
        curves_path = self.output_dir / f"{report_name}_training_curves"
        plot_training_curves(
            train_losses, val_losses, epochs, learning_rates,
            save_path=str(curves_path), save_formats=self.default_formats,
            dpi=self.default_dpi, show=show_plots
        )
        saved_plots['training_curves'] = str(curves_path)
        
        # Latent space visualization
        latent_path = self.output_dir / f"{report_name}_latent_space"
        plot_latent_space(
            latent_representations, labels,
            save_path=str(latent_path), save_formats=self.default_formats,
            dpi=self.default_dpi, show=show_plots
        )
        saved_plots['latent_space'] = str(latent_path)
        
        # Reconstruction errors
        errors = np.mean((originals - reconstructions) ** 2, axis=1)
        error_path = self.output_dir / f"{report_name}_reconstruction_errors"
        plot_reconstruction_errors(
            errors, labels,
            save_path=str(error_path), save_formats=self.default_formats,
            dpi=self.default_dpi, show=show_plots
        )
        saved_plots['reconstruction_errors'] = str(error_path)
        
        # Error heatmap
        heatmap_path = self.output_dir / f"{report_name}_error_heatmap"
        plot_error_heatmap(
            originals, reconstructions, labels,
            save_path=str(heatmap_path), save_formats=self.default_formats,
            dpi=self.default_dpi, show=show_plots
        )
        saved_plots['error_heatmap'] = str(heatmap_path)
        
        # Update registry
        self.plot_registry[report_name] = saved_plots
        
        print(f"Comprehensive visualization report saved to: {self.output_dir}")
        print(f"Generated plots: {list(saved_plots.keys())}")
        
        return saved_plots
    
    def get_plot_registry(self) -> Dict[str, Dict[str, str]]:
        """Get the registry of all created plots."""
        return self.plot_registry


def create_comprehensive_report(*args, **kwargs):
    """
    Convenience function to create a comprehensive visualization report.
    
    This function creates a VisualizationManager and generates all plots.
    """
    manager = VisualizationManager()
    return manager.create_comprehensive_report(*args, **kwargs)
