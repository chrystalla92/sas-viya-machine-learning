"""
Main Plotting Functions for MNIST Autoencoder

This module provides core plotting functionality including:
- MNIST digit visualization (migrated from SAS python_plot.sas)
- Original vs reconstructed image comparisons
- Image grid displays
- Publication-ready image visualization

Migrated and enhanced from the original SAS plotting functionality.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings

# Import utilities
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from utils.plot_utils import (
        create_figure_and_axes, format_axis, save_figure, tensor_to_numpy,
        normalize_for_display, format_mnist_image, get_mnist_colormap,
        setup_subplot_layout, get_figure_size_for_grid, apply_tight_layout,
        COLORS, PlotManager
    )
except ImportError:
    # Fallback with relative import
    from ..utils.plot_utils import (
        create_figure_and_axes, format_axis, save_figure, tensor_to_numpy,
        normalize_for_display, format_mnist_image, get_mnist_colormap,
        setup_subplot_layout, get_figure_size_for_grid, apply_tight_layout,
        COLORS, PlotManager
    )


def plot_mnist_grid(images: Union[torch.Tensor, np.ndarray], 
                   labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                   grid_size: Tuple[int, int] = (5, 5),
                   title: str = "MNIST Digits",
                   save_path: Optional[str] = None,
                   show_labels: bool = True,
                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Plot MNIST images in a grid layout.
    
    This function replicates and enhances the original SAS python_plot.sas functionality,
    which displayed MNIST handwriting images in a 5x5 grid with labels.
    
    Args:
        images: MNIST images tensor/array of shape (N, 784) or (N, 28, 28)
        labels: Optional labels for each image
        grid_size: Grid dimensions (rows, cols)
        title: Main title for the plot
        save_path: Path to save the figure (optional)
        show_labels: Whether to show labels on subplots
        figsize: Figure size (auto-calculated if None)
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    images = tensor_to_numpy(images)
    if labels is not None:
        labels = tensor_to_numpy(labels) if hasattr(labels, 'detach') else np.array(labels)
    
    # Calculate number of images to display
    nrows, ncols = grid_size
    n_images = min(len(images), nrows * ncols)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = get_figure_size_for_grid(n_images, max_cols=ncols, item_size=2.5)
    
    # Create figure and axes
    fig, axes = create_figure_and_axes(
        figsize=figsize,
        nrows=nrows, 
        ncols=ncols,
        subplot_kw={'xticks': [], 'yticks': []}
    )
    
    # Set main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Plot images
    for i in range(n_images):
        row = i // ncols
        col = i % ncols
        
        # Handle different axes array structures
        if nrows == 1 and ncols == 1:
            ax = axes[0] if isinstance(axes, np.ndarray) else axes
        elif nrows == 1 or ncols == 1:
            ax = axes[i]
        else:
            ax = axes[row, col]
        
        # Format image for display
        image = format_mnist_image(images[i])
        
        # Display image
        im = ax.imshow(image, cmap=get_mnist_colormap(), interpolation='none')
        
        # Add label if available
        if show_labels and labels is not None:
            label_text = f"Label: {int(labels[i])}" if i < len(labels) else f"Sample {i+1}"
            ax.set_title(label_text, fontsize=10, pad=8)
        elif show_labels:
            ax.set_title(f"Sample {i+1}", fontsize=10, pad=8)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_images, nrows * ncols):
        row = i // ncols
        col = i % ncols
        
        # Handle different axes array structures  
        if nrows == 1 and ncols == 1:
            ax = axes[0] if isinstance(axes, np.ndarray) else axes
        elif nrows == 1 or ncols == 1:
            ax = axes[i]
        else:
            ax = axes[row, col]
        ax.set_visible(False)
    
    # Adjust layout
    apply_tight_layout(fig, pad=2.0)
    
    # Save figure if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_reconstruction_comparison(original_images: Union[torch.Tensor, np.ndarray],
                                 reconstructed_images: Union[torch.Tensor, np.ndarray],
                                 labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                                 n_samples: int = 10,
                                 title: str = "Original vs Reconstructed Images",
                                 save_path: Optional[str] = None,
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Plot side-by-side comparison of original and reconstructed images.
    
    Args:
        original_images: Original MNIST images
        reconstructed_images: Reconstructed MNIST images  
        labels: Optional labels for each image
        n_samples: Number of samples to display
        title: Main title for the plot
        save_path: Path to save the figure
        figsize: Figure size (auto-calculated if None)
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    original_images = tensor_to_numpy(original_images)
    reconstructed_images = tensor_to_numpy(reconstructed_images)
    if labels is not None:
        labels = tensor_to_numpy(labels) if hasattr(labels, 'detach') else np.array(labels)
    
    # Limit to available samples
    n_samples = min(n_samples, len(original_images), len(reconstructed_images))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (n_samples * 2.5, 6)  # 2 rows, n_samples columns
    
    # Create figure with 2 rows (original, reconstructed)
    fig, axes = create_figure_and_axes(
        figsize=figsize,
        nrows=2,
        ncols=n_samples,
        subplot_kw={'xticks': [], 'yticks': []}
    )
    
    # Set main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Plot comparisons
    for i in range(n_samples):
        # Handle axes indexing properly
        if n_samples == 1:
            orig_ax = axes[0]
            recon_ax = axes[1]
        else:
            orig_ax = axes[0, i]
            recon_ax = axes[1, i]
            
        # Original image (top row)
        orig_image = format_mnist_image(original_images[i])
        orig_ax.imshow(orig_image, cmap=get_mnist_colormap(), interpolation='none')
        
        # Set title for first row
        if i == 0:
            orig_ax.set_ylabel('Original', fontsize=12, fontweight='bold', rotation=0, 
                              labelpad=50, ha='right', va='center')
        
        # Add label if available
        if labels is not None and i < len(labels):
            orig_ax.set_title(f"Label: {int(labels[i])}", fontsize=10, pad=8)
        
        # Reconstructed image (bottom row)  
        recon_image = format_mnist_image(reconstructed_images[i])
        recon_ax.imshow(recon_image, cmap=get_mnist_colormap(), interpolation='none')
        
        # Set title for first column of second row
        if i == 0:
            recon_ax.set_ylabel('Reconstructed', fontsize=12, fontweight='bold', 
                               rotation=0, labelpad=50, ha='right', va='center')
        
        # Remove ticks
        orig_ax.set_xticks([])
        orig_ax.set_yticks([])
        recon_ax.set_xticks([])
        recon_ax.set_yticks([])
    
    # Adjust layout
    apply_tight_layout(fig, pad=1.5)
    
    # Save figure if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_reconstruction_grid(original_images: Union[torch.Tensor, np.ndarray],
                           reconstructed_images: Union[torch.Tensor, np.ndarray],
                           reconstruction_errors: Optional[Union[torch.Tensor, np.ndarray]] = None,
                           grid_size: Tuple[int, int] = (4, 5),
                           title: str = "Reconstruction Quality Grid",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a comprehensive grid showing original, reconstructed, and error images.
    
    Args:
        original_images: Original MNIST images
        reconstructed_images: Reconstructed MNIST images
        reconstruction_errors: Optional pixel-wise reconstruction errors
        grid_size: Grid dimensions for samples
        title: Main title for the plot
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    original_images = tensor_to_numpy(original_images)
    reconstructed_images = tensor_to_numpy(reconstructed_images)
    
    nrows, ncols = grid_size
    n_samples = min(len(original_images), nrows * ncols)
    
    # Calculate reconstruction errors if not provided
    if reconstruction_errors is None:
        reconstruction_errors = np.abs(original_images - reconstructed_images)
    else:
        reconstruction_errors = tensor_to_numpy(reconstruction_errors)
    
    # Create figure with 3 columns per sample (original, reconstructed, error)
    figsize = (ncols * 3 * 2.5, nrows * 2.5)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols*3, figsize=figsize,
                           subplot_kw={'xticks': [], 'yticks': []})
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Add column headers
    headers = ['Original', 'Reconstructed', 'Error']
    for col_group in range(ncols):
        for j, header in enumerate(headers):
            col_idx = col_group * 3 + j
            axes[0, col_idx].text(0.5, 1.15, header, ha='center', va='bottom',
                                transform=axes[0, col_idx].transAxes,
                                fontsize=11, fontweight='bold')
    
    # Plot images
    for i in range(n_samples):
        row = i // ncols
        col_group = i % ncols
        
        # Original image
        orig_col = col_group * 3
        orig_ax = axes[row, orig_col]
        orig_image = format_mnist_image(original_images[i])
        orig_ax.imshow(orig_image, cmap=get_mnist_colormap(), interpolation='none')
        orig_ax.set_xticks([])
        orig_ax.set_yticks([])
        
        # Reconstructed image
        recon_col = col_group * 3 + 1
        recon_ax = axes[row, recon_col]
        recon_image = format_mnist_image(reconstructed_images[i])
        recon_ax.imshow(recon_image, cmap=get_mnist_colormap(), interpolation='none')
        recon_ax.set_xticks([])
        recon_ax.set_yticks([])
        
        # Error image
        error_col = col_group * 3 + 2
        error_ax = axes[row, error_col]
        error_image = format_mnist_image(reconstruction_errors[i])
        error_ax.imshow(error_image, cmap='Reds', interpolation='none')
        error_ax.set_xticks([])
        error_ax.set_yticks([])
    
    # Hide unused axes
    total_axes = nrows * ncols * 3
    for i in range(n_samples * 3, total_axes):
        row = i // (ncols * 3)
        col = i % (ncols * 3)
        axes[row, col].set_visible(False)
    
    # Adjust layout
    apply_tight_layout(fig, pad=1.0)
    
    # Save figure if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def save_reconstruction_samples(original_images: Union[torch.Tensor, np.ndarray],
                              reconstructed_images: Union[torch.Tensor, np.ndarray],
                              labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                              output_dir: str = "./reconstruction_samples",
                              n_samples: int = 50,
                              formats: List[str] = ['png']) -> None:
    """
    Save individual reconstruction samples for detailed analysis.
    
    Args:
        original_images: Original MNIST images
        reconstructed_images: Reconstructed MNIST images
        labels: Optional labels for each image
        output_dir: Output directory for saved images
        n_samples: Number of samples to save
        formats: Image formats to save
    """
    # Convert tensors to numpy
    original_images = tensor_to_numpy(original_images)
    reconstructed_images = tensor_to_numpy(reconstructed_images)
    if labels is not None:
        labels = tensor_to_numpy(labels) if hasattr(labels, 'detach') else np.array(labels)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(n_samples, len(original_images))
    
    # Save individual comparison plots
    for i in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        
        # Original
        orig_image = format_mnist_image(original_images[i])
        axes[0].imshow(orig_image, cmap=get_mnist_colormap(), interpolation='none')
        axes[0].set_title('Original', fontweight='bold')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Reconstructed
        recon_image = format_mnist_image(reconstructed_images[i])
        axes[1].imshow(recon_image, cmap=get_mnist_colormap(), interpolation='none')
        axes[1].set_title('Reconstructed', fontweight='bold')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Error
        error_image = np.abs(orig_image - recon_image)
        im = axes[2].imshow(error_image, cmap='Reds', interpolation='none')
        axes[2].set_title('Absolute Error', fontweight='bold')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        # Add colorbar for error
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        # Set main title
        if labels is not None and i < len(labels):
            fig.suptitle(f"Sample {i+1} - Label: {int(labels[i])}", 
                        fontsize=14, fontweight='bold')
        else:
            fig.suptitle(f"Sample {i+1}", fontsize=14, fontweight='bold')
        
        # Save figure
        sample_path = output_path / f"reconstruction_sample_{i+1:03d}"
        save_figure(fig, str(sample_path), formats=formats, close_after_save=True)
    
    print(f"Saved {n_samples} reconstruction samples to {output_dir}")


def load_and_plot_from_csv(csv_filepath: str,
                          grid_size: Tuple[int, int] = (5, 5),
                          title: str = "MNIST from CSV",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Load MNIST data from CSV and plot (replicates original SAS functionality).
    
    This function replicates the original SAS python_plot.sas behavior of loading
    data from "mnist_train_10_autoencoder_score.csv" and plotting it.
    
    Args:
        csv_filepath: Path to CSV file with MNIST data
        grid_size: Grid size for display
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Read CSV file
        # Assuming format: label, pixel1, pixel2, ..., pixel784
        data = pd.read_csv(csv_filepath, header=None)
        
        # Extract labels (first column) and images (remaining columns)
        if data.shape[1] >= 785:  # Label + 784 pixels
            labels = data.iloc[:, 0].values
            images = data.iloc[:, 1:785].values
        else:
            # No labels, just pixel data
            labels = None
            images = data.values
        
        # Plot using the grid function
        fig = plot_mnist_grid(
            images=images,
            labels=labels,
            grid_size=grid_size,
            title=f"{title} (from {Path(csv_filepath).name})",
            save_path=save_path,
            show_labels=(labels is not None)
        )
        
        print(f"Successfully loaded and plotted {len(images)} images from {csv_filepath}")
        return fig
        
    except Exception as e:
        print(f"Error loading CSV file {csv_filepath}: {e}")
        # Return empty figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error loading {csv_filepath}\n{str(e)}", 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle="round", facecolor='lightcoral'))
        ax.set_title("Error Loading Data", fontweight='bold')
        return fig
