"""
Utility functions for plot formatting, styling, and data handling.

This module provides consistent styling, error handling, and utility functions
for the autoencoder visualization system.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import numpy as np
import warnings
from typing import List, Tuple, Optional, Union, Any, Dict
from contextlib import contextmanager
from pathlib import Path
import gc
import psutil
import os

__all__ = [
    'setup_publication_style',
    'save_plot_multiple_formats', 
    'validate_image_data',
    'get_color_palette',
    'create_subplot_grid',
    'format_plot_labels',
    'handle_plotting_errors',
    'memory_efficient_plot',
    'InteractivePlotter'
]


@contextmanager
def setup_publication_style():
    """
    Context manager for setting up publication-ready plot styling.
    
    Configures matplotlib and seaborn for high-quality, consistent plots
    with proper fonts, colors, and layout parameters.
    """
    # Store original style
    original_style = plt.rcParams.copy()
    
    try:
        # Set seaborn style as base
        sns.set_style("whitegrid", {
            "axes.spines.left": True,
            "axes.spines.bottom": True, 
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5
        })
        
        # Configure matplotlib parameters for publication quality
        plt.rcParams.update({
            # Figure settings
            'figure.facecolor': 'white',
            'figure.edgecolor': 'none',
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            
            # Font settings
            'font.size': 11,
            'font.family': 'DejaVu Sans',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            
            # Line and marker settings
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'patch.linewidth': 0.5,
            'patch.facecolor': 'C0',
            'patch.edgecolor': 'black',
            'patch.force_edgecolor': False,
            
            # Axes settings
            'axes.linewidth': 0.8,
            'axes.titlepad': 12,
            'axes.labelpad': 6,
            'axes.prop_cycle': plt.cycler('color', [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf'
            ]),
            
            # Grid settings
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
            
            # Legend settings
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '0.8',
            
            # Tick settings
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.4,
            'ytick.minor.width': 0.4,
        })
        
        yield
        
    finally:
        # Restore original style
        plt.rcParams.update(original_style)


def save_plot_multiple_formats(fig: plt.Figure, 
                              base_path: str,
                              formats: List[str] = ['png', 'pdf', 'svg'],
                              dpi: int = 300,
                              verbose: bool = True) -> List[str]:
    """
    Save a matplotlib figure in multiple formats.
    
    Args:
        fig (plt.Figure): Figure to save
        base_path (str): Base path without extension
        formats (List[str]): List of formats to save ['png', 'pdf', 'svg', 'eps', 'jpg']
        dpi (int): Resolution for raster formats
        verbose (bool): Whether to print save messages
        
    Returns:
        List[str]: List of saved file paths
    """
    saved_paths = []
    
    # Ensure output directory exists
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fmt = fmt.lower().strip('.')
        file_path = f"{base_path}.{fmt}"
        
        try:
            if fmt in ['png', 'jpg', 'jpeg', 'tiff']:
                fig.savefig(file_path, format=fmt, dpi=dpi, 
                           bbox_inches='tight', facecolor='white', edgecolor='none')
            elif fmt in ['pdf', 'eps', 'ps']:
                fig.savefig(file_path, format=fmt, 
                           bbox_inches='tight', facecolor='white', edgecolor='none')
            elif fmt == 'svg':
                fig.savefig(file_path, format=fmt, 
                           bbox_inches='tight', facecolor='white', edgecolor='none')
            else:
                warnings.warn(f"Unknown format: {fmt}, trying anyway...")
                fig.savefig(file_path, format=fmt, dpi=dpi,
                           bbox_inches='tight', facecolor='white', edgecolor='none')
            
            saved_paths.append(file_path)
            if verbose:
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"Saved {fmt.upper()}: {file_path} ({file_size:.1f} KB)")
                
        except Exception as e:
            warnings.warn(f"Failed to save {fmt.upper()} format: {e}")
    
    return saved_paths


def validate_image_data(originals: np.ndarray, 
                       reconstructions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and preprocess image data for visualization.
    
    Args:
        originals (np.ndarray): Original images
        reconstructions (np.ndarray): Reconstructed images
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Validated and preprocessed image arrays
        
    Raises:
        ValueError: If data validation fails
    """
    # Check if arrays are provided
    if originals is None or reconstructions is None:
        raise ValueError("Both originals and reconstructions must be provided")
    
    # Convert to numpy arrays if needed
    if hasattr(originals, 'numpy'):  # PyTorch tensor
        originals = originals.detach().cpu().numpy()
    if hasattr(reconstructions, 'numpy'):  # PyTorch tensor
        reconstructions = reconstructions.detach().cpu().numpy()
    
    # Ensure numpy arrays
    originals = np.asarray(originals)
    reconstructions = np.asarray(reconstructions)
    
    # Check shapes match
    if originals.shape != reconstructions.shape:
        raise ValueError(f"Shape mismatch: originals {originals.shape} vs reconstructions {reconstructions.shape}")
    
    # Check dimensions
    if originals.ndim not in [2, 3, 4]:
        raise ValueError(f"Invalid number of dimensions: {originals.ndim}. Expected 2, 3, or 4.")
    
    # Handle different input formats
    if originals.ndim == 4:  # (batch, channels, height, width) or (batch, height, width, channels)
        if originals.shape[1] == 1:  # (batch, 1, height, width) - grayscale
            originals = originals.squeeze(1)
            reconstructions = reconstructions.squeeze(1)
        elif originals.shape[-1] == 1:  # (batch, height, width, 1) - grayscale
            originals = originals.squeeze(-1)
            reconstructions = reconstructions.squeeze(-1)
        else:
            raise ValueError(f"Unsupported 4D shape: {originals.shape}")
    
    # Ensure values are in valid range [0, 1] or [-1, 1]
    if np.min(originals) < -1.1 or np.max(originals) > 1.1:
        warnings.warn("Image values outside expected range [-1, 1] or [0, 1]. Consider normalizing.")
    
    # Normalize to [0, 1] if in [-1, 1] range
    if np.min(originals) < -0.1:  # Likely in [-1, 1] range
        originals = (originals + 1) / 2
        reconstructions = (reconstructions + 1) / 2
    
    # Clip to valid range
    originals = np.clip(originals, 0, 1)
    reconstructions = np.clip(reconstructions, 0, 1)
    
    return originals, reconstructions


def get_color_palette(n_colors: int = 10, palette_name: str = 'husl') -> List[str]:
    """
    Get a consistent color palette for plotting.
    
    Args:
        n_colors (int): Number of colors to generate
        palette_name (str): Name of seaborn palette or 'custom'
        
    Returns:
        List[str]: List of color hex codes
    """
    if palette_name == 'custom':
        # Custom publication-friendly palette
        custom_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]
        return (custom_colors * ((n_colors // len(custom_colors)) + 1))[:n_colors]
    else:
        try:
            return sns.color_palette(palette_name, n_colors=n_colors).as_hex()
        except Exception:
            warnings.warn(f"Failed to generate {palette_name} palette, using default")
            return sns.color_palette("husl", n_colors=n_colors).as_hex()


def create_subplot_grid(n_plots: int, 
                       max_cols: int = 4,
                       aspect_ratio: float = 1.0) -> Tuple[int, int, Tuple[float, float]]:
    """
    Calculate optimal subplot grid dimensions and figure size.
    
    Args:
        n_plots (int): Number of subplots needed
        max_cols (int): Maximum number of columns
        aspect_ratio (float): Width/height ratio for each subplot
        
    Returns:
        Tuple[int, int, Tuple[float, float]]: (rows, cols, (figwidth, figheight))
    """
    if n_plots <= 0:
        return 1, 1, (8, 6)
    
    # Calculate grid dimensions
    cols = min(n_plots, max_cols)
    rows = int(np.ceil(n_plots / cols))
    
    # Calculate figure size (base size per subplot: 3x3 inches)
    subplot_width = 3
    subplot_height = 3 / aspect_ratio
    
    fig_width = cols * subplot_width
    fig_height = rows * subplot_height
    
    return rows, cols, (fig_width, fig_height)


def format_plot_labels(ax: plt.Axes, 
                      title: str = "",
                      xlabel: str = "",
                      ylabel: str = "",
                      title_fontsize: int = 14,
                      label_fontsize: int = 12,
                      add_grid: bool = True) -> None:
    """
    Apply consistent formatting to plot labels and titles.
    
    Args:
        ax (plt.Axes): Matplotlib axes object
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        title_fontsize (int): Font size for title
        label_fontsize (int): Font size for axis labels
        add_grid (bool): Whether to add grid
    """
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=10)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    
    if add_grid:
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)


def handle_plotting_errors(func):
    """
    Decorator to handle common plotting errors gracefully.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            gc.collect()  # Force garbage collection
            warnings.warn("Memory error encountered. Try reducing batch size or image resolution.")
            return None
        except ValueError as e:
            warnings.warn(f"Value error in plotting: {e}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error in plotting: {e}")
            return None
    
    return wrapper


def memory_efficient_plot(data_size_mb: float) -> Dict[str, Any]:
    """
    Get memory-efficient plotting parameters based on data size.
    
    Args:
        data_size_mb (float): Estimated data size in MB
        
    Returns:
        Dict[str, Any]: Recommended plotting parameters
    """
    # Get available memory
    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    # Calculate memory usage factor
    memory_factor = data_size_mb / available_memory_mb
    
    if memory_factor > 0.5:  # High memory usage
        return {
            'batch_size': 16,
            'dpi': 150,
            'figure_size_factor': 0.7,
            'use_subsampling': True,
            'max_samples': 100
        }
    elif memory_factor > 0.2:  # Medium memory usage
        return {
            'batch_size': 64,
            'dpi': 200,
            'figure_size_factor': 0.85,
            'use_subsampling': False,
            'max_samples': 500
        }
    else:  # Low memory usage
        return {
            'batch_size': 256,
            'dpi': 300,
            'figure_size_factor': 1.0,
            'use_subsampling': False,
            'max_samples': 1000
        }


class InteractivePlotter:
    """
    Interactive plotting utilities for exploration and analysis.
    
    Provides widgets and interactive features for exploring autoencoder results.
    """
    
    def __init__(self, originals: np.ndarray, 
                 reconstructions: np.ndarray,
                 latent_representations: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None):
        """
        Initialize interactive plotter.
        
        Args:
            originals (np.ndarray): Original images
            reconstructions (np.ndarray): Reconstructed images
            latent_representations (Optional[np.ndarray]): Latent space vectors
            labels (Optional[np.ndarray]): Class labels
        """
        self.originals, self.reconstructions = validate_image_data(originals, reconstructions)
        self.latent_representations = latent_representations
        self.labels = labels
        self.n_samples = len(self.originals)
        
    def create_sample_browser(self, figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
        """
        Create an interactive sample browser (static version for now).
        
        Args:
            figsize (Tuple[float, float]): Figure size
            
        Returns:
            plt.Figure: Figure with browsing interface
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Default sample index
        sample_idx = 0
        
        # Original image
        orig_img = self.originals[sample_idx].reshape(28, 28)
        axes[0].imshow(orig_img, cmap='Greys_r', interpolation='nearest')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Reconstructed image
        recon_img = self.reconstructions[sample_idx].reshape(28, 28)
        axes[1].imshow(recon_img, cmap='Greys_r', interpolation='nearest')
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        # Error visualization
        error_img = np.abs(orig_img - recon_img)
        im = axes[2].imshow(error_img, cmap='Reds', interpolation='nearest')
        axes[2].set_title('Reconstruction Error')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        # Add sample information
        mse_error = np.mean((orig_img - recon_img) ** 2)
        info_text = f'Sample {sample_idx + 1}/{self.n_samples}\nMSE: {mse_error:.6f}'
        if self.labels is not None:
            info_text += f'\nLabel: {self.labels[sample_idx]}'
        
        fig.suptitle(info_text, fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def create_error_distribution_explorer(self, figsize: Tuple[float, float] = (15, 10)) -> plt.Figure:
        """
        Create an interactive error distribution explorer.
        
        Args:
            figsize (Tuple[float, float]): Figure size
            
        Returns:
            plt.Figure: Figure with error analysis
        """
        # Calculate errors
        errors = np.mean((self.originals - self.reconstructions) ** 2, axis=(1, 2) if self.originals.ndim == 3 else 1)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        colors = get_color_palette()
        
        # Overall error distribution
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color=colors[0], edgecolor='black')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].set_xlabel('MSE Error')
        axes[0, 0].set_ylabel('Frequency')
        
        # Error vs sample index
        axes[0, 1].plot(errors, alpha=0.7, color=colors[1])
        axes[0, 1].set_title('Error vs Sample Index')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('MSE Error')
        
        # Best and worst reconstructions
        best_idx = np.argmin(errors)
        worst_idx = np.argmax(errors)
        
        # Best reconstruction
        axes[0, 2].imshow(self.originals[best_idx].reshape(28, 28), cmap='Greys_r')
        axes[0, 2].set_title(f'Best Reconstruction\nMSE: {errors[best_idx]:.6f}')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(self.reconstructions[best_idx].reshape(28, 28), cmap='Greys_r')
        axes[1, 0].set_title('Best Recon (Output)')
        axes[1, 0].axis('off')
        
        # Worst reconstruction
        axes[1, 1].imshow(self.originals[worst_idx].reshape(28, 28), cmap='Greys_r')
        axes[1, 1].set_title(f'Worst Reconstruction\nMSE: {errors[worst_idx]:.6f}')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(self.reconstructions[worst_idx].reshape(28, 28), cmap='Greys_r')
        axes[1, 2].set_title('Worst Recon (Output)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_latent_space_explorer(self, figsize: Tuple[float, float] = (12, 8)) -> Optional[plt.Figure]:
        """
        Create an interactive latent space explorer.
        
        Args:
            figsize (Tuple[float, float]): Figure size
            
        Returns:
            Optional[plt.Figure]: Figure with latent space analysis, None if no latent data
        """
        if self.latent_representations is None:
            warnings.warn("No latent representations provided for latent space exploration")
            return None
        
        from sklearn.decomposition import PCA
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_representations)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        colors = get_color_palette()
        
        # Scatter plot
        if self.labels is not None:
            unique_labels = np.unique(self.labels)
            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                axes[0].scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                              c=colors[i % len(colors)], label=f'Class {label}', alpha=0.7)
            axes[0].legend()
        else:
            axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=colors[0], alpha=0.7)
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0].set_title('Latent Space (PCA Projection)')
        
        # Latent dimension statistics
        latent_means = np.mean(self.latent_representations, axis=0)
        latent_stds = np.std(self.latent_representations, axis=0)
        
        axes[1].bar(range(len(latent_means)), latent_means, alpha=0.7, 
                   color=colors[1], label='Mean')
        axes[1].errorbar(range(len(latent_means)), latent_means, yerr=latent_stds,
                        fmt='none', color='black', alpha=0.5, capsize=3)
        axes[1].set_xlabel('Latent Dimension')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Latent Dimension Statistics')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
