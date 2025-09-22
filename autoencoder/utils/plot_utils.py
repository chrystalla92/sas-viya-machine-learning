"""
Common Plotting Utilities and Styling

This module provides common utilities, styling configurations, and helper functions
for creating publication-ready plots across the autoencoder visualization package.

Key features:
- Consistent styling and color schemes
- Plot configuration and formatting utilities
- File saving and export functions
- Interactive plot support
- Publication-ready formatting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings

# Configure matplotlib for better default appearance
plt.style.use('default')
sns.set_palette("husl")

# Publication-ready style configurations
PUBLICATION_CONFIG = {
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.linewidth': 1,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
}

# Color schemes
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple  
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#7D8491',      # Gray
    'light': '#F5F7FA',        # Light gray
    'dark': '#2C3E50'          # Dark blue-gray
}

COLORMAP_OPTIONS = {
    'mnist': 'Greys',
    'heatmap': 'viridis',
    'diverging': 'RdBu_r',
    'sequential': 'plasma',
    'error': 'Reds'
}


def setup_publication_style():
    """Apply publication-ready matplotlib settings."""
    plt.rcParams.update(PUBLICATION_CONFIG)


def reset_matplotlib_style():
    """Reset matplotlib to default settings."""
    plt.rcdefaults()
    sns.reset_defaults()


def create_figure_and_axes(figsize: Tuple[int, int] = (12, 8), 
                          nrows: int = 1, ncols: int = 1,
                          subplot_kw: Optional[Dict] = None,
                          publication_style: bool = True) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create figure and axes with consistent styling.
    
    Args:
        figsize: Figure size (width, height)
        nrows: Number of subplot rows
        ncols: Number of subplot columns  
        subplot_kw: Subplot keyword arguments
        publication_style: Whether to apply publication styling
        
    Returns:
        Tuple of (figure, axes)
    """
    if publication_style:
        setup_publication_style()
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                           figsize=figsize, subplot_kw=subplot_kw)
    
    # Ensure axes is always iterable
    if nrows * ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    
    return fig, axes


def format_axis(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "",
               grid: bool = True, legend: bool = False) -> plt.Axes:
    """
    Apply consistent formatting to an axis.
    
    Args:
        ax: Matplotlib axis to format
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Whether to show grid
        legend: Whether to show legend
        
    Returns:
        Formatted axis
    """
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend:
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax


def save_figure(fig: plt.Figure, filepath: str, formats: List[str] = ['png'],
               dpi: int = 300, bbox_inches: str = 'tight',
               transparent: bool = False, close_after_save: bool = False):
    """
    Save figure in multiple formats with publication-ready settings.
    
    Args:
        fig: Matplotlib figure to save
        filepath: Base filepath (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
        dpi: Resolution for raster formats  
        bbox_inches: Bounding box mode
        transparent: Whether to use transparent background
        close_after_save: Whether to close figure after saving
    """
    # Create output directory if it doesn't exist
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in each requested format
    for fmt in formats:
        full_filepath = f"{filepath}.{fmt}"
        
        try:
            fig.savefig(
                full_filepath,
                format=fmt,
                dpi=dpi,
                bbox_inches=bbox_inches,
                transparent=transparent,
                facecolor='white',
                edgecolor='none'
            )
            print(f"Figure saved: {full_filepath}")
        except Exception as e:
            print(f"Warning: Could not save figure as {fmt}: {e}")
    
    if close_after_save:
        plt.close(fig)


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array for plotting.
    
    Args:
        tensor: Input tensor or array
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def normalize_for_display(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data for display purposes.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'none')
        
    Returns:
        Normalized data array
    """
    if method == 'minmax':
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        else:
            return data
    elif method == 'zscore':
        return (data - data.mean()) / (data.std() + 1e-8)
    elif method == 'none':
        return data
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_color_legend(colors: Dict[str, str], title: str = "Legend") -> mpatches.Patch:
    """
    Create a color legend for plots.
    
    Args:
        colors: Dictionary mapping labels to colors
        title: Legend title
        
    Returns:
        Legend patches
    """
    patches = [mpatches.Patch(color=color, label=label) 
              for label, color in colors.items()]
    return patches


def setup_subplot_layout(n_plots: int, max_cols: int = 5) -> Tuple[int, int]:
    """
    Calculate optimal subplot layout for given number of plots.
    
    Args:
        n_plots: Number of plots to arrange
        max_cols: Maximum number of columns
        
    Returns:
        Tuple of (nrows, ncols)
    """
    if n_plots <= max_cols:
        return 1, n_plots
    else:
        ncols = min(max_cols, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        return nrows, ncols


def add_colorbar(fig: plt.Figure, mappable, ax: plt.Axes, 
                label: str = "", shrink: float = 0.8) -> plt.colorbar:
    """
    Add a colorbar with consistent styling.
    
    Args:
        fig: Figure object
        mappable: Mappable object (e.g., from imshow)
        ax: Axis to attach colorbar to
        label: Colorbar label
        shrink: Colorbar shrink factor
        
    Returns:
        Colorbar object
    """
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink)
    if label:
        cbar.set_label(label, rotation=270, labelpad=20)
    return cbar


def enable_interactive_mode():
    """Enable interactive matplotlib mode."""
    plt.ion()
    

def disable_interactive_mode():
    """Disable interactive matplotlib mode."""
    plt.ioff()


def get_figure_size_for_grid(n_items: int, max_cols: int = 5, 
                           item_size: float = 3.0) -> Tuple[float, float]:
    """
    Calculate appropriate figure size for a grid layout.
    
    Args:
        n_items: Number of items in grid
        max_cols: Maximum columns in grid
        item_size: Size per grid item
        
    Returns:
        Tuple of (width, height) in inches
    """
    nrows, ncols = setup_subplot_layout(n_items, max_cols)
    width = ncols * item_size
    height = nrows * item_size
    return width, height


def apply_tight_layout(fig: plt.Figure, pad: float = 1.5):
    """
    Apply tight layout with consistent padding.
    
    Args:
        fig: Figure to apply layout to
        pad: Padding between subplots
    """
    try:
        fig.tight_layout(pad=pad)
    except Exception as e:
        warnings.warn(f"Could not apply tight layout: {e}")


# Utility functions for specific plot types
def format_mnist_image(image_data: np.ndarray) -> np.ndarray:
    """
    Format MNIST image data for display.
    
    Args:
        image_data: Flattened MNIST image (784,) or 2D image (28, 28)
        
    Returns:
        2D image array (28, 28)
    """
    if image_data.ndim == 1:
        return image_data.reshape(28, 28)
    return image_data


def get_mnist_colormap() -> str:
    """Get the standard colormap for MNIST images."""
    return COLORMAP_OPTIONS['mnist']


def create_subplot_titles(base_title: str, indices: List[int]) -> List[str]:
    """
    Create subplot titles with consistent formatting.
    
    Args:
        base_title: Base title string
        indices: List of indices
        
    Returns:
        List of formatted titles
    """
    return [f"{base_title} {i}" for i in indices]


class PlotManager:
    """
    Context manager for plot creation and cleanup.
    
    Handles figure creation, styling, and cleanup automatically.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), 
                 publication_style: bool = True,
                 save_path: Optional[str] = None,
                 save_formats: List[str] = ['png']):
        """
        Initialize plot manager.
        
        Args:
            figsize: Figure size
            publication_style: Whether to apply publication styling
            save_path: Path to save figure (optional)
            save_formats: Formats to save figure in
        """
        self.figsize = figsize
        self.publication_style = publication_style
        self.save_path = save_path
        self.save_formats = save_formats
        self.fig = None
        self.axes = None
    
    def __enter__(self):
        """Enter context - create figure."""
        if self.publication_style:
            setup_publication_style()
            
        self.fig = plt.figure(figsize=self.figsize)
        return self.fig
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - save and cleanup figure."""
        if self.save_path and self.fig:
            save_figure(self.fig, self.save_path, self.save_formats)
        
        # Don't automatically close - let user decide
        # plt.close(self.fig)
