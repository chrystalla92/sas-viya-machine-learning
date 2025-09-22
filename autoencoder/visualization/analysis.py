"""
Latent Space Analysis and Diagnostic Visualization

This module provides advanced analysis and diagnostic visualization tools for the
MNIST autoencoder, including:
- Latent space visualization (t-SNE, PCA)
- Reconstruction error analysis
- Weight and bias distribution plots
- Activation heatmaps and hidden layer analysis
- Model diagnostic tools
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot_utils import (
    create_figure_and_axes, format_axis, save_figure, tensor_to_numpy,
    normalize_for_display, COLORS, COLORMAP_OPTIONS, PlotManager,
    apply_tight_layout, add_colorbar
)


def plot_latent_space_2d(hidden_representations: Union[torch.Tensor, np.ndarray],
                        labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                        method: str = "pca",
                        title: Optional[str] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot 2D visualization of latent space using dimensionality reduction.
    
    Args:
        hidden_representations: Hidden layer activations (N, hidden_dim)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('pca', 'tsne')
        title: Plot title (auto-generated if None)
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    hidden_repr = tensor_to_numpy(hidden_representations)
    if labels is not None:
        labels = tensor_to_numpy(labels) if hasattr(labels, 'detach') else np.array(labels)
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(hidden_repr)
        explained_var = reducer.explained_variance_ratio_
        method_title = f"PCA (explained variance: {explained_var[0]:.1%}, {explained_var[1]:.1%})"
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, 
                      n_iter=1000, learning_rate=200)
        reduced_data = reducer.fit_transform(hidden_repr)
        method_title = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create title
    if title is None:
        title = f"Latent Space Visualization - {method_title}"
    
    # Create figure
    fig, ax = create_figure_and_axes(figsize=figsize)
    ax = ax[0] if isinstance(ax, np.ndarray) else ax
    
    # Plot points
    if labels is not None:
        # Color by labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                      c=[colors[i]], label=f'Digit {int(label)}', 
                      alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 frameon=True, fancybox=True, shadow=True)
    else:
        # Single color
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                           c=COLORS['primary'], alpha=0.7, s=30,
                           edgecolors='white', linewidth=0.5)
    
    # Configure axes
    ax = format_axis(ax, title=title, xlabel=f'{method.upper()} Component 1', 
                    ylabel=f'{method.upper()} Component 2', grid=True)
    
    # Add statistics
    stats_text = f"Method: {method.upper()}\n"
    stats_text += f"Samples: {len(hidden_repr)}\n"
    stats_text += f"Original Dim: {hidden_repr.shape[1]}"
    if method.lower() == 'pca':
        stats_text += f"\nTotal Var Explained: {sum(explained_var):.1%}"
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_latent_tsne(hidden_representations: Union[torch.Tensor, np.ndarray],
                    labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                    perplexity: int = 30,
                    title: str = "t-SNE Latent Space Visualization",
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create t-SNE visualization of latent space with custom parameters.
    
    Args:
        hidden_representations: Hidden layer activations
        labels: Optional labels for coloring
        perplexity: t-SNE perplexity parameter
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    return plot_latent_space_2d(
        hidden_representations=hidden_representations,
        labels=labels,
        method='tsne',
        title=f"{title} (perplexity={perplexity})",
        save_path=save_path,
        figsize=figsize
    )


def plot_latent_pca(hidden_representations: Union[torch.Tensor, np.ndarray],
                   labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                   title: str = "PCA Latent Space Visualization",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Create PCA visualization of latent space.
    
    Args:
        hidden_representations: Hidden layer activations
        labels: Optional labels for coloring
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    return plot_latent_space_2d(
        hidden_representations=hidden_representations,
        labels=labels,
        method='pca',
        title=title,
        save_path=save_path,
        figsize=figsize
    )


def plot_reconstruction_errors(original_images: Union[torch.Tensor, np.ndarray],
                             reconstructed_images: Union[torch.Tensor, np.ndarray],
                             labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                             error_type: str = "mse",
                             title: Optional[str] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Analyze and visualize reconstruction errors.
    
    Args:
        original_images: Original MNIST images
        reconstructed_images: Reconstructed MNIST images
        labels: Optional labels for grouping analysis
        error_type: Type of error to compute ('mse', 'mae', 'pixel')
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    original = tensor_to_numpy(original_images)
    reconstructed = tensor_to_numpy(reconstructed_images)
    if labels is not None:
        labels = tensor_to_numpy(labels) if hasattr(labels, 'detach') else np.array(labels)
    
    # Calculate errors
    if error_type == "mse":
        errors = np.mean((original - reconstructed) ** 2, axis=1)
        error_name = "Mean Squared Error"
    elif error_type == "mae":
        errors = np.mean(np.abs(original - reconstructed), axis=1)
        error_name = "Mean Absolute Error"
    elif error_type == "pixel":
        errors = np.abs(original - reconstructed)
        error_name = "Pixel-wise Absolute Error"
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    if title is None:
        title = f"Reconstruction Error Analysis - {error_name}"
    
    # Create subplot layout
    if error_type == "pixel":
        # For pixel-wise errors, show heatmaps
        fig, axes = create_figure_and_axes(figsize=figsize, nrows=2, ncols=3)
    else:
        # For aggregated errors, show distributions and analysis
        fig, axes = create_figure_and_axes(figsize=figsize, nrows=2, ncols=2)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if error_type != "pixel":
        # Plot 1: Error distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(errors, bins=50, alpha=0.7, color=COLORS['primary'], 
                edgecolor='black', density=True)
        ax1.axvline(np.mean(errors), color=COLORS['accent'], linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.6f}')
        ax1.axvline(np.median(errors), color=COLORS['secondary'], linestyle='--',
                   linewidth=2, label=f'Median: {np.median(errors):.6f}')
        ax1 = format_axis(ax1, title="Error Distribution", xlabel=error_name, 
                         ylabel="Density", grid=True, legend=True)
        
        # Plot 2: Box plot by digit (if labels available)
        ax2 = axes[0, 1]
        if labels is not None:
            unique_labels = sorted(np.unique(labels))
            error_by_digit = [errors[labels == label] for label in unique_labels]
            bp = ax2.boxplot(error_by_digit, labels=[f'{int(label)}' for label in unique_labels],
                           patch_artist=True, notch=True)
            
            # Color boxes
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2 = format_axis(ax2, title="Error by Digit", xlabel="Digit", 
                             ylabel=error_name, grid=True)
        else:
            ax2.text(0.5, 0.5, "No labels available\nfor digit analysis", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Error by Digit")
        
        # Plot 3: Error vs sample index (to check for patterns)
        ax3 = axes[1, 0]
        sample_indices = np.arange(len(errors))
        ax3.scatter(sample_indices, errors, alpha=0.6, s=20, c=COLORS['primary'])
        ax3.plot(sample_indices, np.convolve(errors, np.ones(100)/100, mode='same'),
                color=COLORS['accent'], linewidth=2, label='Moving Average (100 samples)')
        ax3 = format_axis(ax3, title="Error vs Sample Index", xlabel="Sample Index", 
                         ylabel=error_name, grid=True, legend=True)
        
        # Plot 4: Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"Error Statistics ({error_name}):\n\n"
        stats_text += f"Mean: {np.mean(errors):.6f}\n"
        stats_text += f"Std: {np.std(errors):.6f}\n"
        stats_text += f"Min: {np.min(errors):.6f}\n"
        stats_text += f"Max: {np.max(errors):.6f}\n"
        stats_text += f"Median: {np.median(errors):.6f}\n"
        stats_text += f"Q25: {np.percentile(errors, 25):.6f}\n"
        stats_text += f"Q75: {np.percentile(errors, 75):.6f}\n\n"
        stats_text += f"Total Samples: {len(errors)}"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                fontsize=11, family='monospace')
        
    else:
        # For pixel-wise errors, show example heatmaps
        n_examples = min(6, len(errors))
        for i in range(n_examples):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            error_image = errors[i].reshape(28, 28)
            im = ax.imshow(error_image, cmap='Reds', interpolation='none')
            ax.set_title(f"Sample {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar
            add_colorbar(fig, im, ax, label="Absolute Error")
        
        # Hide unused subplots
        for i in range(n_examples, 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_weight_distributions(model: torch.nn.Module,
                            title: str = "Model Weight Distributions",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Visualize distributions of model weights and biases.
    
    Args:
        model: PyTorch model to analyze
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Extract weights and biases
    weights_data = {}
    biases_data = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights_data[name] = tensor_to_numpy(param).flatten()
        elif 'bias' in name:
            biases_data[name] = tensor_to_numpy(param).flatten()
    
    # Determine subplot layout
    n_weight_layers = len(weights_data)
    n_bias_layers = len(biases_data)
    n_plots = n_weight_layers + n_bias_layers
    
    if n_plots == 0:
        # No parameters found
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No parameters found in model", 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Create subplots
    nrows = 2 if n_plots > 2 else 1
    ncols = min(n_plots, 3) if nrows == 1 else 3
    fig, axes = create_figure_and_axes(figsize=figsize, nrows=nrows, ncols=ncols)
    
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plot_idx = 0
    
    # Plot weight distributions
    for name, weights in weights_data.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Create histogram
        ax.hist(weights, bins=50, alpha=0.7, color=COLORS['primary'], 
               density=True, edgecolor='black')
        
        # Add statistics
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        ax.axvline(mean_w, color=COLORS['accent'], linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_w:.4f}')
        ax.axvline(mean_w + std_w, color=COLORS['secondary'], linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'±1σ: {std_w:.4f}')
        ax.axvline(mean_w - std_w, color=COLORS['secondary'], linestyle=':', 
                  linewidth=2, alpha=0.7)
        
        # Format axis
        layer_name = name.replace('.weight', '').replace('_', ' ').title()
        ax = format_axis(ax, title=f"{layer_name} Weights", xlabel="Weight Value", 
                        ylabel="Density", grid=True, legend=True)
        
        plot_idx += 1
    
    # Plot bias distributions
    for name, biases in biases_data.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Create histogram
        ax.hist(biases, bins=30, alpha=0.7, color=COLORS['success'], 
               density=True, edgecolor='black')
        
        # Add statistics
        mean_b = np.mean(biases)
        std_b = np.std(biases)
        ax.axvline(mean_b, color=COLORS['accent'], linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_b:.4f}')
        ax.axvline(mean_b + std_b, color=COLORS['secondary'], linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'±1σ: {std_b:.4f}')
        ax.axvline(mean_b - std_b, color=COLORS['secondary'], linestyle=':', 
                  linewidth=2, alpha=0.7)
        
        # Format axis
        layer_name = name.replace('.bias', '').replace('_', ' ').title()
        ax = format_axis(ax, title=f"{layer_name} Biases", xlabel="Bias Value", 
                        ylabel="Density", grid=True, legend=True)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Add summary statistics as text
    if plot_idx < len(axes):
        # Use one of the hidden axes for statistics
        stats_ax = axes[-1]
        stats_ax.set_visible(True)
        stats_ax.axis('off')
        
        stats_text = "Model Parameter Statistics:\n\n"
        total_params = sum(len(w) for w in weights_data.values()) + sum(len(b) for b in biases_data.values())
        stats_text += f"Total Parameters: {total_params:,}\n\n"
        
        for name, weights in weights_data.items():
            layer_name = name.replace('.weight', '')
            stats_text += f"{layer_name} weights: {len(weights):,}\n"
            stats_text += f"  Range: [{np.min(weights):.4f}, {np.max(weights):.4f}]\n"
        
        for name, biases in biases_data.items():
            layer_name = name.replace('.bias', '')
            stats_text += f"{layer_name} biases: {len(biases):,}\n"
            stats_text += f"  Range: [{np.min(biases):.4f}, {np.max(biases):.4f}]\n"
        
        stats_ax.text(0.1, 0.9, stats_text, transform=stats_ax.transAxes,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3),
                     fontsize=10, family='monospace')
    
    # Adjust layout
    apply_tight_layout(fig)
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    return fig


def plot_activation_heatmap(hidden_activations: Union[torch.Tensor, np.ndarray],
                          labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                          n_samples: int = 100,
                          title: str = "Hidden Layer Activation Heatmap",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create heatmap visualization of hidden layer activations.
    
    Args:
        hidden_activations: Hidden layer activations (N, hidden_dim)
        labels: Optional labels for samples
        n_samples: Number of samples to display
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy
    activations = tensor_to_numpy(hidden_activations)
    if labels is not None:
        labels = tensor_to_numpy(labels) if hasattr(labels, 'detach') else np.array(labels)
    
    # Limit number of samples
    n_samples = min(n_samples, len(activations))
    activations = activations[:n_samples]
    if labels is not None:
        labels = labels[:n_samples]
    
    # Create figure with 2 subplots
    fig, axes = create_figure_and_axes(figsize=figsize, nrows=1, ncols=2,
                                      subplot_kw={'aspect': 'auto'})
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Activation heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(activations.T, cmap=COLORMAP_OPTIONS['diverging'], 
                    aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Hidden Unit Index')
    ax1.set_title('Activation Values')
    
    # Add colorbar
    cbar1 = add_colorbar(fig, im1, ax1, label='Activation Value')
    
    # Plot 2: Activation statistics
    ax2 = axes[1]
    
    # Calculate statistics per hidden unit
    mean_activations = np.mean(activations, axis=0)
    std_activations = np.std(activations, axis=0)
    hidden_units = np.arange(len(mean_activations))
    
    # Plot mean and std
    ax2.plot(hidden_units, mean_activations, color=COLORS['primary'], 
            linewidth=2, label='Mean Activation', alpha=0.8)
    ax2.fill_between(hidden_units, 
                    mean_activations - std_activations,
                    mean_activations + std_activations,
                    color=COLORS['primary'], alpha=0.3, label='±1 Std Dev')
    
    ax2 = format_axis(ax2, title='Activation Statistics', 
                     xlabel='Hidden Unit Index', ylabel='Activation Value',
                     grid=True, legend=True)
    
    # Add sample labels if available
    if labels is not None:
        # Add color bar for labels on the first plot
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 10:  # Only if not too many unique labels
            # Create custom colormap for labels
            label_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            # Add label strip at the bottom
            label_strip = np.array([labels]).T
            im_labels = ax1.imshow(label_strip, extent=[0, n_samples, -5, -1],
                                  cmap='tab10', aspect='auto', alpha=0.8)
            ax1.set_ylim(-5, activations.shape[1])
            ax1.text(n_samples/2, -3, 'Labels', ha='center', va='center', 
                    fontweight='bold', color='white')
    
    # Add statistics text
    stats_text = f"Samples: {n_samples}\n"
    stats_text += f"Hidden Units: {activations.shape[1]}\n"
    stats_text += f"Activation Range: [{np.min(activations):.3f}, {np.max(activations):.3f}]\n"
    stats_text += f"Mean Activation: {np.mean(activations):.3f}\n"
    stats_text += f"Std Activation: {np.std(activations):.3f}"
    
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


def analyze_hidden_representations(model: torch.nn.Module,
                                 data: Union[torch.Tensor, np.ndarray],
                                 labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                                 output_dir: str = "./latent_analysis",
                                 n_samples: int = 1000) -> Dict[str, plt.Figure]:
    """
    Comprehensive analysis of hidden layer representations.
    
    Args:
        model: Trained autoencoder model
        data: Input data for analysis
        labels: Optional labels for the data
        output_dir: Directory to save analysis plots
        n_samples: Number of samples to analyze
        
    Returns:
        Dictionary of figure objects created
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Convert data
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)
    data = data[:n_samples]
    
    if labels is not None:
        if isinstance(labels, np.ndarray):
            labels = torch.LongTensor(labels)
        labels = labels[:n_samples]
    
    # Get hidden representations
    with torch.no_grad():
        hidden_repr = model.encode(data)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # PCA analysis
    print("Creating PCA visualization...")
    fig_pca = plot_latent_pca(
        hidden_repr, labels,
        save_path=str(output_path / "latent_pca")
    )
    figures['pca'] = fig_pca
    
    # t-SNE analysis
    print("Creating t-SNE visualization...")
    fig_tsne = plot_latent_tsne(
        hidden_repr, labels,
        save_path=str(output_path / "latent_tsne")
    )
    figures['tsne'] = fig_tsne
    
    # Activation heatmap
    print("Creating activation heatmap...")
    fig_heatmap = plot_activation_heatmap(
        hidden_repr, labels, n_samples=min(100, n_samples),
        save_path=str(output_path / "activation_heatmap")
    )
    figures['heatmap'] = fig_heatmap
    
    # Weight distributions
    print("Creating weight distribution analysis...")
    fig_weights = plot_weight_distributions(
        model,
        save_path=str(output_path / "weight_distributions")
    )
    figures['weights'] = fig_weights
    
    # Reconstruction error analysis
    print("Creating reconstruction error analysis...")
    with torch.no_grad():
        reconstructions = model(data)
    
    fig_errors = plot_reconstruction_errors(
        data, reconstructions, labels,
        save_path=str(output_path / "reconstruction_errors")
    )
    figures['errors'] = fig_errors
    
    print(f"Latent space analysis complete. Results saved to {output_dir}")
    
    return figures
