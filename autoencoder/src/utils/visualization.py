"""
Visualization utilities for autoencoder training and results.

This module provides comprehensive visualization functions for:
- Training progress and loss curves
- Reconstruction quality visualization
- Latent space representation
- Model architecture diagrams
- Performance metrics dashboards
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging

# Configure matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")
logger = logging.getLogger(__name__)


def plot_training_progress(
    loss_history: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Progress",
    convergence_tolerance: Optional[float] = None,
    best_iteration: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comprehensive training progress with loss curve and convergence analysis.
    
    Args:
        loss_history: List of loss values over iterations
        save_path: Path to save the plot (optional)
        title: Plot title
        convergence_tolerance: Convergence tolerance line (optional)
        best_iteration: Iteration with best loss (optional)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    iterations = list(range(1, len(loss_history) + 1))
    
    # 1. Loss curve (linear scale)
    axes[0, 0].plot(iterations, loss_history, 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Progression (Linear Scale)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark best iteration
    if best_iteration is not None and best_iteration <= len(loss_history):
        best_loss = loss_history[best_iteration - 1]
        axes[0, 0].scatter([best_iteration], [best_loss], color='red', s=100, 
                          zorder=5, label=f'Best (Iter {best_iteration})')
    
    axes[0, 0].legend()
    
    # 2. Loss curve (log scale)
    axes[0, 1].plot(iterations, loss_history, 'g-', linewidth=2, label='Training Loss')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Loss Progression (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add convergence tolerance line
    if convergence_tolerance is not None:
        min_loss = min(loss_history)
        tolerance_line = min_loss + convergence_tolerance
        axes[0, 1].axhline(y=tolerance_line, color='r', linestyle='--', alpha=0.7,
                          label=f'Tolerance: {convergence_tolerance:.1e}')
    
    axes[0, 1].legend()
    
    # 3. Loss improvement over time
    if len(loss_history) > 1:
        loss_improvements = []
        for i in range(1, len(loss_history)):
            improvement = loss_history[i-1] - loss_history[i]
            loss_improvements.append(improvement)
        
        axes[1, 0].plot(range(2, len(loss_history) + 1), loss_improvements, 
                       'purple', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss Improvement')
        axes[1, 0].set_title('Loss Improvement per Iteration')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. Moving average and convergence
    if len(loss_history) > 10:
        window_size = min(20, len(loss_history) // 4)
        moving_avg = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        moving_avg_iter = list(range(window_size, len(loss_history) + 1))
        
        axes[1, 1].plot(iterations, loss_history, 'lightblue', alpha=0.6, 
                       label='Original Loss')
        axes[1, 1].plot(moving_avg_iter, moving_avg, 'orange', linewidth=3,
                       label=f'Moving Average (window={window_size})')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss with Moving Average')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor moving average', 
                       transform=axes[1, 1].transAxes, ha='center', va='center',
                       fontsize=12)
        axes[1, 1].set_title('Moving Average (Insufficient Data)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training progress plot saved to {save_path}")
    
    return fig


def plot_reconstruction_comparison(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    n_samples: int = 10,
    save_path: Optional[str] = None,
    title: str = "Reconstruction Comparison",
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot side-by-side comparison of original and reconstructed images.
    
    Args:
        original_images: Original input images tensor
        reconstructed_images: Reconstructed output images tensor
        n_samples: Number of samples to display
        save_path: Path to save the plot (optional)
        title: Plot title
        figsize: Figure size (auto-calculated if None)
        
    Returns:
        matplotlib Figure object
    """
    # Ensure tensors are on CPU
    if original_images.device != torch.device('cpu'):
        original_images = original_images.cpu()
    if reconstructed_images.device != torch.device('cpu'):
        reconstructed_images = reconstructed_images.cpu()
    
    # Convert to numpy and ensure proper shape
    original_np = original_images.detach().numpy()
    reconstructed_np = reconstructed_images.detach().numpy()
    
    n_samples = min(n_samples, original_np.shape[0])
    
    if figsize is None:
        figsize = (2 * n_samples, 6)
    
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if n_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(n_samples):
        # Reshape flat vectors back to 28x28 images
        original_img = original_np[i].reshape(28, 28)
        reconstructed_img = reconstructed_np[i].reshape(28, 28)
        
        # Original image
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(reconstructed_img, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}', fontsize=10)
        axes[1, i].axis('off')
        
        # Difference (error) image
        diff_img = np.abs(original_img - reconstructed_img)
        im = axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=diff_img.max())
        axes[2, i].set_title(f'Error {i+1}', fontsize=10)
        axes[2, i].axis('off')
        
        # Add colorbar for error image
        if i == n_samples - 1:
            plt.colorbar(im, ax=axes[2, i], shrink=0.6)
    
    # Add row labels
    row_labels = ['Original', 'Reconstructed', 'Absolute Error']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.1, 0.5, label, transform=axes[i, 0].transAxes,
                       rotation=90, va='center', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reconstruction comparison plot saved to {save_path}")
    
    return fig


def plot_latent_space(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    labels: Optional[torch.Tensor] = None,
    n_samples: int = 1000,
    save_path: Optional[str] = None,
    title: str = "Latent Space Representation",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot 2D latent space representation using PCA or t-SNE.
    
    Args:
        model: Trained autoencoder model
        data_loader: Data loader for samples
        labels: Optional labels for coloring points
        n_samples: Maximum number of samples to use
        save_path: Path to save the plot (optional)
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    model.eval()
    
    # Collect latent representations
    latent_vectors = []
    sample_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            if sample_count >= n_samples:
                break
                
            batch_data = batch_data.to(model.get_device())
            
            # Get latent representation
            latent = model.encode(batch_data)
            latent_vectors.append(latent.cpu().numpy())
            
            if labels is not None:
                sample_labels.extend(batch_labels.numpy())
            
            sample_count += batch_data.size(0)
    
    # Concatenate all latent vectors
    latent_matrix = np.vstack(latent_vectors)[:n_samples]
    
    if sample_labels:
        sample_labels = np.array(sample_labels[:n_samples])
    
    # Reduce dimensionality for visualization
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # PCA visualization
        pca = PCA(n_components=2)
        latent_2d_pca = pca.fit_transform(latent_matrix)
        
        if sample_labels is not None and len(np.unique(sample_labels)) <= 10:
            scatter_pca = axes[0].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], 
                                         c=sample_labels, cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter_pca, ax=axes[0], label='Class Label')
        else:
            axes[0].scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], alpha=0.6, s=20)
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('PCA Projection')
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE visualization (if small enough dataset)
        if latent_matrix.shape[0] <= 2000:  # t-SNE is expensive for large datasets
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d_tsne = tsne.fit_transform(latent_matrix)
            
            if sample_labels is not None and len(np.unique(sample_labels)) <= 10:
                scatter_tsne = axes[1].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], 
                                              c=sample_labels, cmap='tab10', alpha=0.6, s=20)
                plt.colorbar(scatter_tsne, ax=axes[1], label='Class Label')
            else:
                axes[1].scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], alpha=0.6, s=20)
            
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].set_title('t-SNE Projection')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, f'Dataset too large for t-SNE\n(n_samples={latent_matrix.shape[0]} > 2000)', 
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=12)
            axes[1].set_title('t-SNE (Skipped)')
        
    except ImportError:
        logger.warning("scikit-learn not available. Creating simple 2D projection.")
        
        # Simple 2D projection using first two latent dimensions
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if latent_matrix.shape[1] >= 2:
            if sample_labels is not None and len(np.unique(sample_labels)) <= 10:
                scatter = ax.scatter(latent_matrix[:, 0], latent_matrix[:, 1], 
                                   c=sample_labels, cmap='tab10', alpha=0.6, s=20)
                plt.colorbar(scatter, ax=ax, label='Class Label')
            else:
                ax.scatter(latent_matrix[:, 0], latent_matrix[:, 1], alpha=0.6, s=20)
            
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.set_title('First Two Latent Dimensions')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Latent space has only 1 dimension', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Latent space plot saved to {save_path}")
    
    return fig


def plot_model_architecture(
    model: torch.nn.Module,
    save_path: Optional[str] = None,
    title: str = "Model Architecture",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot a visual representation of the autoencoder architecture.
    
    Args:
        model: Autoencoder model to visualize
        save_path: Path to save the plot (optional)
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Get model architecture info
    if hasattr(model, 'get_architecture_summary'):
        arch_info = model.get_architecture_summary()
        input_dim = arch_info.get('input_dim', 784)
        hidden_dim = arch_info.get('hidden_dim', 400)
        output_dim = arch_info.get('output_dim', 784)
    else:
        # Default values for standard autoencoder
        input_dim = 784
        hidden_dim = 400
        output_dim = 784
    
    # Define layer positions and sizes
    layers = [
        {'name': f'Input\n({input_dim})', 'pos': (1, 3), 'size': input_dim},
        {'name': f'Hidden\n({hidden_dim})', 'pos': (3, 3), 'size': hidden_dim},
        {'name': f'Output\n({output_dim})', 'pos': (5, 3), 'size': output_dim}
    ]
    
    max_size = max(layer['size'] for layer in layers)
    
    # Draw layers as rectangles
    for i, layer in enumerate(layers):
        # Scale rectangle height based on layer size
        height = (layer['size'] / max_size) * 2 + 0.5
        width = 0.8
        
        x, y = layer['pos']
        
        # Create rectangle
        rect = patches.Rectangle(
            (x - width/2, y - height/2), width, height,
            linewidth=2, edgecolor='black', 
            facecolor=f'C{i}', alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add text label
        ax.text(x, y, layer['name'], ha='center', va='center', 
               fontsize=12, fontweight='bold')
    
    # Draw connections between layers
    connections = [
        (layers[0]['pos'], layers[1]['pos'], 'Encoder\n(Tanh)'),
        (layers[1]['pos'], layers[2]['pos'], 'Decoder\n(Linear)')
    ]
    
    for start, end, label in connections:
        # Draw arrow
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        # Add label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.3
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
               fontsize=10, style='italic', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(0, 6)
    ax.set_ylim(1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add additional information
    info_text = f"""
    Architecture: MLP Autoencoder
    Input Dimension: {input_dim}
    Hidden Dimension: {hidden_dim}
    Output Dimension: {output_dim}
    Total Parameters: {sum(p.numel() for p in model.parameters()):,}
    """
    
    ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model architecture plot saved to {save_path}")
    
    return fig


def plot_metrics_dashboard(
    metrics_data: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Training Metrics Dashboard",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive metrics dashboard with multiple subplots.
    
    Args:
        metrics_data: Dictionary containing training metrics
        save_path: Path to save the plot (optional)
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Create grid layout
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data
    loss_stats = metrics_data.get('loss_statistics', {})
    perf_stats = metrics_data.get('performance_statistics', {})
    conv_stats = metrics_data.get('convergence_statistics', {})
    
    # 1. Loss statistics bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    if loss_stats:
        loss_values = [
            loss_stats.get('initial_loss', 0),
            loss_stats.get('final_loss', 0),
            loss_stats.get('best_loss', 0)
        ]
        loss_labels = ['Initial', 'Final', 'Best']
        bars = ax1.bar(loss_labels, loss_values, color=['red', 'blue', 'green'])
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Loss Statistics')
        ax1.set_yscale('log')
        
        # Add value labels on bars
        for bar, value in zip(bars, loss_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 2. Performance metrics
    ax2 = fig.add_subplot(gs[0, 1])
    if perf_stats:
        training_time = perf_stats.get('total_training_time', 0)
        iter_time = perf_stats.get('mean_iteration_time', 0)
        total_iters = perf_stats.get('total_iterations', 0)
        
        metrics_text = f"""
        Training Time: {training_time:.1f}s
        Mean Iter Time: {iter_time:.3f}s
        Total Iterations: {total_iters}
        Iter/sec: {perf_stats.get('iterations_per_second', 0):.2f}
        """
        
        ax2.text(0.1, 0.5, metrics_text.strip(), transform=ax2.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax2.set_title('Performance Metrics')
        ax2.axis('off')
    
    # 3. Convergence information
    ax3 = fig.add_subplot(gs[0, 2])
    if conv_stats.get('convergence_metrics_available', False):
        conv_text = f"""
        Convergence Count: {conv_stats.get('convergence_count', 0)}
        Convergence Ratio: {conv_stats.get('convergence_ratio', 0):.2%}
        Trend: {conv_stats.get('convergence_trend', 'N/A')}
        Mean Rate: {conv_stats.get('mean_convergence_rate', 0):.2e}
        """
        
        ax3.text(0.1, 0.5, conv_text.strip(), transform=ax3.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'Convergence metrics\nnot available', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
    
    ax3.set_title('Convergence Statistics')
    ax3.axis('off')
    
    # 4-6. Additional plots can be added here (loss curves, histograms, etc.)
    # For now, we'll add placeholder plots
    
    # 4. Loss improvement histogram
    ax4 = fig.add_subplot(gs[1, :])
    if 'loss_history' in metrics_data:
        loss_history = metrics_data['loss_history']
        ax4.plot(loss_history, 'b-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss Progression')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # 5. Performance distribution (if available)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.text(0.5, 0.5, 'Performance\nDistribution\n(Placeholder)', 
            transform=ax5.transAxes, ha='center', va='center', fontsize=12)
    ax5.set_title('Performance Distribution')
    ax5.axis('off')
    
    # 6. Model summary
    ax6 = fig.add_subplot(gs[2, 1:])
    summary_text = f"""
    TRAINING SUMMARY
    
    Final Loss: {loss_stats.get('final_loss', 'N/A')}
    Best Loss: {loss_stats.get('best_loss', 'N/A')}
    Training Time: {perf_stats.get('total_training_time', 'N/A')}s
    Total Iterations: {perf_stats.get('total_iterations', 'N/A')}
    Convergence: {conv_stats.get('convergence_trend', 'N/A')}
    """
    
    ax6.text(0.1, 0.5, summary_text.strip(), transform=ax6.transAxes, 
            fontsize=14, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9))
    ax6.set_title('Training Summary')
    ax6.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics dashboard saved to {save_path}")
    
    return fig


def create_training_report(
    model: torch.nn.Module,
    metrics_data: Dict[str, Any],
    sample_data: Optional[torch.Tensor] = None,
    save_dir: Optional[str] = None,
    report_name: str = "training_report"
) -> List[Path]:
    """
    Create a comprehensive training report with multiple visualizations.
    
    Args:
        model: Trained autoencoder model
        metrics_data: Training metrics data
        sample_data: Optional sample data for reconstruction visualization
        save_dir: Directory to save report files
        report_name: Base name for report files
        
    Returns:
        List of paths to generated report files
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # 1. Training progress plot
    if 'loss_history' in metrics_data:
        fig1 = plot_training_progress(
            metrics_data['loss_history'],
            save_path=str(save_dir / f"{report_name}_training_progress.png") if save_dir else None,
            best_iteration=metrics_data.get('loss_statistics', {}).get('best_iteration')
        )
        if save_dir:
            generated_files.append(save_dir / f"{report_name}_training_progress.png")
        plt.close(fig1)
    
    # 2. Metrics dashboard
    fig2 = plot_metrics_dashboard(
        metrics_data,
        save_path=str(save_dir / f"{report_name}_metrics_dashboard.png") if save_dir else None
    )
    if save_dir:
        generated_files.append(save_dir / f"{report_name}_metrics_dashboard.png")
    plt.close(fig2)
    
    # 3. Model architecture
    fig3 = plot_model_architecture(
        model,
        save_path=str(save_dir / f"{report_name}_architecture.png") if save_dir else None
    )
    if save_dir:
        generated_files.append(save_dir / f"{report_name}_architecture.png")
    plt.close(fig3)
    
    # 4. Reconstruction comparison (if sample data provided)
    if sample_data is not None:
        model.eval()
        with torch.no_grad():
            sample_data = sample_data.to(model.get_device() if hasattr(model, 'get_device') 
                                       else next(model.parameters()).device)
            reconstructed = model(sample_data[:10])  # Use first 10 samples
            
            fig4 = plot_reconstruction_comparison(
                sample_data[:10],
                reconstructed,
                save_path=str(save_dir / f"{report_name}_reconstructions.png") if save_dir else None
            )
            if save_dir:
                generated_files.append(save_dir / f"{report_name}_reconstructions.png")
            plt.close(fig4)
    
    logger.info(f"Training report generated with {len(generated_files)} visualizations")
    
    return generated_files
