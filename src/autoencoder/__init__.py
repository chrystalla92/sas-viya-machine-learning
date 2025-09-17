"""
Autoencoder evaluation and visualization package.

This package provides comprehensive evaluation and visualization capabilities
for MLP autoencoder models including:

- Evaluation metrics (MSE, PSNR, SSIM)
- Latent space analysis
- Publication-ready visualizations
- Comprehensive reporting utilities
"""

from .evaluation import (
    mse_loss,
    psnr_metric,
    ssim_metric,
    comprehensive_metrics,
    quality_score,
    latent_analysis,
    reconstruction_error_map,
    evaluate_model_comprehensive,
    compare_models
)

from .visualization import (
    setup_publication_style,
    plot_original_vs_reconstructed,
    plot_training_curves,
    plot_latent_space,
    plot_image_grid,
    plot_error_heatmap,
    plot_metrics_comparison,
    plot_reconstruction_quality_distribution,
    create_evaluation_report
)

__version__ = "1.0.0"
__author__ = "Autoencoder Evaluation Team"

__all__ = [
    # Evaluation functions
    "mse_loss",
    "psnr_metric", 
    "ssim_metric",
    "comprehensive_metrics",
    "quality_score",
    "latent_analysis",
    "reconstruction_error_map",
    "evaluate_model_comprehensive",
    "compare_models",
    
    # Visualization functions
    "setup_publication_style",
    "plot_original_vs_reconstructed",
    "plot_training_curves", 
    "plot_latent_space",
    "plot_image_grid",
    "plot_error_heatmap",
    "plot_metrics_comparison",
    "plot_reconstruction_quality_distribution",
    "create_evaluation_report"
]
