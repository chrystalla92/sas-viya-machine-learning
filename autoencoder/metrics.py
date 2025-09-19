"""
Evaluation metrics for autoencoder models.

This module provides comprehensive evaluation metrics including reconstruction
error calculations, latent space statistics, and analysis utilities.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings


__all__ = [
    'ReconstructionMetrics',
    'LatentSpaceAnalyzer',
    'calculate_reconstruction_errors',
    'calculate_per_sample_errors',
    'calculate_aggregate_errors',
    'compute_latent_statistics',
    'prepare_latent_visualization'
]


class ReconstructionMetrics:
    """
    Comprehensive reconstruction error metrics for autoencoder evaluation.
    
    Provides MSE, MAE, and other reconstruction metrics both per-sample
    and aggregated across the dataset.
    """
    
    @staticmethod
    def mse(original: torch.Tensor, reconstructed: torch.Tensor, 
            per_sample: bool = False, reduce_dims: Optional[List[int]] = None) -> Union[torch.Tensor, float]:
        """
        Calculate Mean Squared Error between original and reconstructed data.
        
        Args:
            original (torch.Tensor): Original data
            reconstructed (torch.Tensor): Reconstructed data
            per_sample (bool): If True, return per-sample errors
            reduce_dims (Optional[List[int]]): Dimensions to reduce over
            
        Returns:
            Union[torch.Tensor, float]: MSE values
        """
        if reduce_dims is None:
            reduce_dims = list(range(1, len(original.shape)))
            
        mse = torch.mean((original - reconstructed) ** 2, dim=reduce_dims)
        
        if not per_sample:
            mse = torch.mean(mse)
            
        return mse.item() if not per_sample and mse.numel() == 1 else mse
    
    @staticmethod
    def mae(original: torch.Tensor, reconstructed: torch.Tensor,
            per_sample: bool = False, reduce_dims: Optional[List[int]] = None) -> Union[torch.Tensor, float]:
        """
        Calculate Mean Absolute Error between original and reconstructed data.
        
        Args:
            original (torch.Tensor): Original data
            reconstructed (torch.Tensor): Reconstructed data
            per_sample (bool): If True, return per-sample errors
            reduce_dims (Optional[List[int]]): Dimensions to reduce over
            
        Returns:
            Union[torch.Tensor, float]: MAE values
        """
        if reduce_dims is None:
            reduce_dims = list(range(1, len(original.shape)))
            
        mae = torch.mean(torch.abs(original - reconstructed), dim=reduce_dims)
        
        if not per_sample:
            mae = torch.mean(mae)
            
        return mae.item() if not per_sample and mae.numel() == 1 else mae
    
    @staticmethod
    def rmse(original: torch.Tensor, reconstructed: torch.Tensor,
             per_sample: bool = False, reduce_dims: Optional[List[int]] = None) -> Union[torch.Tensor, float]:
        """Calculate Root Mean Squared Error."""
        mse = ReconstructionMetrics.mse(original, reconstructed, per_sample, reduce_dims)
        if per_sample:
            return torch.sqrt(mse)
        else:
            return np.sqrt(mse)
    
    @staticmethod
    def ssim_approximation(original: torch.Tensor, reconstructed: torch.Tensor,
                          per_sample: bool = False) -> Union[torch.Tensor, float]:
        """
        Simplified SSIM approximation for batch processing.
        Note: This is a simplified version for efficiency.
        """
        # Constants for SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = torch.mean(original, dim=-1, keepdim=True)
        mu2 = torch.mean(reconstructed, dim=-1, keepdim=True)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.var(original, dim=-1, keepdim=True)
        sigma2_sq = torch.var(reconstructed, dim=-1, keepdim=True)
        sigma12 = torch.mean((original - mu1) * (reconstructed - mu2), dim=-1, keepdim=True)
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim = torch.mean(ssim_map, dim=-1).squeeze()
        
        if not per_sample:
            ssim = torch.mean(ssim)
            
        return ssim.item() if not per_sample and ssim.numel() == 1 else ssim


class LatentSpaceAnalyzer:
    """
    Analysis utilities for latent space representations.
    
    Provides statistical analysis, dimensionality reduction preparation,
    and clustering utilities for latent representations.
    """
    
    @staticmethod
    def compute_statistics(latent_representations: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive statistics for latent representations.
        
        Args:
            latent_representations (torch.Tensor): Latent vectors of shape (N, latent_dim)
            
        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        if len(latent_representations.shape) != 2:
            raise ValueError(f"Expected 2D tensor (N, latent_dim), got shape {latent_representations.shape}")
        
        # Convert to numpy for some operations
        latent_np = latent_representations.detach().cpu().numpy()
        
        statistics = {
            # Basic statistics
            'mean': float(torch.mean(latent_representations).item()),
            'std': float(torch.std(latent_representations).item()),
            'var': float(torch.var(latent_representations).item()),
            'min': float(torch.min(latent_representations).item()),
            'max': float(torch.max(latent_representations).item()),
            
            # Per-dimension statistics
            'mean_per_dim': torch.mean(latent_representations, dim=0).cpu().numpy(),
            'std_per_dim': torch.std(latent_representations, dim=0).cpu().numpy(),
            'var_per_dim': torch.var(latent_representations, dim=0).cpu().numpy(),
            
            # Distribution properties
            'frobenius_norm': float(torch.norm(latent_representations, 'fro').item()),
            'spectral_norm': float(torch.norm(latent_representations, 2).item()),
            
            # Covariance properties
            'covariance_trace': float(np.trace(np.cov(latent_np.T))),
            'covariance_det': float(np.linalg.det(np.cov(latent_np.T))),
            
            # Effective dimensionality (participation ratio)
            'effective_dimensionality': LatentSpaceAnalyzer._compute_effective_dimensionality(latent_representations)
        }
        
        return statistics
    
    @staticmethod
    def _compute_effective_dimensionality(latent_representations: torch.Tensor) -> float:
        """
        Compute effective dimensionality using participation ratio.
        
        This measures how many dimensions actively contribute to the representation.
        """
        # Compute covariance matrix
        latent_centered = latent_representations - torch.mean(latent_representations, dim=0)
        cov_matrix = torch.mm(latent_centered.T, latent_centered) / (latent_representations.size(0) - 1)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(cov_matrix).real
        eigenvalues = torch.maximum(eigenvalues, torch.tensor(0.0))  # Ensure non-negative
        
        # Participation ratio: (sum of eigenvalues)^2 / sum of squared eigenvalues
        sum_evals = torch.sum(eigenvalues)
        sum_squared_evals = torch.sum(eigenvalues ** 2)
        
        if sum_squared_evals > 0:
            participation_ratio = (sum_evals ** 2) / sum_squared_evals
        else:
            participation_ratio = torch.tensor(0.0)
        
        return float(participation_ratio.item())
    
    @staticmethod
    def prepare_pca(latent_representations: torch.Tensor, 
                    n_components: Optional[int] = None) -> Tuple[np.ndarray, PCA]:
        """
        Prepare PCA dimensionality reduction for latent representations.
        
        Args:
            latent_representations (torch.Tensor): Latent vectors
            n_components (Optional[int]): Number of PCA components
            
        Returns:
            Tuple[np.ndarray, PCA]: Transformed data and fitted PCA object
        """
        if n_components is None:
            n_components = min(50, latent_representations.size(1))  # Default to 50 or latent_dim
        
        latent_np = latent_representations.detach().cpu().numpy()
        
        pca = PCA(n_components=n_components)
        latent_pca = pca.fit_transform(latent_np)
        
        return latent_pca, pca
    
    @staticmethod
    def prepare_tsne(latent_representations: torch.Tensor,
                     n_components: int = 2,
                     perplexity: float = 30.0,
                     max_iter: int = 1000,
                     random_state: int = 42) -> np.ndarray:
        """
        Prepare t-SNE dimensionality reduction for latent representations.
        
        Args:
            latent_representations (torch.Tensor): Latent vectors
            n_components (int): Number of t-SNE components (typically 2 or 3)
            perplexity (float): t-SNE perplexity parameter
            max_iter (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: t-SNE transformed data
        """
        latent_np = latent_representations.detach().cpu().numpy()
        
        # For large datasets, we might want to subsample
        if latent_np.shape[0] > 10000:
            warnings.warn("Large dataset detected. Consider subsampling for t-SNE performance.")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        latent_tsne = tsne.fit_transform(latent_np)
        
        return latent_tsne


def calculate_reconstruction_errors(original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive reconstruction errors.
    
    Args:
        original (torch.Tensor): Original data
        reconstructed (torch.Tensor): Reconstructed data
        
    Returns:
        Dict[str, float]: Dictionary of reconstruction errors
    """
    metrics = ReconstructionMetrics()
    
    errors = {
        'mse': metrics.mse(original, reconstructed, per_sample=False),
        'mae': metrics.mae(original, reconstructed, per_sample=False),
        'rmse': metrics.rmse(original, reconstructed, per_sample=False),
        'ssim_approx': metrics.ssim_approximation(original, reconstructed, per_sample=False)
    }
    
    return errors


def calculate_per_sample_errors(original: torch.Tensor,
                               reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculate per-sample reconstruction errors.
    
    Args:
        original (torch.Tensor): Original data
        reconstructed (torch.Tensor): Reconstructed data
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary of per-sample errors
    """
    metrics = ReconstructionMetrics()
    
    errors = {
        'mse_per_sample': metrics.mse(original, reconstructed, per_sample=True),
        'mae_per_sample': metrics.mae(original, reconstructed, per_sample=True),
        'rmse_per_sample': metrics.rmse(original, reconstructed, per_sample=True),
        'ssim_per_sample': metrics.ssim_approximation(original, reconstructed, per_sample=True)
    }
    
    return errors


def calculate_aggregate_errors(per_sample_errors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Calculate aggregate statistics from per-sample errors.
    
    Args:
        per_sample_errors (Dict[str, torch.Tensor]): Per-sample error tensors
        
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary of aggregate statistics
    """
    aggregate_stats = {}
    
    for metric_name, errors in per_sample_errors.items():
        errors_np = errors.detach().cpu().numpy()
        
        aggregate_stats[metric_name] = {
            'mean': float(np.mean(errors_np)),
            'std': float(np.std(errors_np)),
            'min': float(np.min(errors_np)),
            'max': float(np.max(errors_np)),
            'median': float(np.median(errors_np)),
            'q25': float(np.percentile(errors_np, 25)),
            'q75': float(np.percentile(errors_np, 75)),
            'q95': float(np.percentile(errors_np, 95)),
            'q99': float(np.percentile(errors_np, 99))
        }
    
    return aggregate_stats


def compute_latent_statistics(latent_representations: torch.Tensor) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute comprehensive latent space statistics.
    
    Args:
        latent_representations (torch.Tensor): Latent vectors
        
    Returns:
        Dict[str, Union[float, np.ndarray]]: Latent space statistics
    """
    analyzer = LatentSpaceAnalyzer()
    return analyzer.compute_statistics(latent_representations)


def prepare_latent_visualization(latent_representations: torch.Tensor,
                                labels: Optional[torch.Tensor] = None,
                                use_pca: bool = True,
                                use_tsne: bool = True,
                                pca_components: int = 50,
                                tsne_components: int = 2) -> Dict[str, Union[np.ndarray, object]]:
    """
    Prepare latent representations for visualization.
    
    Args:
        latent_representations (torch.Tensor): Latent vectors
        labels (Optional[torch.Tensor]): Optional labels for coloring
        use_pca (bool): Whether to compute PCA
        use_tsne (bool): Whether to compute t-SNE
        pca_components (int): Number of PCA components
        tsne_components (int): Number of t-SNE components
        
    Returns:
        Dict[str, Union[np.ndarray, object]]: Visualization data and fitted objects
    """
    analyzer = LatentSpaceAnalyzer()
    viz_data = {}
    
    # Add original representations
    viz_data['original'] = latent_representations.detach().cpu().numpy()
    
    # Add labels if provided
    if labels is not None:
        viz_data['labels'] = labels.detach().cpu().numpy()
    
    # PCA
    if use_pca:
        pca_data, pca_obj = analyzer.prepare_pca(latent_representations, pca_components)
        viz_data['pca_data'] = pca_data
        viz_data['pca_object'] = pca_obj
        viz_data['pca_explained_variance_ratio'] = pca_obj.explained_variance_ratio_
        viz_data['pca_cumulative_variance'] = np.cumsum(pca_obj.explained_variance_ratio_)
    
    # t-SNE
    if use_tsne:
        # For efficiency, use PCA-reduced data for t-SNE if available and high-dimensional
        if use_pca and latent_representations.size(1) > 50:
            tsne_input = viz_data['pca_data']
        else:
            tsne_input = latent_representations
            
        tsne_data = analyzer.prepare_tsne(
            tsne_input if isinstance(tsne_input, torch.Tensor) else torch.from_numpy(tsne_input),
            n_components=tsne_components
        )
        viz_data['tsne_data'] = tsne_data
    
    return viz_data
