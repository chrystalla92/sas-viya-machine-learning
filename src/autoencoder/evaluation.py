"""
Evaluation metrics and analysis functions for MLP Autoencoder.

This module implements comprehensive evaluation capabilities including:
- Standard image reconstruction metrics (MSE, PSNR, SSIM)
- Latent space analysis and dimensionality reduction
- Quality assessment functions combining multiple metrics
- Batch evaluation utilities for efficient processing
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim_skimage
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def mse_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error between original and reconstructed images.
    
    Args:
        original: Original images tensor of shape (batch_size, features)
        reconstructed: Reconstructed images tensor of same shape
        
    Returns:
        MSE value as float
        
    Raises:
        ValueError: If tensor shapes don't match
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    mse = F.mse_loss(original, reconstructed, reduction='mean')
    return mse.item()


def psnr_metric(original: torch.Tensor, reconstructed: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between original and reconstructed images.
    
    PSNR is defined as: 20 * log10(MAX_VAL / sqrt(MSE))
    Higher values indicate better reconstruction quality.
    
    Args:
        original: Original images tensor of shape (batch_size, features)
        reconstructed: Reconstructed images tensor of same shape
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)
        
    Returns:
        PSNR value in dB
        
    Raises:
        ValueError: If tensor shapes don't match or MSE is zero
    """
    mse = mse_loss(original, reconstructed)
    
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr


def ssim_metric(original: torch.Tensor, reconstructed: torch.Tensor, 
                win_size: int = 7, data_range: float = 1.0) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between original and reconstructed images.
    
    SSIM considers luminance, contrast, and structure to provide perceptually relevant
    quality assessment. Values range from -1 to 1, where 1 indicates perfect similarity.
    
    Args:
        original: Original images tensor of shape (batch_size, 784) for MNIST
        reconstructed: Reconstructed images tensor of same shape
        win_size: Size of sliding window (default: 7)
        data_range: Dynamic range of the images (default: 1.0)
        
    Returns:
        Average SSIM value across batch
        
    Raises:
        ValueError: If tensor shapes don't match
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    batch_size = original.shape[0]
    
    # Reshape to image format (assuming 28x28 MNIST images)
    if original.shape[1] == 784:
        img_size = 28
        original_imgs = original.view(batch_size, img_size, img_size).cpu().numpy()
        reconstructed_imgs = reconstructed.view(batch_size, img_size, img_size).cpu().numpy()
    else:
        # Handle different image sizes
        img_size = int(math.sqrt(original.shape[1]))
        if img_size * img_size != original.shape[1]:
            raise ValueError(f"Cannot reshape {original.shape[1]} features into square image")
        original_imgs = original.view(batch_size, img_size, img_size).cpu().numpy()
        reconstructed_imgs = reconstructed.view(batch_size, img_size, img_size).cpu().numpy()
    
    ssim_values = []
    for i in range(batch_size):
        ssim_val = ssim_skimage(
            original_imgs[i], reconstructed_imgs[i],
            win_size=win_size, data_range=data_range
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def comprehensive_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """
    Calculate comprehensive reconstruction metrics.
    
    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor
        
    Returns:
        Dictionary containing MSE, PSNR, and SSIM metrics
    """
    return {
        'mse': mse_loss(original, reconstructed),
        'psnr': psnr_metric(original, reconstructed),
        'ssim': ssim_metric(original, reconstructed)
    }


def quality_score(metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate a combined quality score from multiple metrics.
    
    The score combines PSNR and SSIM (higher is better) while penalizing MSE (lower is better).
    
    Args:
        metrics: Dictionary containing 'mse', 'psnr', and 'ssim' values
        weights: Optional weights for each metric (default: equal weighting)
        
    Returns:
        Combined quality score (0-1, higher is better)
    """
    if weights is None:
        weights = {'mse': 0.33, 'psnr': 0.33, 'ssim': 0.34}
    
    # Normalize metrics to 0-1 scale
    # MSE: lower is better, so invert
    mse_score = 1.0 / (1.0 + metrics['mse'])
    
    # PSNR: normalize to 0-1 (typical range 0-50 dB)
    psnr_score = min(metrics['psnr'] / 50.0, 1.0) if metrics['psnr'] != float('inf') else 1.0
    
    # SSIM: already in range [-1, 1], normalize to [0, 1]
    ssim_score = (metrics['ssim'] + 1.0) / 2.0
    
    # Weighted combination
    quality = (weights['mse'] * mse_score + 
              weights['psnr'] * psnr_score + 
              weights['ssim'] * ssim_score)
    
    return quality


def latent_analysis(model, data_loader, device: Optional[torch.device] = None, 
                   max_samples: int = 1000) -> Dict[str, Any]:
    """
    Perform latent space analysis using dimensionality reduction.
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader containing input data
        device: Device to run inference on
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary containing latent representations and reduced projections
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    latent_representations = []
    original_data = []
    labels = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                data, batch_labels = batch_data
            else:
                data = batch_data
                batch_labels = None
            
            data = data.to(device)
            
            # Flatten if needed
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            
            # Get latent representation
            latent = model.encode(data)
            
            latent_representations.append(latent.cpu().numpy())
            original_data.append(data.cpu().numpy())
            
            if batch_labels is not None:
                labels.append(batch_labels.numpy())
            
            sample_count += data.size(0)
            if sample_count >= max_samples:
                break
    
    # Concatenate all batches
    latent_matrix = np.vstack(latent_representations)
    original_matrix = np.vstack(original_data)
    
    if labels:
        labels_array = np.hstack(labels)
    else:
        labels_array = None
    
    # Truncate to max_samples
    latent_matrix = latent_matrix[:max_samples]
    original_matrix = original_matrix[:max_samples]
    if labels_array is not None:
        labels_array = labels_array[:max_samples]
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latent_matrix)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(latent_matrix)
    
    return {
        'latent_representations': latent_matrix,
        'original_data': original_matrix,
        'labels': labels_array,
        'pca_projection': pca_result,
        'tsne_projection': tsne_result,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'latent_dim': latent_matrix.shape[1],
        'num_samples': latent_matrix.shape[0]
    }


def reconstruction_error_map(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculate pixel-wise reconstruction error for error heatmap visualization.
    
    Args:
        original: Original images tensor of shape (batch_size, features)
        reconstructed: Reconstructed images tensor of same shape
        
    Returns:
        Error map tensor of same shape as input
    """
    error_map = torch.abs(original - reconstructed)
    return error_map


def evaluate_model_comprehensive(model, test_loader, device: Optional[torch.device] = None,
                               max_batches: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation on test dataset.
    
    Args:
        model: Trained autoencoder model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None for all)
        
    Returns:
        Comprehensive evaluation results
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    all_metrics = []
    total_samples = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, (list, tuple)):
                data, _ = batch_data
            else:
                data = batch_data
            
            data = data.to(device)
            
            # Flatten if needed
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            
            # Get reconstruction
            reconstructed = model(data)
            
            # Calculate metrics for this batch
            batch_metrics = comprehensive_metrics(data, reconstructed)
            all_metrics.append(batch_metrics)
            
            total_samples += data.size(0)
            batch_count += 1
            
            if max_batches is not None and batch_count >= max_batches:
                break
    
    # Calculate aggregate metrics
    avg_metrics = {}
    for metric in ['mse', 'psnr', 'ssim']:
        values = [m[metric] for m in all_metrics]
        avg_metrics[f'avg_{metric}'] = np.mean(values)
        avg_metrics[f'std_{metric}'] = np.std(values)
        avg_metrics[f'min_{metric}'] = np.min(values)
        avg_metrics[f'max_{metric}'] = np.max(values)
    
    # Calculate overall quality score
    overall_quality = quality_score({
        'mse': avg_metrics['avg_mse'],
        'psnr': avg_metrics['avg_psnr'],
        'ssim': avg_metrics['avg_ssim']
    })
    
    return {
        'metrics': avg_metrics,
        'overall_quality': overall_quality,
        'total_samples': total_samples,
        'num_batches': batch_count,
        'detailed_metrics': all_metrics
    }


def compare_models(models: List, test_loader, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Compare multiple trained models using comprehensive metrics.
    
    Args:
        models: List of trained autoencoder models
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Comparison results for all models
    """
    results = {}
    
    for i, model in enumerate(models):
        model_name = f"model_{i+1}"
        results[model_name] = evaluate_model_comprehensive(model, test_loader, device)
    
    # Find best model based on overall quality
    best_model_name = max(results.keys(), 
                         key=lambda k: results[k]['overall_quality'])
    
    results['best_model'] = best_model_name
    results['comparison_summary'] = {
        name: {
            'quality_score': results[name]['overall_quality'],
            'avg_mse': results[name]['metrics']['avg_mse'],
            'avg_psnr': results[name]['metrics']['avg_psnr'],
            'avg_ssim': results[name]['metrics']['avg_ssim']
        }
        for name in results if name not in ['best_model', 'comparison_summary']
    }
    
    return results
