"""
Autoencoder Inference Pipeline for Model Scoring and Reconstruction

This module provides comprehensive inference capabilities for trained autoencoders,
including batch processing, checkpoint loading, tensor preprocessing, and device handling
to match SAS Viya scoring specifications.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

# Handle imports for both package and standalone execution
try:
    # Try relative imports first (when run as package)
    from .models.autoencoder import Autoencoder, create_autoencoder
    from .data.preprocessing import MidrangeScaler, validate_tensor_shape, validate_pixel_range, unflatten_images
    from .utils.device_utils import get_device
except ImportError:
    # Fall back to absolute imports (when run as standalone script)
    import sys
    sys.path.append(str(Path(__file__).parent))
    from models.autoencoder import Autoencoder, create_autoencoder
    from data.preprocessing import MidrangeScaler, validate_tensor_shape, validate_pixel_range, unflatten_images
    from utils.device_utils import get_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoencoderInference:
    """
    High-performance inference pipeline for trained autoencoder models.
    
    This class provides comprehensive scoring capabilities including:
    - Loading trained models from checkpoints
    - Batch processing with configurable sizes
    - Tensor preprocessing with midrange scaling
    - Device-aware operations (CPU/GPU)
    - Image format conversions (784 ↔ 28×28)
    - Input validation and error handling
    - Performance metrics and logging
    """
    
    def __init__(
        self,
        model: Optional[Autoencoder] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 512,
        preprocessing: str = 'midrange',
        validate_input: bool = True,
        log_performance: bool = True
    ):
        """
        Initialize the autoencoder inference pipeline.
        
        Args:
            model: Pre-loaded Autoencoder model (optional)
            checkpoint_path: Path to model checkpoint file (optional)
            device: Device for inference (auto-detect if None)
            batch_size: Default batch size for processing
            preprocessing: Preprocessing method ('midrange', 'none')
            validate_input: Whether to validate input tensors
            log_performance: Whether to log performance metrics
        """
        self.device = device or get_device(prefer_cuda=True)
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.validate_input = validate_input
        self.log_performance = log_performance
        
        # Initialize preprocessing scaler
        self.scaler = MidrangeScaler() if preprocessing == 'midrange' else None
        
        # Model and checkpoint info
        self.model = None
        self.checkpoint_info = {}
        self.model_loaded = False
        
        # Performance tracking
        self.inference_stats = {
            'total_samples_processed': 0,
            'total_inference_time': 0.0,
            'batch_count': 0,
            'average_batch_time': 0.0,
            'throughput_samples_per_sec': 0.0
        }
        
        # Load model
        if model is not None:
            self.load_model(model)
        elif checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        logger.info(f"AutoencoderInference initialized:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Default batch size: {self.batch_size}")
        logger.info(f"  - Preprocessing: {self.preprocessing}")
        logger.info(f"  - Model loaded: {self.model_loaded}")
    
    def load_model(self, model: Autoencoder) -> None:
        """
        Load a pre-initialized autoencoder model.
        
        Args:
            model: Autoencoder model to load
        """
        if not isinstance(model, Autoencoder):
            raise TypeError(f"Expected Autoencoder instance, got {type(model)}")
        
        self.model = model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        
        logger.info("Pre-initialized model loaded successfully")
        logger.info(f"  - Architecture: {model.input_dim}→{model.hidden_dim}→{model.input_dim}")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load trained model from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Validate checkpoint format
            required_keys = ['model_state_dict', 'loss', 'iteration']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise RuntimeError(f"Invalid checkpoint format. Missing keys: {missing_keys}")
            
            # Create model architecture (using standard dimensions)
            # We can infer dimensions from the state dict
            state_dict = checkpoint['model_state_dict']
            input_dim = state_dict['encoder.weight'].shape[1]  # Input features
            hidden_dim = state_dict['encoder.weight'].shape[0]  # Hidden neurons
            
            # Create and load model
            self.model = Autoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seed=checkpoint.get('seed', 23451)
            ).to(self.device)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            
            # Store checkpoint info
            self.checkpoint_info = {
                'path': str(checkpoint_path),
                'iteration': checkpoint['iteration'],
                'loss': checkpoint['loss'],
                'best_loss': checkpoint.get('best_loss', checkpoint['loss']),
                'converged': checkpoint.get('converged', False),
                'training_complete': checkpoint.get('training_complete', False),
                'seed': checkpoint.get('seed', 23451),
                'architecture': f"{input_dim}→{hidden_dim}→{input_dim}"
            }
            
            logger.info(f"Checkpoint loaded successfully from: {checkpoint_path}")
            logger.info(f"  - Architecture: {self.checkpoint_info['architecture']}")
            logger.info(f"  - Training iteration: {self.checkpoint_info['iteration']}")
            logger.info(f"  - Final loss: {self.checkpoint_info['loss']:.6e}")
            logger.info(f"  - Converged: {self.checkpoint_info['converged']}")
            
            return self.checkpoint_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
    
    def _preprocess_input(self, data: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input data for inference.
        
        Args:
            data: Input tensor to preprocess
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        processed_data = data.clone()
        
        # Ensure data is on correct device
        processed_data = processed_data.to(self.device)
        
        # Apply preprocessing if specified
        if self.preprocessing == 'midrange' and self.scaler is not None:
            processed_data = self.scaler.transform(processed_data)
        
        # Validate preprocessing results
        if self.validate_input and self.preprocessing == 'midrange':
            try:
                validate_pixel_range(processed_data, expected_range=(-1.0, 1.0))
            except ValueError as e:
                logger.warning(f"Preprocessing validation warning: {e}")
        
        return processed_data
    
    def _postprocess_output(self, data: torch.Tensor) -> torch.Tensor:
        """
        Postprocess model output (reverse preprocessing if needed).
        
        Args:
            data: Model output tensor
            
        Returns:
            Postprocessed tensor in original scale
        """
        if self.preprocessing == 'midrange' and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
    
    def _validate_input_tensor(self, data: torch.Tensor, expected_features: int = 784) -> None:
        """
        Validate input tensor format and dimensions.
        
        Args:
            data: Input tensor to validate
            expected_features: Expected number of features (default: 784 for MNIST)
            
        Raises:
            ValueError: If tensor format is invalid
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")
        
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D tensor (N, features), got shape {data.shape}")
        
        if data.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {data.shape[1]}. "
                f"Use tensor_to_flat() to reshape image data to ({expected_features},)"
            )
        
        # Check for NaN or infinite values
        if torch.isnan(data).any():
            raise ValueError("Input tensor contains NaN values")
        
        if torch.isinf(data).any():
            raise ValueError("Input tensor contains infinite values")
    
    def predict_single_batch(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on a single batch of data.
        
        Args:
            data: Input tensor of shape (batch_size, 784)
            
        Returns:
            Reconstructed tensor of shape (batch_size, 784)
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Use load_model() or load_checkpoint() first.")
        
        # Validate input
        if self.validate_input:
            self._validate_input_tensor(data)
        
        # Preprocess input
        processed_input = self._preprocess_input(data)
        
        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            reconstructed = self.model(processed_input)
        inference_time = time.time() - start_time
        
        # Postprocess output
        output = self._postprocess_output(reconstructed)
        
        # Update performance stats
        if self.log_performance:
            batch_size = data.shape[0]
            self.inference_stats['total_samples_processed'] += batch_size
            self.inference_stats['total_inference_time'] += inference_time
            self.inference_stats['batch_count'] += 1
            self.inference_stats['average_batch_time'] = (
                self.inference_stats['total_inference_time'] / self.inference_stats['batch_count']
            )
            self.inference_stats['throughput_samples_per_sec'] = (
                self.inference_stats['total_samples_processed'] / self.inference_stats['total_inference_time']
            )
        
        return output
    
    def predict_batched(
        self,
        data: torch.Tensor,
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Perform inference on data using batch processing.
        
        Args:
            data: Input tensor of shape (N, 784)
            batch_size: Batch size for processing (uses default if None)
            show_progress: Whether to show progress logging
            
        Returns:
            Reconstructed tensor of shape (N, 784)
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Use load_model() or load_checkpoint() first.")
        
        batch_size = batch_size or self.batch_size
        num_samples = data.shape[0]
        
        if show_progress:
            logger.info(f"Starting batch inference on {num_samples} samples")
            logger.info(f"  - Batch size: {batch_size}")
            logger.info(f"  - Total batches: {(num_samples + batch_size - 1) // batch_size}")
        
        # Process in batches
        outputs = []
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_data = data[i:batch_end]
            
            batch_output = self.predict_single_batch(batch_data)
            outputs.append(batch_output)
            
            if show_progress and (i // batch_size) % 10 == 0:
                logger.info(f"  - Processed batch {i // batch_size + 1}, samples {i+1}-{batch_end}")
        
        # Concatenate all outputs
        result = torch.cat(outputs, dim=0)
        
        total_time = time.time() - start_time
        if show_progress:
            logger.info(f"Batch inference completed in {total_time:.2f}s")
            logger.info(f"  - Throughput: {num_samples / total_time:.1f} samples/sec")
        
        return result
    
    def predict(
        self,
        data: torch.Tensor,
        batch_size: Optional[int] = None,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Main prediction interface supporting both single and batch processing.
        
        Args:
            data: Input tensor of shape (N, 784) or (784,) for single sample
            batch_size: Batch size (auto-determine if None)
            return_latent: Whether to also return latent representations
            
        Returns:
            Reconstructed tensor, optionally with latent representations
        """
        # Handle single sample case
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Determine processing method
        num_samples = data.shape[0]
        batch_size = batch_size or self.batch_size
        
        if num_samples <= batch_size:
            # Process as single batch
            reconstructed = self.predict_single_batch(data)
        else:
            # Use batch processing
            reconstructed = self.predict_batched(data, batch_size=batch_size)
        
        # Handle latent representation if requested
        if return_latent:
            # Process latent in same batching pattern
            with torch.no_grad():
                if num_samples <= batch_size:
                    processed_input = self._preprocess_input(data)
                    latent = self.model.encode(processed_input)
                else:
                    latent_outputs = []
                    for i in range(0, num_samples, batch_size):
                        batch_end = min(i + batch_size, num_samples)
                        batch_data = data[i:batch_end]
                        processed_batch = self._preprocess_input(batch_data)
                        batch_latent = self.model.encode(processed_batch)
                        latent_outputs.append(batch_latent)
                    latent = torch.cat(latent_outputs, dim=0)
        
        # Handle single sample return format
        if single_sample:
            reconstructed = reconstructed.squeeze(0)
            if return_latent:
                latent = latent.squeeze(0)
        
        if return_latent:
            return reconstructed, latent
        return reconstructed
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive inference performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.inference_stats.copy()
        
        # Add model information if available
        if self.model_loaded:
            stats['model_info'] = {
                'architecture': f"{self.model.input_dim}→{self.model.hidden_dim}→{self.model.input_dim}",
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.model.get_device())
            }
        
        # Add checkpoint info if available
        if self.checkpoint_info:
            stats['checkpoint_info'] = self.checkpoint_info
        
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset all performance tracking statistics."""
        self.inference_stats = {
            'total_samples_processed': 0,
            'total_inference_time': 0.0,
            'batch_count': 0,
            'average_batch_time': 0.0,
            'throughput_samples_per_sec': 0.0
        }
        logger.info("Performance statistics reset")


# Utility functions for tensor/image conversions
def tensor_to_flat(images: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensors to flattened format for autoencoder input.
    
    Args:
        images: Image tensor of shape (N, H, W) or (N, C, H, W)
        
    Returns:
        Flattened tensor of shape (N, H*W) or (N, C*H*W)
    """
    if len(images.shape) < 2:
        raise ValueError(f"Expected at least 2D tensor, got shape {images.shape}")
    
    batch_size = images.shape[0]
    return images.view(batch_size, -1)


def flat_to_images(
    flattened: torch.Tensor,
    image_shape: Tuple[int, ...] = (28, 28)
) -> torch.Tensor:
    """
    Convert flattened tensors back to image format.
    
    Args:
        flattened: Flattened tensor of shape (N, features)
        image_shape: Target image shape (H, W) or (C, H, W)
        
    Returns:
        Reshaped tensor with image dimensions
    """
    if len(flattened.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {flattened.shape}")
    
    batch_size = flattened.shape[0]
    expected_features = np.prod(image_shape)
    
    if flattened.shape[1] != expected_features:
        raise ValueError(
            f"Flattened tensor has {flattened.shape[1]} features, "
            f"but image shape {image_shape} requires {expected_features}"
        )
    
    return flattened.view(batch_size, *image_shape)


# Factory function for easy initialization
def create_inference_pipeline(
    checkpoint_path: Optional[str] = None,
    model: Optional[Autoencoder] = None,
    device: Optional[str] = None,
    **kwargs
) -> AutoencoderInference:
    """
    Factory function to create an autoencoder inference pipeline.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        model: Pre-loaded model (alternative to checkpoint_path)
        device: Device for inference ('cpu', 'cuda', or None for auto)
        **kwargs: Additional arguments for AutoencoderInference
        
    Returns:
        Initialized AutoencoderInference pipeline
    """
    if device is not None:
        device = torch.device(device)
    
    pipeline = AutoencoderInference(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        **kwargs
    )
    
    logger.info(f"Created inference pipeline with {pipeline.inference_stats['total_samples_processed']} samples processed so far")
    
    return pipeline
