"""
Transforms and preprocessing utilities for MNIST data.
"""

from typing import Union, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
from torchvision import transforms


class MNISTTransforms:
    """
    Collection of preprocessing transforms for MNIST data.
    
    Provides standardized transforms that match SAS createData.sas output format.
    """
    
    @staticmethod
    def get_autoencoder_transform(
        normalize_method: str = "01",
        add_noise: bool = False,
        noise_factor: float = 0.1
    ) -> transforms.Compose:
        """
        Get standard transform pipeline for autoencoder training.
        
        Args:
            normalize_method: "01" for [0,1], "11" for [-1,1]
            add_noise: Whether to add noise for denoising autoencoder
            noise_factor: Noise intensity (0.0 to 1.0)
            
        Returns:
            Composed transform pipeline
        """
        transform_list = []
        
        # Convert to tensor if not already
        transform_list.append(transforms.Lambda(lambda x: x if torch.is_tensor(x) else transforms.ToTensor()(x)))
        
        # Flatten to 784 features
        transform_list.append(transforms.Lambda(lambda x: x.reshape(-1)))
        
        # Normalization
        if normalize_method == "01":
            transform_list.append(transforms.Lambda(lambda x: x.clamp(0, 1)))
        elif normalize_method == "11":
            transform_list.append(transforms.Lambda(lambda x: x.clamp(-1, 1)))
        else:
            raise ValueError("normalize_method must be '01' or '11'")
        
        # Add noise if requested
        if add_noise:
            transform_list.append(GaussianNoise(noise_factor))
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_validation_transform(normalize_method: str = "01") -> transforms.Compose:
        """
        Get transform pipeline for validation/testing (no noise).
        
        Args:
            normalize_method: "01" for [0,1], "11" for [-1,1]
            
        Returns:
            Composed transform pipeline
        """
        return MNISTTransforms.get_autoencoder_transform(
            normalize_method=normalize_method,
            add_noise=False
        )
    
    @staticmethod
    def get_sas_compatible_transform() -> transforms.Compose:
        """
        Get transform that matches SAS createData.sas output format.
        
        Based on the SAS code analysis:
        - Flattens to 784 features
        - Keeps pixel values in original range [0, 255]
        - No additional normalization
        
        Returns:
            Transform pipeline compatible with SAS output
        """
        return transforms.Compose([
            transforms.Lambda(lambda x: x if torch.is_tensor(x) else transforms.ToTensor()(x)),
            transforms.Lambda(lambda x: x.reshape(-1) * 255.0),  # Flatten and scale back to [0, 255]
            transforms.Lambda(lambda x: x.clamp(0, 255))
        ])


class GaussianNoise:
    """Add Gaussian noise to tensors."""
    
    def __init__(self, noise_factor: float = 0.1):
        """
        Initialize Gaussian noise transform.
        
        Args:
            noise_factor: Standard deviation of noise (relative to data range)
        """
        self.noise_factor = noise_factor
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor with added noise
        """
        noise = torch.randn_like(tensor) * self.noise_factor
        return tensor + noise
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_factor={self.noise_factor})"


class Flatten:
    """Flatten tensor to 1D."""
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Flatten tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Flattened tensor
        """
        return tensor.reshape(-1)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize:
    """Normalize tensor to specified range."""
    
    def __init__(self, method: str = "01"):
        """
        Initialize normalization.
        
        Args:
            method: "01" for [0,1], "11" for [-1,1]
        """
        if method not in ["01", "11"]:
            raise ValueError("method must be '01' or '11'")
        self.method = method
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor.
        
        Args:
            tensor: Input tensor (assumed to be in [0, 255] range)
            
        Returns:
            Normalized tensor
        """
        # First normalize to [0, 1]
        normalized = tensor / 255.0
        
        if self.method == "01":
            return normalized.clamp(0, 1)
        else:  # "11"
            return (normalized * 2.0 - 1.0).clamp(-1, 1)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method})"


class ValidateShape:
    """Validate tensor shape for MNIST data."""
    
    def __init__(self, expected_features: int = 784):
        """
        Initialize shape validator.
        
        Args:
            expected_features: Expected number of features after flattening
        """
        self.expected_features = expected_features
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate and return tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Input tensor if validation passes
            
        Raises:
            ValueError: If shape validation fails
        """
        if tensor.dim() == 1 and tensor.size(0) != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, got {tensor.size(0)}"
            )
        elif tensor.dim() == 2 and tensor.size(-1) != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features in last dimension, "
                f"got {tensor.size(-1)}"
            )
        elif tensor.dim() == 3 and tensor.shape[-2:] != (28, 28):
            raise ValueError(
                f"Expected (28, 28) image dimensions, got {tensor.shape[-2:]}"
            )
        
        return tensor
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(expected_features={self.expected_features})"


class ToFloat32:
    """Convert tensor to float32 dtype."""
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert to float32.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Float32 tensor
        """
        return tensor.float()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Convenience functions for common transform combinations
def create_training_transforms(
    normalize: str = "01",
    add_noise: bool = False,
    noise_factor: float = 0.1,
    validate_shape: bool = True
) -> transforms.Compose:
    """
    Create training transform pipeline.
    
    Args:
        normalize: "01" for [0,1], "11" for [-1,1] 
        add_noise: Whether to add Gaussian noise
        noise_factor: Noise intensity
        validate_shape: Whether to validate tensor shapes
        
    Returns:
        Composed transform pipeline
    """
    transform_list = [
        ToFloat32(),
        Flatten(),
        Normalize(normalize)
    ]
    
    if add_noise:
        transform_list.append(GaussianNoise(noise_factor))
    
    if validate_shape:
        transform_list.append(ValidateShape(784))
    
    return transforms.Compose(transform_list)


def create_evaluation_transforms(
    normalize: str = "01",
    validate_shape: bool = True
) -> transforms.Compose:
    """
    Create evaluation transform pipeline (no noise).
    
    Args:
        normalize: "01" for [0,1], "11" for [-1,1]
        validate_shape: Whether to validate tensor shapes
        
    Returns:
        Composed transform pipeline
    """
    transform_list = [
        ToFloat32(),
        Flatten(),
        Normalize(normalize)
    ]
    
    if validate_shape:
        transform_list.append(ValidateShape(784))
    
    return transforms.Compose(transform_list)


def create_sas_compatible_transforms() -> transforms.Compose:
    """
    Create transforms that match SAS createData.sas output.
    
    Returns pixel values in [0, 255] range as 784 features.
    
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        ToFloat32(),
        Flatten(),
        ValidateShape(784)
    ])
