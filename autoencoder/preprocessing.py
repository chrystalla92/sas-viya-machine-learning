"""
Preprocessing and Postprocessing Pipeline

This module provides comprehensive input/output processing utilities including
standardization, reshaping, denormalization, and visualization preparation to
ensure data compatibility and numerical precision throughout the pipeline.

Key features:
- Input preprocessing pipeline with standardization and reshaping
- Output postprocessing for denormalization and visualization preparation
- Numerical precision preservation and validation
- Compatibility with SAS midrange standardization
- Data transformation utilities for different input/output formats

Replaces SAS functionality:
- Input data standardization (midrange standardization)
- Output data denormalization for visualization
- Data format conversion and reshaping
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingParams:
    """
    Data class to store preprocessing parameters for reproducible transformations.
    """
    method: str  # 'midrange', 'standard', 'minmax', 'none'
    midrange: Optional[np.ndarray] = None
    range_vals: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method,
            'midrange': self.midrange.tolist() if self.midrange is not None else None,
            'range_vals': self.range_vals.tolist() if self.range_vals is not None else None,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'min_vals': self.min_vals.tolist() if self.min_vals is not None else None,
            'max_vals': self.max_vals.tolist() if self.max_vals is not None else None,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingParams':
        """Create from dictionary."""
        params = cls(method=data['method'])
        
        # Convert lists back to numpy arrays
        for attr in ['midrange', 'range_vals', 'mean', 'std', 'min_vals', 'max_vals']:
            value = data.get(attr)
            if value is not None:
                setattr(params, attr, np.array(value))
        
        params.input_shape = data.get('input_shape')
        params.output_shape = data.get('output_shape')
        
        return params


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.params = None
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BasePreprocessor':
        """Fit preprocessor to data."""
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform preprocessed data."""
        pass
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)


class MidrangeStandardizer(BasePreprocessor):
    """
    Midrange standardization to match SAS behavior exactly.
    
    Formula: standardized_x = (x - midrange) / range
    where:
    - midrange = (max + min) / 2
    - range = max - min
    """
    
    def __init__(self):
        super().__init__()
    
    def fit(self, data: np.ndarray) -> 'MidrangeStandardizer':
        """
        Fit midrange standardization parameters.
        
        Args:
            data: Input data array (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if len(data.shape) != 2:
            raise ValueError("Data must be 2D array (n_samples, n_features)")
        
        # Calculate parameters per feature (column-wise)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        midrange = (max_vals + min_vals) / 2.0
        range_vals = max_vals - min_vals
        
        # Handle constant features (range = 0)
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        self.params = PreprocessingParams(
            method='midrange',
            midrange=midrange,
            range_vals=range_vals,
            min_vals=min_vals,
            max_vals=max_vals,
            input_shape=data.shape,
            output_shape=data.shape
        )
        
        self.is_fitted = True
        logger.info(f"Fitted midrange standardizer on data shape {data.shape}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Input data to transform
            
        Returns:
            Standardized data
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if data.shape[1] != self.params.midrange.shape[0]:
            raise ValueError(f"Feature dimension mismatch: expected {self.params.midrange.shape[0]}, got {data.shape[1]}")
        
        standardized = (data - self.params.midrange) / self.params.range_vals
        
        return standardized
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform standardized data back to original scale.
        
        Args:
            data: Standardized data to inverse transform
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        original = data * self.params.range_vals + self.params.midrange
        
        return original


class StandardScaler(BasePreprocessor):
    """
    Standard (z-score) normalization.
    
    Formula: standardized_x = (x - mean) / std
    """
    
    def __init__(self):
        super().__init__()
    
    def fit(self, data: np.ndarray) -> 'StandardScaler':
        """Fit standard scaler parameters."""
        if len(data.shape) != 2:
            raise ValueError("Data must be 2D array (n_samples, n_features)")
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Handle constant features
        std = np.where(std == 0, 1.0, std)
        
        self.params = PreprocessingParams(
            method='standard',
            mean=mean,
            std=std,
            input_shape=data.shape,
            output_shape=data.shape
        )
        
        self.is_fitted = True
        logger.info(f"Fitted standard scaler on data shape {data.shape}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using z-score normalization."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        standardized = (data - self.params.mean) / self.params.std
        return standardized
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform standardized data."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        original = data * self.params.std + self.params.mean
        return original


class MinMaxScaler(BasePreprocessor):
    """
    Min-max normalization to [0, 1] range.
    
    Formula: normalized_x = (x - min) / (max - min)
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.feature_range = feature_range
    
    def fit(self, data: np.ndarray) -> 'MinMaxScaler':
        """Fit min-max scaler parameters."""
        if len(data.shape) != 2:
            raise ValueError("Data must be 2D array (n_samples, n_features)")
        
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Handle constant features
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        self.params = PreprocessingParams(
            method='minmax',
            min_vals=min_vals,
            max_vals=max_vals,
            range_vals=range_vals,
            input_shape=data.shape,
            output_shape=data.shape
        )
        
        self.is_fitted = True
        logger.info(f"Fitted min-max scaler on data shape {data.shape}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to specified range."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Scale to [0, 1]
        normalized = (data - self.params.min_vals) / self.params.range_vals
        
        # Scale to desired range
        min_range, max_range = self.feature_range
        scaled = normalized * (max_range - min_range) + min_range
        
        return scaled
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        # Scale back from desired range to [0, 1]
        min_range, max_range = self.feature_range
        normalized = (data - min_range) / (max_range - min_range)
        
        # Scale back to original range
        original = normalized * self.params.range_vals + self.params.min_vals
        
        return original


class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline with multiple transformations.
    """
    
    def __init__(self, standardization_method: str = 'midrange'):
        """
        Initialize preprocessing pipeline.
        
        Args:
            standardization_method: Method for standardization ('midrange', 'standard', 'minmax', 'none')
        """
        self.standardization_method = standardization_method
        self.preprocessor = None
        self.is_fitted = False
        
        # Create appropriate preprocessor
        if standardization_method == 'midrange':
            self.preprocessor = MidrangeStandardizer()
        elif standardization_method == 'standard':
            self.preprocessor = StandardScaler()
        elif standardization_method == 'minmax':
            self.preprocessor = MinMaxScaler()
        elif standardization_method == 'none':
            self.preprocessor = None
        else:
            raise ValueError(f"Unknown standardization method: {standardization_method}")
    
    def fit(self, data: np.ndarray) -> 'PreprocessingPipeline':
        """
        Fit preprocessing pipeline to data.
        
        Args:
            data: Training data to fit on
            
        Returns:
            Self for method chaining
        """
        if self.preprocessor is not None:
            self.preprocessor.fit(data)
        
        self.is_fitted = True
        logger.info(f"Fitted preprocessing pipeline with {self.standardization_method} standardization")
        
        return self
    
    def transform(self, data: np.ndarray, 
                  target_format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data through preprocessing pipeline.
        
        Args:
            data: Input data to transform
            target_format: Output format ('numpy' or 'torch')
            
        Returns:
            Transformed data in specified format
        """
        if not self.is_fitted and self.preprocessor is not None:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # Apply standardization if configured
        if self.preprocessor is not None:
            processed_data = self.preprocessor.transform(data)
        else:
            processed_data = data.copy()
        
        # Convert to target format
        if target_format == 'torch':
            return torch.FloatTensor(processed_data)
        elif target_format == 'numpy':
            return processed_data
        else:
            raise ValueError(f"Unknown target format: {target_format}")
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Inverse transform data back to original scale.
        
        Args:
            data: Processed data to inverse transform
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted and self.preprocessor is not None:
            raise RuntimeError("Pipeline must be fitted before inverse_transform")
        
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Apply inverse standardization if configured
        if self.preprocessor is not None:
            original_data = self.preprocessor.inverse_transform(data)
        else:
            original_data = data.copy()
        
        return original_data
    
    def fit_transform(self, data: np.ndarray, 
                     target_format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data, target_format)
    
    def get_params(self) -> Optional[PreprocessingParams]:
        """Get preprocessing parameters."""
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            return self.preprocessor.params
        return None


class VisualizationPreprocessor:
    """
    Utilities for preparing data for visualization.
    """
    
    @staticmethod
    def prepare_for_visualization(data: Union[np.ndarray, torch.Tensor],
                                data_range: Tuple[float, float] = (0.0, 1.0),
                                image_shape: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Prepare data for visualization by reshaping and scaling.
        
        Args:
            data: Flattened image data (n_samples, n_pixels)
            data_range: Target range for pixel values
            image_shape: Shape to reshape images to (height, width)
            
        Returns:
            Reshaped and scaled data for visualization
        """
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Ensure proper shape
        if len(data.shape) != 2:
            raise ValueError("Data must be 2D (n_samples, n_features)")
        
        n_samples = data.shape[0]
        expected_pixels = image_shape[0] * image_shape[1]
        
        if data.shape[1] != expected_pixels:
            raise ValueError(f"Data has {data.shape[1]} features, expected {expected_pixels} for {image_shape} images")
        
        # Scale to visualization range
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            scaled = (data - data_min) / (data_max - data_min)
            scaled = scaled * (data_range[1] - data_range[0]) + data_range[0]
        else:
            scaled = np.full_like(data, data_range[0])
        
        # Reshape for visualization
        visualized = scaled.reshape(n_samples, image_shape[0], image_shape[1])
        
        return visualized
    
    @staticmethod
    def denormalize_for_display(data: Union[np.ndarray, torch.Tensor],
                               preprocessing_params: PreprocessingParams) -> np.ndarray:
        """
        Denormalize data for display using stored preprocessing parameters.
        
        Args:
            data: Normalized data
            preprocessing_params: Parameters from preprocessing step
            
        Returns:
            Denormalized data
        """
        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Apply inverse transformation based on method
        if preprocessing_params.method == 'midrange':
            denormalized = data * preprocessing_params.range_vals + preprocessing_params.midrange
        elif preprocessing_params.method == 'standard':
            denormalized = data * preprocessing_params.std + preprocessing_params.mean
        elif preprocessing_params.method == 'minmax':
            denormalized = data * preprocessing_params.range_vals + preprocessing_params.min_vals
        else:
            denormalized = data.copy()
        
        return denormalized


def create_sas_compatible_preprocessor() -> PreprocessingPipeline:
    """
    Create preprocessor that exactly matches SAS midrange standardization.
    
    Returns:
        PreprocessingPipeline configured for SAS compatibility
    """
    return PreprocessingPipeline(standardization_method='midrange')


def validate_preprocessing(original_data: np.ndarray,
                         preprocessor: BasePreprocessor,
                         tolerance: float = 1e-10) -> Dict[str, Any]:
    """
    Validate that preprocessing maintains numerical precision.
    
    Args:
        original_data: Original input data
        preprocessor: Fitted preprocessor
        tolerance: Numerical tolerance for validation
        
    Returns:
        Validation results
    """
    try:
        # Transform and inverse transform
        transformed = preprocessor.transform(original_data)
        reconstructed = preprocessor.inverse_transform(transformed)
        
        # Calculate differences
        diff = np.abs(original_data - reconstructed)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Check if precision is maintained
        precision_maintained = max_diff <= tolerance
        
        validation_results = {
            'precision_maintained': precision_maintained,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance,
            'original_shape': original_data.shape,
            'transformed_shape': transformed.shape,
            'reconstructed_shape': reconstructed.shape
        }
        
        if not precision_maintained:
            validation_results['warning'] = f"Numerical precision not maintained: max diff {max_diff:.2e} > tolerance {tolerance:.2e}"
        
        return validation_results
        
    except Exception as e:
        return {
            'precision_maintained': False,
            'error': str(e)
        }


def demonstrate_preprocessing():
    """
    Demonstrate preprocessing functionality with sample data.
    """
    print("=== Preprocessing Pipeline Demonstration ===")
    
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.rand(100, 784) * 255  # Simulate MNIST-like data
    
    # Test different preprocessors
    preprocessors = {
        'midrange': MidrangeStandardizer(),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler()
    }
    
    for name, preprocessor in preprocessors.items():
        print(f"\n--- Testing {name} preprocessor ---")
        
        # Fit and transform
        transformed = preprocessor.fit_transform(sample_data)
        
        # Validate
        validation = validate_preprocessing(sample_data, preprocessor)
        
        print(f"Original range: [{np.min(sample_data):.2f}, {np.max(sample_data):.2f}]")
        print(f"Transformed range: [{np.min(transformed):.2f}, {np.max(transformed):.2f}]")
        print(f"Precision maintained: {validation['precision_maintained']}")
        print(f"Max difference: {validation['max_difference']:.2e}")
    
    # Test pipeline
    print(f"\n--- Testing preprocessing pipeline ---")
    pipeline = PreprocessingPipeline('midrange')
    
    # Test numpy output
    numpy_output = pipeline.fit_transform(sample_data, target_format='numpy')
    print(f"NumPy output shape: {numpy_output.shape}")
    
    # Test torch output
    torch_output = pipeline.transform(sample_data, target_format='torch')
    print(f"Torch output shape: {torch_output.shape}")
    print(f"Torch output type: {type(torch_output)}")
    
    # Test inverse transform
    inverse_output = pipeline.inverse_transform(torch_output)
    print(f"Inverse transform max diff: {np.max(np.abs(sample_data - inverse_output)):.2e}")
    
    print("=== Demonstration completed ===")


if __name__ == "__main__":
    demonstrate_preprocessing()
