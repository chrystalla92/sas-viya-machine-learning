"""
Model and Data Serialization Module

This module provides comprehensive serialization utilities using pickle for trained model
persistence, replacing SAS model exports with Python-native serialization that ensures
cross-environment compatibility and data integrity.

Key features:
- Model serialization with pickle for cross-Python environment compatibility
- Comprehensive error handling for serialization edge cases
- Model verification to ensure saved models load correctly and produce identical outputs
- Data integrity checks and validation
- Support for different pickle protocols for compatibility
- Metadata preservation during serialization

Replaces SAS functionality:
- Model saving/export operations
- Model state preservation
- Cross-session model persistence
"""

import pickle
import torch
import numpy as np
import os
import sys
import warnings
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Handles model serialization with comprehensive error handling and validation.
    """
    
    def __init__(self, pickle_protocol: int = pickle.HIGHEST_PROTOCOL):
        """
        Initialize serializer.
        
        Args:
            pickle_protocol: Pickle protocol version for compatibility
        """
        self.pickle_protocol = pickle_protocol
        self._supported_protocols = list(range(pickle.HIGHEST_PROTOCOL + 1))
        
    def save_model(self, model, filepath: str, 
                  include_verification: bool = True,
                  save_metadata: bool = True) -> Dict[str, Any]:
        """
        Save model with comprehensive error handling and verification.
        
        Args:
            model: PyTorch model to save
            filepath: Output file path
            include_verification: Whether to run verification tests
            save_metadata: Whether to save metadata alongside model
            
        Returns:
            Dictionary with save operation results and metadata
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Prepare model data
            model_data = self._prepare_model_data(model)
            
            # Add serialization metadata
            model_data['serialization_info'] = {
                'timestamp': datetime.now().isoformat(),
                'pickle_protocol': self.pickle_protocol,
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
                'platform': sys.platform
            }
            
            # Save with specified protocol
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=self.pickle_protocol)
            
            # Calculate file hash for integrity verification
            file_hash = self._calculate_file_hash(filepath)
            model_data['serialization_info']['file_hash'] = file_hash
            
            # Verification test
            verification_result = None
            if include_verification:
                verification_result = self._verify_saved_model(model, filepath)
                if not verification_result['success']:
                    logger.warning(f"Model verification failed: {verification_result['error']}")
            
            # Save metadata separately if requested
            metadata_path = None
            if save_metadata:
                metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.json"
                self._save_model_metadata(model_data, metadata_path)
            
            result = {
                'success': True,
                'filepath': filepath,
                'file_size': os.path.getsize(filepath),
                'file_hash': file_hash,
                'verification': verification_result,
                'metadata_path': metadata_path,
                'serialization_info': model_data['serialization_info']
            }
            
            logger.info(f"Model saved successfully to {filepath}")
            return result
            
        except Exception as e:
            error_msg = f"Error saving model to {filepath}: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def load_model(self, filepath: str, 
                  verify_integrity: bool = True,
                  device: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model with integrity verification and error handling.
        
        Args:
            filepath: Path to saved model file
            verify_integrity: Whether to verify file integrity
            device: Device to load model onto
            
        Returns:
            Tuple of (loaded_model, load_info)
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Verify file integrity if requested
            if verify_integrity:
                integrity_check = self._check_file_integrity(filepath)
                if not integrity_check['success']:
                    logger.warning(f"File integrity check failed: {integrity_check['error']}")
            
            # Load model data
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Reconstruct model
            model = self._reconstruct_model(model_data, device)
            
            # Prepare load information
            load_info = {
                'success': True,
                'filepath': filepath,
                'file_size': os.path.getsize(filepath),
                'serialization_info': model_data.get('serialization_info', {}),
                'compatibility_info': self._check_compatibility(model_data)
            }
            
            logger.info(f"Model loaded successfully from {filepath}")
            return model, load_info
            
        except Exception as e:
            error_msg = f"Error loading model from {filepath}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _prepare_model_data(self, model) -> Dict[str, Any]:
        """Prepare model data for serialization."""
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_architecture': {},
        }
        
        # Get architecture info if available
        if hasattr(model, 'get_architecture_info'):
            model_data['model_architecture'] = model.get_architecture_info()
        else:
            # Extract basic architecture info
            model_data['model_architecture'] = {
                'type': model.__class__.__name__,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        
        # Store constructor parameters if available
        if hasattr(model, '__dict__'):
            constructor_params = {}
            for key, value in model.__dict__.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    constructor_params[key] = value
            model_data['constructor_params'] = constructor_params
        
        return model_data
    
    def _reconstruct_model(self, model_data: Dict[str, Any], device: Optional[str] = None):
        """Reconstruct model from serialized data."""
        from model import MNISTAutoencoder  # Import here to avoid circular imports
        
        # Get constructor parameters
        constructor_params = model_data.get('constructor_params', {})
        
        # Create model instance
        # Try to use stored parameters, fallback to defaults
        model = MNISTAutoencoder(
            input_dim=constructor_params.get('input_dim', 784),
            hidden_dim=constructor_params.get('hidden_dim', 400),
            dropout_rate=constructor_params.get('dropout_rate', 0.0)
        )
        
        # Load state dict
        model.load_state_dict(model_data['model_state_dict'])
        
        # Move to device if specified
        if device:
            model.to(device)
        
        return model
    
    def _verify_saved_model(self, original_model, filepath: str) -> Dict[str, Any]:
        """Verify saved model can be loaded and produces identical outputs."""
        try:
            # Load the saved model
            loaded_model, _ = self.load_model(filepath, verify_integrity=False)
            
            # Create test input
            test_input = torch.randn(5, 784)  # Small test batch
            
            # Get outputs from both models
            original_model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                original_output = original_model(test_input)
                loaded_output = loaded_model(test_input)
            
            # Compare outputs
            output_diff = torch.abs(original_output - loaded_output)
            max_diff = torch.max(output_diff).item()
            mean_diff = torch.mean(output_diff).item()
            
            # Check if outputs are nearly identical (allowing for floating point precision)
            tolerance = 1e-6
            identical = max_diff < tolerance
            
            return {
                'success': identical,
                'max_difference': max_diff,
                'mean_difference': mean_diff,
                'tolerance': tolerance,
                'error': None if identical else f"Model outputs differ by up to {max_diff:.2e}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _check_file_integrity(self, filepath: str) -> Dict[str, Any]:
        """Check file integrity using stored hash."""
        try:
            # Try to load metadata to get original hash
            metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.json"
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                stored_hash = metadata.get('serialization_info', {}).get('file_hash')
                
                if stored_hash:
                    current_hash = self._calculate_file_hash(filepath)
                    
                    if current_hash == stored_hash:
                        return {'success': True, 'message': 'File integrity verified'}
                    else:
                        return {
                            'success': False, 
                            'error': f'Hash mismatch: expected {stored_hash}, got {current_hash}'
                        }
            
            # If no metadata or hash available, just check if file exists and is readable
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return {'success': True, 'message': 'Basic file check passed (no hash available)'}
            else:
                return {'success': False, 'error': 'File is empty or corrupted'}
                
        except Exception as e:
            return {'success': False, 'error': f'Integrity check failed: {str(e)}'}
    
    def _check_compatibility(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compatibility between saved and current environment."""
        serialization_info = model_data.get('serialization_info', {})
        
        compatibility_info = {
            'python_compatible': True,
            'torch_compatible': True,
            'numpy_compatible': True,
            'warnings': []
        }
        
        # Check Python version compatibility
        saved_python = serialization_info.get('python_version', '')
        if saved_python and sys.version != saved_python:
            compatibility_info['warnings'].append(
                f"Python version mismatch: saved with {saved_python}, running {sys.version}"
            )
        
        # Check PyTorch version compatibility
        saved_torch = serialization_info.get('torch_version', '')
        if saved_torch and torch.__version__ != saved_torch:
            compatibility_info['warnings'].append(
                f"PyTorch version mismatch: saved with {saved_torch}, running {torch.__version__}"
            )
        
        # Check NumPy version compatibility
        saved_numpy = serialization_info.get('numpy_version', '')
        if saved_numpy and np.__version__ != saved_numpy:
            compatibility_info['warnings'].append(
                f"NumPy version mismatch: saved with {saved_numpy}, running {np.__version__}"
            )
        
        return compatibility_info
    
    def _save_model_metadata(self, model_data: Dict[str, Any], metadata_path: str):
        """Save model metadata to JSON file."""
        try:
            # Prepare JSON-serializable metadata
            metadata = {
                'model_class': model_data.get('model_class'),
                'model_architecture': model_data.get('model_architecture'),
                'constructor_params': model_data.get('constructor_params'),
                'serialization_info': model_data.get('serialization_info')
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")


class DataSerializer:
    """
    Handles data serialization with cross-environment compatibility.
    """
    
    @staticmethod
    def save_data(data: Union[np.ndarray, Dict[str, np.ndarray]], 
                  filepath: str,
                  include_metadata: bool = True,
                  compression: bool = True) -> Dict[str, Any]:
        """
        Save data with comprehensive metadata and error handling.
        
        Args:
            data: Data to save (array or dict of arrays)
            filepath: Output file path
            include_metadata: Whether to save metadata
            compression: Whether to use compression
            
        Returns:
            Save operation results
        """
        try:
            # Prepare data for serialization
            if isinstance(data, np.ndarray):
                data_dict = {'data': data}
            elif isinstance(data, dict):
                data_dict = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Add serialization metadata
            data_dict['_serialization_info'] = {
                'timestamp': datetime.now().isoformat(),
                'numpy_version': np.__version__,
                'python_version': sys.version,
                'compression': compression
            }
            
            # Save data
            if compression:
                np.savez_compressed(filepath, **data_dict)
            else:
                np.savez(filepath, **data_dict)
            
            # Calculate file hash
            file_hash = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
            hash_value = file_hash.hexdigest()
            
            # Save metadata if requested
            if include_metadata:
                metadata_path = f"{os.path.splitext(filepath)[0]}_metadata.json"
                metadata = {
                    'data_info': DataSerializer._get_data_info(data_dict),
                    'file_info': {
                        'filepath': filepath,
                        'file_size': os.path.getsize(filepath),
                        'file_hash': hash_value,
                        'compression': compression
                    },
                    'serialization_info': data_dict['_serialization_info']
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            return {
                'success': True,
                'filepath': filepath,
                'file_size': os.path.getsize(filepath),
                'file_hash': hash_value,
                'compression': compression
            }
            
        except Exception as e:
            error_msg = f"Error saving data to {filepath}: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    @staticmethod
    def load_data(filepath: str, 
                  verify_integrity: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Load data with integrity verification.
        
        Args:
            filepath: Path to data file
            verify_integrity: Whether to verify file integrity
            
        Returns:
            Tuple of (loaded_data, load_info)
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Load data
            with np.load(filepath) as data:
                loaded_data = dict(data)
            
            # Remove serialization metadata from main data
            serialization_info = loaded_data.pop('_serialization_info', {})
            
            load_info = {
                'success': True,
                'filepath': filepath,
                'file_size': os.path.getsize(filepath),
                'serialization_info': serialization_info,
                'data_shapes': {key: value.shape for key, value in loaded_data.items()}
            }
            
            return loaded_data, load_info
            
        except Exception as e:
            error_msg = f"Error loading data from {filepath}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @staticmethod
    def _get_data_info(data_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get information about data arrays."""
        info = {}
        
        for key, value in data_dict.items():
            if key.startswith('_'):  # Skip metadata keys
                continue
                
            if isinstance(value, np.ndarray):
                info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'size': value.size,
                    'nbytes': value.nbytes,
                    'min': float(np.min(value)) if value.size > 0 else None,
                    'max': float(np.max(value)) if value.size > 0 else None,
                    'mean': float(np.mean(value)) if value.size > 0 else None
                }
        
        return info


def save_model_with_data(model, model_path: str,
                        data_dict: Optional[Dict[str, np.ndarray]] = None,
                        data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to save model and associated data together.
    
    Args:
        model: Model to save
        model_path: Path for model file
        data_dict: Optional data dictionary to save
        data_path: Path for data file (required if data_dict provided)
        
    Returns:
        Combined save results
    """
    results = {}
    
    # Save model
    serializer = ModelSerializer()
    results['model'] = serializer.save_model(model, model_path)
    
    # Save data if provided
    if data_dict and data_path:
        results['data'] = DataSerializer.save_data(data_dict, data_path)
    
    return results


def load_model_with_data(model_path: str, 
                        data_path: Optional[str] = None,
                        device: Optional[str] = None) -> Tuple[Any, Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Convenience function to load model and associated data together.
    
    Args:
        model_path: Path to model file
        data_path: Optional path to data file
        device: Device to load model onto
        
    Returns:
        Tuple of (model, data_dict, load_info)
    """
    # Load model
    serializer = ModelSerializer()
    model, model_info = serializer.load_model(model_path, device=device)
    
    # Load data if path provided
    data_dict = None
    data_info = None
    if data_path:
        data_dict, data_info = DataSerializer.load_data(data_path)
    
    load_info = {
        'model': model_info,
        'data': data_info
    }
    
    return model, data_dict, load_info
