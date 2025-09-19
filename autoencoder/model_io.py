"""
Model I/O utilities for autoencoder models.

This module provides comprehensive model saving, loading, and checkpoint
management with SAS-compatible output formatting and state management.
"""

import os
import json
import pickle
import datetime
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .autoencoder_model import AutoencoderMLP


__all__ = [
    'ModelSaver',
    'ModelLoader', 
    'CheckpointManager',
    'SASOutputFormatter',
    'save_model_state',
    'load_model_state',
    'convert_checkpoint_to_standalone',
    'export_model_summary',
    'create_sas_compatible_outputs'
]


class ModelSaver:
    """
    Comprehensive model saving utilities with metadata and versioning.
    """
    
    def __init__(self, base_dir: str = "./models", create_dirs: bool = True):
        """
        Initialize ModelSaver.
        
        Args:
            base_dir (str): Base directory for saving models
            create_dirs (bool): Whether to create directories if they don't exist
        """
        self.base_dir = Path(base_dir)
        if create_dirs:
            self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: AutoencoderMLP, 
                   filepath: Optional[str] = None,
                   include_optimizer: bool = False,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   training_history: Optional[Dict[str, Any]] = None) -> str:
        """
        Save complete model with metadata and training state.
        
        Args:
            model (AutoencoderMLP): Model to save
            filepath (Optional[str]): Custom filepath (auto-generated if None)
            include_optimizer (bool): Whether to include optimizer state
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to save
            scheduler (Optional[Any]): Scheduler to save
            metadata (Optional[Dict[str, Any]]): Additional metadata
            training_history (Optional[Dict[str, Any]]): Training history to save
            
        Returns:
            str: Path where model was saved
        """
        # Generate filepath if not provided
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.base_dir / f"autoencoder_{timestamp}.pth"
        else:
            filepath = Path(filepath)
            if not filepath.is_absolute():
                filepath = self.base_dir / filepath
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config(),
            'model_class': model.__class__.__name__,
            'save_timestamp': datetime.datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # Add optimizer state if requested
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['optimizer_class'] = optimizer.__class__.__name__
            
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint['scheduler_class'] = scheduler.__class__.__name__
        
        # Add metadata if provided
        if metadata is not None:
            checkpoint['metadata'] = metadata
            
        # Add training history if provided
        if training_history is not None:
            checkpoint['training_history'] = training_history
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Also save a JSON metadata file for easy inspection
        metadata_file = filepath.with_suffix('.json')
        json_metadata = {
            'model_class': checkpoint['model_class'],
            'model_config': checkpoint['model_config'],
            'save_timestamp': checkpoint['save_timestamp'],
            'pytorch_version': checkpoint['pytorch_version'],
            'filepath': str(filepath),
            'has_optimizer': include_optimizer and optimizer is not None,
            'has_scheduler': scheduler is not None,
            'has_training_history': training_history is not None
        }
        
        if metadata is not None:
            json_metadata['metadata'] = metadata
            
        with open(metadata_file, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        return str(filepath)
    
    def save_weights_only(self, model: AutoencoderMLP, filepath: str) -> str:
        """
        Save only model weights (state_dict).
        
        Args:
            model (AutoencoderMLP): Model to save
            filepath (str): Path to save weights
            
        Returns:
            str: Path where weights were saved
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.base_dir / filepath
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only the state dict
        torch.save(model.state_dict(), filepath)
        
        return str(filepath)


class ModelLoader:
    """
    Comprehensive model loading utilities with validation and compatibility checks.
    """
    
    @staticmethod
    def load_model(filepath: str, 
                   device: Optional[Union[str, torch.device]] = None,
                   strict_loading: bool = True) -> Tuple[AutoencoderMLP, Dict[str, Any]]:
        """
        Load complete model from checkpoint.
        
        Args:
            filepath (str): Path to model checkpoint
            device (Optional[Union[str, torch.device]]): Device to load model on
            strict_loading (bool): Whether to use strict loading for state_dict
            
        Returns:
            Tuple[AutoencoderMLP, Dict[str, Any]]: Loaded model and metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'model_config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint: missing keys {missing_keys}")
        
        # Create model from config
        config = checkpoint['model_config']
        model = AutoencoderMLP(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim'],
            activation=config['activation'],
            init_type=config.get('init_type', 'uniform'),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict_loading)
        
        # Extract metadata
        metadata = {
            'model_config': config,
            'save_timestamp': checkpoint.get('save_timestamp'),
            'pytorch_version': checkpoint.get('pytorch_version'),
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint,
            'has_training_history': 'training_history' in checkpoint,
        }
        
        # Add any additional metadata
        if 'metadata' in checkpoint:
            metadata['additional'] = checkpoint['metadata']
            
        return model, metadata
    
    @staticmethod
    def load_weights_only(model: AutoencoderMLP, 
                         weights_path: str,
                         strict_loading: bool = True) -> AutoencoderMLP:
        """
        Load weights into existing model.
        
        Args:
            model (AutoencoderMLP): Model to load weights into
            weights_path (str): Path to weights file
            strict_loading (bool): Whether to use strict loading
            
        Returns:
            AutoencoderMLP: Model with loaded weights
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load state dict
        state_dict = torch.load(weights_path, map_location=next(model.parameters()).device)
        
        # Load into model
        model.load_state_dict(state_dict, strict=strict_loading)
        
        return model
    
    @staticmethod
    def load_optimizer_state(filepath: str, 
                            optimizer: torch.optim.Optimizer,
                            device: Optional[Union[str, torch.device]] = None) -> torch.optim.Optimizer:
        """
        Load optimizer state from checkpoint.
        
        Args:
            filepath (str): Path to checkpoint containing optimizer state
            optimizer (torch.optim.Optimizer): Optimizer to load state into
            device (Optional[Union[str, torch.device]]): Device to map tensors to
            
        Returns:
            torch.optim.Optimizer: Optimizer with loaded state
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        if 'optimizer_state_dict' not in checkpoint:
            raise ValueError("No optimizer state found in checkpoint")
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return optimizer


class CheckpointManager:
    """
    Utilities for managing and converting model checkpoints.
    """
    
    @staticmethod
    def convert_training_checkpoint_to_standalone(checkpoint_path: str, 
                                                 output_path: str,
                                                 include_training_data: bool = False) -> str:
        """
        Convert training checkpoint to standalone model file.
        
        Args:
            checkpoint_path (str): Path to training checkpoint
            output_path (str): Path for standalone model
            include_training_data (bool): Whether to include training history
            
        Returns:
            str: Path to converted model
        """
        # Load original checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create standalone checkpoint
        standalone = {
            'model_state_dict': checkpoint['model_state_dict'],
            'model_config': checkpoint.get('model_config', {}),
            'model_class': 'AutoencoderMLP',
            'conversion_timestamp': datetime.datetime.now().isoformat(),
            'original_checkpoint': checkpoint_path,
            'pytorch_version': torch.__version__,
        }
        
        # Add training data if requested and available
        if include_training_data:
            training_keys = ['train_loss', 'val_loss', 'epoch', 'training_history']
            for key in training_keys:
                if key in checkpoint:
                    standalone[key] = checkpoint[key]
        
        # Save standalone checkpoint
        torch.save(standalone, output_path)
        
        return output_path
    
    @staticmethod
    def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
        """
        List and analyze checkpoints in a directory.
        
        Args:
            checkpoint_dir (str): Directory containing checkpoints
            
        Returns:
            List[Dict[str, Any]]: List of checkpoint information
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_file in checkpoint_dir.glob("*.pth"):
            try:
                # Load checkpoint metadata
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                
                info = {
                    'filepath': str(checkpoint_file),
                    'filename': checkpoint_file.name,
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                    'modified_time': datetime.datetime.fromtimestamp(checkpoint_file.stat().st_mtime),
                }
                
                # Extract available information
                if 'epoch' in checkpoint:
                    info['epoch'] = checkpoint['epoch']
                if 'val_loss' in checkpoint:
                    info['val_loss'] = checkpoint['val_loss']
                if 'train_loss' in checkpoint:
                    info['train_loss'] = checkpoint['train_loss']
                if 'save_timestamp' in checkpoint:
                    info['save_timestamp'] = checkpoint['save_timestamp']
                if 'model_config' in checkpoint:
                    info['model_config'] = checkpoint['model_config']
                
                checkpoints.append(info)
                
            except Exception as e:
                # Skip corrupted checkpoints
                print(f"Warning: Could not read checkpoint {checkpoint_file}: {e}")
                continue
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return checkpoints


class SASOutputFormatter:
    """
    Utilities for creating SAS-compatible output formats.
    """
    
    @staticmethod
    def format_model_predictions(original_data: np.ndarray,
                                reconstructed_data: np.ndarray,
                                latent_representations: np.ndarray,
                                sample_ids: Optional[np.ndarray] = None,
                                labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Format model outputs in SAS-compatible structure.
        
        Args:
            original_data (np.ndarray): Original input data
            reconstructed_data (np.ndarray): Reconstructed data
            latent_representations (np.ndarray): Latent space representations
            sample_ids (Optional[np.ndarray]): Sample identifiers
            labels (Optional[np.ndarray]): Ground truth labels
            
        Returns:
            pd.DataFrame: SAS-compatible output dataframe
        """
        n_samples = original_data.shape[0]
        
        # Create sample IDs if not provided
        if sample_ids is None:
            sample_ids = np.arange(n_samples)
        
        # Create base dataframe
        df_data = {'sample_id': sample_ids}
        
        # Add labels if provided
        if labels is not None:
            df_data['label'] = labels
        
        # Add original data columns
        for i in range(original_data.shape[1]):
            df_data[f'original_{i+1}'] = original_data[:, i]
        
        # Add reconstructed data columns
        for i in range(reconstructed_data.shape[1]):
            df_data[f'reconstructed_{i+1}'] = reconstructed_data[:, i]
        
        # Add latent representations
        for i in range(latent_representations.shape[1]):
            df_data[f'latent_{i+1}'] = latent_representations[:, i]
        
        # Calculate reconstruction errors per sample
        mse_per_sample = np.mean((original_data - reconstructed_data) ** 2, axis=1)
        mae_per_sample = np.mean(np.abs(original_data - reconstructed_data), axis=1)
        
        df_data['reconstruction_mse'] = mse_per_sample
        df_data['reconstruction_mae'] = mae_per_sample
        
        return pd.DataFrame(df_data)
    
    @staticmethod
    def format_evaluation_summary(evaluation_results: Dict[str, Any],
                                 model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format evaluation results in SAS-compatible summary structure.
        
        Args:
            evaluation_results (Dict[str, Any]): Raw evaluation results
            model_config (Dict[str, Any]): Model configuration
            
        Returns:
            Dict[str, Any]: Formatted evaluation summary
        """
        summary = {
            'model_info': {
                'model_type': 'AutoencoderMLP',
                'input_dim': model_config.get('input_dim', 'unknown'),
                'latent_dim': model_config.get('latent_dim', 'unknown'),
                'activation': model_config.get('activation', 'unknown'),
                'total_parameters': model_config.get('total_parameters', 'unknown'),
            },
            'evaluation_timestamp': datetime.datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': evaluation_results.get('n_samples', 'unknown'),
                'n_features': evaluation_results.get('n_features', 'unknown'),
            }
        }
        
        # Add reconstruction metrics
        if 'reconstruction_metrics' in evaluation_results:
            summary['reconstruction_performance'] = evaluation_results['reconstruction_metrics']
        
        # Add latent space statistics
        if 'latent_statistics' in evaluation_results:
            summary['latent_space_analysis'] = evaluation_results['latent_statistics']
        
        # Add performance metrics
        if 'performance_metrics' in evaluation_results:
            summary['computational_performance'] = evaluation_results['performance_metrics']
        
        return summary


# Convenience functions
def save_model_state(model: AutoencoderMLP, filepath: str, **kwargs) -> str:
    """Convenience function for saving model state."""
    saver = ModelSaver()
    return saver.save_model(model, filepath, **kwargs)


def load_model_state(filepath: str, device: Optional[Union[str, torch.device]] = None) -> Tuple[AutoencoderMLP, Dict[str, Any]]:
    """Convenience function for loading model state."""
    return ModelLoader.load_model(filepath, device)


def convert_checkpoint_to_standalone(checkpoint_path: str, output_path: str) -> str:
    """Convenience function for checkpoint conversion."""
    return CheckpointManager.convert_training_checkpoint_to_standalone(checkpoint_path, output_path)


def export_model_summary(model: AutoencoderMLP, output_path: str, 
                        evaluation_results: Optional[Dict[str, Any]] = None) -> str:
    """
    Export comprehensive model summary to JSON.
    
    Args:
        model (AutoencoderMLP): Model to summarize
        output_path (str): Path for output JSON
        evaluation_results (Optional[Dict[str, Any]]): Optional evaluation results
        
    Returns:
        str: Path to exported summary
    """
    summary = {
        'model_config': model.get_config(),
        'model_summary': model.summary(),
        'export_timestamp': datetime.datetime.now().isoformat(),
    }
    
    if evaluation_results is not None:
        summary['evaluation_results'] = evaluation_results
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)  # default=str to handle numpy types
    
    return output_path


def create_sas_compatible_outputs(original: np.ndarray,
                                 reconstructed: np.ndarray,
                                 latent: np.ndarray,
                                 output_dir: str,
                                 base_filename: str = "autoencoder_results") -> Dict[str, str]:
    """
    Create SAS-compatible output files.
    
    Args:
        original (np.ndarray): Original data
        reconstructed (np.ndarray): Reconstructed data
        latent (np.ndarray): Latent representations
        output_dir (str): Output directory
        base_filename (str): Base filename for outputs
        
    Returns:
        Dict[str, str]: Paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formatter = SASOutputFormatter()
    
    # Create comprehensive dataframe
    df = formatter.format_model_predictions(original, reconstructed, latent)
    
    # Save in multiple formats
    paths = {}
    
    # CSV for SAS import
    csv_path = output_dir / f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    paths['csv'] = str(csv_path)
    
    # JSON for metadata
    json_path = output_dir / f"{base_filename}_metadata.json"
    metadata = {
        'n_samples': len(df),
        'n_original_features': original.shape[1],
        'n_latent_features': latent.shape[1],
        'columns': list(df.columns),
        'creation_timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    paths['metadata'] = str(json_path)
    
    # Parquet for efficient storage
    try:
        parquet_path = output_dir / f"{base_filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        paths['parquet'] = str(parquet_path)
    except ImportError:
        # Parquet not available
        pass
    
    return paths
