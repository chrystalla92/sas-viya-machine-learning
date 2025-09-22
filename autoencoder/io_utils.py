"""
Data I/O Utilities Module

This module provides comprehensive data loading, saving, and file organization utilities
to replace the SAS export_to_csv.sas functionality with Python-native operations.

Key features:
- NumPy array export functions for reconstructions and latent representations
- Structured file organization system for model outputs
- Metadata saving for model parameters, training info, and dataset info
- Batch processing utilities for large datasets
- Cross-platform file path handling and data provenance tracking

Replaces SAS functionality:
- export_to_csv.sas: CSV export operations
- Data organization and naming conventions
- Score dataset export and reconstruction output
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputOrganizer:
    """
    Manages structured file organization for autoencoder outputs.
    
    Provides consistent directory structure and naming conventions
    to replace SAS output organization.
    """
    
    def __init__(self, base_output_dir: str = "./outputs"):
        """
        Initialize output organizer.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.dirs = {
            'models': self.base_dir / 'models',
            'data': self.base_dir / 'data',
            'reconstructions': self.base_dir / 'reconstructions',
            'latent': self.base_dir / 'latent_representations',
            'metadata': self.base_dir / 'metadata',
            'logs': self.base_dir / 'logs',
            'exports': self.base_dir / 'exports'
        }
        
        # Ensure all directories exist
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_file_path(self, category: str, filename: str, 
                     include_timestamp: bool = True) -> Path:
        """
        Generate standardized file path.
        
        Args:
            category: Output category ('models', 'data', 'reconstructions', etc.)
            filename: Base filename
            include_timestamp: Whether to include timestamp in filename
            
        Returns:
            Standardized file path
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.dirs.keys())}")
        
        if include_timestamp:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{self.timestamp}{ext}"
        
        return self.dirs[category] / filename
    
    def create_run_directory(self, run_name: str) -> Dict[str, Path]:
        """
        Create directory structure for a specific training run.
        
        Args:
            run_name: Name of the training run
            
        Returns:
            Dictionary mapping category names to paths
        """
        run_dir = self.base_dir / f"run_{run_name}_{self.timestamp}"
        
        run_dirs = {}
        for category in self.dirs.keys():
            run_dirs[category] = run_dir / category
            run_dirs[category].mkdir(parents=True, exist_ok=True)
        
        return run_dirs


class NumpyExporter:
    """
    Handles NumPy array exports for reconstructions and latent representations.
    
    Replaces CSV export functionality from SAS with more efficient NumPy formats.
    """
    
    @staticmethod
    def export_reconstructions(reconstructions: np.ndarray, 
                              original_data: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              output_path: str = None,
                              include_metadata: bool = True) -> str:
        """
        Export reconstruction data replacing SAS score dataset export.
        
        Args:
            reconstructions: Reconstructed data array (n_samples, n_features)
            original_data: Original input data array (n_samples, n_features)
            labels: Optional labels array (n_samples,)
            output_path: Output file path (auto-generated if None)
            include_metadata: Whether to save metadata alongside data
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"mnist_autoencoder_reconstructions_{timestamp}"
        
        # Prepare data structure matching SAS score output
        output_data = {
            'reconstructions': reconstructions,
            'originals': original_data,
            'reconstruction_errors': np.mean((reconstructions - original_data)**2, axis=1)
        }
        
        if labels is not None:
            output_data['labels'] = labels
        
        # Save as compressed NumPy archive
        np_path = f"{output_path}.npz"
        np.savez_compressed(np_path, **output_data)
        
        # Also save as CSV for SAS compatibility if needed
        csv_path = f"{output_path}.csv"
        NumpyExporter._save_as_csv(output_data, csv_path, include_labels=(labels is not None))
        
        if include_metadata:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'data_shape': reconstructions.shape,
                'n_samples': len(reconstructions),
                'n_features': reconstructions.shape[1] if len(reconstructions.shape) > 1 else 0,
                'has_labels': labels is not None,
                'reconstruction_mse': float(np.mean((reconstructions - original_data)**2)),
                'file_format': 'npz_and_csv'
            }
            
            with open(f"{output_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported reconstructions to {np_path} and {csv_path}")
        return np_path
    
    @staticmethod
    def export_latent_representations(latent_data: np.ndarray,
                                    labels: Optional[np.ndarray] = None,
                                    output_path: str = None,
                                    include_metadata: bool = True) -> str:
        """
        Export latent/hidden representations from autoencoder.
        
        Args:
            latent_data: Latent representations array (n_samples, latent_dim)
            labels: Optional labels array (n_samples,)
            output_path: Output file path (auto-generated if None)
            include_metadata: Whether to save metadata alongside data
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"mnist_autoencoder_latent_{timestamp}"
        
        # Prepare data structure
        output_data = {'latent_representations': latent_data}
        
        if labels is not None:
            output_data['labels'] = labels
        
        # Save as compressed NumPy archive
        np_path = f"{output_path}.npz"
        np.savez_compressed(np_path, **output_data)
        
        # Also save as CSV for compatibility
        csv_path = f"{output_path}.csv"
        NumpyExporter._save_as_csv(output_data, csv_path, include_labels=(labels is not None))
        
        if include_metadata:
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'latent_shape': latent_data.shape,
                'n_samples': len(latent_data),
                'latent_dim': latent_data.shape[1] if len(latent_data.shape) > 1 else 0,
                'has_labels': labels is not None,
                'file_format': 'npz_and_csv'
            }
            
            with open(f"{output_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported latent representations to {np_path} and {csv_path}")
        return np_path
    
    @staticmethod
    def _save_as_csv(data_dict: Dict[str, np.ndarray], csv_path: str, 
                    include_labels: bool = False) -> None:
        """
        Save data dictionary as CSV file matching SAS export format.
        
        Args:
            data_dict: Dictionary containing arrays to export
            csv_path: Output CSV path
            include_labels: Whether to include labels column
        """
        try:
            # Determine main data array and prepare DataFrame
            if 'reconstructions' in data_dict:
                main_data = data_dict['reconstructions']
                prefix = 'recon_'
            elif 'latent_representations' in data_dict:
                main_data = data_dict['latent_representations']
                prefix = 'latent_'
            else:
                # Use first array found
                key = list(data_dict.keys())[0]
                main_data = data_dict[key]
                prefix = f'{key}_'
            
            # Create DataFrame with appropriate column names
            n_features = main_data.shape[1] if len(main_data.shape) > 1 else 1
            
            if include_labels and 'labels' in data_dict:
                # Format: var1 (labels), var2-varN (features) - matching SAS format
                columns = ['var1'] + [f'var{i}' for i in range(2, n_features + 2)]
                df_data = np.column_stack([data_dict['labels'], main_data])
            else:
                # Format: var1-varN (features only)
                columns = [f'var{i}' for i in range(1, n_features + 1)]
                df_data = main_data
            
            df = pd.DataFrame(df_data, columns=columns)
            
            # Save without header to match SAS putnames=no
            df.to_csv(csv_path, index=False, header=False)
            
        except Exception as e:
            logger.error(f"Error saving CSV {csv_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_exported_data(npz_path: str) -> Dict[str, np.ndarray]:
        """
        Load previously exported NumPy data.
        
        Args:
            npz_path: Path to NPZ file
            
        Returns:
            Dictionary containing loaded arrays
        """
        try:
            with np.load(npz_path) as data:
                return dict(data)
        except Exception as e:
            logger.error(f"Error loading {npz_path}: {str(e)}")
            raise


class MetadataManager:
    """
    Manages metadata saving for model parameters, training info, and dataset info.
    """
    
    @staticmethod
    def save_training_metadata(model_info: Dict[str, Any],
                             training_info: Dict[str, Any],
                             dataset_info: Dict[str, Any],
                             output_path: str) -> None:
        """
        Save comprehensive metadata for training session.
        
        Args:
            model_info: Model architecture and parameters
            training_info: Training configuration and results
            dataset_info: Dataset information and preprocessing
            output_path: Output file path
        """
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'model': model_info,
            'training': training_info,
            'dataset': dataset_info,
            'python_env': {
                'python_version': os.sys.version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise
    
    @staticmethod
    def load_metadata(metadata_path: str) -> Dict[str, Any]:
        """
        Load saved metadata.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Loaded metadata dictionary
        """
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise


class BatchProcessor:
    """
    Utilities for batch processing of large datasets.
    """
    
    @staticmethod
    def process_large_dataset_batches(data: np.ndarray,
                                    process_func,
                                    batch_size: int = 1000,
                                    output_dir: str = "./batch_outputs",
                                    **kwargs) -> List[str]:
        """
        Process large dataset in batches and save results.
        
        Args:
            data: Input data array
            process_func: Function to apply to each batch
            batch_size: Size of processing batches
            output_dir: Directory to save batch results
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        n_batches = int(np.ceil(len(data) / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            
            batch_data = data[start_idx:end_idx]
            
            try:
                # Process batch
                result = process_func(batch_data, **kwargs)
                
                # Save batch result
                output_path = os.path.join(output_dir, f"batch_{i:04d}.npz")
                if isinstance(result, dict):
                    np.savez_compressed(output_path, **result)
                else:
                    np.savez_compressed(output_path, result=result)
                
                output_paths.append(output_path)
                logger.info(f"Processed batch {i+1}/{n_batches}: {start_idx}-{end_idx}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                continue
        
        return output_paths
    
    @staticmethod
    def combine_batch_results(batch_paths: List[str], 
                            output_path: str) -> None:
        """
        Combine results from multiple batch files.
        
        Args:
            batch_paths: List of batch result file paths
            output_path: Path for combined output
        """
        combined_data = {}
        
        for batch_path in batch_paths:
            try:
                with np.load(batch_path) as batch_data:
                    for key, value in batch_data.items():
                        if key not in combined_data:
                            combined_data[key] = []
                        combined_data[key].append(value)
            except Exception as e:
                logger.error(f"Error loading batch {batch_path}: {str(e)}")
                continue
        
        # Concatenate arrays
        final_data = {}
        for key, value_list in combined_data.items():
            try:
                final_data[key] = np.concatenate(value_list, axis=0)
            except Exception as e:
                logger.error(f"Error combining {key}: {str(e)}")
                continue
        
        # Save combined result
        np.savez_compressed(output_path, **final_data)
        logger.info(f"Combined {len(batch_paths)} batches into {output_path}")


def export_model_outputs(model, data: np.ndarray, labels: Optional[np.ndarray] = None,
                        output_dir: str = "./outputs", 
                        run_name: str = "autoencoder_export") -> Dict[str, str]:
    """
    Comprehensive export function replacing SAS export functionality.
    
    Args:
        model: Trained autoencoder model
        data: Input data for scoring/reconstruction
        labels: Optional labels array
        output_dir: Base output directory
        run_name: Name for this export run
        
    Returns:
        Dictionary mapping export types to file paths
    """
    # Initialize output organizer
    organizer = OutputOrganizer(output_dir)
    run_dirs = organizer.create_run_directory(run_name)
    
    # Get model outputs
    import torch
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data)
        reconstructions = model(data_tensor).cpu().numpy()
        latent_reps = model.encode(data_tensor).cpu().numpy()
    
    export_paths = {}
    
    # Export reconstructions (replacing SAS score dataset export)
    recon_path = str(run_dirs['reconstructions'] / 'reconstructions')
    export_paths['reconstructions'] = NumpyExporter.export_reconstructions(
        reconstructions, data, labels, recon_path
    )
    
    # Export latent representations
    latent_path = str(run_dirs['latent'] / 'latent_representations')
    export_paths['latent'] = NumpyExporter.export_latent_representations(
        latent_reps, labels, latent_path
    )
    
    # Save model metadata
    model_info = model.get_architecture_info() if hasattr(model, 'get_architecture_info') else {}
    dataset_info = {
        'n_samples': len(data),
        'n_features': data.shape[1] if len(data.shape) > 1 else 0,
        'data_shape': data.shape,
        'has_labels': labels is not None,
        'data_range': {'min': float(np.min(data)), 'max': float(np.max(data))}
    }
    training_info = {'export_only': True, 'export_timestamp': datetime.now().isoformat()}
    
    metadata_path = str(run_dirs['metadata'] / 'export_metadata.json')
    MetadataManager.save_training_metadata(model_info, training_info, dataset_info, metadata_path)
    export_paths['metadata'] = metadata_path
    
    logger.info(f"Model outputs exported to {output_dir}")
    return export_paths
