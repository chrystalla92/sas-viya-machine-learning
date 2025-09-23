"""
Model Checkpointing and State Management for Autoencoder Training

This module provides comprehensive checkpointing functionality for saving
and loading model states, training progress, and configuration data with
full reproducibility support.
"""

import torch
import torch.nn as nn
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import shutil
import hashlib

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Comprehensive checkpoint manager for autoencoder training.
    
    This class handles:
    - Model state saving and loading
    - Optimizer state preservation
    - Training metrics and progress tracking
    - Configuration and hyperparameter storage
    - Automatic checkpoint rotation and cleanup
    - Verification and integrity checking
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_metrics: bool = True,
        compress: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_metrics: Whether to save training metrics
            compress: Whether to compress checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_metrics = save_metrics
        self.compress = compress
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint tracking
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.best_checkpoint = None
        
        logger.info(f"CheckpointManager initialized:")
        logger.info(f"  - Directory: {self.checkpoint_dir}")
        logger.info(f"  - Max checkpoints: {self.max_checkpoints}")
        logger.info(f"  - Save optimizer: {self.save_optimizer}")
        logger.info(f"  - Compression: {self.compress}")
        
    def save_checkpoint(
        self,
        model: nn.Module,
        iteration: int,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save complete checkpoint with model, optimizer, and metadata.
        
        Args:
            model: PyTorch model to save
            iteration: Current training iteration
            loss: Current loss value
            optimizer: Optimizer state (optional)
            metrics: Training metrics (optional)
            metadata: Additional metadata (optional)
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        checkpoint_name = f"checkpoint_iter_{iteration:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            # Core model and training state
            'model_state_dict': model.state_dict(),
            'iteration': iteration,
            'loss': loss,
            'timestamp': timestamp,
            'is_best': is_best,
            
            # Model architecture information
            'model_class': model.__class__.__name__,
            'model_config': self._extract_model_config(model),
            
            # Optional components
            'optimizer_state_dict': optimizer.state_dict() if optimizer and self.save_optimizer else None,
            'metrics': metrics if self.save_metrics else None,
            'metadata': metadata or {},
            
            # Reproducibility information
            'pytorch_version': torch.__version__,
            'random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        # Save checkpoint
        if self.compress:
            torch.save(checkpoint_data, checkpoint_path, pickle_protocol=4)
        else:
            torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate file size and checksum for verification
        file_size = checkpoint_path.stat().st_size
        checksum = self._calculate_checksum(checkpoint_path)
        
        # Update checkpoint tracking
        checkpoint_info = {
            'path': str(checkpoint_path),
            'iteration': iteration,
            'loss': loss,
            'timestamp': timestamp,
            'file_size': file_size,
            'checksum': checksum,
            'is_best': is_best
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # Update best checkpoint tracking
        if is_best or (self.best_checkpoint is None) or (loss < self.best_checkpoint['loss']):
            self.best_checkpoint = checkpoint_info.copy()
            
            # Also save as best checkpoint
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            shutil.copy2(checkpoint_path, best_path)
        
        # Save latest checkpoint link
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        shutil.copy2(checkpoint_path, latest_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save checkpoint manifest
        self._save_checkpoint_manifest()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        logger.info(f"  - Iteration: {iteration}")
        logger.info(f"  - Loss: {loss:.6e}")
        logger.info(f"  - File size: {file_size / 1024:.1f} KB")
        logger.info(f"  - Is best: {is_best}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model/optimizer state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Verify checkpoint integrity
        if not self._verify_checkpoint_integrity(checkpoint_path):
            logger.warning(f"Checkpoint integrity check failed: {checkpoint_path}")
        
        # Load checkpoint data
        device = device or torch.device('cpu')
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
        
        # Load optimizer state if available and requested
        if optimizer and checkpoint_data.get('optimizer_state_dict'):
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("Optimizer state loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        
        # Restore random states for reproducibility
        if 'random_state' in checkpoint_data:
            try:
                torch.set_rng_state(checkpoint_data['random_state'])
                if torch.cuda.is_available() and 'cuda_random_state' in checkpoint_data:
                    if checkpoint_data['cuda_random_state'] is not None:
                        torch.cuda.set_rng_state_all(checkpoint_data['cuda_random_state'])
                logger.info("Random states restored for reproducibility")
            except Exception as e:
                logger.warning(f"Failed to restore random states: {e}")
        
        # Extract key information
        checkpoint_info = {
            'iteration': checkpoint_data['iteration'],
            'loss': checkpoint_data['loss'],
            'timestamp': checkpoint_data['timestamp'],
            'is_best': checkpoint_data.get('is_best', False),
            'model_config': checkpoint_data.get('model_config', {}),
            'metrics': checkpoint_data.get('metrics', {}),
            'metadata': checkpoint_data.get('metadata', {}),
            'pytorch_version': checkpoint_data.get('pytorch_version', 'unknown')
        }
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"  - Iteration: {checkpoint_info['iteration']}")
        logger.info(f"  - Loss: {checkpoint_info['loss']:.6e}")
        logger.info(f"  - Timestamp: {checkpoint_info['timestamp']}")
        
        return checkpoint_info
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint if available.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors to
            
        Returns:
            Checkpoint information or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        
        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None
        
        return self.load_checkpoint(best_path, model, optimizer, device)
    
    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint if available.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors to
            
        Returns:
            Checkpoint information or None if no latest checkpoint exists
        """
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if not latest_path.exists():
            logger.warning("No latest checkpoint found")
            return None
        
        return self.load_checkpoint(latest_path, model, optimizer, device)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information dictionaries
        """
        self._load_checkpoint_manifest()
        return self.checkpoint_history.copy()
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint."""
        return self.best_checkpoint.copy() if self.best_checkpoint else None
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            
            # Update tracking
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history 
                if Path(cp['path']) != checkpoint_path
            ]
            self._save_checkpoint_manifest()
    
    def cleanup_all_checkpoints(self):
        """Delete all checkpoints in the directory."""
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            checkpoint_file.unlink()
            
        self.checkpoint_history.clear()
        self.best_checkpoint = None
        self._save_checkpoint_manifest()
        
        logger.info("All checkpoints cleaned up")
    
    def _extract_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model configuration for checkpoint metadata."""
        config = {
            'class_name': model.__class__.__name__,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Try to extract specific autoencoder config
        if hasattr(model, 'input_dim'):
            config['input_dim'] = model.input_dim
        if hasattr(model, 'hidden_dim'):
            config['hidden_dim'] = model.hidden_dim
        if hasattr(model, 'seed'):
            config['seed'] = model.seed
        
        return config
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint file integrity using stored checksum."""
        # Find checkpoint in history
        checkpoint_info = None
        for cp in self.checkpoint_history:
            if Path(cp['path']) == checkpoint_path:
                checkpoint_info = cp
                break
        
        if not checkpoint_info or 'checksum' not in checkpoint_info:
            return True  # Cannot verify, assume OK
        
        current_checksum = self._calculate_checksum(checkpoint_path)
        return current_checksum == checkpoint_info['checksum']
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on max_checkpoints setting."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by iteration (keep most recent)
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: x['iteration']
        )
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            
            # Don't delete best checkpoint
            if checkpoint_info.get('is_best', False):
                continue
                
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update tracking
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history
            if cp not in checkpoints_to_remove or cp.get('is_best', False)
        ]
    
    def _save_checkpoint_manifest(self):
        """Save checkpoint manifest for tracking."""
        manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        
        manifest_data = {
            'checkpoint_history': self.checkpoint_history,
            'best_checkpoint': self.best_checkpoint,
            'manager_config': {
                'max_checkpoints': self.max_checkpoints,
                'save_optimizer': self.save_optimizer,
                'save_metrics': self.save_metrics,
                'compress': self.compress
            },
            'created_at': datetime.now().isoformat()
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
    def _load_checkpoint_manifest(self):
        """Load checkpoint manifest if it exists."""
        manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        
        if not manifest_path.exists():
            return
        
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            self.checkpoint_history = manifest_data.get('checkpoint_history', [])
            self.best_checkpoint = manifest_data.get('best_checkpoint')
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint manifest: {e}")


def save_training_state(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    loss: float,
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Convenience function to save training state.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        iteration: Current iteration
        loss: Current loss
        metrics: Training metrics
        **kwargs: Additional metadata
    """
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        **kwargs
    }
    
    torch.save(checkpoint_data, filepath)
    logger.info(f"Training state saved to {filepath}")


def load_training_state(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Convenience function to load training state.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    device = device or torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Training state loaded from {filepath}")
    logger.info(f"  - Iteration: {checkpoint['iteration']}")
    logger.info(f"  - Loss: {checkpoint['loss']:.6e}")
    
    return checkpoint


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    **kwargs
) -> CheckpointManager:
    """
    Factory function to create a checkpoint manager.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        **kwargs: Additional arguments for CheckpointManager
        
    Returns:
        Initialized CheckpointManager
    """
    manager = CheckpointManager(checkpoint_dir, **kwargs)
    logger.info(f"Created CheckpointManager for directory: {checkpoint_dir}")
    
    return manager
