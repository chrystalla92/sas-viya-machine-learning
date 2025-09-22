"""
Model Checkpointing Utilities

This module provides comprehensive checkpointing functionality for saving
and loading model states during training, including automatic scheduling,
best model tracking, and resuming training from checkpoints.

Key features:
- Automatic checkpoint saving at regular intervals
- Best model performance tracking and saving
- Training resumption from checkpoints
- Checkpoint cleanup and management
- Metadata preservation (epoch, loss, optimizer state)
"""

import torch
import os
import json
import glob
import shutil
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

from ..model import MNISTAutoencoder


class CheckpointManager:
    """
    Comprehensive checkpoint management for autoencoder training.
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints', 
                 max_checkpoints: int = 10, auto_cleanup: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            auto_cleanup: Whether to automatically clean up old checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track checkpoints
        self.checkpoint_history = []
        self.best_checkpoint = None
        self.best_loss = float('inf')
        
        # Load existing checkpoints info
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """Load existing checkpoint history from directory."""
        history_file = os.path.join(self.checkpoint_dir, 'checkpoint_history.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.checkpoint_history = history_data.get('checkpoints', [])
                    self.best_checkpoint = history_data.get('best_checkpoint', None)
                    self.best_loss = history_data.get('best_loss', float('inf'))
            except Exception as e:
                print(f"Warning: Could not load checkpoint history: {e}")
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to file."""
        history_file = os.path.join(self.checkpoint_dir, 'checkpoint_history.json')
        
        history_data = {
            'checkpoints': self.checkpoint_history,
            'best_checkpoint': self.best_checkpoint,
            'best_loss': self.best_loss,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint history: {e}")
    
    def save_checkpoint(self, model: MNISTAutoencoder, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metrics: Optional[Dict] = None,
                       is_best: bool = False, checkpoint_name: Optional[str] = None) -> str:
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            loss: Current loss value
            metrics: Additional metrics to save
            is_best: Whether this is the best checkpoint so far
            checkpoint_name: Custom checkpoint name (optional)
            
        Returns:
            Path to saved checkpoint file
        """
        # Generate checkpoint filename
        if checkpoint_name is None:
            if is_best:
                checkpoint_name = 'best_model.pt'
            else:
                checkpoint_name = f'checkpoint_epoch_{epoch:04d}.pt'
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'architecture': model.get_architecture_info(),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Add metrics if provided
        if metrics is not None:
            checkpoint_data['metrics'] = metrics
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update tracking
            checkpoint_info = {
                'filename': checkpoint_name,
                'path': checkpoint_path,
                'epoch': epoch,
                'loss': loss,
                'timestamp': checkpoint_data['timestamp'],
                'is_best': is_best
            }
            
            if is_best:
                self.best_checkpoint = checkpoint_info
                self.best_loss = loss
                print(f"New best checkpoint saved: {checkpoint_name} (loss: {loss:.8f})")
            else:
                self.checkpoint_history.append(checkpoint_info)
                print(f"Checkpoint saved: {checkpoint_name} (epoch: {epoch}, loss: {loss:.8f})")
            
            # Cleanup old checkpoints if needed
            if self.auto_cleanup and not is_best:
                self._cleanup_old_checkpoints()
            
            # Save updated history
            self._save_checkpoint_history()
            
            return checkpoint_path
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch (oldest first)
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        while len(sorted_checkpoints) > self.max_checkpoints:
            old_checkpoint = sorted_checkpoints.pop(0)
            
            try:
                if os.path.exists(old_checkpoint['path']):
                    os.remove(old_checkpoint['path'])
                    print(f"Removed old checkpoint: {old_checkpoint['filename']}")
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {old_checkpoint['filename']}: {e}")
        
        # Update history
        self.checkpoint_history = sorted_checkpoints
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, 
                       load_best: bool = False) -> Tuple[Dict, MNISTAutoencoder]:
        """
        Load checkpoint and create model.
        
        Args:
            checkpoint_path: Path to specific checkpoint (optional)
            load_best: Whether to load the best checkpoint
            
        Returns:
            Tuple of (checkpoint_data, model)
        """
        # Determine which checkpoint to load
        if load_best and self.best_checkpoint is not None:
            checkpoint_path = self.best_checkpoint['path']
        elif checkpoint_path is None:
            # Load most recent checkpoint
            if self.checkpoint_history:
                latest = max(self.checkpoint_history, key=lambda x: x['epoch'])
                checkpoint_path = latest['path']
            else:
                raise ValueError("No checkpoints found")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract architecture info
            arch_info = checkpoint_data.get('architecture', {})
            
            # Create model with same architecture
            model = MNISTAutoencoder(
                input_dim=arch_info.get('input_dim', 784),
                hidden_dim=arch_info.get('hidden_dim', 400),
                dropout_rate=arch_info.get('dropout_rate', 0.0)
            )
            
            # Load model weights
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            print(f"Loaded checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"  Epoch: {checkpoint_data['epoch']}")
            print(f"  Loss: {checkpoint_data['loss']:.8f}")
            
            return checkpoint_data, model
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    
    def resume_training(self, checkpoint_path: Optional[str] = None,
                       load_best: bool = False) -> Tuple[MNISTAutoencoder, torch.optim.Optimizer, int, float]:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to specific checkpoint (optional)  
            load_best: Whether to resume from best checkpoint
            
        Returns:
            Tuple of (model, optimizer, start_epoch, best_loss)
        """
        checkpoint_data, model = self.load_checkpoint(checkpoint_path, load_best)
        
        # Create optimizer (L-BFGS with SAS-compatible settings)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            max_iter=500,
            tolerance_grad=1e-10,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Using fresh optimizer state")
        
        start_epoch = checkpoint_data['epoch'] + 1
        best_loss = checkpoint_data['loss']
        
        print(f"Resuming training from epoch {start_epoch}")
        
        return model, optimizer, start_epoch, best_loss
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints with information.
        
        Returns:
            List of checkpoint information dictionaries
        """
        all_checkpoints = self.checkpoint_history.copy()
        
        if self.best_checkpoint is not None:
            all_checkpoints.append(self.best_checkpoint)
        
        # Sort by epoch
        all_checkpoints.sort(key=lambda x: x['epoch'])
        
        return all_checkpoints
    
    def get_best_checkpoint_info(self) -> Optional[Dict]:
        """Get information about the best checkpoint."""
        return self.best_checkpoint
    
    def export_checkpoint(self, checkpoint_path: str, export_path: str,
                         include_optimizer: bool = False):
        """
        Export checkpoint for deployment (smaller file size).
        
        Args:
            checkpoint_path: Source checkpoint path
            export_path: Destination export path  
            include_optimizer: Whether to include optimizer state
        """
        checkpoint_data, model = self.load_checkpoint(checkpoint_path)
        
        # Create export data (minimal)
        export_data = {
            'model_state_dict': model.state_dict(),
            'architecture': checkpoint_data.get('architecture', {}),
            'epoch': checkpoint_data['epoch'],
            'loss': checkpoint_data['loss'],
            'export_timestamp': datetime.now().isoformat()
        }
        
        if include_optimizer and 'optimizer_state_dict' in checkpoint_data:
            export_data['optimizer_state_dict'] = checkpoint_data['optimizer_state_dict']
        
        # Save export
        torch.save(export_data, export_path)
        print(f"Checkpoint exported to: {export_path}")
    
    def cleanup_all_checkpoints(self, keep_best: bool = True):
        """
        Remove all checkpoints except optionally the best one.
        
        Args:
            keep_best: Whether to keep the best checkpoint
        """
        # Remove regular checkpoints
        for checkpoint in self.checkpoint_history:
            try:
                if os.path.exists(checkpoint['path']):
                    os.remove(checkpoint['path'])
                    print(f"Removed checkpoint: {checkpoint['filename']}")
            except Exception as e:
                print(f"Warning: Could not remove {checkpoint['filename']}: {e}")
        
        self.checkpoint_history = []
        
        # Remove best checkpoint if requested
        if not keep_best and self.best_checkpoint is not None:
            try:
                if os.path.exists(self.best_checkpoint['path']):
                    os.remove(self.best_checkpoint['path'])
                    print(f"Removed best checkpoint: {self.best_checkpoint['filename']}")
            except Exception as e:
                print(f"Warning: Could not remove best checkpoint: {e}")
            
            self.best_checkpoint = None
            self.best_loss = float('inf')
        
        # Update history
        self._save_checkpoint_history()
        print("Checkpoint cleanup completed")


def create_checkpoint_schedule(save_intervals: List[int], total_epochs: int) -> List[int]:
    """
    Create a checkpoint saving schedule.
    
    Args:
        save_intervals: List of intervals (e.g., [10, 50, 100])
        total_epochs: Total number of epochs
        
    Returns:
        List of epochs when to save checkpoints
    """
    save_epochs = set()
    
    for interval in save_intervals:
        for epoch in range(interval, total_epochs + 1, interval):
            save_epochs.add(epoch)
    
    # Always save at the end
    save_epochs.add(total_epochs)
    
    return sorted(list(save_epochs))


class CheckpointCallback:
    """
    Callback for automatic checkpoint saving during training.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager,
                 save_interval: int = 50, save_best: bool = True,
                 monitor_metric: str = 'loss', mode: str = 'min'):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_manager: Manager instance to use
            save_interval: Epochs between regular saves
            save_best: Whether to save best model
            monitor_metric: Metric to monitor for best model
            mode: 'min' or 'max' for best model determination
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, epoch: int, model: MNISTAutoencoder, 
                 optimizer: torch.optim.Optimizer, metrics: Dict[str, float]):
        """
        Call during training to handle checkpointing.
        
        Args:
            epoch: Current epoch
            model: Current model
            optimizer: Current optimizer
            metrics: Current metrics dict
        """
        current_metric = metrics.get(self.monitor_metric, 0.0)
        
        # Regular checkpoint saving
        if epoch % self.save_interval == 0:
            self.checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, current_metric, metrics
            )
        
        # Best model saving
        if self.save_best:
            is_better = (
                (self.mode == 'min' and current_metric < self.best_metric) or
                (self.mode == 'max' and current_metric > self.best_metric)
            )
            
            if is_better:
                self.best_metric = current_metric
                self.checkpoint_manager.save_checkpoint(
                    model, optimizer, epoch, current_metric, metrics, is_best=True
                )


def demo_checkpointing():
    """Demonstrate checkpoint functionality."""
    print("=== Checkpoint Manager Demo ===")
    
    # Create checkpoint manager
    manager = CheckpointManager('./demo_checkpoints')
    
    # Create mock model and optimizer
    from ..model import create_sas_compatible_autoencoder
    import torch.optim as optim
    
    model = create_sas_compatible_autoencoder()
    optimizer = optim.LBFGS(model.parameters())
    
    # Mock training loop with checkpoints
    print("Simulating training with checkpoints...")
    
    for epoch in range(1, 6):
        # Mock loss (decreasing)
        mock_loss = 1.0 / epoch
        
        # Save regular checkpoint
        manager.save_checkpoint(
            model, optimizer, epoch, mock_loss,
            metrics={'mock_metric': epoch * 0.1}
        )
        
        # Save best checkpoint (epoch 3)
        if epoch == 3:
            manager.save_checkpoint(
                model, optimizer, epoch, mock_loss,
                is_best=True
            )
    
    # List checkpoints
    print("\nAvailable checkpoints:")
    for checkpoint in manager.list_checkpoints():
        print(f"  {checkpoint['filename']}: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f}")
    
    # Test loading best checkpoint
    print("\nLoading best checkpoint...")
    checkpoint_data, loaded_model = manager.load_checkpoint(load_best=True)
    print(f"Loaded epoch: {checkpoint_data['epoch']}")
    
    # Test resuming training
    print("\nTesting training resumption...")
    model, optimizer, start_epoch, best_loss = manager.resume_training(load_best=True)
    print(f"Would resume from epoch: {start_epoch}")
    
    # Cleanup
    manager.cleanup_all_checkpoints(keep_best=True)
    print("\nDemo completed!")


if __name__ == "__main__":
    demo_checkpointing()
