"""
Training utilities for PyTorch autoencoder training.

This module provides utilities for early stopping, learning rate scheduling,
model checkpointing, logging, and progress tracking.
"""

import os
import time
import json
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'EarlyStopping',
    'TrainingLogger', 
    'ModelCheckpoint',
    'create_lr_scheduler',
    'format_time',
    'print_training_progress',
    'validate_training_config',
    'get_device'
]


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    
    Monitors validation loss and stops training when it stops improving
    for a specified number of epochs (patience).
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights when stopping
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially save weights from
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                
        else:
            # No improvement
            self.wait += 1
            
        # Check if we should stop
        if self.wait >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
                
            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"Restored best weights from epoch {self.stopped_epoch - self.patience}")
                    
            return True
            
        return False
    
    def get_best_loss(self) -> float:
        """Get the best validation loss observed."""
        return self.best_loss
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0


class TrainingLogger:
    """
    Comprehensive logging system for training metrics and progress.
    
    Tracks loss curves, learning rates, epochs, and provides visualization.
    """
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = "autoencoder_training"):
        """
        Initialize training logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log data
        self.reset_logs()
        
    def reset_logs(self) -> None:
        """Reset all logged data."""
        self.logs = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': [],
            'batch_losses': [],
            'training_times': [],
            'config': {},
            'start_time': None,
            'end_time': None
        }
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        self.logs['config'] = config
        self.save_logs()
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  lr: float, epoch_time: float) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss for the epoch
            val_loss (float): Validation loss for the epoch
            lr (float): Current learning rate
            epoch_time (float): Time taken for the epoch in seconds
        """
        self.logs['epochs'].append(epoch)
        self.logs['train_losses'].append(train_loss)
        self.logs['val_losses'].append(val_loss)
        self.logs['learning_rates'].append(lr)
        self.logs['training_times'].append(epoch_time)
        
        self.save_logs()
        
    def log_batch_loss(self, batch_loss: float) -> None:
        """Log loss for a single batch."""
        self.logs['batch_losses'].append(batch_loss)
        
    def set_start_time(self) -> None:
        """Mark the start of training."""
        self.logs['start_time'] = time.time()
        
    def set_end_time(self) -> None:
        """Mark the end of training."""
        self.logs['end_time'] = time.time()
        self.save_logs()
        
    def get_total_training_time(self) -> float:
        """Get total training time in seconds."""
        if self.logs['start_time'] and self.logs['end_time']:
            return self.logs['end_time'] - self.logs['start_time']
        return 0.0
        
    def save_logs(self) -> None:
        """Save logs to JSON file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save logs: {e}")
            
    def load_logs(self) -> Dict[str, Any]:
        """Load logs from JSON file."""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
                    return self.logs
        except Exception as e:
            print(f"Warning: Could not load logs: {e}")
        return self.logs
        
    def plot_loss_curves(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot training and validation loss curves.
        
        Args:
            save_path (Optional[str]): Path to save the plot
            show (bool): Whether to display the plot
        """
        if not self.logs['epochs']:
            print("No training data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.logs['epochs'], self.logs['train_losses'], 'b-', label='Training Loss')
        plt.plot(self.logs['epochs'], self.logs['val_losses'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        plt.plot(self.logs['epochs'], self.logs['learning_rates'], 'g-', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale for learning rate
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curves saved to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        if not self.logs['epochs']:
            return {"status": "No training data"}
            
        summary = {
            "total_epochs": len(self.logs['epochs']),
            "final_train_loss": self.logs['train_losses'][-1],
            "final_val_loss": self.logs['val_losses'][-1],
            "best_val_loss": min(self.logs['val_losses']),
            "best_epoch": self.logs['epochs'][np.argmin(self.logs['val_losses'])],
            "total_training_time": self.get_total_training_time(),
            "average_epoch_time": np.mean(self.logs['training_times']),
            "final_learning_rate": self.logs['learning_rates'][-1]
        }
        
        return summary


class ModelCheckpoint:
    """
    Model checkpointing utility for saving best models and periodic saves.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 model_name: str = "autoencoder", 
                 save_best: bool = True,
                 save_periodic: bool = True,
                 save_frequency: int = 10,
                 verbose: bool = True):
        """
        Initialize model checkpoint utility.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            model_name (str): Base name for model files
            save_best (bool): Whether to save the best model based on validation loss
            save_periodic (bool): Whether to save periodic checkpoints
            save_frequency (int): Frequency of periodic saves (epochs)
            verbose (bool): Whether to print checkpoint messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_best = save_best
        self.save_periodic = save_periodic
        self.save_frequency = save_frequency
        self.verbose = verbose
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_checkpoint_path = None
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], epoch: int, train_loss: float,
                       val_loss: float, is_best: bool = False) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model (nn.Module): Model to save
            optimizer (torch.optim.Optimizer): Optimizer state
            scheduler (Optional[Any]): Learning rate scheduler
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            is_best (bool): Whether this is the best model so far
            
        Returns:
            str: Path to the saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_config': model.get_config() if hasattr(model, 'get_config') else {},
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
            self.best_checkpoint_path = checkpoint_path
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch_{epoch}.pth")
            
        torch.save(checkpoint, checkpoint_path)
        
        if self.verbose:
            checkpoint_type = "best" if is_best else f"epoch {epoch}"
            print(f"Checkpoint saved: {checkpoint_type} model -> {checkpoint_path}")
            
        return checkpoint_path
        
    def save_if_best(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler: Optional[Any], epoch: int, train_loss: float,
                     val_loss: float) -> bool:
        """
        Save checkpoint if this is the best validation loss so far.
        
        Returns:
            bool: True if checkpoint was saved (new best), False otherwise
        """
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, scheduler, epoch, 
                               train_loss, val_loss, is_best=True)
            return True
        return False
        
    def save_periodic(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler: Optional[Any], epoch: int, train_loss: float,
                     val_loss: float) -> bool:
        """
        Save periodic checkpoint based on save frequency.
        
        Returns:
            bool: True if checkpoint was saved, False otherwise
        """
        if self.save_periodic and epoch % self.save_frequency == 0:
            self.save_checkpoint(model, optimizer, scheduler, epoch,
                               train_loss, val_loss, is_best=False)
            return True
        return False
        
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            model (nn.Module): Model to load state into
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into
            scheduler (Optional[Any]): Scheduler to load state into
            device (Optional[torch.device]): Device to map tensors to
            
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.verbose:
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('val_loss', 'unknown')
            print(f"Loaded checkpoint: epoch {epoch}, validation loss: {val_loss}")
            
        return checkpoint
        
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to the best checkpoint if it exists."""
        return self.best_checkpoint_path


def create_lr_scheduler(optimizer: torch.optim.Optimizer, 
                       scheduler_type: str = 'ReduceLROnPlateau',
                       **scheduler_kwargs) -> Any:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        scheduler_type (str): Type of scheduler ('ReduceLROnPlateau' or 'StepLR')
        **scheduler_kwargs: Additional arguments for the scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'ReduceLROnPlateau':
        default_args = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'verbose': True,
            'min_lr': 1e-6
        }
        default_args.update(scheduler_kwargs)
        return ReduceLROnPlateau(optimizer, **default_args)
        
    elif scheduler_type == 'StepLR':
        default_args = {
            'step_size': 30,
            'gamma': 0.1
        }
        default_args.update(scheduler_kwargs)
        return StepLR(optimizer, **default_args)
        
    else:
        supported = ['ReduceLROnPlateau', 'StepLR']
        raise ValueError(f"Unsupported scheduler type '{scheduler_type}'. Supported: {supported}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {secs:.0f}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(mins)}m {secs:.0f}s"


def print_training_progress(epoch: int, total_epochs: int, 
                          train_loss: float, val_loss: float,
                          lr: float, epoch_time: float,
                          best_val_loss: float) -> None:
    """
    Print formatted training progress.
    
    Args:
        epoch (int): Current epoch
        total_epochs (int): Total number of epochs
        train_loss (float): Training loss
        val_loss (float): Validation loss
        lr (float): Current learning rate
        epoch_time (float): Time for this epoch
        best_val_loss (float): Best validation loss so far
    """
    progress = f"[{epoch:3d}/{total_epochs}]"
    losses = f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
    best_indicator = "★" if val_loss <= best_val_loss else " "
    lr_info = f"LR: {lr:.2e}"
    time_info = f"Time: {format_time(epoch_time)}"
    
    print(f"{progress} {losses} {best_indicator} | {lr_info} | {time_info}")


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set defaults for training configuration.
    
    Args:
        config (Dict[str, Any]): Training configuration
        
    Returns:
        Dict[str, Any]: Validated configuration with defaults
        
    Raises:
        ValueError: If configuration is invalid
    """
    default_config = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'early_stopping_patience': 7,
        'early_stopping_min_delta': 0.0,
        'lr_scheduler_type': 'ReduceLROnPlateau',
        'lr_scheduler_patience': 5,
        'lr_scheduler_factor': 0.5,
        'train_val_split': 0.8,
        'save_best_model': True,
        'save_periodic_checkpoints': True,
        'checkpoint_frequency': 10,
        'log_frequency': 1,
        'device': 'auto'  # 'auto', 'cpu', 'cuda', or specific device
    }
    
    # Update with provided config
    validated_config = default_config.copy()
    validated_config.update(config)
    
    # Validation checks
    if validated_config['epochs'] <= 0:
        raise ValueError("epochs must be positive")
    
    if validated_config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
        
    if validated_config['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
        
    if not 0 < validated_config['train_val_split'] < 1:
        raise ValueError("train_val_split must be between 0 and 1")
        
    if validated_config['early_stopping_patience'] < 1:
        raise ValueError("early_stopping_patience must be >= 1")
        
    return validated_config


def get_device(device_spec: str = 'auto') -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_spec (str): Device specification ('auto', 'cpu', 'cuda', or specific device)
        
    Returns:
        torch.device: The selected device
    """
    if device_spec == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_spec)
        
    return device
