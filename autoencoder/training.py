"""
Training Framework for MNIST Autoencoder

This module provides comprehensive training functionality that matches SAS behavior,
including L-BFGS optimization, proper batch processing, convergence tracking,
and early stopping mechanisms.

Key features:
- L-BFGS optimizer with closure function support
- MSE reconstruction loss matching SAS behavior  
- Training metrics logging (loss per epoch, convergence tracking)
- Early stopping based on convergence criteria
- Batch processing for memory efficiency
- Integration with existing data processing modules
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime
import warnings

from model import MNISTAutoencoder, create_sas_compatible_autoencoder
from mnist_data import load_mnist_data, MNISTReader
from data_utils import train_validation_split, create_batches


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Model parameters
        self.input_dim = 784
        self.hidden_dim = 400
        self.dropout_rate = 0.0
        self.seed = 23451
        
        # Optimizer parameters (matching SAS L-BFGS configuration)
        self.optimizer_type = 'lbfgs'
        self.max_iters = 500  # maxiters=500 in SAS
        self.tolerance_grad = 1e-10  # fConv=1E-10 in SAS  
        self.tolerance_change = 1e-9
        self.history_size = 100
        self.line_search_fn = 'strong_wolfe'
        
        # Training parameters
        self.batch_size = None  # None means full batch (matching SAS behavior)
        self.max_epochs = 500
        self.early_stopping_patience = 50
        self.early_stopping_threshold = 1e-6
        
        # Data parameters
        self.validation_ratio = 0.2
        self.standardize_data = True
        
        # Logging and checkpointing
        self.log_interval = 1  # Log every epoch
        self.save_interval = 50  # Save checkpoint every 50 epochs
        self.save_best = True
        
        # Paths
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'


class TrainingMetrics:
    """Class to track and manage training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        self.convergence_metrics = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.start_time = None
        self.total_training_time = 0.0
        
    def update(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
               lr: float = 0.0, convergence_metric: Optional[float] = None):
        """Update metrics with new values."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.learning_rates.append(lr)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            # Track best validation loss
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
        else:
            # Track best training loss if no validation
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_epoch = epoch
                
        if convergence_metric is not None:
            self.convergence_metrics.append(convergence_metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        return {
            'total_epochs': len(self.epochs),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'training_time': self.total_training_time,
            'average_epoch_time': self.total_training_time / len(self.epochs) if self.epochs else 0
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        metrics_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'convergence_metrics': self.convergence_metrics,
            'summary': self.get_summary()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)


class AutoencoderTrainer:
    """
    Comprehensive trainer for MNIST Autoencoder that matches SAS behavior.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = TrainingMetrics()
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def setup_model(self) -> MNISTAutoencoder:
        """Initialize model with SAS-compatible architecture."""
        self.model = create_sas_compatible_autoencoder(seed=self.config.seed)
        self.model.to(self.device)
        
        print(f"Model initialized on device: {self.device}")
        arch_info = self.model.get_architecture_info()
        print("Architecture:")
        for key, value in arch_info.items():
            print(f"  {key}: {value}")
        
        return self.model
    
    def setup_optimizer(self):
        """Setup L-BFGS optimizer matching SAS configuration."""
        if self.config.optimizer_type.lower() == 'lbfgs':
            self.optimizer = optim.LBFGS(
                self.model.parameters(),
                max_iter=self.config.max_iters,
                tolerance_grad=self.config.tolerance_grad,
                tolerance_change=self.config.tolerance_change,
                history_size=self.config.history_size,
                line_search_fn=self.config.line_search_fn
            )
        else:
            # Fallback to Adam if L-BFGS not specified
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
        print(f"Optimizer: {self.config.optimizer_type}")
    
    def create_closure(self, inputs: torch.Tensor, targets: torch.Tensor) -> Callable:
        """
        Create closure function required by L-BFGS optimizer.
        
        Args:
            inputs: Input batch tensor
            targets: Target batch tensor (same as inputs for autoencoder)
            
        Returns:
            Closure function that computes loss and gradients
        """
        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.model.reconstruction_loss(targets, outputs)
            loss.backward()
            return loss
        
        return closure
    
    def train_batch(self, batch_data: torch.Tensor) -> float:
        """
        Train on a single batch using L-BFGS optimizer.
        
        Args:
            batch_data: Input batch tensor
            
        Returns:
            Training loss for this batch
        """
        batch_data = batch_data.to(self.device)
        
        if self.config.optimizer_type.lower() == 'lbfgs':
            # L-BFGS requires a closure function
            closure = self.create_closure(batch_data, batch_data)
            loss = self.optimizer.step(closure)
            
            # Get the actual loss value
            if isinstance(loss, torch.Tensor):
                return loss.item()
            else:
                # Evaluate closure one more time to get current loss
                with torch.no_grad():
                    outputs = self.model(batch_data)
                    current_loss = self.model.reconstruction_loss(batch_data, outputs)
                    return current_loss.item()
        else:
            # Standard optimizer (Adam, etc.)
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.model.reconstruction_loss(batch_data, outputs)
            loss.backward()
            self.optimizer.step()
            return loss.item()
    
    def validate(self, val_data: torch.Tensor) -> float:
        """
        Evaluate model on validation data.
        
        Args:
            val_data: Validation data tensor
            
        Returns:
            Validation loss
        """
        self.model.eval()
        val_data = val_data.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(val_data)
            val_loss = self.model.reconstruction_loss(val_data, outputs)
        
        self.model.train()
        return val_loss.item()
    
    def check_convergence(self, current_loss: float, window_size: int = 10) -> Tuple[bool, float]:
        """
        Check if training has converged based on loss improvement.
        
        Args:
            current_loss: Current epoch loss
            window_size: Number of recent epochs to consider
            
        Returns:
            Tuple of (converged, convergence_metric)
        """
        if len(self.metrics.train_losses) < window_size + 1:
            return False, float('inf')
        
        # Get recent losses
        recent_losses = self.metrics.train_losses[-window_size-1:-1]
        old_loss = np.mean(recent_losses)
        
        # Calculate relative improvement
        if old_loss > 0:
            improvement = (old_loss - current_loss) / old_loss
        else:
            improvement = 0.0
        
        # Check convergence criteria
        converged = improvement < self.config.early_stopping_threshold
        
        return converged, improvement
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_name = f"autoencoder_epoch_{epoch}.pt"
        if is_best:
            checkpoint_name = "autoencoder_best.pt"
        
        filepath = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        
        self.model.save_model_state(
            filepath, 
            epoch=epoch, 
            loss=loss, 
            optimizer_state=self.optimizer.state_dict()
        )
        
        print(f"Checkpoint saved: {checkpoint_name}")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  elapsed_time: float = 0.0, convergence_metric: Optional[float] = None):
        """Log epoch results."""
        log_msg = f"Epoch {epoch:3d}: Train Loss = {train_loss:.8f}"
        
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.8f}"
        
        if convergence_metric is not None:
            log_msg += f", Improvement = {convergence_metric:.2e}"
        
        log_msg += f", Time = {elapsed_time:.2f}s"
        
        print(log_msg)
    
    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None) -> TrainingMetrics:
        """
        Main training loop matching SAS behavior.
        
        Args:
            train_data: Training data array
            val_data: Optional validation data array
            
        Returns:
            Training metrics object
        """
        print("=== Starting Training ===")
        print(f"Training samples: {len(train_data)}")
        if val_data is not None:
            print(f"Validation samples: {len(val_data)}")
        
        # Setup model and optimizer
        if self.model is None:
            self.setup_model()
        if self.optimizer is None:
            self.setup_optimizer()
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data) if val_data is not None else None
        
        # Training setup
        self.metrics.reset()
        self.metrics.start_time = time.time()
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()
            
            # Training step
            self.model.train()
            
            if self.config.batch_size is None:
                # Full batch training (matching SAS behavior)
                train_loss = self.train_batch(train_tensor)
            else:
                # Batch training
                batch_losses = []
                for start_idx in range(0, len(train_tensor), self.config.batch_size):
                    end_idx = min(start_idx + self.config.batch_size, len(train_tensor))
                    batch = train_tensor[start_idx:end_idx]
                    batch_loss = self.train_batch(batch)
                    batch_losses.append(batch_loss)
                
                train_loss = np.mean(batch_losses)
            
            # Validation step
            val_loss = None
            if val_tensor is not None:
                val_loss = self.validate(val_tensor)
            
            # Check convergence
            converged, convergence_metric = self.check_convergence(train_loss)
            
            # Update metrics
            epoch_time = time.time() - epoch_start
            self.metrics.update(epoch, train_loss, val_loss, 
                              convergence_metric=convergence_metric)
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self.log_epoch(epoch, train_loss, val_loss, epoch_time, convergence_metric)
            
            # Save checkpoints
            is_best = (val_loss or train_loss) <= self.metrics.best_loss
            
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_loss)
            
            if is_best and self.config.save_best:
                self.save_checkpoint(epoch, self.metrics.best_loss, is_best=True)
            
            # Early stopping
            if converged:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (convergence achieved)")
                    break
            else:
                patience_counter = 0
        
        # Finalize training
        self.metrics.total_training_time = time.time() - self.metrics.start_time
        
        print("=== Training Completed ===")
        summary = self.metrics.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save final metrics
        metrics_path = os.path.join(self.config.log_dir, 'training_metrics.json')
        self.metrics.save_metrics(metrics_path)
        print(f"Metrics saved to: {metrics_path}")
        
        return self.metrics


def train_mnist_autoencoder(images_path: str, labels_path: str, 
                          config: Optional[TrainingConfig] = None) -> Tuple[MNISTAutoencoder, TrainingMetrics]:
    """
    End-to-end training function for MNIST autoencoder.
    
    Args:
        images_path: Path to MNIST images file
        labels_path: Path to MNIST labels file  
        config: Training configuration (uses defaults if None)
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    if config is None:
        config = TrainingConfig()
    
    print("=== MNIST Autoencoder Training ===")
    print(f"Loading data from: {images_path}, {labels_path}")
    
    # Load and preprocess data
    images, labels = load_mnist_data(images_path, labels_path, 
                                   standardize=config.standardize_data)
    
    print(f"Loaded {len(images)} samples")
    print(f"Image shape: {images.shape}")
    
    # Split data if validation requested
    val_images = None
    if config.validation_ratio > 0:
        train_images, val_images, _, _ = train_validation_split(
            images, labels, 
            validation_ratio=config.validation_ratio,
            random_seed=config.seed,
            stratify=False  # Not needed for autoencoder
        )
    else:
        train_images = images
    
    # Initialize trainer
    trainer = AutoencoderTrainer(config)
    
    # Train model
    metrics = trainer.train(train_images, val_images)
    
    return trainer.model, metrics


def demo_training():
    """Demo training with mock data."""
    print("=== Demo Training (Mock Data) ===")
    
    # Create mock MNIST-like data
    np.random.seed(23451)
    mock_images = np.random.rand(1000, 784).astype(np.float32)
    mock_labels = np.random.randint(0, 10, 1000)
    
    # Create configuration
    config = TrainingConfig()
    config.max_epochs = 10  # Short demo
    config.early_stopping_patience = 5
    config.log_interval = 1
    
    # Initialize trainer
    trainer = AutoencoderTrainer(config)
    
    # Split data for validation
    train_data, val_data, _, _ = train_validation_split(
        mock_images, mock_labels, validation_ratio=0.2, random_seed=23451
    )
    
    # Train
    metrics = trainer.train(train_data, val_data)
    
    print("Demo completed successfully!")
    return trainer.model, metrics


if __name__ == "__main__":
    demo_training()
