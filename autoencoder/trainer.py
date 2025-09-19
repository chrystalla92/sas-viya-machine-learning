"""
PyTorch Training Pipeline for Autoencoder Models.

This module provides a comprehensive training pipeline with early stopping,
learning rate scheduling, model checkpointing, and comprehensive logging.
"""

import time
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .autoencoder_model import AutoencoderMLP
from .datasets import create_mnist_dataloaders
from .training_utils import (
    EarlyStopping,
    TrainingLogger,
    ModelCheckpoint,
    create_lr_scheduler,
    print_training_progress,
    validate_training_config,
    get_device
)


__all__ = [
    'TrainingPipeline',
    'create_training_pipeline',
    'train_autoencoder'
]


class TrainingPipeline:
    """
    Comprehensive training pipeline for autoencoder models.
    
    Features:
    - Automatic GPU detection and usage
    - Early stopping with configurable patience
    - Learning rate scheduling (ReduceLROnPlateau recommended)
    - Model checkpointing (best model and periodic saves)
    - Comprehensive logging and progress tracking
    - Memory-efficient training and validation loops
    """
    
    def __init__(self, model: AutoencoderMLP, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training pipeline.
        
        Args:
            model (AutoencoderMLP): The autoencoder model to train
            config (Optional[Dict[str, Any]]): Training configuration
        """
        # Validate and set configuration
        self.config = validate_training_config(config or {})
        
        # Set device
        self.device = get_device(self.config['device'])
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_loss_function()
        self._setup_scheduler()
        self._setup_utilities()
        
        # Training state
        self.current_epoch = 0
        self.training_complete = False
        
    def _setup_optimizer(self) -> None:
        """Setup the Adam optimizer."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
    def _setup_loss_function(self) -> None:
        """Setup the MSE loss function."""
        self.criterion = nn.MSELoss()
        
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_kwargs = {
            'patience': self.config['lr_scheduler_patience'],
            'factor': self.config['lr_scheduler_factor'],
            'min_lr': 1e-7,
            'verbose': True
        }
        
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            self.config['lr_scheduler_type'],
            **scheduler_kwargs
        )
        
    def _setup_utilities(self) -> None:
        """Setup training utilities (early stopping, logging, checkpointing)."""
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            min_delta=self.config['early_stopping_min_delta'],
            verbose=True
        )
        
        # Logging
        self.logger = TrainingLogger()
        self.logger.log_config(self.config)
        
        # Checkpointing
        self.checkpoint = ModelCheckpoint(
            save_best=self.config['save_best_model'],
            save_periodic=self.config['save_periodic_checkpoints'],
            save_frequency=self.config['checkpoint_frequency']
        )
        
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle different batch formats (with or without labels)
            if isinstance(batch_data, tuple):
                inputs = batch_data[0].to(self.device)
            else:
                inputs = batch_data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstruction = self.model(inputs, return_latent=False)
            
            # Calculate loss
            loss = self.criterion(reconstruction, inputs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional, helps with stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Log batch loss (optional, for detailed monitoring)
            self.logger.log_batch_loss(loss.item())
            
        return total_loss / num_batches
        
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Handle different batch formats (with or without labels)
                if isinstance(batch_data, tuple):
                    inputs = batch_data[0].to(self.device)
                else:
                    inputs = batch_data.to(self.device)
                
                # Forward pass
                reconstruction = self.model(inputs, return_latent=False)
                
                # Calculate loss
                loss = self.criterion(reconstruction, inputs)
                
                # Accumulate loss
                total_loss += loss.item()
                
        return total_loss / num_batches
        
    def train(self, data_dir: str = "./data", 
             train_loader: Optional[DataLoader] = None,
             val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Train the autoencoder model.
        
        Args:
            data_dir (str): Directory containing MNIST data files
            train_loader (Optional[DataLoader]): Custom training data loader
            val_loader (Optional[DataLoader]): Custom validation data loader
            
        Returns:
            Dict[str, Any]: Training results and summary
        """
        print("=" * 60)
        print("Starting Autoencoder Training")
        print("=" * 60)
        
        # Create data loaders if not provided
        if train_loader is None or val_loader is None:
            print("Creating data loaders...")
            train_loader, val_loader, _ = create_mnist_dataloaders(
                data_dir=data_dir,
                batch_size=self.config['batch_size'],
                train_val_split=self.config['train_val_split']
            )
            
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Print model summary
        print(f"\nModel: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        
        # Start logging
        self.logger.set_start_time()
        
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print("-" * 60)
        
        best_val_loss = float('inf')
        
        try:
            for epoch in range(1, self.config['epochs'] + 1):
                epoch_start_time = time.time()
                
                # Train for one epoch
                train_loss = self._train_epoch(train_loader)
                
                # Validate for one epoch
                val_loss = self._validate_epoch(val_loader)
                
                epoch_time = time.time() - epoch_start_time
                
                # Update best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log epoch results
                self.logger.log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time)
                
                # Print progress
                if epoch % self.config['log_frequency'] == 0:
                    print_training_progress(epoch, self.config['epochs'], train_loss, 
                                          val_loss, current_lr, epoch_time, best_val_loss)
                
                # Update learning rate scheduler
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Check for best model and save checkpoint
                is_best = self.checkpoint.save_if_best(
                    self.model, self.optimizer, self.scheduler,
                    epoch, train_loss, val_loss
                )
                
                # Save periodic checkpoint
                self.checkpoint.save_periodic(
                    self.model, self.optimizer, self.scheduler,
                    epoch, train_loss, val_loss
                )
                
                # Check early stopping
                if self.early_stopping(val_loss, self.model):
                    print(f"\nEarly stopping at epoch {epoch}")
                    self.current_epoch = epoch
                    break
                    
                self.current_epoch = epoch
                
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user at epoch {self.current_epoch}")
            
        finally:
            # Mark training as complete
            self.training_complete = True
            self.logger.set_end_time()
            
        print("-" * 60)
        print("Training completed!")
        
        # Get training summary
        summary = self._get_training_summary()
        self._print_training_summary(summary)
        
        return summary
        
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        logger_summary = self.logger.get_summary()
        
        summary = {
            'training_completed': self.training_complete,
            'total_epochs_trained': self.current_epoch,
            'early_stopped': self.current_epoch < self.config['epochs'],
            'best_validation_loss': self.early_stopping.get_best_loss(),
            'final_train_loss': logger_summary.get('final_train_loss', 0),
            'final_val_loss': logger_summary.get('final_val_loss', 0),
            'total_training_time': logger_summary.get('total_training_time', 0),
            'average_epoch_time': logger_summary.get('average_epoch_time', 0),
            'final_learning_rate': logger_summary.get('final_learning_rate', 0),
            'best_checkpoint_path': self.checkpoint.get_best_checkpoint_path(),
            'model_config': self.model.get_config(),
            'training_config': self.config
        }
        
        return summary
        
    def _print_training_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted training summary."""
        print("\nTraining Summary:")
        print("=" * 40)
        print(f"Total epochs: {summary['total_epochs_trained']}")
        print(f"Early stopped: {summary['early_stopped']}")
        print(f"Best validation loss: {summary['best_validation_loss']:.6f}")
        print(f"Final learning rate: {summary['final_learning_rate']:.2e}")
        
        total_time = summary['total_training_time']
        avg_time = summary['average_epoch_time']
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Average epoch time: {avg_time:.1f} seconds")
        
        if summary['best_checkpoint_path']:
            print(f"Best model saved: {summary['best_checkpoint_path']}")
            
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        print("Evaluating model on test data...")
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Handle different batch formats
                if isinstance(batch_data, tuple):
                    inputs = batch_data[0].to(self.device)
                else:
                    inputs = batch_data.to(self.device)
                
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Forward pass
                reconstruction = self.model(inputs, return_latent=False)
                
                # Calculate loss
                loss = self.criterion(reconstruction, inputs)
                total_loss += loss.item() * batch_size
                
        avg_loss = total_loss / total_samples
        
        metrics = {
            'test_loss': avg_loss,
            'test_samples': total_samples
        }
        
        print(f"Test Loss: {avg_loss:.6f}")
        return metrics
        
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'training_config': self.config,
            'training_complete': self.training_complete
        }, save_path)
        print(f"Model saved to: {save_path}")
        
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            load_path (str): Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_complete' in checkpoint:
            self.training_complete = checkpoint['training_complete']
            
        print(f"Model loaded from: {load_path}")
        
    def plot_training_curves(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot training curves.
        
        Args:
            save_path (Optional[str]): Path to save the plot
            show (bool): Whether to display the plot
        """
        self.logger.plot_loss_curves(save_path, show)
        
    def get_model(self) -> AutoencoderMLP:
        """Get the trained model."""
        return self.model
        
    def get_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config
        
    def get_device(self) -> torch.device:
        """Get the device being used for training."""
        return self.device


def create_training_pipeline(model: AutoencoderMLP, 
                           config: Optional[Dict[str, Any]] = None) -> TrainingPipeline:
    """
    Factory function to create a training pipeline.
    
    Args:
        model (AutoencoderMLP): Model to train
        config (Optional[Dict[str, Any]]): Training configuration
        
    Returns:
        TrainingPipeline: Configured training pipeline
    """
    return TrainingPipeline(model, config)


def train_autoencoder(model: AutoencoderMLP,
                     data_dir: str = "./data",
                     config: Optional[Dict[str, Any]] = None) -> Tuple[TrainingPipeline, Dict[str, Any]]:
    """
    Convenience function to train an autoencoder with default settings.
    
    Args:
        model (AutoencoderMLP): Model to train
        data_dir (str): Directory containing MNIST data
        config (Optional[Dict[str, Any]]): Training configuration
        
    Returns:
        Tuple[TrainingPipeline, Dict[str, Any]]: (training_pipeline, summary)
    """
    # Create training pipeline
    pipeline = create_training_pipeline(model, config)
    
    # Train the model
    summary = pipeline.train(data_dir)
    
    return pipeline, summary
