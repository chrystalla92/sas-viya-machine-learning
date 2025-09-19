"""
Comprehensive training orchestration for MLP Autoencoder.

This module implements a complete training pipeline including:
- Loss computation and optimization
- Epoch-based training with batch processing
- Model checkpointing and best model saving
- Training progress logging and metrics tracking
- Early stopping and learning rate scheduling
- Training resumption and graceful shutdown handling
"""

import os
import time
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..models.autoencoder import MLPAutoencoder


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adam"  # "adam", "lbfgs", "sgd"
    weight_decay: float = 0.0
    
    # Loss and validation
    loss_function: str = "mse"  # "mse", "bce"
    validation_split: float = 0.2
    validation_freq: int = 1  # Validate every N epochs
    
    # Checkpointing
    save_dir: Union[str, Path] = "checkpoints"
    save_best_only: bool = True
    save_frequency: int = 10  # Save every N epochs
    checkpoint_format: str = "pytorch"  # "pytorch", "onnx"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    lr_scheduler: Optional[str] = None  # "step", "exponential", "reduce_on_plateau"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Logging and monitoring
    log_frequency: int = 10  # Log every N batches
    verbose: bool = True
    save_logs: bool = True
    log_dir: Union[str, Path] = "logs"
    
    # Device and performance
    device: Optional[str] = None
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class TrainingMetrics:
    """Training metrics and history."""
    
    epoch: int = 0
    train_loss: float = float('inf')
    val_loss: float = float('inf')
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    
    # History tracking
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    learning_rate_history: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # Best metrics tracking
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    epochs_without_improvement: int = 0


class GradientMonitor:
    """Monitor gradient flow and detect gradient problems."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_norms = []
        self.weight_norms = []
    
    def check_gradients(self) -> Dict[str, Any]:
        """Check for gradient explosion/vanishing."""
        total_norm = 0.0
        param_count = 0
        
        gradient_info = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                gradient_info[f"{name}_grad_norm"] = param_norm.item()
                gradient_info[f"{name}_weight_norm"] = param.data.norm(2).item()
        
        total_norm = total_norm ** (1. / 2)
        gradient_info["total_grad_norm"] = total_norm
        gradient_info["param_count"] = param_count
        
        # Detect gradient problems
        gradient_info["gradient_explosion"] = total_norm > 10.0
        gradient_info["gradient_vanishing"] = total_norm < 1e-6
        
        # Store for history
        self.gradient_norms.append(total_norm)
        
        return gradient_info


class Trainer:
    """
    Comprehensive trainer for MLP Autoencoder with full training orchestration.
    
    Features:
    - Multiple optimizer support (Adam, L-BFGS, SGD) to match SAS PROC NNET
    - Automatic checkpointing with best model saving
    - Training progress monitoring with comprehensive logging
    - Early stopping with patience mechanism
    - Learning rate scheduling options
    - Gradient monitoring for training stability
    - Graceful shutdown handling for training interruption
    - Training resumption from checkpoints
    """
    
    def __init__(
        self,
        model: MLPAutoencoder,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer with model and configuration.
        
        Args:
            model: MLPAutoencoder model to train
            config: Training configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or TrainingConfig()
        
        # Initialize training state
        self.metrics = TrainingMetrics()
        self.gradient_monitor = GradientMonitor(model)
        
        # Set up device
        self.device = torch.device(
            self.config.device or 
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.loss_function: Optional[Callable] = None
        
        # Set up directories
        self.save_dir = Path(self.config.save_dir)
        self.log_dir = Path(self.config.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Training state flags
        self._should_stop = False
        self._training_interrupted = False
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Set reproducibility
        if self.config.seed is not None:
            self._set_seed(self.config.seed)
        
        if self.config.verbose:
            self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        self.logger = logging.getLogger(f"Trainer_{id(self)}")
        self.logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if self.config.verbose:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.save_logs:
            log_file = self.log_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
            self._should_stop = True
            self._training_interrupted = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "lbfgs":
            # L-BFGS for SAS PROC NNET compatibility
            return torch.optim.LBFGS(
                params,
                lr=self.config.learning_rate,
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_loss_function(self) -> Callable:
        """Create loss function based on configuration."""
        if self.config.loss_function.lower() == "mse":
            return F.mse_loss
        elif self.config.loss_function.lower() == "bce":
            return F.binary_cross_entropy
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if not self.config.lr_scheduler:
            return None
        
        scheduler_type = self.config.lr_scheduler.lower()
        params = self.config.lr_scheduler_params
        
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 30),
                gamma=params.get("gamma", 0.1)
            )
        elif scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=params.get("gamma", 0.95)
            )
        elif scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 5),
                min_lr=params.get("min_lr", 1e-7)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def prepare_training(self) -> None:
        """Prepare training components (optimizer, scheduler, loss function)."""
        self.optimizer = self._create_optimizer()
        self.loss_function = self._create_loss_function()
        self.scheduler = self._create_scheduler()
        
        if self.config.verbose:
            self.logger.info(f"Training preparation complete:")
            self.logger.info(f"  Optimizer: {self.config.optimizer}")
            self.logger.info(f"  Loss function: {self.config.loss_function}")
            self.logger.info(f"  Learning rate: {self.config.learning_rate}")
            self.logger.info(f"  Scheduler: {self.config.lr_scheduler or 'None'}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Train model for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, training_metrics)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        epoch_start_time = time.time()
        
        batch_losses = []
        gradient_info = {}
        
        for batch_idx, (data, _) in enumerate(train_loader):
            if self._should_stop:
                break
            
            # Move data to device
            data = data.to(self.device)
            
            # Handle different optimizers
            if self.config.optimizer.lower() == "lbfgs":
                # L-BFGS requires closure function
                def closure():
                    self.optimizer.zero_grad()
                    reconstructed = self.model(data)
                    loss = self.loss_function(reconstructed, data)
                    loss.backward()
                    return loss
                
                loss = self.optimizer.step(closure)
                if isinstance(loss, torch.Tensor):
                    loss_value = loss.item()
                else:
                    loss_value = float(loss)
            else:
                # Standard optimizers (Adam, SGD)
                self.optimizer.zero_grad()
                reconstructed = self.model(data)
                loss = self.loss_function(reconstructed, data)
                loss.backward()
                
                # Check gradients before optimization step
                if batch_idx % (self.config.log_frequency * 10) == 0:
                    gradient_info = self.gradient_monitor.check_gradients()
                
                self.optimizer.step()
                loss_value = loss.item()
            
            total_loss += loss_value
            batch_losses.append(loss_value)
            
            # Log training progress
            if batch_idx % self.config.log_frequency == 0 and self.config.verbose:
                progress = 100.0 * batch_idx / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch:3d} [{batch_idx:6d}/{num_batches:6d}] "
                    f"({progress:6.1f}%) | Loss: {loss_value:.6f} | LR: {current_lr:.2e}"
                )
                
                # Log gradient information if available
                if gradient_info and gradient_info.get("gradient_explosion", False):
                    self.logger.warning(f"Gradient explosion detected! Norm: {gradient_info['total_grad_norm']:.2e}")
                if gradient_info and gradient_info.get("gradient_vanishing", False):
                    self.logger.warning(f"Gradient vanishing detected! Norm: {gradient_info['total_grad_norm']:.2e}")
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        training_metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "epoch_time": epoch_time,
            "num_batches": num_batches,
            "batch_losses": batch_losses,
            "gradient_info": gradient_info,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        return avg_loss, training_metrics
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Validate model for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, validation_metrics)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                if self._should_stop:
                    break
                
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                reconstructed = self.model(data)
                loss = self.loss_function(reconstructed, data)
                
                loss_value = loss.item()
                total_loss += loss_value
                val_losses.append(loss_value)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        validation_metrics = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "batch_losses": val_losses
        }
        
        return avg_loss, validation_metrics
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if not self.config.early_stopping:
            return False
        
        # Check if validation loss improved
        if val_loss < self.metrics.best_val_loss - self.config.min_delta:
            self.metrics.best_val_loss = val_loss
            self.metrics.best_epoch = self.metrics.epoch
            self.metrics.epochs_without_improvement = 0
            return False
        else:
            self.metrics.epochs_without_improvement += 1
            
            if self.metrics.epochs_without_improvement >= self.config.patience:
                self.logger.info(
                    f"Early stopping triggered after {self.config.patience} epochs "
                    f"without improvement (best loss: {self.metrics.best_val_loss:.6f} "
                    f"at epoch {self.metrics.best_epoch})"
                )
                return True
        
        return False
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": self.metrics,
            "config": self.config,
            "model_info": self.model.get_model_info(),
        }
        
        if additional_info:
            checkpoint_data.update(additional_info)
        
        # Save regular checkpoint
        if is_best:
            checkpoint_path = self.save_dir / "best_model.pth"
        else:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Also save a latest checkpoint
        latest_path = self.save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint_data, latest_path)
        
        if self.config.verbose:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model checkpoint and resume training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Use weights_only=False for compatibility with custom classes
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer state if available and compatible
            if self.optimizer and "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e:
                    self.logger.warning(f"Could not load optimizer state: {e}")
            
            # Load scheduler state if available and compatible
            if self.scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    self.logger.warning(f"Could not load scheduler state: {e}")
            
            # Load metrics
            if "metrics" in checkpoint:
                self.metrics = checkpoint["metrics"]
            
            self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch: {checkpoint.get('epoch', 0)}")
            
            return checkpoint
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def save_training_history(self) -> Path:
        """
        Save training history to JSON file.
        
        Returns:
            Path to saved history file
        """
        history_data = {
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "loss_function": self.config.loss_function,
            },
            "metrics": {
                "train_loss_history": self.metrics.train_loss_history,
                "val_loss_history": self.metrics.val_loss_history,
                "learning_rate_history": self.metrics.learning_rate_history,
                "epoch_times": self.metrics.epoch_times,
                "best_val_loss": self.metrics.best_val_loss,
                "best_epoch": self.metrics.best_epoch,
            }
        }
        
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        if self.config.verbose:
            self.logger.info(f"Training history saved: {history_path}")
        
        return history_path
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Train the model with full orchestration.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            resume_from: Path to checkpoint to resume from (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Prepare training components
        self.prepare_training()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            start_epoch = checkpoint.get("epoch", 0) + 1
        
        # Initialize training state
        self._should_stop = False
        self._training_interrupted = False
        
        training_start_time = time.time()
        
        self.logger.info("=" * 50)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Epochs: {start_epoch} → {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info("=" * 50)
        
        try:
            # Training loop
            for epoch in range(start_epoch, self.config.epochs):
                if self._should_stop:
                    self.logger.info("Training stopped by user request")
                    break
                
                # Update current epoch
                self.metrics.epoch = epoch
                
                # Training phase
                train_loss, train_metrics = self.train_epoch(train_loader, epoch)
                
                # Update metrics
                self.metrics.train_loss = train_loss
                self.metrics.train_loss_history.append(train_loss)
                self.metrics.learning_rate = self.optimizer.param_groups[0]['lr']
                self.metrics.learning_rate_history.append(self.metrics.learning_rate)
                self.metrics.epoch_time = train_metrics["epoch_time"]
                self.metrics.epoch_times.append(self.metrics.epoch_time)
                
                # Validation phase
                val_loss = float('inf')
                if val_loader and (epoch % self.config.validation_freq == 0):
                    val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
                    self.metrics.val_loss = val_loss
                    self.metrics.val_loss_history.append(val_loss)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Log epoch results
                if self.config.verbose:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.config.epochs:3d} | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"LR: {self.metrics.learning_rate:.2e} | "
                        f"Time: {self.metrics.epoch_time:.1f}s"
                    )
                
                # Check for best model
                is_best = val_loss < self.metrics.best_val_loss
                if is_best:
                    self.metrics.best_val_loss = val_loss
                    self.metrics.best_epoch = epoch
                
                # Save checkpoint
                if (epoch % self.config.save_frequency == 0 or is_best or epoch == self.config.epochs - 1):
                    if self.config.save_best_only:
                        if is_best:
                            self.save_checkpoint(epoch, is_best=True)
                    else:
                        self.save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping check
                if val_loader and self.check_early_stopping(val_loss):
                    self._should_stop = True
                    break
        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._training_interrupted = True
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Save final checkpoint and history
            if not self._training_interrupted:
                self.save_checkpoint(self.metrics.epoch, is_best=False)
            self.save_training_history()
        
        # Calculate training results
        total_training_time = time.time() - training_start_time
        
        results = {
            "training_completed": not self._training_interrupted,
            "total_epochs": self.metrics.epoch + 1,
            "total_training_time": total_training_time,
            "best_val_loss": self.metrics.best_val_loss,
            "best_epoch": self.metrics.best_epoch,
            "final_train_loss": self.metrics.train_loss,
            "final_val_loss": self.metrics.val_loss,
            "metrics": self.metrics,
        }
        
        self.logger.info("=" * 50)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Status: {'Completed' if results['training_completed'] else 'Interrupted'}")
        self.logger.info(f"Total epochs: {results['total_epochs']}")
        self.logger.info(f"Total time: {total_training_time:.1f}s")
        self.logger.info(f"Best validation loss: {results['best_val_loss']:.6f} (epoch {results['best_epoch']})")
        self.logger.info(f"Final train loss: {results['final_train_loss']:.6f}")
        self.logger.info(f"Final validation loss: {results['final_val_loss']:.6f}")
        self.logger.info("=" * 50)
        
        return results
    
    def predict(
        self,
        data_loader: DataLoader,
        return_reconstructions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions/reconstructions for given data.
        
        Args:
            data_loader: Data loader with input data
            return_reconstructions: Whether to return reconstruction outputs
            
        Returns:
            Dictionary containing predictions and optionally reconstructions
        """
        self.model.eval()
        
        all_inputs = []
        all_reconstructions = []
        all_encodings = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                
                # Get encodings and reconstructions
                encoded = self.model.encode(data)
                reconstructed = self.model.decode(encoded)
                
                all_inputs.append(data.cpu())
                all_encodings.append(encoded.cpu())
                if return_reconstructions:
                    all_reconstructions.append(reconstructed.cpu())
        
        results = {
            "inputs": torch.cat(all_inputs, dim=0),
            "encodings": torch.cat(all_encodings, dim=0)
        }
        
        if return_reconstructions:
            results["reconstructions"] = torch.cat(all_reconstructions, dim=0)
        
        return results
    
    def evaluate_model(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model performance on given dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        mse_sum = 0.0
        mae_sum = 0.0
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                reconstructed = self.model(data)
                
                # Calculate various metrics
                batch_size = data.size(0)
                total_samples += batch_size
                
                # MSE loss
                mse = F.mse_loss(reconstructed, data, reduction='sum')
                mse_sum += mse.item()
                
                # MAE loss
                mae = F.l1_loss(reconstructed, data, reduction='sum')
                mae_sum += mae.item()
                
                # Primary loss
                loss = self.loss_function(reconstructed, data, reduction='sum')
                total_loss += loss.item()
        
        # Calculate averages
        avg_loss = total_loss / total_samples
        avg_mse = mse_sum / total_samples
        avg_mae = mae_sum / total_samples
        rmse = (avg_mse) ** 0.5
        
        return {
            "loss": avg_loss,
            "mse": avg_mse,
            "mae": avg_mae,
            "rmse": rmse,
            "num_samples": total_samples
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary containing complete training information
        """
        model_info = self.model.get_model_info()
        
        return {
            "model_info": model_info,
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
                "loss_function": self.config.loss_function,
                "early_stopping": self.config.early_stopping,
                "patience": self.config.patience,
            },
            "training_metrics": {
                "current_epoch": self.metrics.epoch,
                "train_loss": self.metrics.train_loss,
                "val_loss": self.metrics.val_loss,
                "best_val_loss": self.metrics.best_val_loss,
                "best_epoch": self.metrics.best_epoch,
                "epochs_without_improvement": self.metrics.epochs_without_improvement,
                "total_train_losses": len(self.metrics.train_loss_history),
                "total_val_losses": len(self.metrics.val_loss_history),
            },
            "performance": {
                "avg_epoch_time": sum(self.metrics.epoch_times) / len(self.metrics.epoch_times) if self.metrics.epoch_times else 0.0,
                "total_training_time": sum(self.metrics.epoch_times),
                "gradient_norms": self.gradient_monitor.gradient_norms[-10:] if self.gradient_monitor.gradient_norms else [],
            }
        }


def create_data_loaders(
    dataset,
    batch_size: int = 64,
    validation_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: Optional[int] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders from dataset.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size for data loaders
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for GPU transfer
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if validation_split = 0
    """
    if validation_split == 0.0:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, None
    
    # Split dataset
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Set seed for reproducible splits
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
