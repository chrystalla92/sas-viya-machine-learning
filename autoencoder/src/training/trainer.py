"""
PyTorch Autoencoder Training with L-BFGS Optimization

This module implements training for the MLP autoencoder using L-BFGS optimizer
with closure-based optimization, MSE loss, convergence checking, and checkpointing
to match SAS Viya training specifications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import sys

# Handle imports for both package and standalone execution
try:
    # Try relative imports first (when run as package)
    from ..models.autoencoder import Autoencoder
    from .metrics import TrainingMetrics
except ImportError:
    # Fall back to absolute imports (when run as standalone script)
    # Add parent directory to path to find modules
    sys.path.append(str(Path(__file__).parent.parent))
    from models.autoencoder import Autoencoder
    from training.metrics import TrainingMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    L-BFGS-based trainer for MLP autoencoder matching SAS specifications.
    
    This trainer implements:
    - L-BFGS optimizer with max 500 iterations
    - MSE reconstruction loss
    - Closure-based optimization for L-BFGS compatibility
    - Convergence checking with fConv=1E-10 tolerance
    - Training metrics tracking and logging
    - Model checkpointing and state saving
    - Reproducible results with seed=23451
    - Batch processing for larger datasets
    """
    
    def __init__(
        self,
        model: Autoencoder,
        max_iterations: int = 500,
        convergence_tolerance: float = 1e-10,
        seed: int = 23451,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        save_best: bool = True,
        patience: Optional[int] = None
    ):
        """
        Initialize the autoencoder trainer.
        
        Args:
            model: Autoencoder model to train
            max_iterations: Maximum training iterations (500 as per SAS spec)
            convergence_tolerance: Convergence tolerance fConv (1E-10 as per SAS spec)
            seed: Random seed for reproducibility (23451 as per SAS spec)
            device: Device to train on (auto-detect if None)
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model during training
            patience: Early stopping patience (None to disable)
        """
        self.model = model
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.seed = seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_best = save_best
        self.patience = patience
        
        # Ensure model is on correct device
        self.model = self.model.to(self.device)
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize loss function (MSE for reconstruction error)
        self.criterion = nn.MSELoss()
        
        # Initialize L-BFGS optimizer
        self.optimizer = optim.LBFGS(
            self.model.parameters(),
            max_iter=self.max_iterations,
            history_size=100,  # Default L-BFGS history size
            tolerance_grad=1e-7,
            tolerance_change=self.convergence_tolerance,
            line_search_fn='strong_wolfe'  # Recommended for L-BFGS
        )
        
        # Initialize training state
        self.current_iteration = 0
        self.best_loss = float('inf')
        self.converged = False
        self.training_complete = False
        
        # Initialize metrics tracker
        self.metrics = TrainingMetrics()
        
        # Training history
        self.loss_history: List[float] = []
        self.iteration_times: List[float] = []
        
        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"AutoencoderTrainer initialized:")
        logger.info(f"  - Max iterations: {self.max_iterations}")
        logger.info(f"  - Convergence tolerance: {self.convergence_tolerance}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Seed: {self.seed}")
        logger.info(f"  - Checkpoint dir: {self.checkpoint_dir}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _create_closure(self, data_batch: torch.Tensor) -> Callable[[], float]:
        """
        Create closure function required by L-BFGS optimizer.
        
        Args:
            data_batch: Input data batch
            
        Returns:
            Closure function that computes loss and gradients
        """
        def closure() -> float:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.model(data_batch)
            
            # Compute MSE reconstruction loss
            loss = self.criterion(reconstructed, data_batch)
            
            # Backward pass
            loss.backward()
            
            return float(loss.item())
        
        return closure
    
    def _check_convergence(self, current_loss: float) -> bool:
        """
        Check if training has converged based on loss improvement.
        
        Args:
            current_loss: Current iteration loss value
            
        Returns:
            True if converged, False otherwise
        """
        if len(self.loss_history) < 2:
            return False
            
        # Check relative change in loss
        previous_loss = self.loss_history[-2]
        if previous_loss == 0:
            relative_change = abs(current_loss)
        else:
            relative_change = abs((current_loss - previous_loss) / previous_loss)
            
        # Check absolute change
        absolute_change = abs(current_loss - previous_loss)
        
        # Convergence criteria (similar to SAS fConv)
        converged = (relative_change < self.convergence_tolerance and 
                    absolute_change < self.convergence_tolerance)
        
        if converged:
            logger.info(f"Convergence achieved at iteration {self.current_iteration}")
            logger.info(f"  - Relative change: {relative_change:.2e}")
            logger.info(f"  - Absolute change: {absolute_change:.2e}")
            logger.info(f"  - Tolerance: {self.convergence_tolerance:.2e}")
        
        return converged
    
    def _save_checkpoint(self, loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'converged': self.converged,
            'loss_history': self.loss_history,
            'seed': self.seed,
            'training_complete': self.training_complete
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best model
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss {loss:.6e} at iteration {self.current_iteration}")
    
    def _log_training_progress(self, iteration: int, loss: float, iteration_time: float):
        """
        Log training progress information.
        
        Args:
            iteration: Current iteration number
            loss: Current loss value
            iteration_time: Time taken for this iteration
        """
        # Update metrics
        self.metrics.update(
            iteration=iteration,
            loss=loss,
            learning_rate=None,  # L-BFGS doesn't have fixed LR
            convergence_check=self._check_convergence(loss)
        )
        
        # Log every 10 iterations or at convergence
        if iteration % 10 == 0 or self.converged or iteration == 1:
            avg_time = np.mean(self.iteration_times[-10:]) if self.iteration_times else iteration_time
            
            logger.info(f"Iteration {iteration:3d}/{self.max_iterations} | "
                       f"Loss: {loss:.6e} | "
                       f"Time: {iteration_time:.3f}s | "
                       f"Avg Time: {avg_time:.3f}s")
            
            if len(self.loss_history) >= 2:
                improvement = self.loss_history[-2] - loss
                logger.info(f"  Loss improvement: {improvement:.6e}")
    
    def train_single_batch(self, data_batch: torch.Tensor) -> float:
        """
        Train on a single batch using L-BFGS optimizer.
        
        Args:
            data_batch: Input data batch
            
        Returns:
            Final loss value after optimization
        """
        # Move data to device
        data_batch = data_batch.to(self.device)
        
        # Create closure for L-BFGS
        closure = self._create_closure(data_batch)
        
        # Perform optimization step
        start_time = time.time()
        
        # L-BFGS step with closure
        loss_value = self.optimizer.step(closure)
        
        iteration_time = time.time() - start_time
        
        # Update training state
        self.current_iteration += 1
        self.loss_history.append(float(loss_value))
        self.iteration_times.append(iteration_time)
        
        # Check if this is the best loss
        is_best = loss_value < self.best_loss
        if is_best:
            self.best_loss = loss_value
        
        # Check convergence
        if len(self.loss_history) >= 2:
            self.converged = self._check_convergence(loss_value)
        
        # Save checkpoint
        self._save_checkpoint(loss_value, is_best)
        
        # Log progress
        self._log_training_progress(self.current_iteration, loss_value, iteration_time)
        
        return float(loss_value)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch through the dataset.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(dataloader):
            # Train on batch
            batch_loss = self.train_single_batch(data)
            epoch_losses.append(batch_loss)
            
            # Check stopping conditions
            if self.current_iteration >= self.max_iterations:
                logger.info(f"Reached maximum iterations ({self.max_iterations})")
                self.training_complete = True
                break
                
            if self.converged:
                logger.info("Training converged - stopping early")
                self.training_complete = True
                break
        
        epoch_time = time.time() - epoch_start_time
        
        # Calculate epoch statistics
        epoch_stats = {
            'avg_loss': np.mean(epoch_losses),
            'min_loss': np.min(epoch_losses),
            'max_loss': np.max(epoch_losses),
            'num_batches': len(epoch_losses),
            'epoch_time': epoch_time,
            'iterations_completed': self.current_iteration,
            'converged': self.converged,
            'training_complete': self.training_complete
        }
        
        return epoch_stats
    
    def train(
        self, 
        dataloader: DataLoader, 
        num_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Main training loop for the autoencoder.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Maximum number of epochs (None for unlimited)
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with comprehensive training results
        """
        if verbose:
            logger.info("Starting autoencoder training...")
            logger.info(f"  - Dataset size: {len(dataloader.dataset)}")
            logger.info(f"  - Batch size: {dataloader.batch_size}")
            logger.info(f"  - Total batches: {len(dataloader)}")
        
        training_start_time = time.time()
        epoch = 0
        
        # Training loop
        while not self.training_complete:
            epoch += 1
            
            if verbose:
                logger.info(f"\n--- Epoch {epoch} ---")
            
            # Train one epoch
            epoch_stats = self.train_epoch(dataloader)
            
            if verbose:
                logger.info(f"Epoch {epoch} completed:")
                logger.info(f"  - Average loss: {epoch_stats['avg_loss']:.6e}")
                logger.info(f"  - Iterations: {epoch_stats['iterations_completed']}")
                logger.info(f"  - Time: {epoch_stats['epoch_time']:.2f}s")
                logger.info(f"  - Converged: {epoch_stats['converged']}")
            
            # Check epoch limit
            if num_epochs and epoch >= num_epochs:
                logger.info(f"Reached maximum epochs ({num_epochs})")
                self.training_complete = True
        
        total_training_time = time.time() - training_start_time
        
        # Prepare final results
        results = {
            'training_complete': self.training_complete,
            'converged': self.converged,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'best_loss': self.best_loss,
            'total_iterations': self.current_iteration,
            'total_epochs': epoch,
            'total_training_time': total_training_time,
            'average_iteration_time': np.mean(self.iteration_times) if self.iteration_times else None,
            'loss_history': self.loss_history.copy(),
            'metrics_summary': self.metrics.get_summary(),
            'seed': self.seed,
            'max_iterations': self.max_iterations,
            'convergence_tolerance': self.convergence_tolerance
        }
        
        if verbose:
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED")
            logger.info("="*60)
            logger.info(f"Final Loss:        {results['final_loss']:.6e}")
            logger.info(f"Best Loss:         {results['best_loss']:.6e}")
            logger.info(f"Total Iterations:  {results['total_iterations']}")
            logger.info(f"Total Epochs:      {results['total_epochs']}")
            logger.info(f"Training Time:     {results['total_training_time']:.2f}s")
            logger.info(f"Converged:         {results['converged']}")
            logger.info("="*60)
        
        return results
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                reconstructed = self.model(data)
                
                # Compute reconstruction loss
                loss = self.criterion(reconstructed, data)
                
                # Accumulate loss
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        return {
            'reconstruction_loss': avg_loss,
            'total_samples': total_samples,
            'loss_per_sample': avg_loss
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.current_iteration = checkpoint['iteration']
        self.best_loss = checkpoint['best_loss']
        self.converged = checkpoint['converged']
        self.loss_history = checkpoint['loss_history']
        self.training_complete = checkpoint.get('training_complete', False)
        
        logger.info(f"Loaded checkpoint from iteration {self.current_iteration}")
        logger.info(f"  - Loss: {checkpoint['loss']:.6e}")
        logger.info(f"  - Best loss: {self.best_loss:.6e}")
        
        return checkpoint
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training summary information
        """
        return {
            'model_architecture': self.model.get_architecture_summary(),
            'training_configuration': {
                'max_iterations': self.max_iterations,
                'convergence_tolerance': self.convergence_tolerance,
                'optimizer': 'L-BFGS',
                'loss_function': 'MSE',
                'device': str(self.device),
                'seed': self.seed
            },
            'training_state': {
                'current_iteration': self.current_iteration,
                'converged': self.converged,
                'training_complete': self.training_complete,
                'best_loss': self.best_loss,
                'final_loss': self.loss_history[-1] if self.loss_history else None
            },
            'performance': {
                'total_iterations': len(self.loss_history),
                'average_iteration_time': np.mean(self.iteration_times) if self.iteration_times else None,
                'loss_history_length': len(self.loss_history)
            }
        }


def create_trainer(
    model: Autoencoder,
    max_iterations: int = 500,
    convergence_tolerance: float = 1e-10,
    seed: int = 23451,
    **kwargs
) -> AutoencoderTrainer:
    """
    Factory function to create an autoencoder trainer.
    
    Args:
        model: Autoencoder model to train
        max_iterations: Maximum training iterations
        convergence_tolerance: Convergence tolerance
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for AutoencoderTrainer
        
    Returns:
        Initialized AutoencoderTrainer
    """
    trainer = AutoencoderTrainer(
        model=model,
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        seed=seed,
        **kwargs
    )
    
    logger.info(f"Created AutoencoderTrainer with {trainer.current_iteration} iterations completed")
    
    return trainer
