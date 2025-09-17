"""
Training utilities and helper functions for MLP Autoencoder.

This module provides utility functions for training configuration, 
data preparation, model evaluation, and training orchestration helpers.
"""

import os
import json
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import asdict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Import matplotlib only when needed to avoid issues in headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from .trainer import TrainingConfig, TrainingMetrics, Trainer
from ..models.autoencoder import MLPAutoencoder
from ..data.dataset import MNISTDataset


def create_training_config(
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    early_stopping: bool = True,
    patience: int = 10,
    **kwargs
) -> TrainingConfig:
    """
    Create a training configuration with sensible defaults.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ("adam", "lbfgs", "sgd")
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        **kwargs: Additional configuration parameters
        
    Returns:
        TrainingConfig object
    """
    config_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "early_stopping": early_stopping,
        "patience": patience,
    }
    config_dict.update(kwargs)
    
    return TrainingConfig(**config_dict)


def create_sas_compatible_config() -> TrainingConfig:
    """
    Create training configuration that matches SAS PROC NNET behavior.
    
    SAS PROC NNET typically uses:
    - L-BFGS optimizer for better convergence
    - Higher patience for early stopping
    - More conservative learning rates
    
    Returns:
        TrainingConfig optimized for SAS compatibility
    """
    return TrainingConfig(
        epochs=200,
        batch_size=32,  # Smaller batches for L-BFGS
        learning_rate=0.01,  # Higher LR for L-BFGS
        optimizer="lbfgs",
        weight_decay=0.0001,
        early_stopping=True,
        patience=20,  # More patience for L-BFGS convergence
        lr_scheduler="reduce_on_plateau",
        lr_scheduler_params={
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6
        },
        validation_freq=5,  # Less frequent validation for L-BFGS
        log_frequency=1,  # Log every batch for L-BFGS
        save_frequency=25,
        deterministic=True
    )


def prepare_mnist_data(
    data_dir: Union[str, Path] = "./data",
    batch_size: int = 64,
    validation_split: float = 0.2,
    normalize: str = "01",
    num_workers: int = 4,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare MNIST data loaders for training.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        validation_split: Fraction for validation split
        normalize: Normalization method ("01" or "11")
        num_workers: Number of data loader workers
        download: Whether to download data if not found
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .trainer import create_data_loaders
    
    # Create MNIST dataset
    dataset = MNISTDataset(
        root=data_dir,
        train=True,
        download=download,
        flatten=True,
        normalize=normalize,
        cache_data=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        validation_split=validation_split,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def setup_training_environment(
    save_dir: Union[str, Path] = "./training_output",
    log_level: str = "INFO",
    seed: Optional[int] = None
) -> Dict[str, Path]:
    """
    Set up training environment with directories and logging.
    
    Args:
        save_dir: Base directory for training outputs
        log_level: Logging level
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of created directory paths
    """
    save_dir = Path(save_dir)
    
    # Create directories
    directories = {
        "base": save_dir,
        "checkpoints": save_dir / "checkpoints",
        "logs": save_dir / "logs",
        "plots": save_dir / "plots",
        "results": save_dir / "results"
    }
    
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(directories["logs"] / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    return directories


def evaluate_reconstruction_quality(
    model: MLPAutoencoder,
    data_loader: DataLoader,
    num_samples: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality with comprehensive metrics.
    
    Args:
        model: Trained autoencoder model
        data_loader: Data loader for evaluation
        num_samples: Number of samples to evaluate
        device: Device for computation
        
    Returns:
        Dictionary of reconstruction quality metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    mse_losses = []
    mae_losses = []
    ssim_scores = []
    
    samples_processed = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            if samples_processed >= num_samples:
                break
            
            data = data.to(device)
            reconstructed = model(data)
            
            # Calculate per-sample metrics
            batch_size = data.size(0)
            for i in range(min(batch_size, num_samples - samples_processed)):
                original = data[i].cpu()
                recon = reconstructed[i].cpu()
                
                # MSE
                mse = torch.mean((original - recon) ** 2).item()
                mse_losses.append(mse)
                
                # MAE
                mae = torch.mean(torch.abs(original - recon)).item()
                mae_losses.append(mae)
                
                # Simple correlation-based similarity (proxy for SSIM)
                if original.std() > 0 and recon.std() > 0:
                    correlation = torch.corrcoef(torch.stack([original.flatten(), recon.flatten()]))[0, 1]
                    ssim_scores.append(correlation.item() if not torch.isnan(correlation) else 0.0)
                else:
                    ssim_scores.append(0.0)
                
                samples_processed += 1
                if samples_processed >= num_samples:
                    break
    
    return {
        "mse_mean": np.mean(mse_losses),
        "mse_std": np.std(mse_losses),
        "mae_mean": np.mean(mae_losses),
        "mae_std": np.std(mae_losses),
        "similarity_mean": np.mean(ssim_scores),
        "similarity_std": np.std(ssim_scores),
        "rmse": np.sqrt(np.mean(mse_losses)),
        "num_samples": samples_processed
    }


def plot_training_history(
    metrics: TrainingMetrics,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot training history including loss curves and learning rate.
    
    Args:
        metrics: Training metrics object
        save_path: Path to save the plot
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot generation")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot training and validation loss
    axes[0, 0].plot(metrics.train_loss_history, label='Training Loss', color='blue', alpha=0.7)
    if metrics.val_loss_history:
        val_epochs = np.arange(0, len(metrics.train_loss_history), 
                              len(metrics.train_loss_history) // len(metrics.val_loss_history))[:len(metrics.val_loss_history)]
        axes[0, 0].plot(val_epochs, metrics.val_loss_history, label='Validation Loss', color='red', alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if metrics.learning_rate_history:
        axes[0, 1].plot(metrics.learning_rate_history, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot epoch times
    if metrics.epoch_times:
        axes[1, 0].plot(metrics.epoch_times, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot loss distribution (histogram)
    if len(metrics.train_loss_history) > 10:
        axes[1, 1].hist(metrics.train_loss_history, bins=20, alpha=0.7, color='blue', label='Training')
        if metrics.val_loss_history and len(metrics.val_loss_history) > 5:
            axes[1, 1].hist(metrics.val_loss_history, bins=20, alpha=0.7, color='red', label='Validation')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_reconstruction_examples(
    model: MLPAutoencoder,
    data_loader: DataLoader,
    num_examples: int = 8,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot examples of original vs reconstructed images.
    
    Args:
        model: Trained autoencoder model
        data_loader: Data loader with samples
        num_examples: Number of examples to show
        save_path: Path to save the plot
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot generation")
        return None
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get a batch of data
    data, _ = next(iter(data_loader))
    data = data[:num_examples].to(device)
    
    with torch.no_grad():
        reconstructed = model(data)
    
    # Convert to numpy for plotting
    originals = data.cpu().numpy()
    reconstructions = reconstructed.cpu().numpy()
    
    # Create plot
    fig, axes = plt.subplots(2, num_examples, figsize=(2*num_examples, 4))
    fig.suptitle('Original vs Reconstructed Images', fontsize=16)
    
    for i in range(num_examples):
        # Original image
        axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def save_training_config(
    config: TrainingConfig,
    save_path: Union[str, Path]
) -> None:
    """
    Save training configuration to JSON file.
    
    Args:
        config: Training configuration to save
        save_path: Path to save configuration file
    """
    save_path = Path(save_path)
    config_dict = asdict(config)
    
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_training_config(config_path: Union[str, Path]) -> TrainingConfig:
    """
    Load training configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TrainingConfig object
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return TrainingConfig(**config_dict)


def create_trainer_from_config(
    model: MLPAutoencoder,
    config_path: Union[str, Path]
) -> Trainer:
    """
    Create trainer from saved configuration file.
    
    Args:
        model: MLPAutoencoder model
        config_path: Path to configuration file
        
    Returns:
        Configured Trainer object
    """
    config = load_training_config(config_path)
    return Trainer(model, config)


def benchmark_training_performance(
    model: MLPAutoencoder,
    data_loader: DataLoader,
    num_epochs: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark training performance and memory usage.
    
    Args:
        model: Model to benchmark
        data_loader: Data loader for benchmarking
        num_epochs: Number of epochs to run
        device: Device for computation
        
    Returns:
        Performance metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    # Warm up
    for data, _ in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data)
        loss.backward()
        optimizer.step()
        break
    
    # Benchmark
    start_time = time.time()
    total_samples = 0
    total_loss = 0.0
    
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated(device)
    
    for epoch in range(num_epochs):
        for data, _ in data_loader:
            data = data.to(device)
            total_samples += data.size(0)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    end_time = time.time()
    
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated(device)
        memory_used = (end_memory - start_memory) / (1024**2)  # MB
    else:
        memory_used = 0.0
    
    total_time = end_time - start_time
    
    return {
        "total_time": total_time,
        "samples_per_second": total_samples / total_time,
        "epochs_per_second": num_epochs / total_time,
        "average_loss": total_loss / (total_samples / data_loader.batch_size),
        "memory_used_mb": memory_used,
        "total_samples": total_samples,
        "device": str(device)
    }


def compare_optimizers(
    model_factory: Callable[[], MLPAutoencoder],
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizers: List[str] = ["adam", "lbfgs", "sgd"],
    epochs: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different optimizers on the same model and data.
    
    Args:
        model_factory: Function that creates a fresh model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizers: List of optimizer names to compare
        epochs: Number of epochs for each optimizer
        device: Device for computation
        
    Returns:
        Comparison results for each optimizer
    """
    results = {}
    
    for optimizer_name in optimizers:
        print(f"Testing optimizer: {optimizer_name}")
        
        # Create fresh model
        model = model_factory()
        if device:
            model = model.to(device)
        
        # Create training configuration
        config = TrainingConfig(
            epochs=epochs,
            optimizer=optimizer_name,
            batch_size=64,
            learning_rate=0.001 if optimizer_name != "lbfgs" else 0.01,
            early_stopping=False,  # Disable for fair comparison
            verbose=False
        )
        
        # Train model
        trainer = Trainer(model, config)
        training_results = trainer.train(train_loader, val_loader)
        
        # Evaluate final model
        final_metrics = trainer.evaluate_model(val_loader)
        
        results[optimizer_name] = {
            "training_results": training_results,
            "final_metrics": final_metrics,
            "final_train_loss": training_results["final_train_loss"],
            "final_val_loss": training_results["final_val_loss"],
            "total_time": training_results["total_training_time"],
            "converged": training_results["training_completed"]
        }
    
    return results
