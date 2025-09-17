#!/usr/bin/env python3
"""
Quick test script to validate the training implementation.

This script performs a minimal training run to ensure everything works correctly.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

from mnist_autoencoder.models.autoencoder import MLPAutoencoder
from mnist_autoencoder.training import (
    Trainer,
    TrainingConfig,
    create_data_loaders
)


def main():
    """Quick test of training functionality."""
    print("Quick Training Test")
    print("=" * 30)
    
    # Create synthetic data (like MNIST but smaller)
    print("Creating synthetic data...")
    data = torch.randn(100, 784) * 0.5 + 0.5
    data = data.clamp(0, 1)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=16,
        validation_split=0.2,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = MLPAutoencoder(seed=42)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create training configuration (quick test)
    config = TrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=0.01,
        optimizer="adam",
        early_stopping=False,  # Disable for quick test
        verbose=True,
        save_frequency=1,
        log_frequency=1
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(model, config)
    
    # Run quick training
    print("Starting training...")
    try:
        results = trainer.train(train_loader, val_loader)
        
        print("Training Results:")
        print(f"  Completed: {results['training_completed']}")
        print(f"  Epochs: {results['total_epochs']}")
        print(f"  Final train loss: {results['final_train_loss']:.6f}")
        print(f"  Final val loss: {results['final_val_loss']:.6f}")
        print(f"  Training time: {results['total_training_time']:.1f}s")
        
        # Quick evaluation
        print("Evaluating model...")
        eval_metrics = trainer.evaluate_model(val_loader)
        print(f"  MSE: {eval_metrics['mse']:.6f}")
        print(f"  MAE: {eval_metrics['mae']:.6f}")
        print(f"  RMSE: {eval_metrics['rmse']:.6f}")
        
        print("✅ Quick test PASSED!")
        
    except Exception as e:
        print(f"❌ Quick test FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
