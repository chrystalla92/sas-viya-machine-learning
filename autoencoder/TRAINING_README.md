# Autoencoder Training Pipeline

This document describes the comprehensive training pipeline for PyTorch autoencoder models, including setup, configuration, and usage examples.

## Overview

The training pipeline provides:
- **Automatic GPU detection** and device management
- **Early stopping** with configurable patience and delta thresholds
- **Learning rate scheduling** (ReduceLROnPlateau recommended)
- **Model checkpointing** (best model and periodic saves)
- **Comprehensive logging** and progress tracking
- **Memory-efficient training** and validation loops
- **Training curve visualization**

## Quick Start

### Basic Usage

```python
from autoencoder_model import AutoencoderMLP
from trainer import train_autoencoder

# Create model
model = AutoencoderMLP(latent_dim=400)

# Train with default settings
pipeline, summary = train_autoencoder(model, './data')

print(f"Training completed! Best loss: {summary['best_validation_loss']:.6f}")
```

### Custom Configuration

```python
from autoencoder_model import AutoencoderMLP
from trainer import TrainingPipeline

model = AutoencoderMLP(latent_dim=128, activation='relu')

config = {
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'lr_scheduler_patience': 5,
    'device': 'auto'  # 'auto', 'cpu', 'cuda'
}

pipeline = TrainingPipeline(model, config)
summary = pipeline.train('./data')
```

## Configuration Options

### Core Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Maximum number of training epochs |
| `batch_size` | 32 | Batch size for training and validation |
| `learning_rate` | 1e-3 | Initial learning rate for Adam optimizer |
| `weight_decay` | 1e-5 | L2 regularization weight decay |
| `train_val_split` | 0.8 | Fraction of training data for training (rest for validation) |
| `device` | 'auto' | Device to use: 'auto', 'cpu', 'cuda', or specific device |

### Early Stopping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `early_stopping_patience` | 7 | Epochs to wait after last improvement |
| `early_stopping_min_delta` | 0.0 | Minimum change to qualify as improvement |

### Learning Rate Scheduling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_scheduler_type` | 'ReduceLROnPlateau' | Type of scheduler ('ReduceLROnPlateau', 'StepLR') |
| `lr_scheduler_patience` | 5 | Epochs to wait before reducing LR |
| `lr_scheduler_factor` | 0.5 | Factor by which to reduce LR |

### Checkpointing and Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_best_model` | True | Save the best model based on validation loss |
| `save_periodic_checkpoints` | True | Save periodic checkpoints during training |
| `checkpoint_frequency` | 10 | Frequency of periodic saves (epochs) |
| `log_frequency` | 1 | Frequency of progress logging (epochs) |

## Key Features

### 1. Automatic Device Detection

The pipeline automatically detects and uses the best available device:

```python
# Automatically uses GPU if available, otherwise CPU
config = {'device': 'auto'}

# Or specify explicitly
config = {'device': 'cuda'}  # or 'cpu'
```

### 2. Early Stopping

Prevents overfitting by monitoring validation loss:

```python
config = {
    'early_stopping_patience': 10,  # Stop if no improvement for 10 epochs
    'early_stopping_min_delta': 1e-4  # Minimum improvement threshold
}
```

### 3. Learning Rate Scheduling

Reduces learning rate when training plateaus:

```python
config = {
    'lr_scheduler_type': 'ReduceLROnPlateau',
    'lr_scheduler_patience': 5,  # Reduce LR after 5 epochs without improvement
    'lr_scheduler_factor': 0.5   # Multiply LR by 0.5
}
```

### 4. Model Checkpointing

Automatically saves the best model and periodic checkpoints:

```python
# Best model is saved automatically
best_checkpoint = pipeline.checkpoint.get_best_checkpoint_path()

# Load a checkpoint
pipeline.checkpoint.load_checkpoint(
    'checkpoints/autoencoder_best.pth', 
    model, optimizer, scheduler
)
```

### 5. Comprehensive Logging

Training metrics are logged and can be visualized:

```python
# Plot training curves
pipeline.plot_training_curves('training_curves.png')

# Get training summary
summary = pipeline.logger.get_summary()
print(f"Best validation loss: {summary['best_val_loss']}")
```

## Advanced Usage

### Custom Data Loaders

```python
from datasets import create_mnist_dataloaders

# Create custom data loaders
train_loader, val_loader, test_loader = create_mnist_dataloaders(
    data_dir='./data',
    batch_size=64,
    train_val_split=0.85,
    standardize=True
)

# Use with training pipeline
pipeline = TrainingPipeline(model, config)
summary = pipeline.train(train_loader=train_loader, val_loader=val_loader)
```

### Model Evaluation

```python
# Evaluate on test set
test_metrics = pipeline.evaluate(test_loader)
print(f"Test loss: {test_metrics['test_loss']:.6f}")
```

### Resuming Training

```python
# Load from checkpoint and continue training
checkpoint_path = 'checkpoints/autoencoder_epoch_20.pth'
pipeline.checkpoint.load_checkpoint(checkpoint_path, model, optimizer, scheduler)

# Continue training
additional_summary = pipeline.train('./data')
```

## File Structure

The training pipeline creates the following directory structure:

```
./
├── checkpoints/          # Model checkpoints
│   ├── autoencoder_best.pth
│   └── autoencoder_epoch_*.pth
├── logs/                 # Training logs (JSON format)
│   └── autoencoder_training.json
└── plots/               # Training curve plots
    └── loss_curves.png
```

## Training Pipeline Components

### TrainingPipeline Class

Main class that orchestrates the training process:

```python
pipeline = TrainingPipeline(model, config)
summary = pipeline.train(data_dir='./data')
```

### Training Utilities

- **EarlyStopping**: Monitors validation loss and stops training when no improvement
- **TrainingLogger**: Logs metrics, creates visualizations, saves training history
- **ModelCheckpoint**: Saves best models and periodic checkpoints
- **Learning Rate Schedulers**: Automatically adjusts learning rate during training

### Convenience Functions

```python
# Quick training with minimal setup
pipeline, summary = train_autoencoder(model, './data', config)

# Create pipeline with custom configuration
pipeline = create_training_pipeline(model, config)
```

## Success Criteria Verification

The training pipeline meets all specified requirements:

- ✅ **Training loop**: Minimizes reconstruction loss over epochs
- ✅ **Validation metrics**: Calculated without affecting model weights
- ✅ **Early stopping**: Prevents overfitting with configurable parameters
- ✅ **Learning rate scheduling**: Adapts based on training progress
- ✅ **Device handling**: Works on CPU and GPU automatically
- ✅ **Model checkpointing**: Saves and restores models correctly
- ✅ **Comprehensive logging**: Provides detailed training analysis
- ✅ **Memory stability**: No memory leaks during training

## Example Training Output

```
============================================================
Starting Autoencoder Training
============================================================
Creating data loaders...
Training batches: 1875
Validation batches: 469

Model: AutoencoderMLP
Parameters: 628,184
Device: cuda

Starting training for 50 epochs...
------------------------------------------------------------
[  1/50] Train: 0.085432 | Val: 0.082156 ★ | LR: 1.00e-03 | Time: 12.3s
[  2/50] Train: 0.074521 | Val: 0.071834 ★ | LR: 1.00e-03 | Time: 11.8s
[  3/50] Train: 0.068743 | Val: 0.067492 ★ | LR: 1.00e-03 | Time: 12.1s
...
[15/50] Train: 0.048392 | Val: 0.047834 ★ | LR: 5.00e-04 | Time: 11.9s
Early stopping triggered after 7 epochs without improvement
Restored best weights from epoch 15
------------------------------------------------------------
Training completed!

Training Summary:
========================================
Total epochs: 22
Early stopped: True
Best validation loss: 0.047834
Final learning rate: 5.00e-04
Total training time: 4.82 minutes
Average epoch time: 13.1 seconds
Best model saved: ./checkpoints/autoencoder_best.pth
```

## Testing

Run the test suite to verify the training pipeline:

```bash
python test_training_pipeline.py
```

Run example training scripts:

```bash
python training_example.py
```

## Dependencies

- PyTorch >= 1.9.0
- NumPy
- Matplotlib (for plotting)
- MNIST binary data files in `./data/` directory

## Notes

- The training pipeline is optimized for MNIST autoencoder training but can be adapted for other datasets
- GPU training is automatically enabled when CUDA is available
- All training state is preserved in checkpoints for reproducible results
- Memory usage is monitored and optimized for stability during long training runs
