# PyTorch Autoencoder Training System

This document describes the comprehensive L-BFGS training system for MLP autoencoders, designed to match SAS Viya training specifications.

## Overview

The training system provides:
- **L-BFGS Optimization**: Closure-based optimization with max 500 iterations
- **MSE Loss Function**: Mean Squared Error for reconstruction loss
- **Convergence Checking**: Configurable tolerance (default: fConv=1E-10)
- **Comprehensive Metrics**: Training progress tracking and analysis
- **Model Checkpointing**: Automatic state saving and loading
- **Reproducible Results**: Consistent seed=23451 throughout
- **Visualization Suite**: Training progress and reconstruction quality plots

## Quick Start

### Basic Training Example

```python
import torch
from src.models.autoencoder import create_autoencoder
from src.data.mnist_loader import create_mnist_loader
from src.training import create_trainer

# Create model
model = create_autoencoder(seed=23451)

# Load data
data_loader = create_mnist_loader(batch_size=64, random_seed=23451)
train_loader, test_loader, val_loader = data_loader.create_data_loaders()

# Create trainer with SAS-compatible settings
trainer = create_trainer(
    model=model,
    max_iterations=500,
    convergence_tolerance=1e-10,
    seed=23451,
    checkpoint_dir="./checkpoints"
)

# Train the model
results = trainer.train(train_loader)

# Evaluate
test_results = trainer.evaluate(test_loader)
```

### Command Line Training

```bash
cd autoencoder
python src/scripts/train_autoencoder.py \
    --max_iterations 500 \
    --batch_size 64 \
    --checkpoint_dir ./checkpoints \
    --report_dir ./reports
```

## Technical Specifications

### L-BFGS Optimizer Configuration
- **Algorithm**: L-BFGS with strong Wolfe line search
- **Max Iterations**: 500 (configurable)
- **History Size**: 100 (L-BFGS memory)
- **Tolerance**: 1E-10 for convergence detection
- **Gradient Tolerance**: 1E-7

### Loss Function
- **Type**: Mean Squared Error (MSE)
- **Target**: Reconstruction loss between input and output
- **Formula**: `MSE(x, x̂) = (1/n) * Σ(x_i - x̂_i)²`

### Convergence Criteria
- **Relative Change**: `|loss_current - loss_previous| / |loss_previous| < tolerance`
- **Absolute Change**: `|loss_current - loss_previous| < tolerance`
- **Early Stopping**: Automatic when convergence criteria are met

### Model Architecture
- **Input**: 784 dimensions (28×28 MNIST images, flattened)
- **Hidden**: 400 dimensions with tanh activation
- **Output**: 784 dimensions with linear activation
- **Weight Initialization**: Uniform distribution [-1, 1] with seed=23451

## Core Components

### 1. AutoencoderTrainer

The main training class that orchestrates the entire training process.

```python
from src.training import AutoencoderTrainer

trainer = AutoencoderTrainer(
    model=model,
    max_iterations=500,
    convergence_tolerance=1e-10,
    seed=23451,
    device=device,
    checkpoint_dir="./checkpoints"
)
```

**Key Methods**:
- `train(dataloader)`: Main training loop
- `evaluate(dataloader)`: Model evaluation
- `train_single_batch(batch)`: Single batch training with L-BFGS
- `load_checkpoint(path)`: Resume training from checkpoint

### 2. TrainingMetrics

Comprehensive metrics tracking and analysis.

```python
from src.training import TrainingMetrics

metrics = TrainingMetrics()
# Metrics are automatically updated during training

# Get comprehensive summary
summary = metrics.get_summary()

# Create visualizations
fig = metrics.plot_training_progress(save_path="progress.png")

# Print detailed summary
metrics.print_summary()
```

**Tracked Metrics**:
- Loss progression and convergence rates
- Training performance and timing
- Iteration statistics and improvements
- Convergence trend analysis

### 3. CheckpointManager

Robust model state management with integrity checking.

```python
from src.utils.checkpoints import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    max_checkpoints=5,
    save_best=True
)

# Save checkpoint
checkpoint_manager.save_checkpoint(
    model=model,
    iteration=iteration,
    loss=loss,
    optimizer=optimizer,
    is_best=True
)

# Load best checkpoint
checkpoint_manager.load_best_checkpoint(model, optimizer)
```

**Features**:
- Automatic checkpoint rotation
- Best model tracking
- Integrity verification with checksums
- Reproducibility state preservation

### 4. Visualization Suite

Comprehensive visualization tools for training analysis.

```python
from src.utils.visualization import (
    plot_training_progress,
    plot_reconstruction_comparison,
    create_training_report
)

# Training progress
fig1 = plot_training_progress(loss_history, save_path="progress.png")

# Reconstruction quality
fig2 = plot_reconstruction_comparison(original, reconstructed)

# Complete training report
report_files = create_training_report(
    model=model,
    metrics_data=results,
    sample_data=test_data,
    save_dir="./reports"
)
```

## Training Configuration

### Default Configuration (SAS-Compatible)

```python
DEFAULT_CONFIG = {
    'max_iterations': 500,
    'convergence_tolerance': 1e-10,  # fConv parameter
    'seed': 23451,
    'optimizer': 'L-BFGS',
    'loss_function': 'MSE',
    'batch_processing': True,
    'early_stopping': True
}
```

### Advanced Configuration Options

```python
trainer = AutoencoderTrainer(
    model=model,
    max_iterations=500,              # Maximum iterations
    convergence_tolerance=1e-10,     # Convergence threshold
    seed=23451,                      # Reproducibility seed
    device=device,                   # Training device
    checkpoint_dir="./checkpoints",  # Checkpoint directory
    save_best=True,                  # Save best model
    patience=None                    # Early stopping patience
)
```

## Training Process

### 1. Initialization
- Model weight initialization with uniform distribution [-1, 1]
- Random seed setting for reproducibility (seed=23451)
- L-BFGS optimizer configuration
- Metrics tracking initialization

### 2. Training Loop
```python
for epoch in range(max_epochs):
    for batch in dataloader:
        # Create closure for L-BFGS
        def closure():
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            return loss.item()
        
        # L-BFGS optimization step
        loss = optimizer.step(closure)
        
        # Update metrics and check convergence
        metrics.update(iteration, loss)
        if check_convergence(loss):
            break
```

### 3. Convergence Checking
- Relative loss change: `|Δloss| / |loss_prev| < tolerance`
- Absolute loss change: `|Δloss| < tolerance`
- Both criteria must be satisfied
- Early stopping when converged

### 4. Checkpointing
- Automatic checkpoint saving every iteration
- Best model tracking based on lowest loss
- Latest checkpoint for resuming training
- State preservation for reproducibility

## Performance Optimization

### Memory Management
- Batch processing for large datasets
- Gradient accumulation support
- Automatic device management (CPU/GPU)

### Training Speed
- L-BFGS optimization for fast convergence
- Efficient closure-based gradient computation
- Minimal overhead metrics tracking

### Reproducibility
- Fixed random seeds (23451) throughout
- Deterministic operations
- State preservation in checkpoints
- Cross-platform consistency

## Output and Results

### Training Results Dictionary
```python
results = {
    'training_complete': True,
    'converged': True,
    'final_loss': 0.001234,
    'best_loss': 0.001200,
    'total_iterations': 156,
    'total_epochs': 3,
    'total_training_time': 45.2,
    'loss_history': [...],
    'metrics_summary': {...},
    'seed': 23451
}
```

### Generated Files
- **Checkpoints**: Model states and training progress
- **Metrics**: JSON files with comprehensive statistics
- **Visualizations**: Training progress and reconstruction plots
- **Reports**: Complete training analysis documents

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Use CPU instead of GPU
   - Enable gradient checkpointing

2. **Convergence Problems**
   - Adjust convergence tolerance
   - Increase max iterations
   - Check data preprocessing

3. **Reproducibility Issues**
   - Ensure seed=23451 is used consistently
   - Check PyTorch version compatibility
   - Verify deterministic settings

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

trainer = create_trainer(model, checkpoint_dir="./debug_checkpoints")
```

## Integration with SAS

The training system is designed to produce results compatible with SAS Viya autoencoder training:

- **Same Architecture**: 784→400→784 MLP with tanh activation
- **Same Initialization**: Uniform distribution [-1, 1]
- **Same Loss Function**: Mean Squared Error
- **Same Convergence**: fConv=1E-10 tolerance
- **Same Seed**: 23451 for reproducible results

## API Reference

See individual module documentation for detailed API reference:
- `src/training/trainer.py` - Core training functionality
- `src/training/metrics.py` - Metrics tracking and analysis
- `src/utils/checkpoints.py` - Model state management
- `src/utils/visualization.py` - Visualization tools

## Examples

Complete examples are available in:
- `src/scripts/train_autoencoder.py` - Command-line training script
- `examples/` directory - Additional usage examples
- Test files - Unit tests demonstrating functionality

---

For questions or issues, please refer to the documentation or create an issue in the project repository.
