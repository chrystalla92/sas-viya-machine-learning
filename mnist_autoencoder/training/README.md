# MNIST Autoencoder Training Module

This module provides comprehensive training orchestration for the MLP Autoencoder, including loss computation, optimization, checkpointing, and progress monitoring.

## Features

### Core Training Components

- **Trainer Class**: Complete training orchestration with epoch management and batch processing
- **TrainingConfig**: Comprehensive configuration management for all training parameters
- **TrainingMetrics**: Training progress tracking with history and best model monitoring
- **GradientMonitor**: Gradient flow monitoring for detecting training issues

### Optimization Support

- **Multiple Optimizers**: Adam, L-BFGS, SGD with SAS PROC NNET compatibility
- **Learning Rate Scheduling**: Step, exponential, and reduce-on-plateau schedulers
- **Loss Functions**: MSE and BCE loss for reconstruction training

### Training Management

- **Checkpointing**: Automatic model saving with best model tracking
- **Early Stopping**: Configurable early stopping with patience mechanism
- **Training Resumption**: Resume training from any checkpoint
- **Graceful Shutdown**: Signal handling for safe training interruption

### Monitoring and Evaluation

- **Progress Logging**: Comprehensive training progress with loss tracking
- **Model Evaluation**: Multiple metrics for reconstruction quality assessment
- **Visualization**: Training history plots and reconstruction examples
- **Performance Benchmarking**: Training performance and memory usage analysis

## Quick Start

### Basic Training

```python
import torch
from mnist_autoencoder.models.autoencoder import MLPAutoencoder
from mnist_autoencoder.training import Trainer, TrainingConfig, prepare_mnist_data

# Create model
model = MLPAutoencoder()

# Prepare data
train_loader, val_loader = prepare_mnist_data(
    batch_size=64,
    validation_split=0.2
)

# Configure training
config = TrainingConfig(
    epochs=100,
    learning_rate=0.001,
    optimizer="adam",
    early_stopping=True
)

# Train model
trainer = Trainer(model, config)
results = trainer.train(train_loader, val_loader)
```

### SAS PROC NNET Compatible Training

```python
from mnist_autoencoder.training import create_sas_compatible_config

# Use SAS-compatible configuration
config = create_sas_compatible_config()
trainer = Trainer(model, config)
results = trainer.train(train_loader, val_loader)
```

### Training with Checkpointing

```python
# Resume from checkpoint
results = trainer.train(
    train_loader, 
    val_loader, 
    resume_from="checkpoints/best_model.pth"
)

# Save custom checkpoint
trainer.save_checkpoint(epoch=50, is_best=True)
```

## Configuration Options

### TrainingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 64 | Batch size for training |
| `learning_rate` | float | 0.001 | Learning rate for optimizer |
| `optimizer` | str | "adam" | Optimizer type ("adam", "lbfgs", "sgd") |
| `weight_decay` | float | 0.0 | L2 regularization weight decay |
| `loss_function` | str | "mse" | Loss function ("mse", "bce") |
| `validation_split` | float | 0.2 | Fraction for validation split |
| `early_stopping` | bool | True | Enable early stopping |
| `patience` | int | 10 | Early stopping patience |
| `lr_scheduler` | str | None | Learning rate scheduler |
| `save_frequency` | int | 10 | Checkpoint save frequency |
| `log_frequency` | int | 10 | Training progress log frequency |

### Learning Rate Schedulers

#### Step Scheduler
```python
config = TrainingConfig(
    lr_scheduler="step",
    lr_scheduler_params={
        "step_size": 30,
        "gamma": 0.1
    }
)
```

#### Reduce on Plateau
```python
config = TrainingConfig(
    lr_scheduler="reduce_on_plateau",
    lr_scheduler_params={
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-7
    }
)
```

## Training Utilities

### Data Preparation

```python
from mnist_autoencoder.training import prepare_mnist_data

# Prepare MNIST data with custom settings
train_loader, val_loader = prepare_mnist_data(
    data_dir="./data",
    batch_size=32,
    validation_split=0.15,
    normalize="01",  # [0,1] normalization
    num_workers=4
)
```

### Environment Setup

```python
from mnist_autoencoder.training import setup_training_environment

# Set up training directories and logging
directories = setup_training_environment(
    save_dir="./training_output",
    log_level="INFO",
    seed=42
)
```

### Model Evaluation

```python
from mnist_autoencoder.training import evaluate_reconstruction_quality

# Evaluate reconstruction quality
metrics = evaluate_reconstruction_quality(
    model=trained_model,
    data_loader=test_loader,
    num_samples=100
)

print(f"Mean MSE: {metrics['mse_mean']:.6f}")
print(f"Mean MAE: {metrics['mae_mean']:.6f}")
print(f"Similarity: {metrics['similarity_mean']:.3f}")
```

### Visualization

```python
from mnist_autoencoder.training import plot_training_history, plot_reconstruction_examples

# Plot training history
plot_training_history(
    metrics=trainer.metrics,
    save_path="training_history.png"
)

# Plot reconstruction examples
plot_reconstruction_examples(
    model=trainer.model,
    data_loader=val_loader,
    num_examples=8,
    save_path="reconstructions.png"
)
```

## Advanced Usage

### Custom Training Loop

```python
# Manual training control
trainer = Trainer(model, config)
trainer.prepare_training()

for epoch in range(config.epochs):
    # Train epoch
    train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
    
    # Validate epoch
    val_loss, val_metrics = trainer.validate_epoch(val_loader, epoch)
    
    # Check early stopping
    if trainer.check_early_stopping(val_loss):
        break
    
    # Save checkpoint
    if epoch % 10 == 0:
        trainer.save_checkpoint(epoch, is_best=False)
```

### Optimizer Comparison

```python
from mnist_autoencoder.training import compare_optimizers

# Compare different optimizers
def model_factory():
    return MLPAutoencoder(seed=42)

results = compare_optimizers(
    model_factory=model_factory,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizers=["adam", "lbfgs", "sgd"],
    epochs=20
)

for optimizer, metrics in results.items():
    print(f"{optimizer}: Final loss = {metrics['final_val_loss']:.6f}")
```

### Performance Benchmarking

```python
from mnist_autoencoder.training import benchmark_training_performance

# Benchmark training performance
perf_metrics = benchmark_training_performance(
    model=model,
    data_loader=train_loader,
    num_epochs=5
)

print(f"Samples/second: {perf_metrics['samples_per_second']:.1f}")
print(f"Memory used: {perf_metrics['memory_used_mb']:.1f} MB")
```

## Example Scripts

### Complete Training Example
See `examples/train_autoencoder.py` for a complete training script with:
- Command-line argument parsing
- Multiple configuration options
- Comprehensive logging and evaluation
- Visualization generation

### Quick Test
See `examples/quick_test.py` for a minimal training validation script.

## Error Handling

The training module includes robust error handling for:

- **Gradient Problems**: Automatic detection of gradient explosion/vanishing
- **Checkpoint Issues**: Graceful handling of corrupted or incompatible checkpoints
- **Training Interruption**: Signal handling for safe shutdown with checkpoint saving
- **Memory Issues**: Memory usage monitoring and warnings
- **Data Loading Errors**: Comprehensive data validation and error messages

## Performance Considerations

### Memory Optimization
- Use `pin_memory=True` for GPU training
- Adjust `num_workers` based on system capabilities
- Monitor memory usage during training

### Training Speed
- L-BFGS optimizer may be slower but can achieve better convergence
- Smaller batch sizes recommended for L-BFGS
- Adam optimizer generally faster for large datasets

### Checkpointing Strategy
- Set `save_best_only=True` to save disk space
- Use reasonable `save_frequency` to balance safety and storage
- Consider using `validation_freq` to reduce validation overhead

## Integration with SAS

This implementation is designed to match SAS PROC NNET behavior:

- **L-BFGS Optimizer**: Primary optimizer choice for SAS compatibility
- **Convergence Criteria**: Similar early stopping and tolerance settings  
- **Weight Initialization**: Xavier/Glorot initialization matching SAS defaults
- **Architecture**: 784→400→784 structure matching SAS implementation

Use `create_sas_compatible_config()` for best compatibility with SAS PROC NNET results.
