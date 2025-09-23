# Autoencoder Inference Pipeline

This module provides a comprehensive inference pipeline for trained autoencoder models, enabling efficient batch processing, model scoring, and image reconstruction.

## Features

- **Model Checkpoint Loading**: Load trained models from PyTorch checkpoint files
- **Batch Processing**: Efficient processing of large datasets with configurable batch sizes
- **Tensor Preprocessing**: Automatic midrange scaling matching training pipeline
- **Format Conversions**: Easy conversion between flattened (784) and image (28×28) formats
- **Device Support**: Automatic CPU/GPU handling for optimal performance
- **Input Validation**: Comprehensive validation with clear error messages
- **Performance Tracking**: Built-in metrics for throughput and timing analysis

## Quick Start

```python
from src.inference import create_inference_pipeline, tensor_to_flat, flat_to_images
import torch

# Method 1: Load from checkpoint
pipeline = create_inference_pipeline(
    checkpoint_path="path/to/model_checkpoint.pt",
    batch_size=32,
    preprocessing='midrange'
)

# Method 2: Use existing model
from src.models.autoencoder import create_autoencoder
model = create_autoencoder()
pipeline = create_inference_pipeline(model=model)

# Single sample inference
sample = torch.randn(784)  # Flattened image
reconstruction = pipeline.predict(sample)

# Batch inference
batch = torch.randn(100, 784)  # 100 samples
reconstructions = pipeline.predict(batch)

# Extract latent representations
reconstructions, latents = pipeline.predict(batch, return_latent=True)
```

## Tensor Format Conversions

```python
# Convert images to flat format for autoencoder input
images = torch.randn(10, 28, 28)  # 10 images of 28×28
flat_data = tensor_to_flat(images)  # Shape: (10, 784)

# Convert reconstructions back to image format
reconstructed_images = flat_to_images(flat_data, (28, 28))  # Shape: (10, 28, 28)
```

## Checkpoint Format

The inference pipeline expects checkpoints with the following structure:
```python
{
    'model_state_dict': model.state_dict(),
    'loss': final_loss,
    'iteration': training_iteration,
    'converged': convergence_status,
    'seed': random_seed
}
```

## API Reference

### AutoencoderInference

Main inference class providing comprehensive scoring capabilities.

**Constructor Parameters:**
- `model`: Pre-loaded Autoencoder model (optional)
- `checkpoint_path`: Path to checkpoint file (optional)
- `device`: Device for inference ('cpu', 'cuda', or None for auto)
- `batch_size`: Default batch size (default: 512)
- `preprocessing`: Preprocessing method ('midrange' or 'none')
- `validate_input`: Enable input validation (default: True)
- `log_performance`: Enable performance tracking (default: True)

**Key Methods:**
- `predict(data, batch_size=None, return_latent=False)`: Main prediction interface
- `load_checkpoint(path)`: Load model from checkpoint
- `get_performance_stats()`: Get performance metrics
- `reset_performance_stats()`: Reset performance tracking

### Utility Functions

- `create_inference_pipeline(**kwargs)`: Factory function for easy setup
- `tensor_to_flat(images)`: Convert images to flat format
- `flat_to_images(flattened, shape)`: Convert flat tensors to images

## Performance Optimization

- **Batch Processing**: Use appropriate batch sizes based on available memory
- **Device Selection**: Automatically uses GPU when available
- **Memory Management**: Efficient tensor operations with minimal copying
- **Preprocessing**: Optimized midrange scaling matching training pipeline

## Error Handling

The pipeline includes comprehensive error handling for:
- Missing checkpoint files
- Invalid tensor dimensions
- NaN/infinite input values
- Memory issues with large batches
- Device compatibility problems

## Testing

Run the complete test suite:
```bash
python test_inference_pipeline.py
```

Run a simple example:
```bash
python example_inference.py
```

## Integration with Training Pipeline

The inference pipeline is designed to work seamlessly with the training pipeline:

1. **Preprocessing Consistency**: Uses the same midrange scaling as training
2. **Checkpoint Compatibility**: Loads checkpoints saved by the trainer
3. **Architecture Matching**: Automatically detects model dimensions from checkpoints
4. **Device Handling**: Consistent CPU/GPU behavior across training and inference

## Production Deployment

For production use:
```python
# Load model once and reuse
pipeline = create_inference_pipeline(
    checkpoint_path="best_model.pt",
    batch_size=64,
    validate_input=True,
    log_performance=False  # Disable for production
)

# Process data efficiently
results = pipeline.predict(input_data, batch_size=32)
```

## Dependencies

- PyTorch >= 1.7.0
- NumPy
- Existing autoencoder training pipeline components
