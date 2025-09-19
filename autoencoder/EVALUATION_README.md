# Autoencoder Evaluation and Inference Pipeline

This document provides comprehensive documentation for the autoencoder evaluation and inference functionality, including metrics calculation, batch processing, model persistence, and SAS-compatible output generation.

## Overview

The evaluation pipeline provides:

- **Comprehensive Metrics**: MSE, MAE, RMSE, SSIM approximation, and latent space statistics
- **Batch Inference**: Efficient processing of large datasets with memory management
- **Model Persistence**: Save/load functionality with proper state management
- **SAS Compatibility**: Output formats compatible with SAS for comparison purposes
- **Performance Benchmarking**: Inference speed and memory usage tracking
- **Latent Space Analysis**: PCA, t-SNE preparation and statistical analysis
- **Model Comparison**: Compare different model states and configurations

## Quick Start

### Basic Model Evaluation

```python
from autoencoder import AutoencoderMLP, ModelEvaluator
import numpy as np

# Create or load your model
model = AutoencoderMLP(input_dim=784, latent_dim=400)

# Create evaluator
evaluator = ModelEvaluator(model, batch_size=256, output_dir="./results")

# Load your test data
test_data = np.random.randn(1000, 784)  # Replace with your actual data

# Run comprehensive evaluation
results = evaluator.evaluate_dataset(
    dataset=test_data,
    save_results=True,
    include_visualization=True
)

# Print key metrics
print(f"MSE: {results['reconstruction_metrics']['aggregate']['mse']:.6f}")
print(f"MAE: {results['reconstruction_metrics']['aggregate']['mae']:.6f}")

# Generate detailed report
evaluator.generate_evaluation_report()
```

## Core Components

### 1. ModelEvaluator Class

The main interface for model evaluation and inference.

```python
from autoencoder import ModelEvaluator

evaluator = ModelEvaluator(
    model=your_model,
    batch_size=256,
    device='cuda',  # or 'cpu'
    output_dir="./evaluation_results"
)

# Comprehensive evaluation
results = evaluator.evaluate_dataset(
    dataset=test_data,
    labels=test_labels,  # Optional
    save_results=True,
    include_visualization=True,
    results_prefix="my_evaluation"
)
```

### 2. Reconstruction Metrics

Calculate various reconstruction error metrics:

```python
from autoencoder import calculate_reconstruction_errors
import torch

original = torch.randn(100, 784)
reconstructed = model(original, return_latent=False)

# Calculate all metrics at once
errors = calculate_reconstruction_errors(original, reconstructed)
print(f"MSE: {errors['mse']:.6f}")
print(f"MAE: {errors['mae']:.6f}")
print(f"RMSE: {errors['rmse']:.6f}")
print(f"SSIM: {errors['ssim_approx']:.4f}")

# Per-sample errors
from autoencoder import calculate_per_sample_errors, calculate_aggregate_errors

per_sample = calculate_per_sample_errors(original, reconstructed)
aggregated = calculate_aggregate_errors(per_sample)

# Access statistics
print(f"MSE mean: {aggregated['mse_per_sample']['mean']:.6f}")
print(f"MSE std: {aggregated['mse_per_sample']['std']:.6f}")
```

### 3. Batch Inference Processing

Efficient processing of large datasets:

```python
from autoencoder import BatchInferenceProcessor

processor = BatchInferenceProcessor(
    model=model,
    batch_size=512,
    device='cuda',
    num_workers=4
)

# Process large dataset
original, reconstructed, latent = processor.process_data_arrays(
    large_dataset,
    return_latent=True,
    track_performance=True
)

# Get performance metrics
perf = processor.get_performance_summary()
print(f"Throughput: {perf['total_throughput_samples_per_sec']:.1f} samples/sec")
print(f"Memory usage: {perf['avg_memory_usage_mb']:.1f} MB")
```

### 4. Latent Space Analysis

Analyze and prepare latent representations:

```python
from autoencoder import compute_latent_statistics, prepare_latent_visualization
import torch

# Extract latent representations
latent = model.encode(test_data)

# Compute statistics
stats = compute_latent_statistics(latent)
print(f"Effective dimensionality: {stats['effective_dimensionality']:.2f}")
print(f"Mean activation: {stats['mean']:.4f}")

# Prepare for visualization
viz_data = prepare_latent_visualization(
    latent,
    labels=test_labels,
    use_pca=True,
    use_tsne=True,
    pca_components=50,
    tsne_components=2
)

# Access visualization data
pca_2d = viz_data['pca_data'][:, :2]  # First 2 PCA components
tsne_2d = viz_data['tsne_data']       # t-SNE embedding
explained_var = viz_data['pca_explained_variance_ratio']
```

### 5. Model Persistence

Save and load models with comprehensive metadata:

```python
from autoencoder import ModelSaver, ModelLoader

# Saving models
saver = ModelSaver(base_dir="./models")

# Save with metadata
model_path = saver.save_model(
    model=model,
    include_optimizer=True,
    optimizer=optimizer,
    scheduler=scheduler,
    metadata={
        'description': 'MNIST autoencoder after 50 epochs',
        'dataset': 'MNIST',
        'training_time': '45 minutes'
    },
    training_history=training_logs
)

# Loading models
model, metadata = ModelLoader.load_model(model_path, device='cuda')
print(f"Model saved: {metadata['save_timestamp']}")
print(f"Config: {metadata['model_config']}")
```

### 6. SAS-Compatible Output

Generate outputs compatible with SAS for comparison:

```python
from autoencoder import create_sas_compatible_outputs

# Generate SAS-compatible files
output_paths = create_sas_compatible_outputs(
    original=original_data,
    reconstructed=reconstructed_data,
    latent=latent_data,
    output_dir="./sas_outputs",
    base_filename="autoencoder_results"
)

print(f"CSV file: {output_paths['csv']}")
print(f"Metadata: {output_paths['metadata']}")
if 'parquet' in output_paths:
    print(f"Parquet file: {output_paths['parquet']}")
```

## Performance Benchmarking

### Inference Speed Testing

```python
# Benchmark different batch sizes
benchmark_results = evaluator.benchmark_inference_speed(
    data=test_data,
    batch_sizes=[32, 64, 128, 256, 512],
    n_runs=5
)

# Find optimal configuration
optimal_bs = benchmark_results['optimal_batch_size']
optimal_throughput = benchmark_results['optimal_throughput']

print(f"Optimal batch size: {optimal_bs}")
print(f"Max throughput: {optimal_throughput:.1f} samples/sec")

# Detailed results
for batch_size, metrics in benchmark_results['benchmark_results'].items():
    print(f"BS {batch_size}: {metrics['avg_throughput']:.1f} ± "
          f"{metrics['std_throughput']:.1f} samples/sec")
```

### Memory Usage Monitoring

```python
from autoencoder import PerformanceBenchmark

# Manual performance tracking
benchmark = PerformanceBenchmark()

for batch in data_loader:
    benchmark.start_measurement()
    
    # Your processing code here
    output = model(batch)
    
    benchmark.end_measurement(len(batch))

# Get detailed summary
summary = benchmark.get_summary()
print(f"Peak memory: {summary['max_memory_usage_mb']:.1f} MB")
print(f"Average time per batch: {summary['avg_batch_time']:.3f} seconds")
```

## Model Comparison

Compare different models or training states:

```python
from autoencoder import ModelComparator

# Load two different models
model1 = ModelLoader.load_model("model_epoch_10.pth")[0]
model2 = ModelLoader.load_model("model_epoch_50.pth")[0]

# Compare performance
comparator = ModelComparator()
comparison = comparator.compare_models(
    model1, model2, test_data,
    "Early Training", "Final Model"
)

# Results
summary = comparison['comparison_summary']
print(f"Better MSE: {summary['better_mse']}")
print(f"Better Speed: {summary['faster_inference']}")
print(f"MSE improvement: {summary['mse_improvement']:.6f}")

# Using evaluator for comparison
results = evaluator.compare_with_checkpoint(
    checkpoint_path="old_model.pth",
    dataset=test_data
)
```

## Advanced Usage

### Custom Metrics

```python
from autoencoder.metrics import ReconstructionMetrics

class CustomMetrics(ReconstructionMetrics):
    @staticmethod
    def custom_metric(original, reconstructed):
        # Your custom metric calculation
        return torch.mean(torch.abs(original - reconstructed) ** 1.5)

# Use in evaluation
metrics = CustomMetrics()
custom_error = metrics.custom_metric(original, reconstructed)
```

### Evaluation Pipeline Integration

```python
# Load and evaluate trained model
results = evaluator.load_and_evaluate(
    model_path="trained_model.pth",
    dataset=test_data,
    save_results=True,
    results_prefix="final_evaluation"
)

# Add to existing training pipeline
from autoencoder import TrainingPipeline

pipeline = TrainingPipeline(model, config)
training_results = pipeline.train(data_dir="./data")

# Evaluate final model
eval_results = evaluator.evaluate_dataset(
    dataset=test_data,
    save_results=True,
    results_prefix="post_training"
)

# Generate comprehensive report
report_path = evaluator.generate_evaluation_report()
```

## Output Files and Formats

The evaluation pipeline generates several types of output files:

### SAS-Compatible CSV

```csv
sample_id,label,original_1,original_2,...,reconstructed_1,reconstructed_2,...,latent_1,latent_2,...,reconstruction_mse,reconstruction_mae
0,7,0.123,-0.456,...,0.134,-0.445,...,0.234,0.567,...,0.001234,0.0345
1,2,-0.234,0.678,...,-0.223,0.687,...,-0.456,-0.123,...,0.002345,0.0456
...
```

### Evaluation Summary JSON

```json
{
  "evaluation_info": {
    "timestamp": "2024-01-15T10:30:45",
    "model_config": {
      "input_dim": 784,
      "latent_dim": 400,
      "activation": "tanh"
    },
    "n_samples": 10000,
    "device": "cuda"
  },
  "reconstruction_metrics": {
    "aggregate": {
      "mse": 0.012345,
      "mae": 0.067890,
      "rmse": 0.111213,
      "ssim_approx": 0.8456
    }
  },
  "latent_statistics": {
    "effective_dimensionality": 156.78,
    "mean": 0.0234,
    "std": 0.5678
  },
  "performance_metrics": {
    "total_throughput_samples_per_sec": 1234.5,
    "avg_memory_usage_mb": 512.3
  }
}
```

### Evaluation Report

```
================================================================================
AUTOENCODER MODEL EVALUATION REPORT
================================================================================

Evaluation Date: 2024-01-15T10:30:45
Model Configuration: {'input_dim': 784, 'latent_dim': 400, 'activation': 'tanh'}
Dataset Size: 10000 samples
Input Features: 784
Device Used: cuda

RECONSTRUCTION PERFORMANCE
----------------------------------------
Mean Squared Error (MSE): 0.012345
Mean Absolute Error (MAE): 0.067890
Root Mean Squared Error (RMSE): 0.111213
SSIM Approximation: 0.8456

LATENT SPACE ANALYSIS
----------------------------------------
Latent Dimension: 400
Mean Activation: 0.0234
Standard Deviation: 0.5678
Effective Dimensionality: 156.78

COMPUTATIONAL PERFORMANCE
----------------------------------------
Total Samples: 10000
Total Processing Time: 8.12 seconds
Average Throughput: 1234.5 samples/sec
Average Memory Usage: 512.3 MB

================================================================================
```

## Error Handling and Best Practices

### Memory Management

```python
# For large datasets, use smaller batch sizes
processor = BatchInferenceProcessor(model, batch_size=128)

# Enable garbage collection for long-running evaluations
import gc
results = evaluator.evaluate_dataset(data)
gc.collect()
```

### Device Management

```python
# Automatic device detection
evaluator = ModelEvaluator(model)  # Uses model's device

# Explicit device specification
evaluator = ModelEvaluator(model, device='cuda:1')

# CPU fallback for memory-constrained environments
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    device = 'cuda'
else:
    device = 'cpu'
    
evaluator = ModelEvaluator(model, device=device)
```

### Validation and Error Checking

```python
try:
    results = evaluator.evaluate_dataset(data)
except ValueError as e:
    print(f"Data validation error: {e}")
except RuntimeError as e:
    print(f"Runtime error during evaluation: {e}")
    # Try with smaller batch size
    evaluator.batch_size = evaluator.batch_size // 2
    results = evaluator.evaluate_dataset(data)
```

## Integration with Existing Workflows

### With Training Pipeline

```python
# Train model
pipeline = TrainingPipeline(model, config)
training_summary = pipeline.train("./data")

# Evaluate trained model
evaluator = ModelEvaluator(model, output_dir="./evaluation")
eval_results = evaluator.evaluate_dataset(test_data)

# Compare with checkpoint
best_checkpoint = training_summary['best_checkpoint_path']
comparison = evaluator.compare_with_checkpoint(best_checkpoint, test_data)
```

### Batch Processing Workflow

```python
# Process multiple datasets
datasets = ['test_set_1.npz', 'test_set_2.npz', 'validation_set.npz']

for dataset_path in datasets:
    data = np.load(dataset_path)
    results = evaluator.evaluate_dataset(
        dataset=data['images'],
        labels=data['labels'],
        results_prefix=Path(dataset_path).stem
    )
    print(f"Dataset {dataset_path}: MSE = {results['reconstruction_metrics']['aggregate']['mse']:.6f}")
```

## Testing

Run the test suite to verify functionality:

```bash
python test_evaluation_pipeline.py
```

Or run individual test classes:

```python
python -m unittest autoencoder.test_evaluation_pipeline.TestModelEvaluator
```

## Examples

See `evaluation_example.py` for comprehensive usage examples covering all functionality.

## Dependencies

The evaluation pipeline requires:

- PyTorch >= 1.9.0
- NumPy
- Pandas
- Scikit-learn (for PCA, t-SNE)
- Matplotlib (for visualization preparation)
- psutil (for memory monitoring)

Optional dependencies:

- pyarrow (for Parquet output format)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU device
2. **Slow t-SNE**: Reduce dataset size or use PCA preprocessing
3. **Large output files**: Use Parquet format instead of CSV for large datasets
4. **Import errors**: Ensure all dependencies are installed

### Performance Optimization

1. **Use appropriate batch sizes**: Run `benchmark_inference_speed()` to find optimal batch size
2. **Enable mixed precision**: For CUDA, use `torch.cuda.amp` for faster inference
3. **Parallel data loading**: Set `num_workers > 0` in BatchInferenceProcessor
4. **Memory pinning**: Automatic for CUDA devices, improves transfer speed

## API Reference

For detailed API documentation, see the docstrings in each module:

- `metrics.py`: Reconstruction metrics and latent space analysis
- `model_io.py`: Model saving, loading, and output formatting
- `evaluator.py`: Main evaluation and inference pipeline

## Contributing

When adding new evaluation metrics or functionality:

1. Add comprehensive docstrings
2. Include unit tests in `test_evaluation_pipeline.py`
3. Update this documentation
4. Follow the existing code style and patterns
5. Ensure SAS compatibility for output formats

## Success Criteria Verification

The evaluation pipeline meets all specified requirements:

- ✅ **Reconstruction errors**: MSE, MAE calculated correctly with per-sample and aggregate statistics
- ✅ **Model persistence**: Complete save/load functionality with proper state management
- ✅ **Batch inference**: Efficient processing of large datasets without memory issues
- ✅ **SAS compatibility**: Output formats enable direct comparison with SAS results
- ✅ **Latent space analysis**: Comprehensive statistical analysis and visualization preparation
- ✅ **Performance metrics**: Detailed benchmarking of inference speed and memory usage
- ✅ **Model comparison**: Utilities for comparing different model states and configurations
- ✅ **Evaluation reporting**: Comprehensive and interpretable evaluation reports
