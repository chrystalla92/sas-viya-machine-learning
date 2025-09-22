# MNIST Autoencoder Visualization Framework

This comprehensive visualization framework provides publication-ready plotting capabilities for the MNIST autoencoder project. The framework migrates and enhances the original SAS plotting functionality while adding advanced analysis and diagnostic tools.

## Overview

The visualization package consists of four main modules:

1. **`visualization/plots.py`** - Core image visualization and reconstruction comparisons
2. **`visualization/training_plots.py`** - Training metrics and progress monitoring
3. **`visualization/analysis.py`** - Latent space analysis and diagnostic tools
4. **`utils/plot_utils.py`** - Common utilities and publication-ready styling

## Key Features

### ✅ Migrated SAS Functionality
- **MNIST Grid Display**: Direct migration of `python_plot.sas` functionality
- **Enhanced Grid Layouts**: Flexible grid sizes and improved formatting
- **CSV Data Loading**: Compatible with SAS output files

### ✅ Training Visualization
- **Loss Curves**: Training and validation loss tracking
- **Convergence Analysis**: Automatic convergence detection and analysis
- **Progress Monitoring**: Comprehensive training overview dashboards
- **Batch Analysis**: Within-epoch loss progression

### ✅ Image Analysis
- **Reconstruction Comparisons**: Side-by-side original vs reconstructed
- **Error Visualization**: Pixel-wise error heatmaps
- **Quality Assessment**: Statistical reconstruction analysis
- **Batch Processing**: Automated sample generation

### ✅ Latent Space Analysis
- **Dimensionality Reduction**: PCA and t-SNE visualizations
- **Hidden Representations**: Activation pattern analysis
- **Cluster Analysis**: Digit grouping in latent space
- **Interactive Exploration**: Customizable analysis parameters

### ✅ Diagnostic Tools
- **Weight Distributions**: Model parameter analysis
- **Activation Heatmaps**: Hidden layer visualization
- **Error Distributions**: Reconstruction error statistics
- **Model Diagnostics**: Comprehensive health checks

### ✅ Publication Features
- **Professional Styling**: Consistent, publication-ready appearance
- **Multi-format Export**: PNG, PDF, SVG support
- **Interactive Mode**: Real-time plot updates
- **Customizable Themes**: Color schemes and styling options

## Installation Requirements

```python
# Core dependencies
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
torch>=1.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

## Quick Start

### Basic Usage

```python
from visualization.plots import plot_mnist_grid, plot_reconstruction_comparison
from visualization.training_plots import plot_training_curves
from visualization.analysis import plot_latent_pca

# Display MNIST images (replicating SAS functionality)
fig = plot_mnist_grid(images, labels, title="MNIST Samples")

# Compare original vs reconstructed
fig = plot_reconstruction_comparison(original, reconstructed, labels)

# Plot training progress
fig = plot_training_curves(train_losses, val_losses)

# Visualize latent space
fig = plot_latent_pca(hidden_representations, labels)
```

### Complete Analysis Pipeline

```python
from visualization.analysis import analyze_hidden_representations

# Run comprehensive analysis
figures = analyze_hidden_representations(
    model=trained_model,
    data=test_images,
    labels=test_labels,
    output_dir="./analysis_results"
)
```

## Module Documentation

### 1. Core Image Visualization (`visualization/plots.py`)

#### `plot_mnist_grid(images, labels, grid_size=(5,5))`
Replicates the original SAS `python_plot.sas` functionality with enhancements.

**Features:**
- Flexible grid layouts
- Automatic image formatting (28x28 reshape)
- Label display options
- Publication-ready styling

**Example:**
```python
# Basic 5x5 grid (matching SAS behavior)
fig = plot_mnist_grid(mnist_images[:25], mnist_labels[:25])

# Custom grid with enhanced formatting
fig = plot_mnist_grid(
    images=mnist_data,
    labels=mnist_labels,
    grid_size=(4, 6),
    title="MNIST Test Samples",
    save_path="./results/mnist_grid"
)
```

#### `plot_reconstruction_comparison(original, reconstructed, labels)`
Side-by-side comparison of original and reconstructed images.

**Features:**
- Aligned original/reconstructed pairs
- Quality metrics display
- Batch processing support
- Error highlighting

#### `plot_reconstruction_grid(original, reconstructed, reconstruction_errors)`
Comprehensive three-column layout showing original, reconstructed, and error images.

### 2. Training Visualization (`visualization/training_plots.py`)

#### `plot_training_curves(train_losses, val_losses, epochs)`
Professional training progress visualization.

**Features:**
- Dual-axis loss plotting
- Best model marking
- Statistical annotations
- Log-scale support

**Example:**
```python
fig = plot_training_curves(
    train_losses=trainer.metrics.train_losses,
    val_losses=trainer.metrics.val_losses,
    show_best=True,
    save_path="./results/training_curves"
)
```

#### `plot_loss_convergence(losses, convergence_threshold)`
Advanced convergence analysis with automatic detection.

#### `plot_training_progress(metrics_data)`
Comprehensive 2x2 dashboard with multiple training insights.

### 3. Latent Space Analysis (`visualization/analysis.py`)

#### `plot_latent_pca(hidden_representations, labels)`
PCA visualization of hidden layer activations.

**Features:**
- Explained variance reporting
- Digit-wise coloring
- Interactive legends
- Statistical summaries

#### `plot_latent_tsne(hidden_representations, labels, perplexity=30)`
t-SNE visualization for non-linear dimensionality reduction.

#### `plot_reconstruction_errors(original, reconstructed, labels, error_type="mse")`
Comprehensive error analysis with multiple error metrics.

**Error Types:**
- `"mse"`: Mean Squared Error
- `"mae"`: Mean Absolute Error
- `"pixel"`: Pixel-wise error heatmaps

#### `plot_weight_distributions(model)`
Model parameter analysis and distribution visualization.

#### `plot_activation_heatmap(hidden_activations, labels)`
Hidden layer activation pattern visualization.

### 4. Utilities (`utils/plot_utils.py`)

#### Publication Styling
```python
from utils.plot_utils import setup_publication_style

setup_publication_style()  # Apply consistent formatting
```

#### Figure Management
```python
from utils.plot_utils import PlotManager

with PlotManager(figsize=(12, 8), save_path="./results/plot") as fig:
    # Create plots
    plt.plot(data)
```

## Advanced Usage

### CSV Data Loading (SAS Migration)
```python
from visualization.plots import load_and_plot_from_csv

# Load and plot directly from SAS output CSV
fig = load_and_plot_from_csv(
    csv_filepath="mnist_train_10_autoencoder_score.csv",
    grid_size=(5, 5),
    title="MNIST from SAS Output"
)
```

### Training Metrics from JSON
```python
from visualization.training_plots import load_and_plot_training_metrics

# Load metrics saved by training framework
fig = load_and_plot_training_metrics(
    metrics_file="./logs/training_metrics.json",
    plot_type="progress"
)
```

### Batch Reconstruction Analysis
```python
from visualization.plots import save_reconstruction_samples

# Save individual comparison plots for detailed analysis
save_reconstruction_samples(
    original_images=test_data,
    reconstructed_images=reconstructions,
    labels=test_labels,
    output_dir="./detailed_analysis",
    n_samples=50
)
```

### Comprehensive Model Analysis
```python
from visualization.analysis import analyze_hidden_representations

# Complete latent space analysis pipeline
results = analyze_hidden_representations(
    model=autoencoder_model,
    data=evaluation_data,
    labels=evaluation_labels,
    output_dir="./comprehensive_analysis",
    n_samples=1000
)

# Results contain: 'pca', 'tsne', 'heatmap', 'weights', 'errors'
for analysis_type, figure in results.items():
    print(f"Generated {analysis_type} analysis")
```

## Configuration Options

### Color Schemes
```python
from utils.plot_utils import COLORS, COLORMAP_OPTIONS

# Available color schemes
COLORS['primary']    # Main plotting color
COLORS['secondary']  # Secondary color
COLORS['accent']     # Highlight color

COLORMAP_OPTIONS['mnist']     # Grayscale for MNIST
COLORMAP_OPTIONS['heatmap']   # Viridis for heatmaps
COLORMAP_OPTIONS['error']     # Reds for error visualization
```

### Export Formats
```python
# Multiple format export
save_figure(fig, "results/plot", formats=['png', 'pdf', 'svg'])

# High-resolution export
save_figure(fig, "results/plot", dpi=300, formats=['png'])
```

### Interactive Mode
```python
from utils.plot_utils import enable_interactive_mode

enable_interactive_mode()  # Enable real-time updates
```

## Example Scripts

### Complete Visualization Demo
```bash
# Run comprehensive demonstration
python example_visualization.py --output_dir ./demo_results

# Use real MNIST data
python example_visualization.py --data_path ./data/mnist --output_dir ./results

# Load trained model
python example_visualization.py --model_path ./checkpoints/best_model.pth --output_dir ./results
```

### Integration with Training Framework
```python
from training import AutoencoderTrainer
from visualization.training_plots import plot_training_progress

# Train model with visualization
trainer = AutoencoderTrainer(config)
trainer.train(train_data, val_data)

# Visualize training progress
fig = plot_training_progress(
    trainer.metrics.get_summary(),
    save_path="./results/training_analysis"
)
```

## Migration from SAS

The original SAS plotting functionality in `python_plot.sas` has been fully migrated and enhanced:

### Original SAS Code:
```python
# Original python_plot.sas functionality
f = open("mnist_train_10_autoencoder_score.csv", 'r')
a = f.readlines()
f.close()

f = pl.figure(figsize=(15,15));
count=1
for line in a:
    linebits = line.split(',')
    imarray = np.asfarray(linebits[1:]).reshape((28,28))
    pl.subplot(5,5,count)
    pl.subplots_adjust(hspace=0.5)
    count += 1
    pl.title("Label is " + linebits[0])
    pl.imshow(imarray, cmap='Greys', interpolation='None')

pl.show()
```

### New Enhanced Version:
```python
# Enhanced Python version with additional features
from visualization.plots import load_and_plot_from_csv

fig = load_and_plot_from_csv(
    csv_filepath="mnist_train_10_autoencoder_score.csv",
    grid_size=(5, 5),
    title="MNIST Handwriting Recognition Results",
    save_path="./results/mnist_analysis"
)
```

**Enhancements:**
- ✅ Error handling for file I/O
- ✅ Flexible grid layouts
- ✅ Publication-ready formatting
- ✅ Multi-format export (PNG, PDF, SVG)
- ✅ Consistent styling
- ✅ Statistical annotations
- ✅ Interactive mode support

## Performance Considerations

### Memory Management
- Large datasets are processed in batches
- Tensor to numpy conversion is optimized
- Memory cleanup after plot generation

### Optimization Tips
```python
# For large datasets, limit samples
plot_latent_tsne(hidden_repr[:500], labels[:500])  # Limit for t-SNE

# Use batch processing for reconstructions
save_reconstruction_samples(data, reconstructions, n_samples=20)

# Enable memory cleanup
save_figure(fig, path, close_after_save=True)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure proper path setup
   import sys
   sys.path.append('/path/to/autoencoder')
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Reduce sample sizes for analysis
   analyze_hidden_representations(model, data[:1000], n_samples=500)
   ```

3. **Display Issues in Headless Environments**
   ```python
   # Use non-interactive backend
   import matplotlib
   matplotlib.use('Agg')
   ```

## Contributing

The visualization framework is designed to be extensible. Key extension points:

1. **Custom Colormaps**: Add to `COLORMAP_OPTIONS`
2. **New Analysis Types**: Extend `analysis.py`
3. **Additional Metrics**: Enhance training visualization
4. **Export Formats**: Add support for new formats

## Success Criteria Verification

✅ **All original SAS plotting capabilities replicated**
- `plot_mnist_grid()` fully replicates `python_plot.sas`
- Enhanced with flexible layouts and error handling
- CSV loading functionality maintained

✅ **Training visualizations provide clear insights**
- Loss curves with convergence detection
- Multi-metric progress dashboards
- Batch-level analysis capabilities

✅ **Image reconstructions clearly displayed**
- Side-by-side comparisons
- Error heatmaps and statistics
- Quality assessment metrics

✅ **Latent space analysis reveals patterns**
- PCA and t-SNE dimensionality reduction
- Activation pattern visualization
- Cluster analysis by digit class

✅ **Training progress monitored comprehensively**
- Real-time loss tracking
- Convergence analysis
- Parameter distribution monitoring

✅ **Publication-ready formatting**
- Consistent styling and color schemes
- Multi-format export capabilities
- Professional layout and typography

This visualization framework provides a complete solution for analyzing and presenting MNIST autoencoder results, fully replacing the original SAS functionality while adding significant enhancements and new capabilities.
