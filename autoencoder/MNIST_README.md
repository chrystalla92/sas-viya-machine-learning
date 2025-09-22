# MNIST Data Processing Implementation

This implementation provides Python utilities to read and process MNIST dataset files in a way that exactly matches the behavior of the SAS `createData.sas` script.

## Overview

The implementation consists of two main modules:

- **`mnist_data.py`**: Core MNIST IDX file reading and processing
- **`data_utils.py`**: Utility functions for dataset handling and validation

## Key Features

### ✅ IDX Binary Format Parser
- Handles big-endian byte order
- Correctly parses magic numbers and dimensions  
- Skips appropriate header bytes (16 for images, 8 for labels)
- Compatible with MNIST IDX3 (images) and IDX1 (labels) formats

### ✅ Image Processing
- Flattens 28×28 pixel images to 784-dimensional vectors
- Preserves pixel intensity values as floating-point
- Memory-efficient processing for large datasets

### ✅ Midrange Standardization
- Implements SAS `standardize=midrange` algorithm exactly
- Formula: `(x - midrange) / range` where `midrange = (max + min) / 2`
- Applied per-pixel across all images
- Handles edge cases (constant pixels with range = 0)

### ✅ SAS-Compatible Output
- Creates datasets matching SAS `mnist_train` format
- Column 0: labels (var1)
- Columns 1-784: pixel values (var2-var785)
- Maintains proper image-label correspondence

## Usage Examples

### Basic Usage

```python
from mnist_data import load_mnist_data, create_sas_format_dataset

# Load MNIST data with standardization
images, labels = load_mnist_data(
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte', 
    standardize=True
)

# Create SAS-compatible dataset format
sas_dataset = create_sas_format_dataset(
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    standardize=True
)

print(f"Loaded {len(images)} training examples")
print(f"Dataset shape: {sas_dataset.shape}")  # (60000, 785)
```

### Advanced Usage

```python
from mnist_data import MNISTReader
from data_utils import train_validation_split, save_dataset_csv

# Initialize reader
reader = MNISTReader()

# Load data
images, labels = reader.load_mnist_dataset(
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    standardize=True
)

# Split data
train_img, val_img, train_lbl, val_lbl = train_validation_split(
    images, labels, 
    validation_ratio=0.2,
    random_seed=42,
    stratify=True
)

# Create SAS format and save
sas_data = reader.create_sas_compatible_dataset(train_img, train_lbl)
save_dataset_csv(sas_data, 'mnist_train.csv', include_header=False)
```

### Memory-Efficient Processing

```python
from data_utils import create_batches, memory_efficient_process

# Process in batches for large datasets
batches = create_batches(images, labels, batch_size=1000, shuffle=True)

for batch_images, batch_labels in batches:
    # Process each batch
    pass
```

## File Structure

```
autoencoder/
├── mnist_data.py              # Core MNIST processing
├── data_utils.py              # Utility functions  
├── test_mnist_implementation.py   # Test and demo script
├── MNIST_README.md            # This documentation
├── createData.sas             # Original SAS implementation
└── [other SAS files]
```

## API Reference

### MNISTReader Class

```python
class MNISTReader:
    def read_idx_images(filepath: str) -> np.ndarray
    def read_idx_labels(filepath: str) -> np.ndarray  
    def apply_midrange_standardization(images: np.ndarray = None) -> np.ndarray
    def load_mnist_dataset(images_path: str, labels_path: str, standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]
    def create_sas_compatible_dataset(images: np.ndarray, labels: np.ndarray) -> np.ndarray
```

### Utility Functions

```python
# Data loading
load_mnist_data(images_path, labels_path, standardize=True) -> Tuple[np.ndarray, np.ndarray]
create_sas_format_dataset(images_path, labels_path, standardize=True) -> np.ndarray

# Dataset utilities  
train_validation_split(images, labels, validation_ratio=0.2, random_seed=None, stratify=True)
create_batches(images, labels=None, batch_size=32, shuffle=True, random_seed=None)
validate_dataset_format(dataset, expected_shape=None, check_sas_format=True)
save_dataset_csv(dataset, filepath, include_header=False, sas_compatible=True)
dataset_statistics(images, labels) -> Dict
```

## Data Format Specifications

### IDX File Format
- **Images (IDX3)**: `[magic:4][n_images:4][rows:4][cols:4][pixel_data...]`
- **Labels (IDX1)**: `[magic:4][n_labels:4][label_data...]`
- All multi-byte integers are big-endian
- Magic numbers: 2051 (images), 2049 (labels)

### SAS Compatible Format
- Shape: `(n_samples, 785)`
- Column 0: Integer labels (0-9)
- Columns 1-784: Standardized pixel values
- Matches SAS `var1` (label) + `var2-var785` (pixels)

## Standardization Algorithm

The midrange standardization matches SAS `proc nnet standardize=midrange`:

```python
# Per-pixel across all images:
min_vals = np.min(images, axis=0)
max_vals = np.max(images, axis=0)
midrange = (max_vals + min_vals) / 2.0
range_vals = max_vals - min_vals

# Standardize:
standardized = (images - midrange) / range_vals
```

## Error Handling

The implementation includes comprehensive error handling:

- **File Validation**: Checks file existence and accessibility
- **Format Validation**: Verifies magic numbers and expected dimensions
- **Data Integrity**: Validates complete reads and correct array shapes
- **Edge Cases**: Handles constant pixels, empty datasets, mismatched sizes

## Testing

Run the test script to verify implementation:

```bash
python test_mnist_implementation.py
```

Tests include:
- IDX format parsing verification
- Midrange standardization accuracy
- SAS format compatibility
- Dataset validation
- Train/validation splits

## Performance Characteristics

### Memory Usage
- **Efficient**: Loads full MNIST dataset (~180MB) comfortably in standard memory
- **Scalable**: Batch processing support for larger datasets
- **Optimized**: Uses NumPy arrays for efficient numerical operations

### Processing Speed
- **Fast**: Leverages NumPy vectorized operations
- **Parallel**: Standardization computed across all pixels simultaneously
- **Cached**: Standardization parameters stored for reuse

## Compatibility

### SAS Equivalence
- ✅ Identical IDX file parsing (header skipping, byte order)
- ✅ Exact midrange standardization algorithm
- ✅ Matching output format (785 columns)
- ✅ Same image-label correspondence

### Python Requirements
- NumPy (required)
- Python 3.6+ (for type hints and f-strings)
- Standard library only (struct, os, warnings)

## Common Use Cases

### 1. Direct SAS Replacement
```python
# Replace SAS createData.sas with:
dataset = create_sas_format_dataset(
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    standardize=True
)
# Result matches SAS mnist_train exactly
```

### 2. PyTorch Integration
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

images, labels = load_mnist_data(..., standardize=True)
dataset = TensorDataset(torch.FloatTensor(images), torch.LongTensor(labels))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3. Preprocessing Pipeline
```python
# Complete preprocessing pipeline
images, labels = load_mnist_data(..., standardize=True)
train_img, val_img, train_lbl, val_lbl = train_validation_split(images, labels)
train_batches = create_batches(train_img, train_lbl, batch_size=64)
```

## Troubleshooting

### Common Issues

**FileNotFoundError**: Ensure MNIST IDX files are in correct location
```python
# Check file paths
import os
print(os.path.exists('train-images-idx3-ubyte'))
```

**Shape Mismatch**: Verify IDX files are not corrupted
```python
# Validate dataset
from data_utils import validate_dataset_format
validation = validate_dataset_format(dataset)
print(validation['issues'])
```

**Memory Error**: Use batch processing for large datasets
```python
# Process in smaller batches
batches = create_batches(images, batch_size=1000)
```

### Getting MNIST Data

Download MNIST IDX files from:
- [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
- Files needed:
  - `train-images-idx3-ubyte` (training images)
  - `train-labels-idx1-ubyte` (training labels)  
  - `t10k-images-idx3-ubyte` (test images)
  - `t10k-labels-idx1-ubyte` (test labels)

## Integration with Autoencoder

This implementation is designed to feed into PyTorch autoencoder training:

```python
# Prepare data for autoencoder
images, labels = load_mnist_data(..., standardize=True)

# For autoencoder: use images as both input and target
# Labels can be used for visualization/analysis
autoencoder_data = images  # Shape: (60000, 784)
```

The standardized pixel values are ready for neural network training with appropriate range and distribution.
