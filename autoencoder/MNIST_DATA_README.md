# MNIST Data Loading for PyTorch Autoencoders

This module provides functionality to load MNIST binary files directly and convert them to PyTorch tensors with proper standardization, compatible with the SAS implementation.

## Features

- **Direct Binary File Reading**: Reads IDX3-UBYTE (images) and IDX1-UBYTE (labels) files
- **SAS-Compatible Standardization**: Implements midrange standardization `(x - midrange) / (max - min)`
- **PyTorch Integration**: Provides Dataset and DataLoader classes
- **Memory Efficient**: Optimized for typical development machines
- **Error Handling**: Comprehensive validation and error handling

## Files

- `data_loader.py`: Core data loading and standardization functions
- `datasets.py`: PyTorch Dataset and DataLoader classes
- `test_data_loading.py`: Test suite to verify functionality
- `__init__.py`: Package initialization

## Quick Start

### Basic Data Loading

```python
from autoencoder import load_mnist_training_data, load_mnist_test_data

# Load training data (60,000 samples)
train_images, train_labels = load_mnist_training_data("./data")
print(f"Training: {train_images.shape}, {train_labels.shape}")
# Output: Training: torch.Size([60000, 784]), torch.Size([60000])

# Load test data (10,000 samples) 
test_images, test_labels = load_mnist_test_data("./data")
print(f"Test: {test_images.shape}, {test_labels.shape}")
# Output: Test: torch.Size([10000, 784]), torch.Size([10000])
```

### Using DataLoaders for Training

```python
from autoencoder import get_train_dataloader, get_test_dataloader

# Simple DataLoaders
train_loader = get_train_dataloader("./data", batch_size=64)
test_loader = get_test_dataloader("./data", batch_size=64)

# For autoencoder training (input = target)
for batch in train_loader:
    images = batch  # Shape: (64, 784)
    # Train autoencoder: reconstruct images from images
    reconstructed = model(images)
    loss = criterion(reconstructed, images)
```

### Advanced DataLoader with Train/Validation Split

```python
from autoencoder import create_mnist_dataloaders

train_loader, val_loader, test_loader = create_mnist_dataloaders(
    data_dir="./data",
    batch_size=32,
    train_val_split=0.8,  # 80% train, 20% validation
    standardize=True,
    shuffle=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")
```

## Data Format

### Input Files Required

Place these MNIST binary files in your data directory:

- `train-images.idx3-ubyte` (60,000 training images)
- `train-labels.idx1-ubyte` (60,000 training labels)
- `t10k-images.idx3-ubyte` (10,000 test images)
- `t10k-labels.idx1-ubyte` (10,000 test labels)

### Output Format

- **Images**: `torch.Tensor` of shape `(N, 784)` with dtype `float32`
  - Flattened 28×28 pixel images
  - Values standardized using midrange scaling by default
- **Labels**: `torch.Tensor` of shape `(N,)` with dtype `int64`
  - Integer labels from 0 to 9

### Standardization

The midrange standardization matches the SAS `standardize=midrange` option:

```
standardized = (x - midrange) / (max - min)
where midrange = (max + min) / 2
```

For MNIST pixels (range 0-255):
- `midrange = (0 + 255) / 2 = 127.5`
- `standardized_range = [-0.5, 0.5]`

## API Reference

### Core Functions

#### `load_mnist_training_data(data_dir, standardize=True)`
Load MNIST training data (60,000 samples).

#### `load_mnist_test_data(data_dir, standardize=True)`  
Load MNIST test data (10,000 samples).

#### `midrange_standardize(data)`
Apply midrange standardization to numpy array.

### Dataset Classes

#### `MNISTAutoencoderDataset(images, labels=None)`
PyTorch Dataset for autoencoder training where input = target.

#### `MNISTDatasetFromFiles(images_path, labels_path, **kwargs)`
Dataset that loads from binary files (memory efficient for larger datasets).

### DataLoader Utilities

#### `create_mnist_dataloaders(**kwargs)`
Create train/validation/test DataLoaders with configurable split.

#### `get_train_dataloader(data_dir, batch_size, **kwargs)`
Quick access to training DataLoader.

#### `get_test_dataloader(data_dir, batch_size, **kwargs)`
Quick access to test DataLoader.

## Testing

Run the test suite to verify your installation:

```python
python test_data_loading.py
```

This will test:
- Basic data loading functionality
- Standardization correctness
- DataLoader creation and batching
- Memory usage efficiency

## Example: Complete Autoencoder Training Setup

```python
import torch
import torch.nn as nn
from autoencoder import create_mnist_dataloaders

# Create DataLoaders
train_loader, val_loader, test_loader = create_mnist_dataloaders(
    data_dir="./data",
    batch_size=128,
    train_val_split=0.9
)

# Simple autoencoder model
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.Tanh()
        )
        self.decoder = nn.Linear(400, 784)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = SimpleAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    for batch in train_loader:
        images = batch  # Input images
        
        # Forward pass
        reconstructed = model(images)
        loss = criterion(reconstructed, images)  # Reconstruct same images
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Compatibility

This implementation is designed to be compatible with the existing SAS autoencoder workflow while providing the flexibility and performance benefits of PyTorch for deep learning applications.

## Error Handling

The module includes comprehensive error handling for:
- Missing MNIST binary files
- Corrupted or invalid file formats
- Incorrect data shapes or types
- File I/O errors

All errors provide clear messages indicating the issue and suggested solutions.
