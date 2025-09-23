# Data Directory

This directory contains datasets and data-related files for autoencoder training and evaluation.

## Structure

```
data/
├── raw/              # Original, unprocessed datasets
├── processed/        # Cleaned and preprocessed datasets  
├── external/         # External datasets from third parties
└── interim/          # Intermediate data during processing
```

## Datasets

- **MNIST**: Handwritten digits dataset for basic autoencoder training
- **CIFAR-10**: Natural images for convolutional autoencoder experiments
- **Custom datasets**: Domain-specific datasets as needed

## Usage Notes

- Raw data files are typically downloaded automatically by data loaders
- Processed data files are generated during preprocessing steps
- Large datasets are excluded from version control (see .gitignore)
- Use `torchvision.datasets` for standard datasets when possible
