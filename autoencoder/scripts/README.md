# Scripts Directory

This directory contains standalone Python scripts for various autoencoder operations:

- **Training Scripts**: Scripts for training different autoencoder architectures
- **Evaluation Scripts**: Scripts for model evaluation and testing
- **Data Processing Scripts**: Utilities for data preprocessing and management
- **Visualization Scripts**: Scripts for generating plots and visualizations

## Usage

Scripts in this directory are designed to be run from the project root:

```bash
python scripts/train_mnist_autoencoder.py --config configs/mnist_config.yaml
python scripts/evaluate_model.py --model-path models/checkpoints/best_model.pth
```

## Structure

- `train_*.py`: Training scripts for different datasets/architectures
- `evaluate_*.py`: Evaluation and testing scripts  
- `visualize_*.py`: Visualization and plotting scripts
- `preprocess_*.py`: Data preprocessing utilities
