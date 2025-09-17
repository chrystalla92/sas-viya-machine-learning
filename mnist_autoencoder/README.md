# MNIST Autoencoder

A modern Python package for training and evaluating autoencoders on MNIST handwritten digit data. This package provides a comprehensive framework for implementing autoencoder neural networks with clean, modular code organization.

## Features

- 🏗️ **Modular Architecture**: Well-organized package structure with separate modules for data, models, training, and evaluation
- 🔥 **PyTorch Integration**: Built on PyTorch for flexible and efficient neural network implementation  
- 📊 **Comprehensive Evaluation**: Built-in metrics and visualization tools for model assessment
- 🧪 **Full Test Coverage**: Complete test suite with pytest for reliability
- 📚 **Rich Documentation**: Comprehensive API documentation with Sphinx
- 🐍 **Modern Python**: Type hints, modern packaging with pyproject.toml, and Python 3.8+ support

## Installation

### From Source (Development)

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd sas-viya-machine-learning
```

2. Install in development mode:
```bash
pip install -e ./mnist_autoencoder
```

This will install the package in editable mode along with all dependencies.

### With Development Dependencies

For development with testing, linting, and documentation tools:
```bash
pip install -e "mnist_autoencoder[dev]"
```

For documentation building:
```bash
pip install -e "mnist_autoencoder[docs]"
```

For Jupyter notebook support:
```bash
pip install -e "mnist_autoencoder[jupyter]"
```

For all optional dependencies:
```bash
pip install -e "mnist_autoencoder[all]"
```

## Quick Start

```python
import mnist_autoencoder

# Import specific modules
from mnist_autoencoder import data, models, training, evaluation

# The package follows a modular structure:
# - data: Data loading and preprocessing utilities
# - models: Autoencoder architectures and model definitions  
# - training: Training loops and optimization strategies
# - evaluation: Metrics, visualization, and model assessment
```

## Package Structure

```
mnist_autoencoder/
├── __init__.py              # Package initialization and exports
├── data/                    # Data loading and preprocessing
│   └── __init__.py
├── models/                  # Autoencoder model architectures
│   └── __init__.py
├── training/               # Training utilities and procedures
│   └── __init__.py
├── evaluation/             # Evaluation metrics and visualization
│   └── __init__.py
└── README.md               # This file
```

## Development

### Setup Development Environment

1. Clone the repository
2. Install in development mode with all dependencies:
```bash
pip install -e "mnist_autoencoder[all]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=mnist_autoencoder

# Run specific test file
pytest tests/test_package.py
```

### Code Formatting and Linting

```bash
# Format code with black
black mnist_autoencoder/

# Sort imports with isort
isort mnist_autoencoder/

# Lint with flake8
flake8 mnist_autoencoder/

# Type checking with mypy
mypy mnist_autoencoder/
```

### Building Documentation

```bash
cd docs/
make html
# Documentation will be built in docs/_build/html/
```

## Requirements

- Python 3.8+
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- tqdm >= 4.64.0

## Development Dependencies

- pytest >= 7.0.0 (testing)
- black >= 22.0.0 (code formatting)
- flake8 >= 5.0.0 (linting)
- mypy >= 0.991 (type checking)
- sphinx >= 5.0.0 (documentation)

## Usage Examples

*Note: Detailed usage examples will be added as the package modules are implemented in future development phases.*

## Contributing

This package is part of the SAS Viya Machine Learning collection. Contributions should follow the established coding standards and include appropriate tests and documentation.

## License

This project is licensed under the same terms as the parent SAS Viya Machine Learning repository.

## Support

For support and questions, please refer to the main repository documentation and support channels.
