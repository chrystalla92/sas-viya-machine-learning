"""
Training module for MLP Autoencoder.

This module provides comprehensive training orchestration including:
- Trainer class for complete training management
- Training configuration and metrics tracking
- Utility functions for data preparation and evaluation
- Checkpointing and training resumption capabilities
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    GradientMonitor,
    create_data_loaders
)

from .utils import (
    create_training_config,
    create_sas_compatible_config,
    prepare_mnist_data,
    setup_training_environment,
    evaluate_reconstruction_quality,
    plot_training_history,
    plot_reconstruction_examples,
    save_training_config,
    load_training_config,
    create_trainer_from_config,
    benchmark_training_performance,
    compare_optimizers
)

__all__ = [
    # Core training classes
    "Trainer",
    "TrainingConfig", 
    "TrainingMetrics",
    "GradientMonitor",
    
    # Data utilities
    "create_data_loaders",
    "prepare_mnist_data",
    
    # Configuration utilities
    "create_training_config",
    "create_sas_compatible_config",
    "save_training_config",
    "load_training_config",
    "create_trainer_from_config",
    
    # Environment and evaluation utilities
    "setup_training_environment",
    "evaluate_reconstruction_quality",
    "benchmark_training_performance",
    "compare_optimizers",
    
    # Visualization utilities
    "plot_training_history",
    "plot_reconstruction_examples"
]
