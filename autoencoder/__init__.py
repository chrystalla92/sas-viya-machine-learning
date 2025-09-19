"""
Autoencoder package for MNIST data processing and neural network training.

This package provides utilities for loading MNIST binary data files,
converting them to PyTorch tensors, creating DataLoaders, and building
autoencoder models for training and inference.
"""

from .data_loader import (
    read_idx3_ubyte,
    read_idx1_ubyte,
    midrange_standardize,
    load_mnist_data,
    load_mnist_training_data,
    load_mnist_test_data
)

from .datasets import (
    MNISTAutoencoderDataset,
    MNISTDatasetFromFiles,
    create_mnist_dataloaders,
    create_simple_dataloader,
    get_train_dataloader,
    get_test_dataloader
)

from .autoencoder_model import (
    Encoder,
    Decoder,
    AutoencoderMLP,
    create_mnist_autoencoder
)

from .model_utils import (
    xavier_uniform_init,
    xavier_normal_init,
    kaiming_uniform_init,
    kaiming_normal_init,
    get_activation_function,
    get_initialization_function,
    count_parameters,
    get_model_summary,
    print_model_summary,
    validate_model_config,
    move_model_to_device
)

from .training_utils import (
    EarlyStopping,
    TrainingLogger,
    ModelCheckpoint,
    create_lr_scheduler,
    format_time,
    print_training_progress,
    validate_training_config,
    get_device
)

from .trainer import (
    TrainingPipeline,
    create_training_pipeline,
    train_autoencoder
)

from .metrics import (
    ReconstructionMetrics,
    LatentSpaceAnalyzer,
    calculate_reconstruction_errors,
    calculate_per_sample_errors,
    calculate_aggregate_errors,
    compute_latent_statistics,
    prepare_latent_visualization
)

from .model_io import (
    ModelSaver,
    ModelLoader,
    CheckpointManager,
    SASOutputFormatter,
    save_model_state,
    load_model_state,
    convert_checkpoint_to_standalone,
    export_model_summary,
    create_sas_compatible_outputs
)

from .evaluator import (
    ModelEvaluator,
    BatchInferenceProcessor,
    PerformanceBenchmark,
    ModelComparator
)

__version__ = "1.0.0"
__all__ = [
    # Data loading functions
    "read_idx3_ubyte",
    "read_idx1_ubyte", 
    "midrange_standardize",
    "load_mnist_data",
    "load_mnist_training_data",
    "load_mnist_test_data",
    
    # Dataset classes
    "MNISTAutoencoderDataset",
    "MNISTDatasetFromFiles",
    
    # DataLoader utilities
    "create_mnist_dataloaders",
    "create_simple_dataloader", 
    "get_train_dataloader",
    "get_test_dataloader",
    
    # Model classes
    "Encoder",
    "Decoder", 
    "AutoencoderMLP",
    "create_mnist_autoencoder",
    
    # Model utilities
    "xavier_uniform_init",
    "xavier_normal_init",
    "kaiming_uniform_init",
    "kaiming_normal_init",
    "get_activation_function",
    "get_initialization_function",
    "count_parameters",
    "get_model_summary",
    "print_model_summary",
    "validate_model_config",
    "move_model_to_device",
    
    # Training utilities
    "EarlyStopping",
    "TrainingLogger",
    "ModelCheckpoint",
    "create_lr_scheduler",
    "format_time",
    "print_training_progress",
    "validate_training_config",
    "get_device",
    
    # Training pipeline
    "TrainingPipeline",
    "create_training_pipeline",
    "train_autoencoder",
    
    # Evaluation metrics
    "ReconstructionMetrics",
    "LatentSpaceAnalyzer",
    "calculate_reconstruction_errors",
    "calculate_per_sample_errors",
    "calculate_aggregate_errors",
    "compute_latent_statistics",
    "prepare_latent_visualization",
    
    # Model I/O utilities
    "ModelSaver",
    "ModelLoader",
    "CheckpointManager",
    "SASOutputFormatter",
    "save_model_state",
    "load_model_state",
    "convert_checkpoint_to_standalone",
    "export_model_summary",
    "create_sas_compatible_outputs",
    
    # Evaluation and inference
    "ModelEvaluator",
    "BatchInferenceProcessor",
    "PerformanceBenchmark",
    "ModelComparator"
]
