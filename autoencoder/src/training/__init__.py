"""
Training Module for PyTorch Autoencoder

This module provides comprehensive training functionality for MLP autoencoders
with L-BFGS optimization, matching SAS Viya training specifications.

Key Features:
- L-BFGS optimizer with closure-based optimization
- MSE reconstruction loss
- Convergence checking with configurable tolerance (default: 1E-10)
- Comprehensive metrics tracking and visualization
- Model checkpointing and state management
- Reproducible results with seed=23451
- Max 500 iterations with early stopping
"""

# Import core training classes
from .trainer import AutoencoderTrainer, create_trainer
from .metrics import (
    TrainingMetrics, 
    TrainingSnapshot,
    calculate_reconstruction_metrics,
    compare_training_runs
)

# Import utility functions
try:
    # Try relative imports first (when installed as package)
    from ..utils.checkpoints import (
        CheckpointManager,
        create_checkpoint_manager,
        save_training_state,
        load_training_state
    )
    
    from ..utils.visualization import (
        plot_training_progress,
        plot_reconstruction_comparison,
        plot_latent_space,
        plot_model_architecture,
        plot_metrics_dashboard,
        create_training_report
    )
except ImportError:
    # Fallback to absolute imports (when imported directly)
    try:
        from utils.checkpoints import (
            CheckpointManager,
            create_checkpoint_manager,
            save_training_state,
            load_training_state
        )
        
        from utils.visualization import (
            plot_training_progress,
            plot_reconstruction_comparison,
            plot_latent_space,
            plot_model_architecture,
            plot_metrics_dashboard,
            create_training_report
        )
    except ImportError as e:
        # If both fail, provide informative error message
        raise ImportError(
            "Could not import utils modules. Make sure the utils package is available "
            "either as a sibling module (utils) or as a parent relative import (..utils). "
            f"Original error: {e}"
        )

# Training configuration constants matching SAS specifications
DEFAULT_CONFIG = {
    'max_iterations': 500,
    'convergence_tolerance': 1e-10,  # fConv parameter
    'seed': 23451,
    'optimizer': 'L-BFGS',
    'loss_function': 'MSE',
    'batch_processing': True,
    'early_stopping': True
}

__all__ = [
    # Core training classes
    'AutoencoderTrainer',
    'create_trainer',
    
    # Metrics and monitoring
    'TrainingMetrics',
    'TrainingSnapshot',
    'calculate_reconstruction_metrics',
    'compare_training_runs',
    
    # Checkpointing
    'CheckpointManager',
    'create_checkpoint_manager',
    'save_training_state',
    'load_training_state',
    
    # Visualization
    'plot_training_progress',
    'plot_reconstruction_comparison',
    'plot_latent_space',
    'plot_model_architecture',
    'plot_metrics_dashboard',
    'create_training_report',
    
    # Configuration
    'DEFAULT_CONFIG'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "PyTorch Autoencoder Training System"
__description__ = "L-BFGS training implementation matching SAS Viya specifications"
