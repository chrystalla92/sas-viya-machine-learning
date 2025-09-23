"""
Utility functions and helpers for autoencoder implementations.

This module provides common utilities for visualization, metrics,
device detection, checkpointing, and other helper functions.
"""

# Import utility functions
try:
    from .device_utils import get_device, setup_cuda, print_device_info
    from .visualization import (
        plot_training_progress,
        plot_reconstruction_comparison, 
        plot_latent_space,
        plot_model_architecture,
        plot_metrics_dashboard,
        create_training_report
    )
    from .checkpoints import (
        CheckpointManager,
        create_checkpoint_manager,
        save_training_state,
        load_training_state
    )
    
    __all__ = [
        # Device utilities
        "get_device",
        "setup_cuda",
        "print_device_info",
        
        # Visualization utilities
        "plot_training_progress",
        "plot_reconstruction_comparison",
        "plot_latent_space", 
        "plot_model_architecture",
        "plot_metrics_dashboard",
        "create_training_report",
        
        # Checkpoint utilities
        "CheckpointManager",
        "create_checkpoint_manager",
        "save_training_state",
        "load_training_state",
    ]
except ImportError as e:
    # Graceful handling during development
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some utility imports failed: {e}")
    __all__ = []

# Module metadata
__version__ = "1.0.0"
