"""
Training Package for MNIST Autoencoder

This package provides comprehensive training functionality including:
- Main training orchestration (trainer.py)
- Model evaluation and metrics (evaluator.py)
"""

from .trainer import AutoencoderTrainer, TrainingConfig, train_mnist_autoencoder, demo_training
from .evaluator import AutoencoderEvaluator, EvaluationMetrics, evaluate_model_checkpoint

__all__ = [
    'AutoencoderTrainer',
    'TrainingConfig', 
    'train_mnist_autoencoder',
    'demo_training',
    'AutoencoderEvaluator',
    'EvaluationMetrics',
    'evaluate_model_checkpoint'
]
