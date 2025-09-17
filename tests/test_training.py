"""
Comprehensive tests for training orchestration functionality.

This module tests the complete training pipeline including:
- Trainer class functionality
- Training configuration management
- Data loader integration
- Checkpointing and resumption
- Training utilities and helpers
"""

import os
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from mnist_autoencoder.training.trainer import (
    Trainer, TrainingConfig, TrainingMetrics, GradientMonitor, create_data_loaders
)
from mnist_autoencoder.training.utils import (
    create_training_config, create_sas_compatible_config, prepare_mnist_data,
    setup_training_environment, evaluate_reconstruction_quality, plot_training_history,
    save_training_config, load_training_config, benchmark_training_performance
)
from mnist_autoencoder.models.autoencoder import MLPAutoencoder


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return MLPAutoencoder(input_size=784, hidden_size=400, seed=42)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    # Create synthetic MNIST-like data
    data = torch.randn(100, 784) * 0.5 + 0.5
    data = data.clamp(0, 1)
    labels = torch.randint(0, 10, (100,))
    return TensorDataset(data, labels)


@pytest.fixture
def sample_data_loader(sample_data):
    """Create sample data loader for testing."""
    return DataLoader(sample_data, batch_size=16, shuffle=True)


class TestTrainingConfig:
    """Test TrainingConfig class and configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TrainingConfig()
        
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.early_stopping is True
        assert config.patience == 10
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.01,
            optimizer="lbfgs",
            early_stopping=False
        )
        
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 0.01
        assert config.optimizer == "lbfgs"
        assert config.early_stopping is False
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid configurations should not raise errors
        TrainingConfig(epochs=1, batch_size=1, learning_rate=0.1)
        
        # Test reasonable ranges
        assert TrainingConfig(epochs=1000).epochs == 1000
        assert TrainingConfig(learning_rate=1e-6).learning_rate == 1e-6


class TestTrainingMetrics:
    """Test TrainingMetrics class and metrics tracking."""
    
    def test_default_metrics(self):
        """Test default metrics initialization."""
        metrics = TrainingMetrics()
        
        assert metrics.epoch == 0
        assert metrics.train_loss == float('inf')
        assert metrics.val_loss == float('inf')
        assert metrics.best_val_loss == float('inf')
        assert len(metrics.train_loss_history) == 0
    
    def test_metrics_update(self):
        """Test metrics updating."""
        metrics = TrainingMetrics()
        
        # Update metrics
        metrics.train_loss = 0.5
        metrics.val_loss = 0.6
        metrics.epoch = 1
        metrics.train_loss_history.append(0.5)
        metrics.val_loss_history.append(0.6)
        
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.epoch == 1
        assert len(metrics.train_loss_history) == 1
        assert len(metrics.val_loss_history) == 1


class TestGradientMonitor:
    """Test GradientMonitor class."""
    
    def test_gradient_monitor_creation(self, sample_model):
        """Test gradient monitor creation."""
        monitor = GradientMonitor(sample_model)
        
        assert monitor.model == sample_model
        assert len(monitor.gradient_norms) == 0
        assert len(monitor.weight_norms) == 0
    
    def test_gradient_checking(self, sample_model, sample_data_loader):
        """Test gradient checking functionality."""
        monitor = GradientMonitor(sample_model)
        
        # Perform a forward/backward pass to create gradients
        data, _ = next(iter(sample_data_loader))
        output = sample_model(data)
        loss = nn.MSELoss()(output, data)
        loss.backward()
        
        # Check gradients
        gradient_info = monitor.check_gradients()
        
        assert "total_grad_norm" in gradient_info
        assert "param_count" in gradient_info
        assert "gradient_explosion" in gradient_info
        assert "gradient_vanishing" in gradient_info
        assert gradient_info["param_count"] > 0
        assert len(monitor.gradient_norms) == 1


class TestTrainer:
    """Test Trainer class functionality."""
    
    def test_trainer_creation(self, sample_model, temp_dir):
        """Test trainer creation with default config."""
        config = TrainingConfig(save_dir=temp_dir / "checkpoints")
        trainer = Trainer(sample_model, config)
        
        assert trainer.model == sample_model
        assert trainer.config == config
        assert trainer.device is not None
        assert trainer.optimizer is None  # Not initialized until prepare_training
    
    def test_trainer_preparation(self, sample_model, temp_dir):
        """Test trainer preparation."""
        config = TrainingConfig(save_dir=temp_dir / "checkpoints")
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        assert trainer.optimizer is not None
        assert trainer.loss_function is not None
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_optimizer_creation(self, sample_model, temp_dir):
        """Test different optimizer creation."""
        # Test Adam optimizer
        config = TrainingConfig(optimizer="adam", save_dir=temp_dir)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        
        # Test L-BFGS optimizer
        config = TrainingConfig(optimizer="lbfgs", save_dir=temp_dir)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        assert isinstance(trainer.optimizer, torch.optim.LBFGS)
        
        # Test SGD optimizer
        config = TrainingConfig(optimizer="sgd", save_dir=temp_dir)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        assert isinstance(trainer.optimizer, torch.optim.SGD)
    
    def test_invalid_optimizer(self, sample_model, temp_dir):
        """Test invalid optimizer handling."""
        config = TrainingConfig(optimizer="invalid", save_dir=temp_dir)
        trainer = Trainer(sample_model, config)
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            trainer.prepare_training()
    
    def test_train_epoch(self, sample_model, sample_data_loader, temp_dir):
        """Test single epoch training."""
        config = TrainingConfig(
            epochs=1,
            batch_size=16,
            log_frequency=1,
            save_dir=temp_dir,
            verbose=False
        )
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        # Train one epoch
        train_loss, train_metrics = trainer.train_epoch(sample_data_loader, 0)
        
        assert isinstance(train_loss, float)
        assert train_loss > 0
        assert "epoch" in train_metrics
        assert "avg_loss" in train_metrics
        assert "epoch_time" in train_metrics
        assert "num_batches" in train_metrics
    
    def test_validate_epoch(self, sample_model, sample_data_loader, temp_dir):
        """Test validation epoch."""
        config = TrainingConfig(save_dir=temp_dir, verbose=False)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        # Validate one epoch
        val_loss, val_metrics = trainer.validate_epoch(sample_data_loader, 0)
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert "epoch" in val_metrics
        assert "avg_loss" in val_metrics
        assert "num_batches" in val_metrics
    
    def test_early_stopping(self, sample_model, temp_dir):
        """Test early stopping mechanism."""
        config = TrainingConfig(
            early_stopping=True,
            patience=2,
            min_delta=0.01,
            save_dir=temp_dir
        )
        trainer = Trainer(sample_model, config)
        
        # Test no early stopping when loss improves
        trainer.metrics.best_val_loss = 1.0
        should_stop = trainer.check_early_stopping(0.5)
        assert not should_stop
        assert trainer.metrics.epochs_without_improvement == 0
        
        # Test early stopping when loss doesn't improve
        should_stop = trainer.check_early_stopping(0.6)
        assert not should_stop
        assert trainer.metrics.epochs_without_improvement == 1
        
        should_stop = trainer.check_early_stopping(0.7)
        assert not should_stop
        assert trainer.metrics.epochs_without_improvement == 2
        
        should_stop = trainer.check_early_stopping(0.8)
        assert should_stop  # Should trigger early stopping
    
    def test_checkpoint_save_load(self, sample_model, temp_dir):
        """Test checkpoint saving and loading."""
        config = TrainingConfig(save_dir=temp_dir / "checkpoints", verbose=False)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(epoch=5, is_best=True)
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        new_model = MLPAutoencoder(input_size=784, hidden_size=400)
        new_trainer = Trainer(new_model, config)
        new_trainer.prepare_training()
        
        checkpoint_info = new_trainer.load_checkpoint(checkpoint_path)
        
        assert "epoch" in checkpoint_info
        assert checkpoint_info["epoch"] == 5
        
        # Check that model states match
        original_state = sample_model.state_dict()
        loaded_state = new_model.state_dict()
        
        for key in original_state:
            assert torch.allclose(original_state[key], loaded_state[key])
    
    def test_full_training_loop(self, sample_model, sample_data_loader, temp_dir):
        """Test complete training loop."""
        config = TrainingConfig(
            epochs=2,
            batch_size=16,
            save_dir=temp_dir / "checkpoints",
            log_dir=temp_dir / "logs",
            verbose=False,
            early_stopping=False
        )
        trainer = Trainer(sample_model, config)
        
        # Split data loader into train/val
        train_loader, val_loader = create_data_loaders(
            sample_data_loader.dataset,
            batch_size=16,
            validation_split=0.3,
            num_workers=0
        )
        
        # Run training
        results = trainer.train(train_loader, val_loader)
        
        assert results["training_completed"]
        assert results["total_epochs"] == 2
        assert results["final_train_loss"] > 0
        assert results["final_val_loss"] > 0
        assert results["total_training_time"] > 0
    
    def test_model_evaluation(self, sample_model, sample_data_loader, temp_dir):
        """Test model evaluation functionality."""
        config = TrainingConfig(save_dir=temp_dir, verbose=False)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        # Evaluate model
        metrics = trainer.evaluate_model(sample_data_loader)
        
        assert "loss" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] > 0
    
    def test_prediction_generation(self, sample_model, sample_data_loader, temp_dir):
        """Test prediction/reconstruction generation."""
        config = TrainingConfig(save_dir=temp_dir, verbose=False)
        trainer = Trainer(sample_model, config)
        
        # Generate predictions
        predictions = trainer.predict(sample_data_loader)
        
        assert "inputs" in predictions
        assert "encodings" in predictions
        assert "reconstructions" in predictions
        assert predictions["inputs"].shape[0] == len(sample_data_loader.dataset)
        assert predictions["encodings"].shape[1] == 400  # hidden size


class TestTrainingUtils:
    """Test training utility functions."""
    
    def test_create_training_config(self):
        """Test training config creation utility."""
        config = create_training_config(
            epochs=50,
            batch_size=32,
            optimizer="lbfgs"
        )
        
        assert isinstance(config, TrainingConfig)
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.optimizer == "lbfgs"
    
    def test_sas_compatible_config(self):
        """Test SAS-compatible configuration."""
        config = create_sas_compatible_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.optimizer == "lbfgs"
        assert config.lr_scheduler == "reduce_on_plateau"
        assert config.patience == 20
        assert config.deterministic is True
    
    def test_setup_training_environment(self, temp_dir):
        """Test training environment setup."""
        directories = setup_training_environment(
            save_dir=temp_dir,
            log_level="INFO",
            seed=42
        )
        
        assert "base" in directories
        assert "checkpoints" in directories
        assert "logs" in directories
        assert "plots" in directories
        assert "results" in directories
        
        # Check directories exist
        for path in directories.values():
            assert path.exists()
    
    def test_config_save_load(self, temp_dir):
        """Test configuration save and load."""
        config = TrainingConfig(epochs=123, batch_size=45)
        save_path = temp_dir / "config.json"
        
        # Save config
        save_training_config(config, save_path)
        assert save_path.exists()
        
        # Load config
        loaded_config = load_training_config(save_path)
        
        assert loaded_config.epochs == 123
        assert loaded_config.batch_size == 45
    
    def test_reconstruction_quality_evaluation(self, sample_model, sample_data_loader):
        """Test reconstruction quality evaluation."""
        metrics = evaluate_reconstruction_quality(
            model=sample_model,
            data_loader=sample_data_loader,
            num_samples=10
        )
        
        assert "mse_mean" in metrics
        assert "mae_mean" in metrics
        assert "similarity_mean" in metrics
        assert "rmse" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 10
    
    def test_benchmark_training_performance(self, sample_model, sample_data_loader):
        """Test training performance benchmarking."""
        metrics = benchmark_training_performance(
            model=sample_model,
            data_loader=sample_data_loader,
            num_epochs=2
        )
        
        assert "total_time" in metrics
        assert "samples_per_second" in metrics
        assert "epochs_per_second" in metrics
        assert "average_loss" in metrics
        assert "total_samples" in metrics
        assert metrics["total_time"] > 0
        assert metrics["samples_per_second"] > 0


class TestDataLoaderCreation:
    """Test data loader creation utilities."""
    
    def test_create_data_loaders_no_validation(self, sample_data):
        """Test data loader creation without validation split."""
        train_loader, val_loader = create_data_loaders(
            dataset=sample_data,
            batch_size=16,
            validation_split=0.0,
            num_workers=0
        )
        
        assert train_loader is not None
        assert val_loader is None
        assert len(train_loader.dataset) == len(sample_data)
    
    def test_create_data_loaders_with_validation(self, sample_data):
        """Test data loader creation with validation split."""
        train_loader, val_loader = create_data_loaders(
            dataset=sample_data,
            batch_size=16,
            validation_split=0.2,
            num_workers=0,
            seed=42
        )
        
        assert train_loader is not None
        assert val_loader is not None
        
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        assert total_samples == len(sample_data)
        
        # Check approximate split ratio
        val_ratio = len(val_loader.dataset) / len(sample_data)
        assert abs(val_ratio - 0.2) < 0.05  # Allow some tolerance


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data_loader(self, sample_model, temp_dir):
        """Test handling of empty data loader."""
        # Create empty dataset
        empty_data = TensorDataset(torch.empty(0, 784), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_data, batch_size=16)
        
        config = TrainingConfig(epochs=1, save_dir=temp_dir, verbose=False)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        # Should handle empty data gracefully
        val_loss, val_metrics = trainer.validate_epoch(empty_loader, 0)
        assert val_loss == float('inf')
    
    def test_single_sample_training(self, sample_model, temp_dir):
        """Test training with single sample."""
        # Create dataset with single sample
        single_data = TensorDataset(torch.randn(1, 784), torch.tensor([0]))
        single_loader = DataLoader(single_data, batch_size=1)
        
        config = TrainingConfig(
            epochs=1,
            batch_size=1,
            save_dir=temp_dir,
            verbose=False,
            early_stopping=False
        )
        trainer = Trainer(sample_model, config)
        
        # Should handle single sample training
        results = trainer.train(single_loader)
        assert results["training_completed"]
    
    def test_training_interruption_handling(self, sample_model, sample_data_loader, temp_dir):
        """Test graceful handling of training interruption."""
        config = TrainingConfig(
            epochs=10,
            save_dir=temp_dir,
            verbose=False
        )
        trainer = Trainer(sample_model, config)
        
        # Simulate interruption
        trainer._should_stop = True
        
        results = trainer.train(sample_data_loader)
        
        # Should handle interruption gracefully
        assert results["total_epochs"] < 10  # Should stop early
    
    def test_checkpoint_loading_errors(self, sample_model, temp_dir):
        """Test checkpoint loading error handling."""
        config = TrainingConfig(save_dir=temp_dir, verbose=False)
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        
        # Test loading non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(temp_dir / "nonexistent.pth")
    
    def test_invalid_loss_function(self, sample_model, temp_dir):
        """Test invalid loss function handling."""
        config = TrainingConfig(loss_function="invalid", save_dir=temp_dir)
        trainer = Trainer(sample_model, config)
        
        with pytest.raises(ValueError, match="Unsupported loss function"):
            trainer.prepare_training()
    
    def test_lr_scheduler_creation(self, sample_model, temp_dir):
        """Test learning rate scheduler creation."""
        # Test step scheduler
        config = TrainingConfig(
            lr_scheduler="step",
            lr_scheduler_params={"step_size": 10, "gamma": 0.5},
            save_dir=temp_dir
        )
        trainer = Trainer(sample_model, config)
        trainer.prepare_training()
        assert trainer.scheduler is not None
        
        # Test invalid scheduler
        config = TrainingConfig(lr_scheduler="invalid", save_dir=temp_dir)
        trainer = Trainer(sample_model, config)
        
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            trainer.prepare_training()


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete training workflows."""
    
    def test_end_to_end_training_workflow(self, temp_dir):
        """Test complete end-to-end training workflow."""
        # Create model and data
        model = MLPAutoencoder(seed=42)
        data = torch.randn(50, 784) * 0.5 + 0.5
        data = data.clamp(0, 1)
        labels = torch.randint(0, 10, (50,))
        dataset = TensorDataset(data, labels)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dataset, batch_size=8, validation_split=0.3, num_workers=0
        )
        
        # Create and configure trainer
        config = TrainingConfig(
            epochs=3,
            batch_size=8,
            save_dir=temp_dir / "checkpoints",
            log_dir=temp_dir / "logs",
            verbose=False,
            early_stopping=True,
            patience=2
        )
        
        trainer = Trainer(model, config)
        
        # Run complete training
        results = trainer.train(train_loader, val_loader)
        
        # Verify results
        assert results["training_completed"]
        assert results["total_epochs"] <= 3
        assert results["final_train_loss"] > 0
        assert results["final_val_loss"] > 0
        
        # Verify checkpoints were created
        checkpoint_dir = temp_dir / "checkpoints"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "latest_checkpoint.pth").exists()
        
        # Verify logs were created
        log_dir = temp_dir / "logs"
        assert log_dir.exists()
        assert (log_dir / "training_history.json").exists()
        
        # Test model evaluation
        eval_metrics = trainer.evaluate_model(val_loader)
        assert eval_metrics["num_samples"] > 0
        
        # Test predictions
        predictions = trainer.predict(val_loader)
        assert predictions["inputs"].shape[0] > 0
        assert predictions["reconstructions"].shape[0] > 0
    
    def test_training_resumption(self, temp_dir):
        """Test training resumption from checkpoint."""
        # Create model and data
        model = MLPAutoencoder(seed=42)
        data = torch.randn(30, 784) * 0.5 + 0.5
        data = data.clamp(0, 1)
        labels = torch.randint(0, 10, (30,))
        dataset = TensorDataset(data, labels)
        
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Initial training
        config = TrainingConfig(
            epochs=2,
            batch_size=8,
            save_dir=temp_dir / "checkpoints",
            verbose=False,
            early_stopping=False
        )
        
        trainer1 = Trainer(model, config)
        results1 = trainer1.train(train_loader)
        
        # Save checkpoint
        checkpoint_path = trainer1.save_checkpoint(epoch=1, is_best=True)
        
        # Resume training with new trainer
        new_model = MLPAutoencoder(seed=123)  # Different seed
        trainer2 = Trainer(new_model, config)
        
        # Train for 1 more epoch
        config.epochs = 3
        results2 = trainer2.train(train_loader, resume_from=checkpoint_path)
        
        # Verify resumption worked
        assert results2["training_completed"]
        # Note: exact epoch count may vary due to resumption logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
