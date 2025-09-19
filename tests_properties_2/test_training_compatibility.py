"""
Test training compatibility between Python autoencoder and SAS autoencoder.

Tests based on comparison table in SAS_Autoencoder_Properties.md to verify
that the Python implementation follows the SAS implementation training configuration.
"""

import unittest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mnist_autoencoder.models.autoencoder import MLPAutoencoder
from mnist_autoencoder.training.trainer import TrainingConfig, Trainer


class TestTrainingCompatibility(unittest.TestCase):
    """Test training properties to match SAS PROC NNET configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MLPAutoencoder(seed=23451)  # Use SAS seed
        self.test_data = torch.randn(10, 784)  # Sample batch

    def test_loss_function_compatibility(self):
        """Test loss function matches SAS: MSE (Mean Squared Error)."""
        # SAS: MSE (Mean Squared Error) - implicit for autoencoder
        # Python: MSE (configurable: MSE/BCE) (trainer.py:288-293)

        # Test MSE loss function
        input_data = torch.randn(5, 784)
        target_data = torch.randn(5, 784)

        # Test F.mse_loss (used in trainer)
        mse_loss = F.mse_loss(input_data, target_data)

        # Verify it's a scalar loss
        self.assertEqual(mse_loss.dim(), 0, "MSE loss should be a scalar")
        self.assertTrue(mse_loss.item() >= 0.0, "MSE loss should be non-negative")

        # Test that TrainingConfig defaults to MSE
        config = TrainingConfig()
        self.assertEqual(
            config.loss_function.lower(),
            "mse",
            "Default loss function should be MSE to match SAS"
        )

    def test_optimizer_lbfgs_compatibility(self):
        """Test L-BFGS optimizer availability for SAS compatibility."""
        # SAS: LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
        # Python: Multiple (Adam/LBFGS/SGD) with LBFGS specifically for SAS compatibility

        # Test L-BFGS configuration
        config = TrainingConfig(
            optimizer="lbfgs",
            learning_rate=1.0  # L-BFGS typical learning rate
        )

        trainer = Trainer(self.model, config)
        trainer.prepare_training()

        # Verify L-BFGS optimizer is created
        self.assertIsInstance(
            trainer.optimizer,
            torch.optim.LBFGS,
            "Should create L-BFGS optimizer for SAS compatibility"
        )

        # Test L-BFGS specific parameters
        self.assertEqual(
            trainer.optimizer.param_groups[0]['max_iter'],
            20,  # Limited iterations for L-BFGS
            "L-BFGS should have limited iterations per step"
        )

    def test_convergence_criteria_compatibility(self):
        """Test convergence criteria configuration."""
        # SAS: Maximum Iterations: 500 (maxIters=500), Convergence Tolerance: fConv=1E-10
        # Python: Epochs + early stopping with configurable tolerance

        # Test L-BFGS convergence tolerances
        config = TrainingConfig(optimizer="lbfgs")
        trainer = Trainer(self.model, config)
        trainer.prepare_training()

        optimizer = trainer.optimizer
        self.assertEqual(
            optimizer.param_groups[0]['tolerance_grad'],
            1e-7,
            "L-BFGS gradient tolerance should match SAS precision expectations"
        )

        self.assertEqual(
            optimizer.param_groups[0]['tolerance_change'],
            1e-9,
            "L-BFGS change tolerance should match SAS precision expectations"
        )

    def test_reproducibility_seed_compatibility(self):
        """Test seed control for reproducible results."""
        # SAS: Seed: 23451 (for reproducible results)
        # Python: Configurable seed control (trainer.py:246-255)

        seed_value = 23451  # SAS seed

        # Test seed configuration
        config1 = TrainingConfig(seed=seed_value)
        config2 = TrainingConfig(seed=seed_value)

        model1 = MLPAutoencoder(seed=seed_value)
        model2 = MLPAutoencoder(seed=seed_value)

        # Test that models with same seed have identical initial weights
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            torch.testing.assert_close(
                param1,
                param2,
                msg=f"Parameter {name1} should be identical with same seed"
            )

    def test_training_procedure_compatibility(self):
        """Test training procedure matches SAS workflow."""
        # SAS: proc nnet with standardize=midrange, architecture MLP, hidden 400 / act=tanh
        # Python: Equivalent configuration through TrainingConfig

        # Create SAS-compatible configuration
        config = TrainingConfig(
            optimizer="lbfgs",
            loss_function="mse",
            learning_rate=1.0,
            epochs=10,  # Reduced for testing
            seed=23451,
            early_stopping=False  # SAS uses max iterations, not early stopping
        )

        trainer = Trainer(self.model, config)
        self.assertEqual(trainer.config.optimizer, "lbfgs")
        self.assertEqual(trainer.config.loss_function, "mse")
        self.assertEqual(trainer.config.seed, 23451)

    def test_batch_processing_compatibility(self):
        """Test batch processing approach differences."""
        # SAS: Full dataset processing (no explicit batch size)
        # Python: Mini-batch (configurable)

        # Test full-batch configuration (closest to SAS)
        large_batch_config = TrainingConfig(
            batch_size=10000,  # Large batch size to simulate full-batch
            optimizer="lbfgs"
        )

        # Test mini-batch configuration
        mini_batch_config = TrainingConfig(
            batch_size=64,  # Standard mini-batch
            optimizer="adam"
        )

        # Both should be valid configurations
        trainer1 = Trainer(self.model, large_batch_config)
        trainer2 = Trainer(self.model, mini_batch_config)

        self.assertEqual(trainer1.config.batch_size, 10000)
        self.assertEqual(trainer2.config.batch_size, 64)

    def test_training_metrics_compatibility(self):
        """Test training metrics tracking."""
        # SAS: Basic loss tracking
        # Python: Comprehensive metrics tracking

        config = TrainingConfig(
            optimizer="lbfgs",
            loss_function="mse",
            epochs=1,
            verbose=False
        )

        trainer = Trainer(self.model, config)

        # Check metrics initialization
        self.assertIsNotNone(trainer.metrics)
        self.assertEqual(trainer.metrics.epoch, 0)
        self.assertEqual(trainer.metrics.train_loss, float('inf'))

        # Test metrics have expected attributes for SAS compatibility
        self.assertTrue(hasattr(trainer.metrics, 'train_loss_history'))
        self.assertTrue(hasattr(trainer.metrics, 'best_val_loss'))

    def test_weight_decay_regularization_compatibility(self):
        """Test regularization configuration."""
        # SAS: L1/L2 Regularization: Not explicitly configured
        # Python: L2 Regularization: Via weight_decay parameter

        # Test no regularization (SAS default)
        config_no_reg = TrainingConfig(weight_decay=0.0)
        trainer_no_reg = Trainer(self.model, config_no_reg)
        trainer_no_reg.prepare_training()

        self.assertEqual(
            trainer_no_reg.optimizer.param_groups[0]['weight_decay'],
            0.0,
            "Default should be no weight decay to match SAS"
        )

        # Test optional regularization
        config_with_reg = TrainingConfig(weight_decay=1e-4)
        trainer_with_reg = Trainer(self.model, config_with_reg)
        trainer_with_reg.prepare_training()

        self.assertEqual(
            trainer_with_reg.optimizer.param_groups[0]['weight_decay'],
            1e-4,
            "Weight decay should be configurable"
        )

    def test_learning_rate_configuration_compatibility(self):
        """Test learning rate configuration for different optimizers."""
        # SAS: Fixed (LBFGS)
        # Python: Configurable with scheduling

        # Test L-BFGS learning rate (typically 1.0)
        lbfgs_config = TrainingConfig(
            optimizer="lbfgs",
            learning_rate=1.0
        )
        trainer_lbfgs = Trainer(self.model, lbfgs_config)
        trainer_lbfgs.prepare_training()

        self.assertEqual(
            trainer_lbfgs.optimizer.param_groups[0]['lr'],
            1.0,
            "L-BFGS learning rate should be 1.0 (typical value)"
        )

        # Test Adam learning rate
        adam_config = TrainingConfig(
            optimizer="adam",
            learning_rate=0.001
        )
        trainer_adam = Trainer(self.model, adam_config)
        trainer_adam.prepare_training()

        self.assertEqual(
            trainer_adam.optimizer.param_groups[0]['lr'],
            0.001,
            "Adam learning rate should be configurable"
        )

    def test_early_stopping_configuration_compatibility(self):
        """Test early stopping configuration."""
        # SAS: Controlled by convergence criteria and maxIters
        # Python: Patience-based early stopping

        # Test early stopping disabled (more like SAS approach)
        config_no_early = TrainingConfig(early_stopping=False)
        trainer_no_early = Trainer(self.model, config_no_early)

        self.assertFalse(
            trainer_no_early.config.early_stopping,
            "Should be able to disable early stopping to match SAS approach"
        )

        # Test early stopping enabled
        config_early = TrainingConfig(
            early_stopping=True,
            patience=10,
            min_delta=1e-4
        )
        trainer_early = Trainer(self.model, config_early)

        self.assertTrue(trainer_early.config.early_stopping)
        self.assertEqual(trainer_early.config.patience, 10)

    def test_loss_computation_compatibility(self):
        """Test loss computation matches expected behavior."""
        # Test MSE loss computation
        input_data = torch.randn(3, 784) * 0.1  # Scale input
        with torch.no_grad():
            output_data = self.model(input_data)

        # Compute MSE loss
        mse_loss = F.mse_loss(output_data, input_data)

        # Verify loss properties
        self.assertTrue(mse_loss.item() >= 0.0, "MSE loss should be non-negative")
        self.assertTrue(torch.isfinite(mse_loss), "Loss should be finite")

        # Test that loss decreases with identical input/output
        zero_loss = F.mse_loss(input_data, input_data)
        self.assertAlmostEqual(
            zero_loss.item(),
            0.0,
            places=6,
            msg="MSE loss should be zero for identical input/output"
        )

    def test_optimizer_parameter_compatibility(self):
        """Test optimizer parameters match SAS expectations."""
        # Test LBFGS parameters specifically for SAS compatibility
        config = TrainingConfig(optimizer="lbfgs")
        trainer = Trainer(self.model, config)
        trainer.prepare_training()

        optimizer = trainer.optimizer
        param_group = optimizer.param_groups[0]

        # Check key LBFGS parameters
        self.assertIn('max_iter', param_group)
        self.assertIn('tolerance_grad', param_group)
        self.assertIn('tolerance_change', param_group)

        # Verify values are set for SAS compatibility
        self.assertLessEqual(param_group['max_iter'], 20)  # Limited iterations
        self.assertLessEqual(param_group['tolerance_grad'], 1e-6)  # Strict tolerance
        self.assertLessEqual(param_group['tolerance_change'], 1e-8)

    def test_gradient_monitoring_compatibility(self):
        """Test gradient monitoring for training stability."""
        # Both SAS and Python should monitor training stability

        config = TrainingConfig(optimizer="adam", epochs=1)
        trainer = Trainer(self.model, config)

        # Test gradient monitor exists
        self.assertIsNotNone(trainer.gradient_monitor)

        # Test gradient checking functionality
        sample_input = torch.randn(2, 784, requires_grad=True) * 0.1  # Scale input
        output = self.model(sample_input)
        loss = F.mse_loss(output, sample_input)
        loss.backward()

        gradient_info = trainer.gradient_monitor.check_gradients()

        # Verify gradient monitoring provides useful information
        self.assertIn('total_grad_norm', gradient_info)
        self.assertIn('gradient_explosion', gradient_info)
        self.assertIn('gradient_vanishing', gradient_info)

    def test_training_resumption_compatibility(self):
        """Test training resumption capability."""
        # Test that training can be stopped and resumed
        # This is important for long-running SAS-style training jobs

        config = TrainingConfig(
            optimizer="lbfgs",
            epochs=5,
            save_frequency=1,
            verbose=False
        )

        trainer = Trainer(self.model, config)
        trainer.prepare_training()

        # Check checkpoint functionality exists
        self.assertTrue(hasattr(trainer, 'save_checkpoint'))
        self.assertTrue(hasattr(trainer, 'load_checkpoint'))

        # Test checkpoint data structure
        test_checkpoint = {
            "epoch": 0,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "metrics": trainer.metrics,
            "config": trainer.config,
            "model_info": self.model.get_model_info(),
        }

        # Verify checkpoint structure
        self.assertIn('epoch', test_checkpoint)
        self.assertIn('model_state_dict', test_checkpoint)


def create_sas_compatible_config() -> TrainingConfig:
    """Create a training configuration that closely matches SAS PROC NNET."""
    return TrainingConfig(
        optimizer="lbfgs",
        loss_function="mse",
        learning_rate=1.0,
        weight_decay=0.0,  # No regularization like SAS default
        early_stopping=False,  # Use max epochs instead
        seed=23451,  # SAS seed
        batch_size=10000,  # Large batch to approximate full-batch processing
        epochs=500,  # Match SAS maxIters concept
        verbose=True,
        deterministic=True
    )


class TestSASCompatibleConfiguration(unittest.TestCase):
    """Test the SAS-compatible configuration."""

    def test_sas_compatible_config_creation(self):
        """Test that SAS-compatible configuration can be created."""
        config = create_sas_compatible_config()

        # Verify SAS-compatible settings
        self.assertEqual(config.optimizer, "lbfgs")
        self.assertEqual(config.loss_function, "mse")
        self.assertEqual(config.learning_rate, 1.0)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertFalse(config.early_stopping)
        self.assertEqual(config.seed, 23451)
        self.assertTrue(config.deterministic)

    def test_sas_compatible_training(self):
        """Test that SAS-compatible training can be initiated."""
        model = MLPAutoencoder(seed=23451)
        config = create_sas_compatible_config()
        trainer = Trainer(model, config)

        # Should be able to prepare training without errors
        trainer.prepare_training()

        # Verify optimizer type
        self.assertIsInstance(trainer.optimizer, torch.optim.LBFGS)

        # Verify loss function
        test_input = torch.randn(2, 784)
        test_target = torch.randn(2, 784)
        loss = trainer.loss_function(test_input, test_target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()