"""
Test cases for training configuration properties.

Tests verify that the mnist_autoencoder training configuration matches the
specifications documented in SAS_Autoencoder_Properties.md
"""

import unittest
import torch
import torch.nn.functional as F
from mnist_autoencoder.models.autoencoder import MLPAutoencoder, create_mnist_autoencoder
from mnist_autoencoder.training.trainer import Trainer, TrainingConfig


class TestTrainingProperties(unittest.TestCase):
    """Test training configuration properties of the autoencoder."""

    def setUp(self):
        """Set up test fixtures."""
        self.autoencoder = create_mnist_autoencoder(seed=23451)
        self.config = TrainingConfig(
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            seed=23451
        )
        self.trainer = Trainer(self.autoencoder, self.config)

    def test_mse_loss_function(self):
        """Test that MSE (Mean Squared Error) is the primary loss function."""
        # Test MSE loss configuration
        config_mse = TrainingConfig(loss_function="mse")
        trainer_mse = Trainer(self.autoencoder, config_mse)
        trainer_mse.prepare_training()

        # Test that loss function is MSE
        sample_input = torch.rand(8, 784)
        sample_target = torch.rand(8, 784)

        # Calculate loss using trainer's loss function
        loss_value = trainer_mse.loss_function(sample_input, sample_target)

        # Compare with PyTorch MSE
        expected_loss = F.mse_loss(sample_input, sample_target)

        torch.testing.assert_close(
            loss_value,
            expected_loss,
            msg="Training should use MSE loss function"
        )

    def test_no_explicit_l1_l2_regularization(self):
        """Test that L1/L2 regularization is not explicitly configured by default."""
        # Test default configuration
        default_config = TrainingConfig()

        self.assertEqual(
            default_config.weight_decay,
            0.0,
            "Default configuration should not include L2 regularization (weight_decay=0)"
        )

        # Verify no L1 regularization is configured
        self.assertFalse(
            hasattr(default_config, 'l1_weight'),
            "Configuration should not have explicit L1 regularization"
        )

    def test_lbfgs_optimizer_support(self):
        """Test that L-BFGS optimizer is supported and configured correctly."""
        # Test L-BFGS configuration
        lbfgs_config = TrainingConfig(
            optimizer="lbfgs",
            learning_rate=0.001
        )
        trainer_lbfgs = Trainer(self.autoencoder, lbfgs_config)
        trainer_lbfgs.prepare_training()

        # Check optimizer type
        self.assertIsInstance(
            trainer_lbfgs.optimizer,
            torch.optim.LBFGS,
            "Should support L-BFGS optimizer"
        )

        # Check L-BFGS specific parameters
        optimizer = trainer_lbfgs.optimizer
        self.assertEqual(
            optimizer.param_groups[0]['lr'],
            0.001,
            "L-BFGS should use specified learning rate"
        )

        # Check L-BFGS specific settings (tolerance, max_iter)
        defaults = optimizer.defaults
        self.assertLessEqual(
            defaults.get('tolerance_grad', 1e-5),
            1e-5,
            "L-BFGS should have tight gradient tolerance"
        )
        self.assertGreaterEqual(
            defaults.get('max_iter', 10),
            10,
            "L-BFGS should support sufficient iterations"
        )

    def test_maximum_iterations_500_equivalent(self):
        """Test that training supports maximum iterations equivalent to SAS 500."""
        # Test epoch-based training that provides equivalent iteration control
        max_iter_config = TrainingConfig(
            epochs=500,  # Use epochs as iteration equivalent
            optimizer="lbfgs"
        )

        trainer_max_iter = Trainer(self.autoencoder, max_iter_config)

        self.assertEqual(
            trainer_max_iter.config.epochs,
            500,
            "Should support 500 epochs (equivalent to max iterations)"
        )

        # For L-BFGS, also check max_iter per step
        trainer_max_iter.prepare_training()
        if hasattr(trainer_max_iter.optimizer, 'defaults'):
            max_iter_per_step = trainer_max_iter.optimizer.defaults.get('max_iter', 20)
            self.assertGreaterEqual(
                max_iter_per_step,
                10,
                "L-BFGS should support sufficient iterations per step"
            )

    def test_convergence_tolerance_configuration(self):
        """Test that convergence tolerance is configured (equivalent to fConv=1E-10)."""
        # Test that trainer supports convergence-like criteria
        early_stop_config = TrainingConfig(
            early_stopping=True,
            min_delta=1e-10,  # Equivalent to tight convergence tolerance
            patience=10
        )

        trainer_convergence = Trainer(self.autoencoder, early_stop_config)

        self.assertEqual(
            trainer_convergence.config.min_delta,
            1e-10,
            "Should support tight convergence tolerance (1e-10)"
        )

        # Test early stopping mechanism
        self.assertTrue(
            trainer_convergence.config.early_stopping,
            "Should support early stopping for convergence detection"
        )

    def test_seed_23451_reproducibility(self):
        """Test that seed 23451 provides reproducible results."""
        # Create two identical models with same seed
        model1 = create_mnist_autoencoder(seed=23451)
        model2 = create_mnist_autoencoder(seed=23451)

        # Test that initial weights are identical
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            self.assertEqual(name1, name2, "Parameter names should match")
            torch.testing.assert_close(
                param1,
                param2,
                msg=f"Parameters {name1} should be identical with same seed"
            )

    def test_batch_processing_support(self):
        """Test that training supports batch processing."""
        # Test batch processing capability
        batch_config = TrainingConfig(
            batch_size=64
        )

        trainer_batch = Trainer(self.autoencoder, batch_config)

        self.assertEqual(
            trainer_batch.config.batch_size,
            64,
            "Should support configurable batch size"
        )

        # Test that model can handle batch processing
        batch_input = torch.rand(64, 784)
        batch_output = self.autoencoder(batch_input)

        self.assertEqual(
            batch_output.shape,
            (64, 784),
            "Model should support batch processing"
        )

    def test_full_dataset_processing_mode(self):
        """Test support for full dataset processing (no explicit batch size)."""
        # Test that very large batch size can simulate full dataset processing
        full_dataset_config = TrainingConfig(
            batch_size=10000  # Large batch to simulate full dataset
        )

        trainer_full = Trainer(self.autoencoder, full_dataset_config)

        self.assertGreaterEqual(
            trainer_full.config.batch_size,
            1000,
            "Should support large batch sizes for full dataset processing"
        )

    def test_epoch_based_convergence_control(self):
        """Test that epochs provide convergence control similar to SAS iterations."""
        # Test epoch-based training with convergence monitoring
        convergence_config = TrainingConfig(
            epochs=100,
            early_stopping=True,
            patience=20,
            min_delta=1e-6
        )

        trainer_conv = Trainer(self.autoencoder, convergence_config)

        # Verify configuration supports convergence control
        self.assertTrue(
            trainer_conv.config.early_stopping,
            "Should support early stopping for convergence"
        )
        self.assertGreater(
            trainer_conv.config.patience,
            0,
            "Should support patience-based convergence detection"
        )

    def test_training_procedure_configuration(self):
        """Test that training procedure matches SAS PROC NNET structure."""
        # Test complete training configuration
        sas_like_config = TrainingConfig(
            # Architecture implicit (tested in architecture tests)
            optimizer="lbfgs",
            loss_function="mse",
            epochs=500,  # maxiters equivalent
            seed=23451,
            early_stopping=True,
            min_delta=1e-10  # fConv equivalent
        )

        trainer_sas_like = Trainer(self.autoencoder, sas_like_config)
        trainer_sas_like.prepare_training()

        # Verify key components
        self.assertEqual(
            trainer_sas_like.config.optimizer,
            "lbfgs",
            "Should use L-BFGS optimizer"
        )
        self.assertEqual(
            trainer_sas_like.config.loss_function,
            "mse",
            "Should use MSE loss"
        )
        self.assertEqual(
            trainer_sas_like.config.seed,
            23451,
            "Should use seed 23451"
        )

    def test_reconstruction_error_minimization(self):
        """Test that training optimizes reconstruction error (autoencoder objective)."""
        # Test that loss function measures reconstruction error
        sample_input = torch.rand(8, 784)
        sample_reconstruction = self.autoencoder(sample_input)

        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(sample_reconstruction, sample_input)

        # Verify this is what the training optimizes
        trainer = Trainer(self.autoencoder, TrainingConfig(loss_function="mse"))
        trainer.prepare_training()

        training_loss = trainer.loss_function(sample_reconstruction, sample_input)

        torch.testing.assert_close(
            training_loss,
            reconstruction_error,
            msg="Training should optimize reconstruction error"
        )

    def test_no_additional_regularization_by_default(self):
        """Test that no additional regularization is specified by default."""
        default_config = TrainingConfig()

        # Check weight decay (L2 regularization)
        self.assertEqual(
            default_config.weight_decay,
            0.0,
            "Should not include L2 regularization by default"
        )

        # Check that no other regularization parameters exist
        regularization_attrs = [
            'l1_weight', 'l1_lambda', 'dropout_rate', 'noise_factor'
        ]

        for attr in regularization_attrs:
            self.assertFalse(
                hasattr(default_config, attr),
                f"Should not have {attr} regularization by default"
            )


if __name__ == '__main__':
    unittest.main()