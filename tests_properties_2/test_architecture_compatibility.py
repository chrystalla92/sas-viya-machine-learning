"""
Test architecture compatibility between Python autoencoder and SAS autoencoder.

Tests based on comparison table in SAS_Autoencoder_Properties.md to verify
that the Python implementation follows the SAS implementation architecture.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mnist_autoencoder.models.autoencoder import MLPAutoencoder


class TestArchitectureCompatibility(unittest.TestCase):
    """Test architecture properties to match SAS PROC NNET configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MLPAutoencoder(seed=23451)  # Use SAS seed for reproducibility

    def test_input_size_compatibility(self):
        """Test input size matches SAS: 784 features (28×28 MNIST images flattened)."""
        # SAS: 784 features (var2-var785 in createData.sas)
        # Python: 784 features (default input_size)
        expected_input_size = 784
        actual_input_size = self.model.input_size

        self.assertEqual(
            actual_input_size,
            expected_input_size,
            f"Input size mismatch: expected {expected_input_size} (SAS), got {actual_input_size} (Python)"
        )

    def test_hidden_size_compatibility(self):
        """Test hidden layer size matches SAS: 400 neurons."""
        # SAS: hidden 400 / act=tanh; (nnet.sas:10)
        # Python: hidden_size=400 (default in autoencoder.py:40)
        expected_hidden_size = 400
        actual_hidden_size = self.model.hidden_size

        self.assertEqual(
            actual_hidden_size,
            expected_hidden_size,
            f"Hidden size mismatch: expected {expected_hidden_size} (SAS), got {actual_hidden_size} (Python)"
        )

    def test_output_size_compatibility(self):
        """Test output size matches SAS: 784 features (reconstruction)."""
        # SAS: Output layer: 784 features (reconstruction)
        # Python: decoder Linear(400→784)
        expected_output_size = 784
        actual_output_size = self.model.decoder.out_features

        self.assertEqual(
            actual_output_size,
            expected_output_size,
            f"Output size mismatch: expected {expected_output_size} (SAS), got {actual_output_size} (Python)"
        )

    def test_network_type_compatibility(self):
        """Test that network type is MLP (Multi-Layer Perceptron)."""
        # SAS: architecture MLP; (nnet.sas:8)
        # Python: MLPAutoencoder class (autoencoder.py:15)
        self.assertIsInstance(
            self.model,
            MLPAutoencoder,
            "Model should be MLPAutoencoder to match SAS MLP architecture"
        )

    def test_hidden_activation_compatibility(self):
        """Test hidden layer activation function matches SAS: tanh."""
        # SAS: hidden 400 / act=tanh; (nnet.sas:10)
        # Python: torch.tanh (autoencoder.py:112)

        # Test by checking the activation in encode method
        test_input = torch.randn(1, 784) * 0.1  # Scale to avoid input validation issues

        # Get the raw encoder output (before activation)
        with torch.no_grad():
            raw_encoder_output = self.model.encoder(test_input)
            activated_output = self.model.encode(test_input)
            expected_output = torch.tanh(raw_encoder_output)

        torch.testing.assert_close(
            activated_output,
            expected_output,
            msg="Hidden layer should use tanh activation to match SAS"
        )

    def test_output_activation_compatibility(self):
        """Test output layer activation function matches SAS: sigmoid."""
        # SAS: Output Layer: sigmoid activation (implicit for autoencoder reconstruction)
        # Python: torch.sigmoid (autoencoder.py:131)

        test_input = torch.randn(1, 400)  # Hidden representation

        with torch.no_grad():
            raw_decoder_output = self.model.decoder(test_input)
            activated_output = self.model.decode(test_input)
            expected_output = torch.sigmoid(raw_decoder_output)

        torch.testing.assert_close(
            activated_output,
            expected_output,
            msg="Output layer should use sigmoid activation to match SAS"
        )

    def test_network_structure_compatibility(self):
        """Test complete network structure: Input (784) → Hidden (400, tanh) → Output (784, sigmoid)."""
        # SAS: Input (784) → Hidden (400, tanh) → Output (784, sigmoid)
        # Python: Input (784) → Encoder: Linear(784→400) + tanh → Decoder: Linear(400→784) + sigmoid

        test_input = torch.randn(5, 784) * 0.1  # Batch of 5 samples, scaled

        with torch.no_grad():
            # Test forward pass
            output = self.model(test_input)

            # Check output shape
            self.assertEqual(
                output.shape,
                test_input.shape,
                "Output shape should match input shape (784 features)"
            )

            # Check output range (sigmoid should produce values in [0,1])
            self.assertTrue(
                torch.all(output >= 0.0) and torch.all(output <= 1.0),
                "Output values should be in [0,1] range due to sigmoid activation"
            )

            # Test intermediate representation
            encoded = self.model.encode(test_input)
            self.assertEqual(
                encoded.shape,
                (5, 400),
                "Encoded representation should have 400 features"
            )

            # Check tanh range (should produce values in [-1,1])
            self.assertTrue(
                torch.all(encoded >= -1.0) and torch.all(encoded <= 1.0),
                "Encoded values should be in [-1,1] range due to tanh activation"
            )

    def test_parameter_count_compatibility(self):
        """Test total parameter count matches expected for SAS architecture."""
        # Expected parameters:
        # Encoder: 784×400 + 400 (bias) = 314,000
        # Decoder: 400×784 + 784 (bias) = 314,784
        # Total: 628,784 parameters

        expected_encoder_params = 784 * 400 + 400  # weights + bias
        expected_decoder_params = 400 * 784 + 784  # weights + bias
        expected_total_params = expected_encoder_params + expected_decoder_params

        actual_total_params = self.model.count_parameters()

        self.assertEqual(
            actual_total_params,
            expected_total_params,
            f"Total parameter count mismatch: expected {expected_total_params}, got {actual_total_params}"
        )

    def test_weight_initialization_compatibility(self):
        """Test weight initialization method matches SAS approach."""
        # SAS: Xavier/Glorot uniform initialization (default for neural networks)
        # Python: nn.init.xavier_uniform_ (autoencoder.py:91, 95)

        # Check that weights are not all zeros (initialized)
        encoder_weights = self.model.encoder.weight
        decoder_weights = self.model.decoder.weight

        self.assertFalse(
            torch.allclose(encoder_weights, torch.zeros_like(encoder_weights)),
            "Encoder weights should be initialized (not all zeros)"
        )

        self.assertFalse(
            torch.allclose(decoder_weights, torch.zeros_like(decoder_weights)),
            "Decoder weights should be initialized (not all zeros)"
        )

        # Check bias initialization (should be zeros)
        encoder_bias = self.model.encoder.bias
        decoder_bias = self.model.decoder.bias

        self.assertTrue(
            torch.allclose(encoder_bias, torch.zeros_like(encoder_bias)),
            "Encoder bias should be initialized to zeros to match SAS"
        )

        self.assertTrue(
            torch.allclose(decoder_bias, torch.zeros_like(decoder_bias)),
            "Decoder bias should be initialized to zeros to match SAS"
        )

    def test_tied_weights_compatibility(self):
        """Test tied weights configuration matches SAS."""
        # SAS: Tied Weights: Not explicitly specified in configuration
        # Python: Tied Weights: Not implemented (separate encoder and decoder weights)

        encoder_weights = self.model.encoder.weight
        decoder_weights = self.model.decoder.weight

        # Check that weights are separate (not tied)
        self.assertFalse(
            torch.allclose(encoder_weights.T, decoder_weights),
            "Weights should not be tied to match SAS configuration"
        )

        # Check that they are different parameter objects
        self.assertIsNot(
            encoder_weights,
            decoder_weights,
            "Encoder and decoder should have separate weight parameters"
        )

    def test_layer_types_compatibility(self):
        """Test that layer types match SAS specification."""
        # SAS: Linear layers (implicit in MLP architecture)
        # Python: nn.Linear layers

        self.assertIsInstance(
            self.model.encoder,
            nn.Linear,
            "Encoder should be a Linear layer to match SAS MLP"
        )

        self.assertIsInstance(
            self.model.decoder,
            nn.Linear,
            "Decoder should be a Linear layer to match SAS MLP"
        )

    def test_forward_pass_compatibility(self):
        """Test that forward pass produces expected behavior."""
        # Test deterministic behavior with same seed
        torch.manual_seed(23451)  # SAS seed
        model1 = MLPAutoencoder(seed=23451)

        torch.manual_seed(23451)  # Reset seed
        model2 = MLPAutoencoder(seed=23451)

        test_input = torch.randn(3, 784) * 0.1  # Scale input

        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)

        torch.testing.assert_close(
            output1,
            output2,
            msg="Models with same seed should produce identical outputs"
        )

    def test_gradient_flow_compatibility(self):
        """Test that gradients flow properly through the network."""
        test_input = torch.randn(2, 784, requires_grad=True) * 0.1  # Scale input

        # Forward pass
        output = self.model(test_input)
        loss = torch.mean((output - test_input) ** 2)  # MSE loss

        # Backward pass
        loss.backward()

        # Check that all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"Parameter {name} should have gradients after backward pass"
            )
            self.assertFalse(
                torch.allclose(param.grad, torch.zeros_like(param.grad)),
                f"Parameter {name} should have non-zero gradients"
            )


if __name__ == "__main__":
    unittest.main()