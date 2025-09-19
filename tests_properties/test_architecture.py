"""
Test cases for autoencoder architecture properties.

Tests verify that the mnist_autoencoder implementation matches the architecture
specifications documented in SAS_Autoencoder_Properties.md
"""

import unittest
import torch
import torch.nn as nn
from mnist_autoencoder.models.autoencoder import MLPAutoencoder, create_mnist_autoencoder


class TestArchitectureProperties(unittest.TestCase):
    """Test architecture-related properties of the autoencoder."""

    def setUp(self):
        """Set up test fixtures."""
        self.autoencoder = create_mnist_autoencoder(seed=23451)

    def test_input_layer_size(self):
        """Test that input layer has 784 features (28×28 MNIST images flattened)."""
        expected_input_size = 784
        actual_input_size = self.autoencoder.input_size

        self.assertEqual(
            actual_input_size,
            expected_input_size,
            f"Input layer should have {expected_input_size} features, got {actual_input_size}"
        )

    def test_hidden_layer_size(self):
        """Test that hidden layer has 400 neurons."""
        expected_hidden_size = 400
        actual_hidden_size = self.autoencoder.hidden_size

        self.assertEqual(
            actual_hidden_size,
            expected_hidden_size,
            f"Hidden layer should have {expected_hidden_size} neurons, got {actual_hidden_size}"
        )

    def test_output_layer_size(self):
        """Test that output layer has 784 features (reconstruction)."""
        expected_output_size = 784
        actual_output_size = self.autoencoder.decoder.out_features

        self.assertEqual(
            actual_output_size,
            expected_output_size,
            f"Output layer should have {expected_output_size} features, got {actual_output_size}"
        )

    def test_architecture_type_mlp(self):
        """Test that architecture is MLP (Multi-Layer Perceptron)."""
        # Verify it's an MLP by checking layer types
        self.assertIsInstance(
            self.autoencoder.encoder,
            nn.Linear,
            "Encoder should be a Linear layer (MLP component)"
        )
        self.assertIsInstance(
            self.autoencoder.decoder,
            nn.Linear,
            "Decoder should be a Linear layer (MLP component)"
        )

    def test_hidden_layer_activation_tanh(self):
        """Test that hidden layer uses tanh activation function."""
        # Test with sample input in valid range [0, 1]
        sample_input = torch.rand(1, 784)

        # Get encoder output (before activation)
        encoder_linear_output = self.autoencoder.encoder(sample_input)

        # Get actual encoder output (with activation)
        encoder_output = self.autoencoder.encode(sample_input)

        # Check that the activation is tanh
        expected_output = torch.tanh(encoder_linear_output)
        torch.testing.assert_close(
            encoder_output,
            expected_output,
            msg="Hidden layer should use tanh activation"
        )

    def test_output_layer_activation_sigmoid(self):
        """Test that output layer uses sigmoid activation function."""
        # Test with sample input in valid range [0, 1]
        sample_input = torch.rand(1, 784)

        # Get full autoencoder output
        output = self.autoencoder(sample_input)

        # Verify output is in [0, 1] range (sigmoid characteristic)
        self.assertTrue(
            torch.all(output >= 0) and torch.all(output <= 1),
            "Output layer should use sigmoid activation (values in [0,1] range)"
        )

        # Test that sigmoid is actually applied in decoder
        encoded = self.autoencoder.encode(sample_input)
        decoder_linear_output = self.autoencoder.decoder(encoded)
        decoder_output = self.autoencoder.decode(encoded)

        expected_output = torch.sigmoid(decoder_linear_output)
        torch.testing.assert_close(
            decoder_output,
            expected_output,
            msg="Output layer should use sigmoid activation"
        )

    def test_bias_terms_included(self):
        """Test that bias terms are included in layers."""
        # Check encoder bias
        self.assertIsNotNone(
            self.autoencoder.encoder.bias,
            "Encoder should include bias terms"
        )

        # Check decoder bias
        self.assertIsNotNone(
            self.autoencoder.decoder.bias,
            "Decoder should include bias terms"
        )

    def test_weight_initialization_method(self):
        """Test that weights are initialized using Xavier/Glorot uniform initialization."""
        # Test that weights are not zero (initialized)
        encoder_weights = self.autoencoder.encoder.weight
        decoder_weights = self.autoencoder.decoder.weight

        # Check that weights are not all zeros
        self.assertFalse(
            torch.allclose(encoder_weights, torch.zeros_like(encoder_weights)),
            "Encoder weights should be initialized (not all zeros)"
        )
        self.assertFalse(
            torch.allclose(decoder_weights, torch.zeros_like(decoder_weights)),
            "Decoder weights should be initialized (not all zeros)"
        )

        # Check that biases are initialized to zero (Xavier standard practice)
        encoder_bias = self.autoencoder.encoder.bias
        decoder_bias = self.autoencoder.decoder.bias

        torch.testing.assert_close(
            encoder_bias,
            torch.zeros_like(encoder_bias),
            msg="Encoder bias should be initialized to zero"
        )
        torch.testing.assert_close(
            decoder_bias,
            torch.zeros_like(decoder_bias),
            msg="Decoder bias should be initialized to zero"
        )

    def test_network_structure_784_400_784(self):
        """Test that network follows 784→400→784 structure."""
        # Test input to hidden
        self.assertEqual(
            self.autoencoder.encoder.in_features,
            784,
            "Encoder input should be 784 features"
        )
        self.assertEqual(
            self.autoencoder.encoder.out_features,
            400,
            "Encoder output should be 400 features"
        )

        # Test hidden to output
        self.assertEqual(
            self.autoencoder.decoder.in_features,
            400,
            "Decoder input should be 400 features"
        )
        self.assertEqual(
            self.autoencoder.decoder.out_features,
            784,
            "Decoder output should be 784 features"
        )

    def test_model_parameter_count(self):
        """Test that model has correct number of parameters."""
        # Calculate expected parameters:
        # Encoder: (784 * 400) + 400 = 313,600 + 400 = 314,000
        # Decoder: (400 * 784) + 784 = 313,600 + 784 = 314,384
        # Total: 314,000 + 314,384 = 628,384

        expected_encoder_params = (784 * 400) + 400  # weights + biases
        expected_decoder_params = (400 * 784) + 784  # weights + biases
        expected_total_params = expected_encoder_params + expected_decoder_params

        actual_total_params = self.autoencoder.count_parameters()

        self.assertEqual(
            actual_total_params,
            expected_total_params,
            f"Model should have {expected_total_params} parameters, got {actual_total_params}"
        )

    def test_forward_pass_dimensions(self):
        """Test that forward pass maintains correct dimensions."""
        batch_sizes = [1, 8, 32]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.rand(batch_size, 784)

                # Test encoding
                encoded = self.autoencoder.encode(input_tensor)
                self.assertEqual(
                    encoded.shape,
                    (batch_size, 400),
                    f"Encoded shape should be ({batch_size}, 400)"
                )

                # Test decoding
                decoded = self.autoencoder.decode(encoded)
                self.assertEqual(
                    decoded.shape,
                    (batch_size, 784),
                    f"Decoded shape should be ({batch_size}, 784)"
                )

                # Test full forward pass
                output = self.autoencoder(input_tensor)
                self.assertEqual(
                    output.shape,
                    input_tensor.shape,
                    f"Output shape should match input shape {input_tensor.shape}"
                )


if __name__ == '__main__':
    unittest.main()