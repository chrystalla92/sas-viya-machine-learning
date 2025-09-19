"""
Test cases for output transform properties.

Tests verify that the mnist_autoencoder output transforms match the
specifications documented in SAS_Autoencoder_Properties.md
"""

import unittest
import torch
import torch.nn.functional as F
from mnist_autoencoder.models.autoencoder import MLPAutoencoder, create_mnist_autoencoder


class TestOutputTransformProperties(unittest.TestCase):
    """Test output transform properties of the autoencoder."""

    def setUp(self):
        """Set up test fixtures."""
        self.autoencoder = create_mnist_autoencoder(seed=23451)

    def test_encoder_output_tanh_activation(self):
        """Test that encoder output uses tanh activation."""
        # Create sample input in valid range [0, 1]
        sample_input = torch.rand(8, 784)

        # Get encoder output
        encoded = self.autoencoder.encode(sample_input)

        # Verify tanh characteristics: output in [-1, 1] range
        self.assertTrue(
            torch.all(encoded >= -1.0) and torch.all(encoded <= 1.0),
            "Encoder output should be in [-1, 1] range (tanh activation)"
        )

        # Test that it's actually tanh by comparing with manual calculation
        linear_output = self.autoencoder.encoder(sample_input)
        expected_output = torch.tanh(linear_output)

        torch.testing.assert_close(
            encoded,
            expected_output,
            msg="Encoder should use tanh activation"
        )

    def test_decoder_output_sigmoid_activation(self):
        """Test that decoder output uses sigmoid activation."""
        # Create sample hidden representation in tanh range [-1, 1]
        sample_hidden = torch.rand(8, 400) * 2 - 1

        # Get decoder output
        decoded = self.autoencoder.decode(sample_hidden)

        # Verify sigmoid characteristics: output in [0, 1] range
        self.assertTrue(
            torch.all(decoded >= 0.0) and torch.all(decoded <= 1.0),
            "Decoder output should be in [0, 1] range (sigmoid activation)"
        )

        # Test that it's actually sigmoid by comparing with manual calculation
        linear_output = self.autoencoder.decoder(sample_hidden)
        expected_output = torch.sigmoid(linear_output)

        torch.testing.assert_close(
            decoded,
            expected_output,
            msg="Decoder should use sigmoid activation"
        )

    def test_output_range_0_to_1(self):
        """Test that final output is in [0, 1] range."""
        # Test with various inputs
        test_inputs = [
            torch.rand(1, 784),       # Single sample
            torch.rand(16, 784),      # Small batch
            torch.rand(64, 784),      # Large batch
            torch.zeros(1, 784),      # Zero input
            torch.ones(1, 784),       # Ones input
        ]

        for i, test_input in enumerate(test_inputs):
            with self.subTest(input_case=i):
                output = self.autoencoder(test_input)

                self.assertTrue(
                    torch.all(output >= 0.0),
                    f"All output values should be >= 0.0 for input case {i}"
                )
                self.assertTrue(
                    torch.all(output <= 1.0),
                    f"All output values should be <= 1.0 for input case {i}"
                )

    def test_no_additional_post_scaling(self):
        """Test that no additional post-scaling is applied beyond sigmoid."""
        # Test that output is direct sigmoid result without additional scaling
        sample_input = torch.rand(4, 784)

        # Get full forward pass
        full_output = self.autoencoder(sample_input)

        # Get manual calculation: encode -> decode
        encoded = self.autoencoder.encode(sample_input)
        decoded = self.autoencoder.decode(encoded)

        torch.testing.assert_close(
            full_output,
            decoded,
            msg="Full forward should be same as encode->decode (no additional scaling)"
        )

        # Verify sigmoid is the final transform (no scaling after)
        linear_decoder_output = self.autoencoder.decoder(encoded)
        expected_final = torch.sigmoid(linear_decoder_output)

        torch.testing.assert_close(
            full_output,
            expected_final,
            msg="Final output should be sigmoid without additional post-scaling"
        )

    def test_sigmoid_vs_identity_preference(self):
        """Test that sigmoid is preferred over identity transform for output."""
        # Create decoder output before activation
        sample_hidden = torch.rand(8, 400) * 2 - 1  # Range [-1, 1] like tanh output
        linear_output = self.autoencoder.decoder(sample_hidden)

        # Compare sigmoid vs identity
        sigmoid_output = torch.sigmoid(linear_output)
        identity_output = linear_output  # Identity transform

        # Actual decoder output should match sigmoid, not identity
        actual_output = self.autoencoder.decode(sample_hidden)

        torch.testing.assert_close(
            actual_output,
            sigmoid_output,
            msg="Should use sigmoid activation (not identity transform)"
        )

        # Verify it's NOT identity transform
        self.assertFalse(
            torch.allclose(actual_output, identity_output),
            "Should NOT use identity transform for decoder output"
        )

    def test_pixel_reconstruction_suitability(self):
        """Test that sigmoid output is suitable for pixel reconstruction."""
        # Test with pixel-like input (normalized to [0, 1])
        pixel_input = torch.rand(8, 784)  # Pixel values in [0, 1]

        output = self.autoencoder(pixel_input)

        # Output should be in valid pixel range
        self.assertTrue(
            torch.all(output >= 0.0) and torch.all(output <= 1.0),
            "Output should be in valid pixel range [0, 1]"
        )

        # Output should be continuous (not binary)
        unique_values = torch.unique(output)
        self.assertGreater(
            len(unique_values),
            2,
            "Output should be continuous (suitable for grayscale pixels)"
        )

    def test_activation_function_gradient_flow(self):
        """Test that activation functions support proper gradient flow."""
        # Test gradient flow through tanh (encoder)
        sample_input = torch.rand(4, 784, requires_grad=True)

        encoded = self.autoencoder.encode(sample_input)
        loss_encoded = encoded.sum()
        loss_encoded.backward(retain_graph=True)

        self.assertIsNotNone(
            sample_input.grad,
            "Gradients should flow through tanh activation"
        )

        # Reset gradients
        sample_input.grad = None

        # Test gradient flow through sigmoid (decoder)
        full_output = self.autoencoder(sample_input)
        loss_full = full_output.sum()
        loss_full.backward()

        self.assertIsNotNone(
            sample_input.grad,
            "Gradients should flow through sigmoid activation"
        )

    def test_encoder_decoder_activation_consistency(self):
        """Test that encoder and decoder activations are consistently applied."""
        sample_input = torch.rand(8, 784)

        # Test encoder consistency
        encoded1 = self.autoencoder.encode(sample_input)

        # Manual encoding
        linear_encoded = self.autoencoder.encoder(sample_input)
        manual_encoded = torch.tanh(linear_encoded)

        torch.testing.assert_close(
            encoded1,
            manual_encoded,
            msg="Encoder activation should be consistently applied"
        )

        # Test decoder consistency
        decoded1 = self.autoencoder.decode(encoded1)

        # Manual decoding
        linear_decoded = self.autoencoder.decoder(encoded1)
        manual_decoded = torch.sigmoid(linear_decoded)

        torch.testing.assert_close(
            decoded1,
            manual_decoded,
            msg="Decoder activation should be consistently applied"
        )

    def test_no_identity_transform_used(self):
        """Test that identity transform is not used for either encoder or decoder."""
        sample_input = torch.rand(8, 784)

        # Test encoder doesn't use identity
        linear_encoded = self.autoencoder.encoder(sample_input)
        actual_encoded = self.autoencoder.encode(sample_input)

        self.assertFalse(
            torch.allclose(linear_encoded, actual_encoded),
            "Encoder should not use identity activation (should use tanh)"
        )

        # Test decoder doesn't use identity
        sample_hidden = torch.rand(8, 400) * 2 - 1  # Range [-1, 1] like tanh output
        linear_decoded = self.autoencoder.decoder(sample_hidden)
        actual_decoded = self.autoencoder.decode(sample_hidden)

        self.assertFalse(
            torch.allclose(linear_decoded, actual_decoded),
            "Decoder should not use identity activation (should use sigmoid)"
        )

    def test_output_transform_numerical_stability(self):
        """Test that output transforms are numerically stable."""
        # Test with extreme values
        extreme_inputs = [
            torch.full((4, 784), -10.0),  # Very negative
            torch.full((4, 784), 10.0),   # Very positive
            torch.full((4, 784), 0.0),    # Zero
        ]

        for i, extreme_input in enumerate(extreme_inputs):
            with self.subTest(extreme_case=i):
                try:
                    output = self.autoencoder(extreme_input)

                    # Should not produce NaN or Inf
                    self.assertFalse(
                        torch.isnan(output).any(),
                        f"Output should not contain NaN for extreme input {i}"
                    )
                    self.assertFalse(
                        torch.isinf(output).any(),
                        f"Output should not contain Inf for extreme input {i}"
                    )

                    # Should still be in valid range
                    self.assertTrue(
                        torch.all(output >= 0.0) and torch.all(output <= 1.0),
                        f"Output should be in [0,1] range for extreme input {i}"
                    )

                except Exception as e:
                    self.fail(f"Output transform should be stable for extreme input {i}: {e}")

    def test_batch_consistency_output_transforms(self):
        """Test that output transforms work consistently across different batch sizes."""
        # Create same input with different batch sizes
        base_input = torch.rand(784)

        batch_sizes = [1, 8, 16, 32, 64]
        outputs = []

        for batch_size in batch_sizes:
            batch_input = base_input.unsqueeze(0).expand(batch_size, -1)
            batch_output = self.autoencoder(batch_input)

            # All outputs in batch should be identical (same input)
            for i in range(batch_size):
                if i == 0:
                    single_output = batch_output[i]
                else:
                    torch.testing.assert_close(
                        batch_output[i],
                        single_output,
                        msg=f"Batch processing should be consistent for batch size {batch_size}"
                    )

            outputs.append(single_output)

        # All single outputs should be identical across different batch sizes
        reference_output = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            torch.testing.assert_close(
                output,
                reference_output,
                msg=f"Output should be consistent across batch sizes (batch size {batch_sizes[i]})"
            )


if __name__ == '__main__':
    unittest.main()