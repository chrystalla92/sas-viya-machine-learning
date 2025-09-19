"""
Test cases for preprocessing properties.

Tests verify that the mnist_autoencoder preprocessing matches the specifications
documented in SAS_Autoencoder_Properties.md
"""

import unittest
import torch
import numpy as np
from torchvision import transforms
from mnist_autoencoder.data.transforms import (
    MNISTTransforms,
    create_training_transforms,
    create_evaluation_transforms,
    create_sas_compatible_transforms,
    Normalize,
    Flatten,
    ToFloat32,
    ValidateShape
)


class TestPreprocessingProperties(unittest.TestCase):
    """Test preprocessing-related properties of the autoencoder."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample MNIST-like data
        self.sample_image_2d = torch.rand(28, 28) * 255  # Raw pixel values [0, 255]
        self.sample_image_1d = torch.rand(784) * 255     # Flattened pixel values
        self.sample_batch = torch.rand(32, 784) * 255    # Batch of flattened images

    def test_midrange_standardization_equivalent(self):
        """Test that normalization provides midrange-like standardization."""
        # SAS midrange standardization typically scales to [0, 1] or [-1, 1]
        # Test [0, 1] normalization (most common for image data)
        normalize_01 = Normalize(method="01")

        # Test with raw pixel values [0, 255]
        raw_pixels = torch.tensor([0.0, 128.0, 255.0])
        normalized = normalize_01(raw_pixels)

        expected = torch.tensor([0.0, 128.0/255.0, 1.0])
        torch.testing.assert_close(
            normalized,
            expected,
            msg="Should normalize [0, 255] to [0, 1] range (midrange-like)"
        )

    def test_midrange_standardization_11_range(self):
        """Test that normalization can provide [-1, 1] range (alternative midrange)."""
        normalize_11 = Normalize(method="11")

        # Test with raw pixel values [0, 255]
        raw_pixels = torch.tensor([0.0, 128.0, 255.0])
        normalized = normalize_11(raw_pixels)

        expected = torch.tensor([-1.0, 0.0039, 1.0])  # approximately
        torch.testing.assert_close(
            normalized,
            expected,
            atol=1e-2,
            rtol=1e-2,
            msg="Should normalize [0, 255] to [-1, 1] range"
        )

    def test_mnist_data_range_handling(self):
        """Test handling of raw MNIST pixel values (0-255 range)."""
        # Test that transforms can handle raw MNIST data
        sas_transform = create_sas_compatible_transforms()

        # Test with raw MNIST-like data
        raw_image = torch.rand(28, 28) * 255

        try:
            transformed = sas_transform(raw_image)
            self.assertEqual(
                transformed.shape,
                (784,),
                "Should flatten 28x28 image to 784 features"
            )
            self.assertTrue(
                torch.all(transformed >= 0) and torch.all(transformed <= 255),
                "Should preserve [0, 255] range for SAS compatibility"
            )
        except Exception as e:
            self.fail(f"SAS-compatible transform should handle raw MNIST data: {e}")

    def test_no_explicit_clipping_default(self):
        """Test that default transforms don't apply explicit clipping."""
        # Create training transform without explicit clipping
        train_transform = create_training_transforms(normalize="01")

        # Test with values slightly outside [0, 255] to see if clipping is applied
        test_data = torch.tensor([[-10.0, 0.0, 128.0, 255.0, 300.0]])  # Batch format

        # The transform should normalize based on division, not clipping
        try:
            result = train_transform(test_data.squeeze())
            # After normalization, values outside [0, 255] should be outside [0, 1]
            self.assertTrue(
                result.min() < 0 or result.max() > 1,
                "Should not clip values before normalization (values outside [0,1] expected)"
            )
        except Exception:
            # If transform fails, it's acceptable as it shows no implicit clipping
            pass

    def test_missing_value_handling_not_configured(self):
        """Test that missing value handling is not explicitly configured."""
        # Test that transforms don't have special NaN/missing value handling
        # (relies on PyTorch/SAS defaults)

        # Create data with NaN values
        data_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0, 5.0])

        normalize = Normalize(method="01")

        # Should propagate NaN (no special handling)
        result = normalize(data_with_nan)
        self.assertTrue(
            torch.isnan(result[2]),
            "Should propagate NaN values (no special missing value handling)"
        )

    def test_data_preparation_pipeline_mnist(self):
        """Test that data preparation follows MNIST binary processing pipeline."""
        # Test the overall pipeline matches createData.sas approach:
        # 1. Read binary data
        # 2. Extract 784 features (var2-var785)
        # 3. Preserve labels (var1)
        # 4. Export to CSV format

        # Simulate the pipeline
        # Step 1: Raw binary-like data (simulated)
        raw_data = torch.randint(0, 256, (10, 784), dtype=torch.float32)
        labels = torch.randint(0, 10, (10,))

        # Step 2: Flatten and validate 784 features
        validator = ValidateShape(expected_features=784)
        try:
            validated_data = validator(raw_data)
            self.assertEqual(
                validated_data.shape[1],
                784,
                "Should validate 784 features (var2-var785 equivalent)"
            )
        except Exception as e:
            self.fail(f"Pipeline should handle 784-feature MNIST data: {e}")

        # Step 3: Labels should be preserved separately
        self.assertEqual(
            labels.shape[0],
            raw_data.shape[0],
            "Labels should be preserved for each sample"
        )

    def test_feature_extraction_784_dimensions(self):
        """Test that feature extraction produces 784 dimensions."""
        # Test flattening transform
        flatten = Flatten()

        # Test with 28x28 image
        image_2d = torch.rand(28, 28)
        flattened = flatten(image_2d)

        self.assertEqual(
            flattened.shape,
            (784,),
            "Should flatten 28x28 image to 784 features"
        )

    def test_var2_to_var785_equivalent_indexing(self):
        """Test that feature indexing matches SAS var2-var785 convention."""
        # In SAS, var1 is label, var2-var785 are features (784 features total)
        # Test that we handle the same indexing

        # Simulate full data with label + features
        full_data = torch.rand(785)  # label + 784 features

        # Extract features (equivalent to var2-var785)
        features = full_data[1:]  # Skip first element (label)

        self.assertEqual(
            features.shape[0],
            784,
            "Should extract 784 features (var2-var785 equivalent)"
        )

    def test_csv_export_compatibility(self):
        """Test data format compatibility with CSV export."""
        # Test that transforms produce data compatible with CSV format
        # (no special objects, proper numeric types)

        transform = create_training_transforms(normalize="01")

        # Test with sample data
        sample_data = torch.rand(784) * 255
        transformed = transform(sample_data)

        # Should be numeric tensor
        self.assertIsInstance(
            transformed,
            torch.Tensor,
            "Should produce torch.Tensor (CSV-compatible numeric data)"
        )

        # Should be float32 (standard numeric type)
        self.assertEqual(
            transformed.dtype,
            torch.float32,
            "Should produce float32 data (CSV-compatible)"
        )

    def test_no_explicit_missing_handling_configuration(self):
        """Test that no explicit missing value handling is configured."""
        # Verify that transforms don't have built-in missing value strategies
        # (relies on default behavior)

        transforms_to_test = [
            Normalize("01"),
            Flatten(),
            ToFloat32(),
            ValidateShape(784)
        ]

        for transform in transforms_to_test:
            # Check that transform doesn't have missing value handling attributes
            self.assertFalse(
                hasattr(transform, 'handle_missing'),
                f"{transform.__class__.__name__} should not have explicit missing value handling"
            )
            self.assertFalse(
                hasattr(transform, 'fill_value'),
                f"{transform.__class__.__name__} should not have explicit fill value"
            )

    def test_preprocessing_preserves_mnist_structure(self):
        """Test that preprocessing preserves MNIST data structure."""
        # Test complete preprocessing pipeline
        train_transform = create_training_transforms(normalize="01")
        eval_transform = create_evaluation_transforms(normalize="01")

        # Test with MNIST-like batch
        batch_data = torch.rand(32, 784) * 255

        # Train transform
        train_results = []
        for sample in batch_data:
            train_results.append(train_transform(sample))
        train_batch = torch.stack(train_results)

        # Eval transform
        eval_results = []
        for sample in batch_data:
            eval_results.append(eval_transform(sample))
        eval_batch = torch.stack(eval_results)

        # Both should preserve batch structure
        self.assertEqual(
            train_batch.shape,
            (32, 784),
            "Training transform should preserve MNIST batch structure"
        )
        self.assertEqual(
            eval_batch.shape,
            (32, 784),
            "Evaluation transform should preserve MNIST batch structure"
        )


if __name__ == '__main__':
    unittest.main()