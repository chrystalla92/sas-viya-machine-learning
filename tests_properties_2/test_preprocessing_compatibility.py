"""
Test preprocessing compatibility between Python autoencoder and SAS autoencoder.

Tests based on comparison table in SAS_Autoencoder_Properties.md to verify
that the Python implementation follows the SAS implementation preprocessing.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mnist_autoencoder.data.transforms import (
    Normalize,
    Flatten,
    create_sas_compatible_transforms,
    create_training_transforms,
    create_evaluation_transforms
)


class TestPreprocessingCompatibility(unittest.TestCase):
    """Test preprocessing properties to match SAS createData.sas configuration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample MNIST-like data (28x28 images)
        self.sample_image_2d = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
        self.sample_image_1d = self.sample_image_2d.reshape(-1).float()
        self.sample_batch = torch.randint(0, 256, (5, 28, 28), dtype=torch.uint8)

    def test_input_flattening_compatibility(self):
        """Test that 28×28 images are flattened to 784 features like SAS."""
        # SAS: Pixel values extracted as 784 features (var2-var785)
        # Python: Image flattening to 784 features via reshape(-1)

        expected_features = 784
        expected_shape = (784,)

        # Test flatten transform
        flatten_transform = Flatten()
        flattened = flatten_transform(self.sample_image_2d)

        self.assertEqual(
            flattened.shape,
            expected_shape,
            f"Flattened shape should be {expected_shape}, got {flattened.shape}"
        )

        self.assertEqual(
            flattened.numel(),
            expected_features,
            f"Should have {expected_features} features after flattening"
        )

    def test_data_range_compatibility(self):
        """Test input data range handling matches SAS approach."""
        # SAS: Raw pixel values from MNIST data (0-255 range)
        # Python: Various normalization options but should support raw range

        # Test SAS-compatible transforms (should preserve [0, 255] range)
        sas_transforms = create_sas_compatible_transforms()

        # Convert to tensor first
        sample_tensor = self.sample_image_2d.float()
        transformed = sas_transforms(sample_tensor)

        # Should be flattened
        self.assertEqual(transformed.shape, (784,))

        # Should preserve original range for SAS compatibility
        # Note: The values should be in a range suitable for SAS processing
        self.assertTrue(
            torch.all(transformed >= 0.0),
            "Transformed values should be non-negative"
        )

    def test_normalization_methods_compatibility(self):
        """Test normalization methods - this should FAIL to show the difference."""
        # SAS: midrange standardization (standardize=midrange)
        # This is NOT the same as simple min-max normalization
        # SAS midrange: (value - midrange) / range where midrange = (min + max) / 2
        # Python: Simple [0,1] or [-1,1] scaling

        # This test is designed to FAIL to highlight the normalization difference

        test_data = torch.tensor([0.0, 127.5, 255.0])  # Min, mid, max values

        # Python [0,1] normalization: x / 255
        normalize_01 = Normalize(method="01")
        python_normalized = normalize_01(test_data)

        # What SAS midrange would actually produce:
        # For MNIST [0, 255]: midrange = (0 + 255) / 2 = 127.5, range = 255 - 0 = 255
        # SAS formula: (value - midrange) / (range/2) = (value - 127.5) / 127.5
        midrange = (0.0 + 255.0) / 2.0  # 127.5
        half_range = (255.0 - 0.0) / 2.0  # 127.5
        sas_expected = (test_data - midrange) / half_range  # [-1, 0, 1]

        print(f"\nNormalization Difference Detected:")
        print(f"Test data: {test_data}")
        print(f"Python [0,1] result: {python_normalized}")
        print(f"SAS midrange would be: {sas_expected}")
        print(f"Difference: {python_normalized - torch.tensor([0.0, 0.5, 1.0])}")

        # This assertion will FAIL because Python normalization != SAS midrange
        self.fail(
            f"EXPECTED FAILURE: Python normalization ({python_normalized.tolist()}) "
            f"does NOT match SAS midrange standardization ({sas_expected.tolist()}). "
            f"SAS uses midrange standardization: (value - midrange) / half_range, "
            f"while Python uses simple scaling: value / max_value. "
            f"This is a fundamental difference in preprocessing approaches."
        )

    def test_missing_value_handling_compatibility(self):
        """Test missing value handling approach."""
        # SAS: Default missing value handling applies
        # Python: Built-in NaN and infinite value detection

        # Test with NaN values
        test_data_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0])

        # The transforms should handle or detect NaN values
        normalize = Normalize(method="01")
        result = normalize(test_data_with_nan)

        # Check that NaN is preserved or properly handled
        self.assertTrue(
            torch.isnan(result[2]),
            "NaN values should be preserved in the normalized output"
        )

    def test_data_preparation_pipeline_compatibility(self):
        """Test complete data preparation pipeline matches SAS approach."""
        # SAS Pipeline:
        # 1. MNIST binary files read with recfm=n (binary format)
        # 2. Pixel values extracted as 784 features (var2-var785)
        # 3. Labels preserved as var1
        # 4. Data exported to CSV format for processing

        # Python Pipeline:
        # 1. Torchvision automatic download and loading OR Direct IDX file reading
        # 2. Image flattening to 784 features via reshape(-1)
        # 3. Normalization to specified range
        # 4. Optional Gaussian noise addition for denoising

        # Test training transforms
        training_transforms = create_training_transforms(
            normalize="11",  # Use [-1,1] to be closer to SAS midrange
            add_noise=False
        )

        sample_tensor = self.sample_image_2d.float()
        transformed = training_transforms(sample_tensor)

        # Should be flattened to 784 features
        self.assertEqual(transformed.shape, (784,))

        # Should be in [-1,1] range (closer to SAS midrange scaling)
        self.assertTrue(
            torch.all(transformed >= -1.0) and torch.all(transformed <= 1.0),
            "Training transforms should produce values in [-1,1] range"
        )

    def test_feature_count_validation_compatibility(self):
        """Test feature count validation matches SAS expectations."""
        # SAS: 784 features (28×28 flattened)
        # Python: Shape validation for 784 features

        from mnist_autoencoder.data.transforms import ValidateShape

        validator = ValidateShape(expected_features=784)

        # Test valid input
        valid_input = torch.randn(784)
        validated = validator(valid_input)
        self.assertEqual(validated.shape, (784,))

        # Test invalid input
        invalid_input = torch.randn(400)  # Wrong number of features
        with self.assertRaises(ValueError):
            validator(invalid_input)

    def test_batch_processing_compatibility(self):
        """Test batch processing maintains feature structure."""
        # Both SAS and Python should handle batched data properly

        batch_transforms = create_evaluation_transforms(normalize="01")

        # Process batch
        batch_tensor = self.sample_batch.float()
        batch_size = batch_tensor.shape[0]

        # Apply transforms to each sample in batch
        transformed_batch = []
        for i in range(batch_size):
            sample = batch_tensor[i]
            transformed = batch_transforms(sample)
            transformed_batch.append(transformed)

        transformed_batch = torch.stack(transformed_batch)

        # Check batch dimensions
        expected_shape = (batch_size, 784)
        self.assertEqual(
            transformed_batch.shape,
            expected_shape,
            f"Batch shape should be {expected_shape}, got {transformed_batch.shape}"
        )

    def test_data_type_compatibility(self):
        """Test data type handling matches expectations."""
        # SAS: Handles numeric data types appropriately
        # Python: Converts to float32 for neural network compatibility

        from mnist_autoencoder.data.transforms import ToFloat32

        # Test different input types
        int_data = torch.randint(0, 256, (784,), dtype=torch.int32)
        float_converter = ToFloat32()
        converted = float_converter(int_data)

        self.assertEqual(
            converted.dtype,
            torch.float32,
            "Data should be converted to float32 for neural network compatibility"
        )

    def test_preprocessing_determinism_compatibility(self):
        """Test that preprocessing is deterministic for reproducibility."""
        # Both SAS and Python should produce consistent results

        # Set seed for reproducible transforms
        torch.manual_seed(23451)  # Use SAS seed
        transform1 = create_training_transforms(normalize="01", add_noise=True)

        torch.manual_seed(23451)  # Reset seed
        transform2 = create_training_transforms(normalize="01", add_noise=True)

        sample = self.sample_image_2d.float()

        # Apply same transforms with same seed
        torch.manual_seed(42)
        result1 = transform1(sample)

        torch.manual_seed(42)
        result2 = transform2(sample)

        torch.testing.assert_close(
            result1,
            result2,
            msg="Transforms with same seed should produce identical results"
        )

    def test_value_range_validation_compatibility(self):
        """Test value range validation for different input ranges."""
        # Test that transforms handle expected value ranges properly

        # Test [0, 255] input range
        max_value_input = torch.full((784,), 255.0)
        min_value_input = torch.zeros(784)

        normalize_01 = Normalize(method="01")

        # Test maximum values
        max_normalized = normalize_01(max_value_input)
        self.assertTrue(
            torch.allclose(max_normalized, torch.ones(784)),
            "Maximum input values should normalize to 1.0"
        )

        # Test minimum values
        min_normalized = normalize_01(min_value_input)
        self.assertTrue(
            torch.allclose(min_normalized, torch.zeros(784)),
            "Minimum input values should normalize to 0.0"
        )

    def test_preprocessing_chain_compatibility(self):
        """Test that preprocessing chain maintains data integrity."""
        # Test complete preprocessing chain similar to SAS workflow

        original_2d = self.sample_image_2d.float()
        original_flat = original_2d.reshape(-1)

        # Apply preprocessing chain
        transforms = create_evaluation_transforms(normalize="01")
        processed = transforms(original_2d)

        # Verify data integrity
        self.assertEqual(processed.shape, original_flat.shape)
        self.assertTrue(torch.all(processed >= 0.0) and torch.all(processed <= 1.0))

        # Test that we can recover approximate original values
        denormalized = processed * 255.0
        torch.testing.assert_close(
            denormalized,
            original_flat,
            rtol=1e-5,
            atol=1e-5,
            msg="Should be able to recover original values from normalized data"
        )

    def test_sas_midrange_approximation_compatibility(self):
        """Test approximation of SAS midrange standardization."""
        # SAS uses midrange standardization which normalizes to [-1, 1]
        # Test that Python [-1,1] normalization approximates this

        # Create test data with known range
        test_data = torch.linspace(0, 255, 10)

        normalize_11 = Normalize(method="11")
        result = normalize_11(test_data)

        # Should map [0, 255] to [-1, 1]
        self.assertAlmostEqual(result.min().item(), -1.0, places=5)
        self.assertAlmostEqual(result.max().item(), 1.0, places=5)

        # Midpoint should be close to 0 (allowing for discrete sampling differences)
        midpoint_idx = len(test_data) // 2
        self.assertLess(
            abs(result[midpoint_idx].item()),
            0.2,
            "Midpoint of [-1,1] normalization should be close to 0"
        )


if __name__ == "__main__":
    unittest.main()