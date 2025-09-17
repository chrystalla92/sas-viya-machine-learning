"""
Comprehensive tests for MNIST autoencoder models.

This module tests all aspects of the MLPAutoencoder implementation including
architecture validation, forward pass, gradient flow, device management,
and model persistence.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import tempfile
import shutil

from mnist_autoencoder.models.autoencoder import MLPAutoencoder, create_mnist_autoencoder
from mnist_autoencoder.models.utils import (
    ModelCheckpoint, ModelValidator, ModelSummary,
    save_model, load_model, validate_model, print_model_summary
)


class TestMLPAutoencoder:
    """Test suite for MLPAutoencoder class."""
    
    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = MLPAutoencoder()
        
        assert model.input_size == 784
        assert model.hidden_size == 400
        assert isinstance(model.encoder, nn.Linear)
        assert isinstance(model.decoder, nn.Linear)
        assert model.encoder.in_features == 784
        assert model.encoder.out_features == 400
        assert model.decoder.in_features == 400
        assert model.decoder.out_features == 784
    
    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = MLPAutoencoder(input_size=512, hidden_size=256, seed=42)
        
        assert model.input_size == 512
        assert model.hidden_size == 256
        assert model.encoder.in_features == 512
        assert model.encoder.out_features == 256
        assert model.decoder.in_features == 256
        assert model.decoder.out_features == 512
    
    def test_model_initialization_validation(self):
        """Test model initialization parameter validation."""
        # Test invalid input_size
        with pytest.raises(ValueError, match="input_size must be a positive integer"):
            MLPAutoencoder(input_size=0)
        
        with pytest.raises(ValueError, match="input_size must be a positive integer"):
            MLPAutoencoder(input_size=-784)
        
        # Test invalid hidden_size
        with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
            MLPAutoencoder(hidden_size=0)
        
        with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
            MLPAutoencoder(hidden_size=-400)
    
    def test_device_management(self, device):
        """Test device management functionality."""
        model = MLPAutoencoder(device=device)
        
        assert model.device == device
        assert next(model.parameters()).device == device
        
        # Test moving to different device
        cpu_device = torch.device("cpu")
        model_on_cpu = model.to_device(cpu_device)
        assert model_on_cpu.device == cpu_device
        assert next(model_on_cpu.parameters()).device == cpu_device
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_forward_pass_shapes(self, batch_size):
        """Test forward pass with various batch sizes."""
        model = MLPAutoencoder()
        input_tensor = torch.randn(batch_size, 784) * 0.5 + 0.5
        input_tensor = input_tensor.clamp(0, 1)
        
        output = model(input_tensor)
        
        assert output.shape == input_tensor.shape
        assert output.shape == (batch_size, 784)
    
    def test_encoder_decoder_components(self):
        """Test individual encoder and decoder components."""
        model = MLPAutoencoder()
        input_tensor = torch.randn(4, 784) * 0.5 + 0.5
        input_tensor = input_tensor.clamp(0, 1)
        
        # Test encoder
        encoded = model.encode(input_tensor)
        assert encoded.shape == (4, 400)
        
        # Check tanh activation (values should be in [-1, 1])
        assert encoded.min() >= -1.1  # Small tolerance for numerical precision
        assert encoded.max() <= 1.1
        
        # Test decoder
        decoded = model.decode(encoded)
        assert decoded.shape == (4, 784)
        
        # Check sigmoid activation (values should be in [0, 1])
        assert decoded.min() >= -0.1  # Small tolerance
        assert decoded.max() <= 1.1
    
    def test_input_validation(self):
        """Test input validation for various edge cases."""
        model = MLPAutoencoder()
        
        # Test invalid tensor type
        with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
            model.forward([1, 2, 3])
        
        # Test wrong dimensionality
        with pytest.raises(ValueError, match="Input must be 2D"):
            model.forward(torch.randn(784))  # 1D tensor
        
        with pytest.raises(ValueError, match="Input must be 2D"):
            model.forward(torch.randn(1, 28, 28))  # 3D tensor
        
        # Test wrong input size
        with pytest.raises(ValueError, match="Input size .* doesn't match expected"):
            model.forward(torch.randn(4, 512))  # Wrong feature size
        
        # Test out of range values
        with pytest.raises(ValueError, match="Input values out of expected range"):
            model.forward(torch.randn(4, 784) * 10)  # Values too large
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = MLPAutoencoder(seed=42)
        input_tensor = torch.randn(4, 784) * 0.5 + 0.5
        input_tensor = input_tensor.clamp(0, 1)
        
        # Test gradient flow verification
        gradient_status = model.verify_gradient_flow(input_tensor)
        
        # All parameters should have gradients
        for param_name, has_gradient in gradient_status.items():
            assert has_gradient, f"No gradient for {param_name}"
        
        # Check specific parameters
        expected_params = ["encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias"]
        for param_name in expected_params:
            assert param_name in gradient_status
            assert gradient_status[param_name]
    
    def test_model_info(self):
        """Test model information generation."""
        model = MLPAutoencoder(seed=42)
        info = model.get_model_info()
        
        assert info["architecture"] == "MLPAutoencoder"
        assert info["input_size"] == 784
        assert info["hidden_size"] == 400
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["activation_functions"]["encoder"] == "tanh"
        assert info["activation_functions"]["decoder"] == "sigmoid"
    
    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        model = MLPAutoencoder()
        
        total_params = model.count_parameters()
        trainable_params = model.count_parameters(trainable_only=True)
        
        # Calculate expected parameters
        # Encoder: 784 * 400 + 400 = 314,000
        # Decoder: 400 * 784 + 784 = 314,184
        # Total: 628,584
        expected_total = 784 * 400 + 400 + 400 * 784 + 784
        assert total_params == expected_total
        assert trainable_params == expected_total  # All parameters are trainable by default
    
    def test_weight_initialization(self):
        """Test weight initialization consistency."""
        # Test reproducibility with seed
        model1 = MLPAutoencoder(seed=42)
        model2 = MLPAutoencoder(seed=42)
        
        # Weights should be identical with same seed
        assert torch.allclose(model1.encoder.weight, model2.encoder.weight)
        assert torch.allclose(model1.encoder.bias, model2.encoder.bias)
        assert torch.allclose(model1.decoder.weight, model2.decoder.weight)
        assert torch.allclose(model1.decoder.bias, model2.decoder.bias)
        
        # Test different seeds produce different weights
        model3 = MLPAutoencoder(seed=123)
        assert not torch.allclose(model1.encoder.weight, model3.encoder.weight)
    
    def test_model_repr(self):
        """Test string representation of model."""
        model = MLPAutoencoder()
        repr_str = repr(model)
        
        assert "MLPAutoencoder" in repr_str
        assert "input_size=784" in repr_str
        assert "hidden_size=400" in repr_str
        assert "parameters=" in repr_str


class TestModelUtils:
    """Test suite for model utilities."""
    
    def test_model_checkpoint_save_load(self, tmp_model_path):
        """Test model checkpoint saving and loading."""
        # Create and save model
        original_model = MLPAutoencoder(seed=42)
        metadata = {"experiment": "test", "version": "1.0"}
        training_info = {"epoch": 10, "loss": 0.05}
        
        checkpoint_manager = ModelCheckpoint()
        checkpoint_manager.save_checkpoint(
            original_model, tmp_model_path, metadata, training_info
        )
        
        # Load model
        loaded_model, checkpoint_metadata = checkpoint_manager.load_checkpoint(
            tmp_model_path
        )
        
        # Verify model architecture
        assert loaded_model.input_size == original_model.input_size
        assert loaded_model.hidden_size == original_model.hidden_size
        
        # Verify weights are identical
        assert torch.allclose(loaded_model.encoder.weight, original_model.encoder.weight)
        assert torch.allclose(loaded_model.encoder.bias, original_model.encoder.bias)
        assert torch.allclose(loaded_model.decoder.weight, original_model.decoder.weight)
        assert torch.allclose(loaded_model.decoder.bias, original_model.decoder.bias)
        
        # Verify metadata
        assert checkpoint_metadata["metadata"]["experiment"] == "test"
        assert checkpoint_metadata["training_info"]["epoch"] == 10
    
    def test_convenience_functions(self, tmp_model_path):
        """Test convenience functions for save/load."""
        model = MLPAutoencoder(seed=42)
        
        # Test save
        save_model(model, tmp_model_path, metadata={"test": "data"})
        
        # Test load
        loaded_model, metadata = load_model(tmp_model_path)
        
        assert torch.allclose(model.encoder.weight, loaded_model.encoder.weight)
        assert metadata["metadata"]["test"] == "data"
    
    def test_model_validation(self):
        """Test comprehensive model validation."""
        model = MLPAutoencoder(seed=42)
        
        # Test validation
        validation_results = validate_model(model)
        
        assert validation_results["overall_valid"] is True
        assert validation_results["architecture"]["valid"] is True
        assert validation_results["forward_pass"]["valid"] is True
        assert validation_results["gradient_flow"]["valid"] is True
    
    def test_model_validator_architecture(self):
        """Test architecture validation."""
        model = MLPAutoencoder()
        
        validation = ModelValidator.validate_architecture(model)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert "architecture_info" in validation
    
    def test_model_validator_forward_pass(self):
        """Test forward pass validation."""
        model = MLPAutoencoder()
        
        validation = ModelValidator.validate_forward_pass(model)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert "shapes" in validation
        assert validation["shapes"]["input"] == (4, 784)  # Default batch size
        assert validation["shapes"]["encoded"] == (4, 400)
        assert validation["shapes"]["output"] == (4, 784)
    
    def test_model_validator_gradient_flow(self):
        """Test gradient flow validation."""
        model = MLPAutoencoder()
        
        validation = ModelValidator.validate_gradient_flow(model)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert "gradients" in validation
        
        # All parameters should have gradients
        for param_name, has_grad in validation["gradients"].items():
            assert has_grad, f"No gradient for {param_name}"
    
    def test_model_summary_generation(self):
        """Test model summary generation."""
        model = MLPAutoencoder()
        
        # Test text summary
        summary = ModelSummary.generate_summary(model)
        assert "MLP Autoencoder Summary" in summary
        assert "784" in summary
        assert "400" in summary
        assert "tanh" in summary
        assert "sigmoid" in summary
        
        # Test parameter analysis
        analysis = ModelSummary.parameter_analysis(model)
        assert "total_parameters" in analysis
        assert "layer_breakdown" in analysis
        assert "weight_statistics" in analysis
        
        # Check layer breakdown
        assert "encoder" in analysis["layer_breakdown"]
        assert "decoder" in analysis["layer_breakdown"]
    
    def test_print_model_summary(self, capsys):
        """Test print model summary function."""
        model = MLPAutoencoder()
        print_model_summary(model)
        
        captured = capsys.readouterr()
        assert "MLP Autoencoder Summary" in captured.out
        assert "784" in captured.out
        assert "400" in captured.out


class TestModelIntegration:
    """Integration tests for complete model workflow."""
    
    def test_training_simulation(self, device):
        """Test a simulated training workflow."""
        model = MLPAutoencoder(device=device, seed=42)
        
        # Create sample data
        batch_size = 32
        sample_data = torch.randn(batch_size, 784, device=device) * 0.5 + 0.5
        sample_data = sample_data.clamp(0, 1)
        
        # Simulate training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        reconstructed = model(sample_data)
        loss = F.mse_loss(reconstructed, sample_data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss is reasonable
        assert loss.item() > 0
        assert torch.isfinite(loss)
        
        # Verify gradients were applied
        for param in model.parameters():
            assert param.grad is not None
    
    def test_inference_mode(self, device):
        """Test model in inference mode."""
        model = MLPAutoencoder(device=device, seed=42)
        
        # Create sample data
        sample_data = torch.randn(10, 784, device=device) * 0.5 + 0.5
        sample_data = sample_data.clamp(0, 1)
        
        # Test inference
        model.eval()
        with torch.no_grad():
            reconstructed = model(sample_data)
        
        assert reconstructed.shape == sample_data.shape
        assert not reconstructed.requires_grad
    
    def test_full_workflow_with_save_load(self, tmp_path):
        """Test complete workflow including save/load."""
        save_path = tmp_path / "test_model.pth"
        
        # Create and train model
        model = MLPAutoencoder(seed=42)
        sample_data = torch.randn(16, 784) * 0.5 + 0.5
        sample_data = sample_data.clamp(0, 1)
        
        # Get initial output
        model.eval()
        with torch.no_grad():
            initial_output = model(sample_data)
        
        # Save model
        save_model(model, save_path, metadata={"test": "workflow"})
        
        # Load model
        loaded_model, metadata = load_model(save_path)
        
        # Test loaded model produces same output
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(sample_data)
        
        assert torch.allclose(initial_output, loaded_output, atol=1e-6)
        assert metadata["metadata"]["test"] == "workflow"


class TestFactoryFunction:
    """Test factory function for creating models."""
    
    def test_create_mnist_autoencoder(self):
        """Test factory function for creating standard MNIST autoencoder."""
        model = create_mnist_autoencoder()
        
        assert isinstance(model, MLPAutoencoder)
        assert model.input_size == 784
        assert model.hidden_size == 400
    
    def test_create_mnist_autoencoder_with_params(self, device):
        """Test factory function with custom parameters."""
        model = create_mnist_autoencoder(device=device, seed=42)
        
        assert model.device == device
        assert next(model.parameters()).device == device
    
    def test_create_mnist_autoencoder_reproducibility(self):
        """Test reproducibility of factory function."""
        model1 = create_mnist_autoencoder(seed=42)
        model2 = create_mnist_autoencoder(seed=42)
        
        # Should have same weights with same seed
        assert torch.allclose(model1.encoder.weight, model2.encoder.weight)
        assert torch.allclose(model1.decoder.weight, model2.decoder.weight)


@pytest.fixture
def sample_model():
    """Fixture providing a sample model for testing."""
    return MLPAutoencoder(seed=42)


@pytest.fixture
def sample_data():
    """Fixture providing sample MNIST-like data."""
    batch_size = 8
    data = torch.randn(batch_size, 784) * 0.5 + 0.5
    return data.clamp(0, 1)


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input_validation(self, sample_model):
        """Test behavior with edge case inputs."""
        # Test minimum batch size
        single_sample = torch.randn(1, 784) * 0.5 + 0.5
        single_sample = single_sample.clamp(0, 1)
        
        output = sample_model(single_sample)
        assert output.shape == (1, 784)
    
    def test_large_batch_processing(self, sample_model):
        """Test model with large batch size."""
        large_batch = torch.randn(1000, 784) * 0.5 + 0.5
        large_batch = large_batch.clamp(0, 1)
        
        # Should handle large batches without issues
        output = sample_model(large_batch)
        assert output.shape == (1000, 784)
    
    def test_numerical_stability(self, sample_model):
        """Test numerical stability with extreme inputs."""
        # Test with all zeros (edge case)
        zeros_input = torch.zeros(4, 784)
        output = sample_model(zeros_input)
        assert torch.isfinite(output).all()
        
        # Test with all ones (edge case)
        ones_input = torch.ones(4, 784)
        output = sample_model(ones_input)
        assert torch.isfinite(output).all()
