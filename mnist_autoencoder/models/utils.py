"""
Model utilities for saving, loading, and validation of MNIST autoencoders.

This module provides comprehensive utilities for model persistence, validation,
and analysis to support the MLPAutoencoder implementation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from .autoencoder import MLPAutoencoder


class ModelCheckpoint:
    """
    Model checkpoint manager with metadata tracking.
    
    Handles saving and loading of model states with comprehensive metadata
    including architecture details, training information, and validation metrics.
    """
    
    def __init__(self, save_dir: Union[str, Path] = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: MLPAutoencoder,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            filepath: Path to save checkpoint
            metadata: Additional metadata to store
            training_info: Training information (epoch, loss, etc.)
            
        Raises:
            RuntimeError: If save operation fails
        """
        filepath = Path(filepath)
        
        try:
            # Prepare checkpoint data
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_info": model.get_model_info(),
                "save_timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "model_class": model.__class__.__name__
            }
            
            # Add optional metadata
            if metadata:
                checkpoint["metadata"] = metadata
            if training_info:
                checkpoint["training_info"] = training_info
            
            # Save checkpoint
            torch.save(checkpoint, filepath)
            
            # Also save human-readable metadata as JSON
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    "model_info": checkpoint["model_info"],
                    "save_timestamp": checkpoint["save_timestamp"],
                    "pytorch_version": checkpoint["pytorch_version"],
                    "model_class": checkpoint["model_class"],
                    "metadata": metadata or {},
                    "training_info": training_info or {}
                }, f, indent=2)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint to {filepath}: {e}")
    
    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True
    ) -> Tuple[MLPAutoencoder, Dict[str, Any]]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            Tuple of (loaded_model, checkpoint_metadata)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If loading fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            
            # Extract model information
            model_info = checkpoint["model_info"]
            
            # Create model with saved architecture
            model = MLPAutoencoder(
                input_size=model_info["input_size"],
                hidden_size=model_info["hidden_size"],
                device=device
            )
            
            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            
            # Return model and metadata
            metadata = {
                "model_info": model_info,
                "save_timestamp": checkpoint.get("save_timestamp"),
                "pytorch_version": checkpoint.get("pytorch_version"),
                "metadata": checkpoint.get("metadata", {}),
                "training_info": checkpoint.get("training_info", {})
            }
            
            return model, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {filepath}: {e}")


class ModelValidator:
    """
    Comprehensive model validation utilities.
    
    Provides methods to validate model architecture, integrity, and functionality
    with detailed reporting and error handling.
    """
    
    @staticmethod
    def validate_architecture(model: MLPAutoencoder) -> Dict[str, Any]:
        """
        Validate model architecture matches expected specifications.
        
        Args:
            model: Model to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "architecture_info": {}
        }
        
        try:
            # Check basic architecture
            if not isinstance(model, MLPAutoencoder):
                validation_results["errors"].append(
                    f"Expected MLPAutoencoder, got {type(model)}"
                )
                validation_results["valid"] = False
            
            # Check layer structure
            expected_layers = ["encoder", "decoder"]
            for layer_name in expected_layers:
                if not hasattr(model, layer_name):
                    validation_results["errors"].append(
                        f"Missing layer: {layer_name}"
                    )
                    validation_results["valid"] = False
            
            # Check encoder architecture
            if hasattr(model, 'encoder'):
                encoder = model.encoder
                if not isinstance(encoder, nn.Linear):
                    validation_results["errors"].append(
                        f"Encoder should be nn.Linear, got {type(encoder)}"
                    )
                    validation_results["valid"] = False
                else:
                    # Check dimensions
                    if encoder.in_features != 784:
                        validation_results["errors"].append(
                            f"Encoder input size should be 784, got {encoder.in_features}"
                        )
                        validation_results["valid"] = False
                    
                    if encoder.out_features != 400:
                        validation_results["errors"].append(
                            f"Encoder output size should be 400, got {encoder.out_features}"
                        )
                        validation_results["valid"] = False
            
            # Check decoder architecture
            if hasattr(model, 'decoder'):
                decoder = model.decoder
                if not isinstance(decoder, nn.Linear):
                    validation_results["errors"].append(
                        f"Decoder should be nn.Linear, got {type(decoder)}"
                    )
                    validation_results["valid"] = False
                else:
                    # Check dimensions
                    if decoder.in_features != 400:
                        validation_results["errors"].append(
                            f"Decoder input size should be 400, got {decoder.in_features}"
                        )
                        validation_results["valid"] = False
                    
                    if decoder.out_features != 784:
                        validation_results["errors"].append(
                            f"Decoder output size should be 784, got {decoder.out_features}"
                        )
                        validation_results["valid"] = False
            
            # Store architecture info
            if validation_results["valid"]:
                validation_results["architecture_info"] = model.get_model_info()
            
        except Exception as e:
            validation_results["errors"].append(f"Validation failed with exception: {e}")
            validation_results["valid"] = False
        
        return validation_results
    
    @staticmethod
    def validate_forward_pass(model: MLPAutoencoder, batch_size: int = 4) -> Dict[str, Any]:
        """
        Validate model forward pass with sample data.
        
        Args:
            model: Model to validate
            batch_size: Size of test batch
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "shapes": {}
        }
        
        try:
            # Create sample input
            sample_input = torch.randn(batch_size, 784, device=model.device) * 0.5 + 0.5
            sample_input = sample_input.clamp(0, 1)  # Ensure valid range
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(sample_input)
            
            # Validate output shape
            if output.shape != sample_input.shape:
                validation_results["errors"].append(
                    f"Output shape {output.shape} doesn't match input shape {sample_input.shape}"
                )
                validation_results["valid"] = False
            
            # Validate output range
            if output.min() < -0.1 or output.max() > 1.1:
                validation_results["warnings"].append(
                    f"Output values outside expected [0,1] range: [{output.min():.3f}, {output.max():.3f}]"
                )
            
            # Test individual components
            encoded = model.encode(sample_input)
            if encoded.shape != (batch_size, model.hidden_size):
                validation_results["errors"].append(
                    f"Encoded shape {encoded.shape} doesn't match expected ({batch_size}, {model.hidden_size})"
                )
                validation_results["valid"] = False
            
            decoded = model.decode(encoded)
            if decoded.shape != sample_input.shape:
                validation_results["errors"].append(
                    f"Decoded shape {decoded.shape} doesn't match input shape {sample_input.shape}"
                )
                validation_results["valid"] = False
            
            # Store shape information
            validation_results["shapes"] = {
                "input": tuple(sample_input.shape),
                "encoded": tuple(encoded.shape),
                "output": tuple(output.shape)
            }
            
        except Exception as e:
            validation_results["errors"].append(f"Forward pass validation failed: {e}")
            validation_results["valid"] = False
        
        return validation_results
    
    @staticmethod
    def validate_gradient_flow(model: MLPAutoencoder, batch_size: int = 4) -> Dict[str, Any]:
        """
        Validate gradient flow through the model.
        
        Args:
            model: Model to validate
            batch_size: Size of test batch
            
        Returns:
            Dictionary with gradient validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "gradients": {}
        }
        
        try:
            # Create sample data
            sample_input = torch.randn(batch_size, 784, device=model.device) * 0.5 + 0.5
            sample_input = sample_input.clamp(0, 1)
            
            # Test gradient flow
            gradient_status = model.verify_gradient_flow(sample_input)
            
            # Check all parameters have gradients
            for param_name, has_grad in gradient_status.items():
                if not has_grad:
                    validation_results["errors"].append(
                        f"No gradient flow detected for parameter: {param_name}"
                    )
                    validation_results["valid"] = False
            
            validation_results["gradients"] = gradient_status
            
        except Exception as e:
            validation_results["errors"].append(f"Gradient validation failed: {e}")
            validation_results["valid"] = False
        
        return validation_results
    
    @staticmethod
    def comprehensive_validation(model: MLPAutoencoder) -> Dict[str, Any]:
        """
        Run comprehensive model validation.
        
        Args:
            model: Model to validate
            
        Returns:
            Dictionary with complete validation results
        """
        results = {
            "overall_valid": True,
            "architecture": ModelValidator.validate_architecture(model),
            "forward_pass": ModelValidator.validate_forward_pass(model),
            "gradient_flow": ModelValidator.validate_gradient_flow(model)
        }
        
        # Check overall validity
        for category in ["architecture", "forward_pass", "gradient_flow"]:
            if not results[category]["valid"]:
                results["overall_valid"] = False
                break
        
        return results


class ModelSummary:
    """
    Generate detailed model summaries and statistics.
    """
    
    @staticmethod
    def generate_summary(model: MLPAutoencoder) -> str:
        """
        Generate a detailed text summary of the model.
        
        Args:
            model: Model to summarize
            
        Returns:
            Formatted summary string
        """
        info = model.get_model_info()
        
        summary = f"""
═══════════════════════════════════════════════════════════════════════════════
                              MLP Autoencoder Summary
═══════════════════════════════════════════════════════════════════════════════

Architecture: {info['architecture']}
Input Size:   {info['input_size']} features (28×28 MNIST images flattened)
Hidden Size:  {info['hidden_size']} neurons
Device:       {info['device']}

Network Structure:
┌─────────────┬──────────────┬─────────────┬──────────────────┐
│ Layer       │ Input Shape  │ Output Shape│ Activation       │
├─────────────┼──────────────┼─────────────┼──────────────────┤
│ Encoder     │ ({info['input_size']},)       │ ({info['hidden_size']},)       │ tanh             │
│ Decoder     │ ({info['hidden_size']},)       │ ({info['input_size']},)       │ sigmoid          │
└─────────────┴──────────────┴─────────────┴──────────────────┘

Parameters:
  Total:      {info['total_parameters']:,} parameters
  Trainable:  {info['trainable_parameters']:,} parameters

Parameter Breakdown:
  Encoder:    {model.encoder.weight.numel() + model.encoder.bias.numel():,} parameters
              └─ Weights: {model.encoder.weight.numel():,} ({info['input_size']} × {info['hidden_size']})
              └─ Biases:  {model.encoder.bias.numel():,}
  
  Decoder:    {model.decoder.weight.numel() + model.decoder.bias.numel():,} parameters  
              └─ Weights: {model.decoder.weight.numel():,} ({info['hidden_size']} × {info['input_size']})
              └─ Biases:  {model.decoder.bias.numel():,}

Memory Usage (estimated):
  Model Size: ~{(info['total_parameters'] * 4) / (1024**2):.2f} MB (float32)

═══════════════════════════════════════════════════════════════════════════════
"""
        return summary
    
    @staticmethod
    def parameter_analysis(model: MLPAutoencoder) -> Dict[str, Any]:
        """
        Analyze model parameters in detail.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with parameter analysis
        """
        analysis = {
            "total_parameters": model.count_parameters(),
            "trainable_parameters": model.count_parameters(trainable_only=True),
            "layer_breakdown": {},
            "weight_statistics": {}
        }
        
        # Analyze each layer
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            param_type = name.split('.')[-1]
            
            if layer_name not in analysis["layer_breakdown"]:
                analysis["layer_breakdown"][layer_name] = {}
            
            analysis["layer_breakdown"][layer_name][param_type] = {
                "shape": list(param.shape),
                "parameters": param.numel(),
                "requires_grad": param.requires_grad,
                "dtype": str(param.dtype),
                "device": str(param.device)
            }
            
            # Weight statistics
            analysis["weight_statistics"][name] = {
                "mean": float(param.data.mean()),
                "std": float(param.data.std()),
                "min": float(param.data.min()),
                "max": float(param.data.max()),
                "norm": float(param.data.norm())
            }
        
        return analysis


# Convenience functions
def save_model(
    model: MLPAutoencoder,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model with metadata (convenience function).
    
    Args:
        model: Model to save
        filepath: Save path
        metadata: Additional metadata
        training_info: Training information
    """
    checkpoint_manager = ModelCheckpoint()
    checkpoint_manager.save_checkpoint(model, filepath, metadata, training_info)


def load_model(
    filepath: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[MLPAutoencoder, Dict[str, Any]]:
    """
    Load model from checkpoint (convenience function).
    
    Args:
        filepath: Path to checkpoint
        device: Target device
        
    Returns:
        Tuple of (model, metadata)
    """
    checkpoint_manager = ModelCheckpoint()
    return checkpoint_manager.load_checkpoint(filepath, device)


def validate_model(model: MLPAutoencoder) -> Dict[str, Any]:
    """
    Run comprehensive model validation (convenience function).
    
    Args:
        model: Model to validate
        
    Returns:
        Validation results
    """
    return ModelValidator.comprehensive_validation(model)


def print_model_summary(model: MLPAutoencoder) -> None:
    """
    Print detailed model summary (convenience function).
    
    Args:
        model: Model to summarize
    """
    print(ModelSummary.generate_summary(model))
