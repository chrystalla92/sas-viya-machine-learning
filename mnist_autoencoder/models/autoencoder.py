"""
MLP Autoencoder implementation for MNIST digit reconstruction.

This module implements a Multi-Layer Perceptron (MLP) autoencoder architecture
that matches the SAS PROC NNET implementation with 784→400→784 structure.
"""

from typing import Dict, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPAutoencoder(nn.Module):
    """
    Multi-Layer Perceptron Autoencoder for MNIST digit reconstruction.
    
    Architecture:
    - Input: 784 features (28x28 MNIST images flattened)
    - Encoder: Linear(784, 400) → tanh activation
    - Decoder: Linear(400, 784) → sigmoid activation
    
    This implementation matches the SAS PROC NNET architecture with proper
    weight initialization and device management.
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 400,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize MLP Autoencoder.
        
        Args:
            input_size: Size of input layer (default: 784 for MNIST)
            hidden_size: Size of hidden layer (default: 400)
            device: Device to run model on ('cpu', 'cuda', or torch.device)
            seed: Random seed for weight initialization
            
        Raises:
            ValueError: If input_size or hidden_size are not positive integers
        """
        super(MLPAutoencoder, self).__init__()
        
        # Validate inputs
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seed for reproducible initialization
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Define network layers
        # Encoder: 784 → 400 with tanh activation
        self.encoder = nn.Linear(input_size, hidden_size)
        
        # Decoder: 400 → 784 with sigmoid activation (for [0,1] outputs)
        self.decoder = nn.Linear(hidden_size, input_size)
        
        # Initialize weights with SAS-compatible method
        self._initialize_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights to match SAS PROC NNET defaults.
        
        SAS PROC NNET typically uses Xavier/Glorot initialization for neural networks.
        This provides good initial weights for tanh and sigmoid activations.
        """
        # Xavier initialization for encoder (good for tanh activation)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Xavier initialization for decoder (good for sigmoid activation)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Encoded representation of shape (batch_size, hidden_size)
            
        Raises:
            ValueError: If input shape is invalid
        """
        self._validate_input(x)
        return torch.tanh(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to output.
        
        Args:
            z: Hidden representation of shape (batch_size, hidden_size)
            
        Returns:
            Reconstructed output of shape (batch_size, input_size)
            
        Raises:
            ValueError: If hidden representation shape is invalid
        """
        if z.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {z.size(-1)}"
            )
        return torch.sigmoid(self.decoder(z))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder-decoder pipeline.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Reconstructed output of same shape as input
            
        Raises:
            ValueError: If input shape is invalid
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape and properties.
        
        Args:
            x: Input tensor to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D (batch_size, features), got {x.dim()}D")
        
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Input size {x.size(-1)} doesn't match expected {self.input_size}"
            )
        
        # Check for valid pixel values (assuming normalized data)
        if x.min() < -1.1 or x.max() > 1.1:
            raise ValueError(
                f"Input values out of expected range [-1.1, 1.1]: [{x.min():.3f}, {x.max():.3f}]"
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information and metadata.
        
        Returns:
            Dictionary containing model architecture and parameters
        """
        total_params = self.count_parameters()
        trainable_params = self.count_parameters(trainable_only=True)
        
        return {
            "architecture": "MLPAutoencoder",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "encoder_shape": f"{self.input_size} → {self.hidden_size}",
            "decoder_shape": f"{self.hidden_size} → {self.input_size}",
            "activation_functions": {
                "encoder": "tanh",
                "decoder": "sigmoid"
            }
        }
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def verify_gradient_flow(self, x: torch.Tensor) -> Dict[str, bool]:
        """
        Verify that gradients flow properly through the network.
        
        Args:
            x: Sample input tensor
            
        Returns:
            Dictionary indicating gradient flow status for each layer
        """
        # Ensure model is in training mode and gradients are enabled
        self.train()
        
        # Forward pass
        output = self.forward(x)
        loss = F.mse_loss(output, x)  # Simple reconstruction loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        gradient_status = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                has_gradient = param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad))
                gradient_status[name] = has_gradient
        
        return gradient_status
    
    def to_device(self, device: Union[str, torch.device]) -> 'MLPAutoencoder':
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
            
        Returns:
            Model on target device
        """
        self.device = torch.device(device)
        return self.to(self.device)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"MLPAutoencoder(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  device={self.device},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


def create_mnist_autoencoder(
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None
) -> MLPAutoencoder:
    """
    Create a standard MNIST autoencoder with default parameters.
    
    Args:
        device: Device to run model on
        seed: Random seed for initialization
        
    Returns:
        Initialized MLPAutoencoder instance
    """
    return MLPAutoencoder(
        input_size=784,
        hidden_size=400,
        device=device,
        seed=seed
    )
