"""
PyTorch Autoencoder Model

This module implements a PyTorch autoencoder that matches the SAS neural network
configuration for MNIST data processing. The architecture mirrors the SAS
implementation with proper layer dimensions and activation functions.

Architecture:
- Input layer: 784 neurons (28x28 flattened MNIST images)
- Hidden layer: 400 neurons with tanh activation  
- Output layer: 784 neurons (reconstruction)

Key features:
- MSE reconstruction loss
- Compatible with L-BFGS optimizer
- Proper weight initialization matching SAS behavior
- Support for midrange-standardized input data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class MNISTAutoencoder(nn.Module):
    """
    PyTorch implementation of MNIST autoencoder matching SAS nnet/CAS configuration.
    
    This autoencoder implements the same architecture as the SAS implementation:
    - MLP architecture (multilayer perceptron) 
    - Hidden layer with 400 neurons and tanh activation
    - Input and output layers with 784 neurons each
    - Suitable for reconstruction of 28x28 MNIST images
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, 
                 dropout_rate: float = 0.0, seed: Optional[int] = 23451):
        """
        Initialize the autoencoder architecture.
        
        Args:
            input_dim: Input dimension (default 784 for flattened MNIST)
            hidden_dim: Hidden layer dimension (default 400 to match SAS)
            dropout_rate: Dropout rate for regularization (default 0.0, matching SAS)
            seed: Random seed for weight initialization (default 23451 to match SAS)
        """
        super(MNISTAutoencoder, self).__init__()
        
        # Store architecture parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Set random seed for reproducible initialization
        if seed is not None:
            torch.manual_seed(seed)
            
        # Define encoder (input -> hidden)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Define decoder (hidden -> output)  
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Dropout layer (if specified)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights to match SAS behavior
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights to match SAS neural network initialization.
        
        SAS uses uniform distribution for weight initialization with scaleInit=1.
        This method approximates that behavior.
        """
        # Initialize encoder weights
        nn.init.uniform_(self.encoder.weight, -1.0, 1.0)
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder weights  
        nn.init.uniform_(self.decoder.weight, -1.0, 1.0)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Hidden representation of shape (batch_size, hidden_dim)
        """
        # Linear transformation followed by tanh activation
        hidden = torch.tanh(self.encoder(x))
        
        # Apply dropout if specified
        if self.dropout is not None:
            hidden = self.dropout(hidden)
            
        return hidden
    
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to output.
        
        Args:
            hidden: Hidden tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Reconstructed output of shape (batch_size, input_dim)
        """
        # Linear transformation (no activation on output layer)
        output = self.decoder(hidden)
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed output tensor of shape (batch_size, input_dim)
        """
        # Encode input to hidden representation
        hidden = self.encode(x)
        
        # Decode hidden representation to output
        reconstructed = self.decode(hidden)
        
        return reconstructed
    
    def reconstruction_loss(self, input_data: torch.Tensor, 
                          reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE reconstruction loss matching SAS behavior.
        
        Args:
            input_data: Original input tensor
            reconstructed: Reconstructed output tensor
            
        Returns:
            MSE loss tensor
        """
        return F.mse_loss(reconstructed, input_data, reduction='mean')
    
    def get_architecture_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing architecture details
        """
        return {
            'type': 'MLP Autoencoder',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim, 
            'output_dim': self.input_dim,
            'activation': 'tanh',
            'dropout_rate': self.dropout_rate,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model_state(self, filepath: str, epoch: int = 0, 
                        loss: float = 0.0, optimizer_state: Optional[dict] = None):
        """
        Save model state for checkpointing.
        
        Args:
            filepath: Path to save the model state
            epoch: Current epoch number
            loss: Current loss value
            optimizer_state: Optimizer state dict (optional)
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'architecture': self.get_architecture_info()
        }
        
        if optimizer_state is not None:
            state['optimizer_state_dict'] = optimizer_state
            
        torch.save(state, filepath)
    
    @classmethod
    def load_model_state(cls, filepath: str, map_location: Optional[str] = None) -> Tuple['MNISTAutoencoder', dict]:
        """
        Load model state from checkpoint.
        
        Args:
            filepath: Path to the saved model state
            map_location: Device to map tensors to
            
        Returns:
            Tuple of (model, state_info) where state_info contains epoch, loss, etc.
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Extract architecture info
        arch_info = checkpoint.get('architecture', {})
        
        # Create model with same architecture
        model = cls(
            input_dim=arch_info.get('input_dim', 784),
            hidden_dim=arch_info.get('hidden_dim', 400),
            dropout_rate=arch_info.get('dropout_rate', 0.0)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return model and metadata
        state_info = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None)
        }
        
        return model, state_info


def create_sas_compatible_autoencoder(seed: Optional[int] = 23451) -> MNISTAutoencoder:
    """
    Create autoencoder with exact SAS configuration.
    
    This function creates an autoencoder that matches the SAS nnet/CAS
    implementation exactly:
    - Input: 784 neurons (var2-var785)
    - Hidden: 400 neurons with tanh activation
    - Output: 784 neurons (reconstruction)
    - Weight initialization: uniform distribution with seed=23451
    
    Args:
        seed: Random seed for reproducible results (default 23451, matching SAS)
        
    Returns:
        Configured MNISTAutoencoder instance
    """
    return MNISTAutoencoder(
        input_dim=784,
        hidden_dim=400,
        dropout_rate=0.0,  # No dropout in SAS implementation
        seed=seed
    )


def test_model_architecture():
    """Test the model architecture with mock data."""
    print("=== Testing MNIST Autoencoder Architecture ===")
    
    # Create model
    model = create_sas_compatible_autoencoder()
    
    # Print architecture info
    arch_info = model.get_architecture_info()
    print("Architecture Information:")
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass with mock data
    batch_size = 10
    mock_input = torch.randn(batch_size, 784)
    
    print(f"\nTesting forward pass:")
    print(f"Input shape: {mock_input.shape}")
    
    with torch.no_grad():
        # Test encoding
        encoded = model.encode(mock_input)
        print(f"Encoded shape: {encoded.shape}")
        
        # Test decoding
        decoded = model.decode(encoded)
        print(f"Decoded shape: {decoded.shape}")
        
        # Test full forward pass
        reconstructed = model(mock_input)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Test loss calculation
        loss = model.reconstruction_loss(mock_input, reconstructed)
        print(f"Reconstruction loss: {loss.item():.6f}")
    
    print("✓ Model architecture test completed successfully!")


if __name__ == "__main__":
    test_model_architecture()
