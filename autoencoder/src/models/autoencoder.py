"""
MLP Autoencoder implementation matching SAS Viya specifications.

This module implements a simple multilayer perceptron autoencoder with
784→400→784 architecture, tanh activation for the hidden layer, and
uniform weight initialization matching SAS parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class Autoencoder(nn.Module):
    """
    MLP Autoencoder with 784→400→784 architecture.
    
    This implementation matches the SAS Viya annTrain configuration:
    - hiddens={400}: Single hidden layer with 400 neurons
    - acts={'tanh'}: Tanh activation for hidden layer
    - arch='mlp': Multilayer perceptron architecture
    - randDist='uniform': Uniform weight initialization
    - scaleInit=1: Weight initialization scale
    - seed=23451: Random seed for reproducibility
    
    Architecture:
        Input (784) → Encoder (784→400, tanh) → Decoder (400→784, linear) → Output (784)
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, seed: int = 23451):
        """
        Initialize the MLP Autoencoder.
        
        Args:
            input_dim: Input dimension (784 for MNIST)
            hidden_dim: Hidden/latent dimension (400 as per SAS spec)
            seed: Random seed for weight initialization (23451 as per SAS spec)
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        
        # Define layers
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights with SAS-compatible uniform distribution
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using uniform distribution matching SAS specifications.
        
        SAS uses:
        - randDist='uniform': Uniform random distribution
        - scaleInit=1: Weight initialization scale factor
        - seed=23451: Random seed
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Initialize encoder weights uniformly between [-scaleInit, scaleInit]
        # where scaleInit = 1 as per SAS configuration
        scale_init = 1.0
        
        with torch.no_grad():
            # Encoder layer initialization
            nn.init.uniform_(self.encoder.weight, -scale_init, scale_init)
            nn.init.uniform_(self.encoder.bias, -scale_init, scale_init)
            
            # Decoder layer initialization  
            nn.init.uniform_(self.decoder.weight, -scale_init, scale_init)
            nn.init.uniform_(self.decoder.bias, -scale_init, scale_init)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, 784)
            
        Returns:
            Latent representation tensor of shape (batch_size, 400)
        """
        # Apply encoder with tanh activation (as per SAS acts={'tanh'})
        return torch.tanh(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            z: Latent tensor of shape (batch_size, 400)
            
        Returns:
            Reconstructed tensor of shape (batch_size, 784)
        """
        # Apply decoder with linear activation (no activation function)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, 784)
            
        Returns:
            Reconstructed tensor of shape (batch_size, 784)
        """
        # Encode input to latent space
        latent = self.encode(x)
        # Decode latent back to input space
        reconstructed = self.decode(latent)
        return reconstructed
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without decoding (alias for encode).
        
        Args:
            x: Input tensor of shape (batch_size, 784)
            
        Returns:
            Latent representation tensor of shape (batch_size, 400)
        """
        return self.encode(x)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters()) 
        total_params = encoder_params + decoder_params
        
        return {
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'total_parameters': total_params,
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get architecture summary information.
        
        Returns:
            Dictionary with architecture details
        """
        param_counts = self.count_parameters()
        
        return {
            'architecture': 'MLP Autoencoder',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.input_dim,
            'layers': [
                f'Encoder: Linear({self.input_dim} → {self.hidden_dim}) + Tanh',
                f'Decoder: Linear({self.hidden_dim} → {self.input_dim}) + Linear'
            ],
            'activation_functions': {
                'encoder': 'tanh',
                'decoder': 'linear'
            },
            'weight_initialization': {
                'distribution': 'uniform',
                'range': '[-1.0, 1.0]',
                'seed': self.seed
            },
            **param_counts
        }
    
    def to_device(self, device: torch.device) -> 'Autoencoder':
        """
        Move model to specified device.
        
        Args:
            device: Target device (CPU or GPU)
            
        Returns:
            Self for method chaining
        """
        return self.to(device)
    
    def get_device(self) -> torch.device:
        """
        Get the device the model is currently on.
        
        Returns:
            Current device
        """
        return next(self.parameters()).device
    
    def print_architecture(self):
        """Print a detailed architecture summary."""
        summary = self.get_architecture_summary()
        
        print("=" * 60)
        print(f"Model Architecture: {summary['architecture']}")
        print("=" * 60)
        print(f"Input Dimension:    {summary['input_dim']}")
        print(f"Hidden Dimension:   {summary['hidden_dim']}")
        print(f"Output Dimension:   {summary['output_dim']}")
        print("\nLayer Details:")
        for i, layer in enumerate(summary['layers'], 1):
            print(f"  {i}. {layer}")
        
        print(f"\nActivation Functions:")
        for layer, activation in summary['activation_functions'].items():
            print(f"  {layer.capitalize()}: {activation}")
        
        print(f"\nWeight Initialization:")
        init_info = summary['weight_initialization']
        print(f"  Distribution: {init_info['distribution']}")
        print(f"  Range: {init_info['range']}")
        print(f"  Seed: {init_info['seed']}")
        
        print(f"\nParameter Counts:")
        print(f"  Encoder Parameters:    {summary['encoder_parameters']:,}")
        print(f"  Decoder Parameters:    {summary['decoder_parameters']:,}")
        print(f"  Total Parameters:      {summary['total_parameters']:,}")
        print(f"  Trainable Parameters:  {summary['trainable_parameters']:,}")
        print("=" * 60)


def create_autoencoder(input_dim: int = 784, hidden_dim: int = 400, 
                      seed: int = 23451, device: Optional[str] = None) -> Autoencoder:
    """
    Factory function to create and initialize an autoencoder.
    
    Args:
        input_dim: Input dimension (default: 784 for MNIST)
        hidden_dim: Hidden dimension (default: 400 as per SAS spec)
        seed: Random seed (default: 23451 as per SAS spec)
        device: Device to place model on ('cpu', 'cuda', or None for auto)
        
    Returns:
        Initialized Autoencoder model
    """
    # Create model
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, seed=seed)
    
    # Handle device placement
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device_obj = torch.device(device)
    model = model.to(device_obj)
    
    print(f"Created MLP Autoencoder on device: {device_obj}")
    model.print_architecture()
    
    return model
