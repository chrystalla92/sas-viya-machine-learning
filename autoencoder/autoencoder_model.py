"""
Autoencoder MLP Model Implementation for MNIST.

This module provides a modular autoencoder architecture with separate encoder
and decoder components, configurable activation functions, and proper weight
initialization for compatibility with various training scenarios.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from .model_utils import (
    get_activation_function,
    get_initialization_function,
    validate_model_config,
    move_model_to_device,
    get_model_summary,
    print_model_summary
)


class Encoder(nn.Module):
    """
    Encoder module for the autoencoder.
    
    Compresses input data from input_dim to latent_dim dimensions.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 400, activation: str = 'tanh'):
        """
        Initialize the encoder.
        
        Args:
            input_dim (int): Input dimension (default: 784 for MNIST)
            latent_dim (int): Latent dimension (default: 400)
            activation (str): Activation function name (default: 'tanh')
        """
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_name = activation
        
        # Build encoder layers
        self.linear = nn.Linear(input_dim, latent_dim)
        self.activation = get_activation_function(activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, latent_dim)
        """
        # Ensure input is properly flattened
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    """
    Decoder module for the autoencoder.
    
    Reconstructs data from latent_dim back to output_dim dimensions.
    """
    
    def __init__(self, latent_dim: int = 400, output_dim: int = 784, activation: str = 'tanh'):
        """
        Initialize the decoder.
        
        Args:
            latent_dim (int): Latent dimension (default: 400)
            output_dim (int): Output dimension (default: 784 for MNIST)
            activation (str): Activation function name (default: 'tanh')
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Build decoder layers
        self.linear = nn.Linear(latent_dim, output_dim)
        self.activation = get_activation_function(activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x (torch.Tensor): Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, output_dim)
        """
        x = self.linear(x)
        x = self.activation(x)
        return x


class AutoencoderMLP(nn.Module):
    """
    Multi-Layer Perceptron Autoencoder for MNIST data.
    
    Architecture: 784 → 400 → 784 with configurable activation functions
    and proper weight initialization.
    """
    
    def __init__(self, 
                 input_dim: int = 784,
                 latent_dim: int = 400,
                 activation: str = 'tanh',
                 init_type: str = 'uniform',
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the autoencoder model.
        
        Args:
            input_dim (int): Input dimension (default: 784 for MNIST)
            latent_dim (int): Latent dimension (default: 400)
            activation (str): Activation function name (default: 'tanh' for SAS compatibility)
            init_type (str): Weight initialization type ('uniform' or 'normal')
            device (Optional[Union[str, torch.device]]): Device to place model on
        """
        super(AutoencoderMLP, self).__init__()
        
        # Validate configuration
        validate_model_config(input_dim, latent_dim, activation)
        
        # Store configuration
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.init_type = init_type
        
        # Build encoder and decoder
        self.encoder = Encoder(input_dim, latent_dim, activation)
        self.decoder = Decoder(latent_dim, input_dim, activation)  # output_dim = input_dim for autoencoder
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device if specified
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.to(device)
        
    def _initialize_weights(self) -> None:
        """Initialize model weights based on activation function."""
        init_fn = get_initialization_function(self.activation_name, self.init_type)
        
        # Apply initialization to all modules
        self.apply(init_fn)
    
    def forward(self, x: torch.Tensor, return_latent: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim) or (batch_size, 28, 28)
            return_latent (bool): Whether to return latent representation along with reconstruction
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                If return_latent=True: (reconstruction, latent_representation)
                If return_latent=False: reconstruction only
        """
        # Ensure input is properly shaped
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(latent)
        
        if return_latent:
            return reconstruction, latent
        else:
            return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            latent (torch.Tensor): Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, input_dim)
        """
        return self.decoder(latent)
    
    def get_latent_dim(self) -> int:
        """Get the latent dimension size."""
        return self.latent_dim
    
    def get_input_dim(self) -> int:
        """Get the input dimension size."""
        return self.input_dim
    
    def get_config(self) -> dict:
        """
        Get model configuration.
        
        Returns:
            dict: Model configuration parameters
        """
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'activation': self.activation_name,
            'init_type': self.init_type,
            'device': str(next(self.parameters()).device)
        }
    
    def summary(self, input_shape: tuple = None) -> dict:
        """
        Get model summary.
        
        Args:
            input_shape (tuple): Input shape for testing (default: (1, input_dim))
            
        Returns:
            dict: Model summary information
        """
        if input_shape is None:
            input_shape = (1, self.input_dim)
        
        return get_model_summary(self, input_shape)
    
    def print_summary(self, input_shape: tuple = None) -> None:
        """
        Print formatted model summary.
        
        Args:
            input_shape (tuple): Input shape for testing (default: (1, input_dim))
        """
        if input_shape is None:
            input_shape = (1, self.input_dim)
        
        print_model_summary(self, input_shape)
    
    def to_device(self, device: Optional[Union[str, torch.device]] = None) -> 'AutoencoderMLP':
        """
        Move model to specified device or auto-detect best available device.
        
        Args:
            device (Optional[Union[str, torch.device]]): Target device. If None, auto-detect.
            
        Returns:
            AutoencoderMLP: Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        move_model_to_device(self, device)
        return self
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = False, 
                       optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath (str): Path to save checkpoint
            include_optimizer (bool): Whether to include optimizer state
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'model_class': self.__class__.__name__
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['optimizer_class'] = optimizer.__class__.__name__
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: Optional[Union[str, torch.device]] = None) -> 'AutoencoderMLP':
        """
        Load model from checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
            device (Optional[Union[str, torch.device]]): Device to load model on
            
        Returns:
            AutoencoderMLP: Loaded model
        """
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['config']
        
        # Create model with saved configuration
        model = cls(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim'],
            activation=config['activation'],
            init_type=config.get('init_type', 'uniform'),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


# Factory function for easy model creation
def create_mnist_autoencoder(latent_dim: int = 400,
                           activation: str = 'tanh',
                           init_type: str = 'uniform',
                           device: Optional[Union[str, torch.device]] = None) -> AutoencoderMLP:
    """
    Factory function to create MNIST autoencoder with default parameters.
    
    Args:
        latent_dim (int): Latent dimension (default: 400)
        activation (str): Activation function (default: 'tanh')
        init_type (str): Weight initialization type (default: 'uniform')
        device (Optional[Union[str, torch.device]]): Device to place model on
        
    Returns:
        AutoencoderMLP: Configured autoencoder model
    """
    return AutoencoderMLP(
        input_dim=784,  # MNIST standard
        latent_dim=latent_dim,
        activation=activation,
        init_type=init_type,
        device=device
    )
