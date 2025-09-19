"""
Model utilities for autoencoder initialization and analysis.

This module provides utility functions for proper weight initialization,
model parameter counting, and model summary functionality.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def xavier_uniform_init(module: nn.Module) -> None:
    """
    Apply Xavier uniform initialization to linear layers.
    
    Best for tanh and sigmoid activations.
    
    Args:
        module (nn.Module): Module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def xavier_normal_init(module: nn.Module) -> None:
    """
    Apply Xavier normal initialization to linear layers.
    
    Best for tanh and sigmoid activations.
    
    Args:
        module (nn.Module): Module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def kaiming_uniform_init(module: nn.Module) -> None:
    """
    Apply Kaiming uniform initialization to linear layers.
    
    Best for ReLU and its variants.
    
    Args:
        module (nn.Module): Module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def kaiming_normal_init(module: nn.Module) -> None:
    """
    Apply Kaiming normal initialization to linear layers.
    
    Best for ReLU and its variants.
    
    Args:
        module (nn.Module): Module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_activation_function(activation_name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation_name (str): Name of activation function
        
    Returns:
        nn.Module: Activation function
        
    Raises:
        ValueError: If activation name is not supported
    """
    activation_map = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),  # SiLU is the same as Swish
        'none': nn.Identity(),
        'identity': nn.Identity()
    }
    
    if activation_name.lower() not in activation_map:
        supported = ', '.join(activation_map.keys())
        raise ValueError(f"Unsupported activation '{activation_name}'. Supported: {supported}")
    
    return activation_map[activation_name.lower()]


def get_initialization_function(activation_name: str, init_type: str = 'uniform') -> callable:
    """
    Get appropriate initialization function for an activation function.
    
    Args:
        activation_name (str): Name of activation function
        init_type (str): Type of initialization ('uniform' or 'normal')
        
    Returns:
        callable: Initialization function
        
    Raises:
        ValueError: If activation or init_type is not supported
    """
    # Activations that benefit from Xavier initialization
    xavier_activations = ['tanh', 'sigmoid', 'none', 'identity']
    # Activations that benefit from Kaiming initialization  
    kaiming_activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish']
    
    activation_lower = activation_name.lower()
    
    if activation_lower in xavier_activations:
        if init_type == 'uniform':
            return xavier_uniform_init
        elif init_type == 'normal':
            return xavier_normal_init
        else:
            raise ValueError(f"Unsupported init_type '{init_type}'. Use 'uniform' or 'normal'")
    
    elif activation_lower in kaiming_activations:
        if init_type == 'uniform':
            return kaiming_uniform_init
        elif init_type == 'normal':
            return kaiming_normal_init
        else:
            raise ValueError(f"Unsupported init_type '{init_type}'. Use 'uniform' or 'normal'")
    
    else:
        supported = xavier_activations + kaiming_activations
        raise ValueError(f"Unsupported activation '{activation_name}'. Supported: {supported}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        trainable_only (bool): If True, count only trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module, input_shape: tuple = (1, 784)) -> Dict[str, Any]:
    """
    Get a comprehensive summary of model architecture and parameters.
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape for testing
        
    Returns:
        Dict[str, Any]: Model summary information
    """
    # Count parameters
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Get device info
    device = next(model.parameters()).device
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        # Handle different output formats
        if isinstance(output, tuple):
            if len(output) == 2:
                reconstruction, latent = output
                output_info = {
                    'reconstruction_shape': tuple(reconstruction.shape),
                    'latent_shape': tuple(latent.shape)
                }
            else:
                output_info = {'output_shapes': [tuple(o.shape) for o in output]}
        else:
            output_info = {'output_shape': tuple(output.shape)}
            
    except Exception as e:
        output_info = {'forward_pass_error': str(e)}
    
    summary = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'device': str(device),
        'input_shape': input_shape,
        **output_info
    }
    
    return summary


def print_model_summary(model: nn.Module, input_shape: tuple = (1, 784)) -> None:
    """
    Print a formatted model summary.
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape for testing
    """
    summary = get_model_summary(model, input_shape)
    
    print("=" * 60)
    print(f"Model Summary: {summary['model_name']}")
    print("=" * 60)
    
    print(f"Input Shape:              {summary['input_shape']}")
    
    if 'reconstruction_shape' in summary:
        print(f"Reconstruction Shape:     {summary['reconstruction_shape']}")
        print(f"Latent Shape:             {summary['latent_shape']}")
    elif 'output_shape' in summary:
        print(f"Output Shape:             {summary['output_shape']}")
    elif 'output_shapes' in summary:
        for i, shape in enumerate(summary['output_shapes']):
            print(f"Output {i+1} Shape:          {shape}")
    
    print(f"Total Parameters:         {summary['total_parameters']:,}")
    print(f"Trainable Parameters:     {summary['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {summary['non_trainable_parameters']:,}")
    print(f"Device:                   {summary['device']}")
    
    if 'forward_pass_error' in summary:
        print(f"Forward Pass Error:       {summary['forward_pass_error']}")
    
    print("=" * 60)


def validate_model_config(input_dim: int, latent_dim: int, activation: str) -> None:
    """
    Validate model configuration parameters.
    
    Args:
        input_dim (int): Input dimension
        latent_dim (int): Latent dimension  
        activation (str): Activation function name
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
    
    if not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError(f"latent_dim must be a positive integer, got {latent_dim}")
    
    if latent_dim >= input_dim:
        raise ValueError(f"latent_dim ({latent_dim}) should be smaller than input_dim ({input_dim}) for compression")
    
    # Validate activation by trying to get it
    try:
        get_activation_function(activation)
    except ValueError as e:
        raise ValueError(f"Invalid activation function: {e}")


def move_model_to_device(model: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
    """
    Move model to specified device or auto-detect best available device.
    
    Args:
        model (nn.Module): PyTorch model
        device (Optional[torch.device]): Target device. If None, auto-detect.
        
    Returns:
        nn.Module: Model moved to device
    """
    if device is None:
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
        else:
            device = torch.device('cpu')
    
    return model.to(device)
