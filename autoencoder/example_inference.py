#!/usr/bin/env python3
"""
Example: Using the Autoencoder Inference Pipeline

This script demonstrates how to use the inference pipeline for scoring
and reconstructing images from a trained autoencoder model.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')
from src.inference import create_inference_pipeline, tensor_to_flat, flat_to_images
from src.models.autoencoder import create_autoencoder


def main():
    """Demonstrate the inference pipeline usage."""
    print("Autoencoder Inference Pipeline Example")
    print("=" * 50)
    
    # 1. Create a simple model for demonstration
    print("1. Creating demonstration model...")
    model = create_autoencoder(
        input_dim=784,
        hidden_dim=400, 
        device='cpu'
    )
    
    # 2. Create inference pipeline
    print("2. Setting up inference pipeline...")
    pipeline = create_inference_pipeline(
        model=model,
        batch_size=32,
        preprocessing='midrange',  # Match training preprocessing
        validate_input=True,
        log_performance=True
    )
    
    # 3. Prepare sample data (simulating MNIST-like data)
    print("3. Preparing sample data...")
    # Create synthetic data that looks like normalized MNIST (values in [0,1])
    sample_images = torch.rand(10, 28, 28)  # 10 random 28x28 images
    print(f"   Sample images shape: {sample_images.shape}")
    
    # Convert to flat format for autoencoder input
    flat_data = tensor_to_flat(sample_images)  # Shape: (10, 784)
    print(f"   Flattened data shape: {flat_data.shape}")
    
    # 4. Single sample inference
    print("4. Running single sample inference...")
    single_input = flat_data[0]  # First sample
    single_reconstruction = pipeline.predict(single_input)
    print(f"   Input shape: {single_input.shape}")
    print(f"   Reconstruction shape: {single_reconstruction.shape}")
    
    # 5. Batch inference
    print("5. Running batch inference...")
    batch_reconstructions = pipeline.predict(flat_data)
    print(f"   Batch input shape: {flat_data.shape}")
    print(f"   Batch reconstructions shape: {batch_reconstructions.shape}")
    
    # 6. Inference with latent representation extraction
    print("6. Extracting latent representations...")
    reconstructions, latent_codes = pipeline.predict(
        flat_data, 
        return_latent=True
    )
    print(f"   Reconstructions shape: {reconstructions.shape}")
    print(f"   Latent codes shape: {latent_codes.shape}")
    
    # 7. Convert reconstructions back to image format
    print("7. Converting back to image format...")
    reconstructed_images = flat_to_images(batch_reconstructions, (28, 28))
    print(f"   Reconstructed images shape: {reconstructed_images.shape}")
    
    # 8. Process larger dataset with automatic batching
    print("8. Processing larger dataset...")
    large_dataset = torch.rand(100, 784)  # 100 samples
    large_reconstructions = pipeline.predict(
        large_dataset, 
        batch_size=16,  # Custom batch size
    )
    print(f"   Large dataset shape: {large_dataset.shape}")
    print(f"   Large reconstructions shape: {large_reconstructions.shape}")
    
    # 9. Show performance statistics
    print("9. Performance statistics...")
    stats = pipeline.get_performance_stats()
    print(f"   Total samples processed: {stats['total_samples_processed']}")
    print(f"   Total inference time: {stats['total_inference_time']:.4f}s")
    print(f"   Average batch time: {stats['average_batch_time']:.4f}s")
    print(f"   Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
    
    # 10. Demonstrate reconstruction quality (using MSE)
    print("10. Checking reconstruction quality...")
    original = flat_data[:5]  # First 5 samples
    reconstructed = batch_reconstructions[:5]
    
    mse = torch.mean((original - reconstructed) ** 2, dim=1)
    print(f"   Mean squared error per sample:")
    for i, error in enumerate(mse):
        print(f"     Sample {i+1}: {error.item():.6f}")
    
    avg_mse = torch.mean(mse)
    print(f"   Average MSE: {avg_mse.item():.6f}")
    
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print("\nThe inference pipeline demonstrates:")
    print("• Loading models for scoring")
    print("• Efficient batch processing")  
    print("• Automatic tensor format handling")
    print("• Performance tracking")
    print("• Device-aware computation")
    print("• Input validation and preprocessing")


if __name__ == "__main__":
    main()
