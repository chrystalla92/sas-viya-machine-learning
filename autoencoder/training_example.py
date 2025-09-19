"""
Example usage of the autoencoder training pipeline.

This script demonstrates how to train an autoencoder using the comprehensive
training pipeline with various configuration options.
"""

import torch
from autoencoder_model import AutoencoderMLP
from trainer import TrainingPipeline, train_autoencoder
from datasets import create_mnist_dataloaders


def basic_training_example():
    """Basic training example with default settings."""
    print("=" * 60)
    print("Basic Training Example")
    print("=" * 60)
    
    # Create model
    model = AutoencoderMLP(
        input_dim=784,
        latent_dim=400,
        activation='tanh'
    )
    
    # Use convenience function with minimal configuration
    config = {
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 1e-3
    }
    
    # Train the model
    pipeline, summary = train_autoencoder(model, './data', config)
    
    print(f"Training completed in {summary['total_training_time']:.1f} seconds")
    print(f"Best validation loss: {summary['best_validation_loss']:.6f}")
    
    # Save the trained model
    pipeline.save_model('./models/basic_autoencoder.pth')
    
    return pipeline


def advanced_training_example():
    """Advanced training example with custom configuration."""
    print("\n" + "=" * 60)
    print("Advanced Training Example")
    print("=" * 60)
    
    # Create model with different architecture
    model = AutoencoderMLP(
        input_dim=784,
        latent_dim=128,  # Smaller latent space
        activation='relu',  # Different activation
        init_type='normal'
    )
    
    # Advanced configuration
    config = {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 2e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 1e-4,
        'lr_scheduler_type': 'ReduceLROnPlateau',
        'lr_scheduler_patience': 5,
        'lr_scheduler_factor': 0.5,
        'train_val_split': 0.85,
        'save_best_model': True,
        'save_periodic_checkpoints': True,
        'checkpoint_frequency': 10,
        'device': 'auto'
    }
    
    # Create pipeline manually for more control
    pipeline = TrainingPipeline(model, config)
    
    print(f"Training on device: {pipeline.get_device()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    summary = pipeline.train('./data')
    
    # Evaluate on test set
    _, _, test_loader = create_mnist_dataloaders('./data', batch_size=128)
    test_metrics = pipeline.evaluate(test_loader)
    
    print(f"Test loss: {test_metrics['test_loss']:.6f}")
    
    # Plot training curves
    pipeline.plot_training_curves('./plots/advanced_training_curves.png', show=False)
    
    # Save final model
    pipeline.save_model('./models/advanced_autoencoder.pth')
    
    return pipeline


def custom_dataloader_example():
    """Example using custom data loaders."""
    print("\n" + "=" * 60)
    print("Custom DataLoader Example")
    print("=" * 60)
    
    # Create custom data loaders
    train_loader, val_loader, test_loader = create_mnist_dataloaders(
        data_dir='./data',
        batch_size=64,
        train_val_split=0.9,  # Use more data for training
        standardize=True,
        shuffle=True
    )
    
    print(f"Custom data split - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Create model
    model = AutoencoderMLP(latent_dim=256, activation='elu')
    
    # Configuration optimized for the custom split
    config = {
        'epochs': 25,
        'learning_rate': 1e-3,
        'early_stopping_patience': 8,
        'lr_scheduler_patience': 4
    }
    
    # Create pipeline
    pipeline = TrainingPipeline(model, config)
    
    # Train with custom data loaders
    summary = pipeline.train(train_loader=train_loader, val_loader=val_loader)
    
    # Evaluate on test set
    test_metrics = pipeline.evaluate(test_loader)
    print(f"Final test loss: {test_metrics['test_loss']:.6f}")
    
    return pipeline


def gpu_training_example():
    """Example of training with explicit GPU configuration."""
    print("\n" + "=" * 60)
    print("GPU Training Example")
    print("=" * 60)
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device_config = 'cuda'
        print("GPU detected - using CUDA")
    else:
        device_config = 'cpu'
        print("No GPU detected - using CPU")
    
    model = AutoencoderMLP(latent_dim=512)  # Larger model for GPU
    
    config = {
        'epochs': 30,
        'batch_size': 256,  # Larger batch size for GPU
        'learning_rate': 2e-3,
        'device': device_config,
        'early_stopping_patience': 12
    }
    
    pipeline, summary = train_autoencoder(model, './data', config)
    
    print(f"Trained on: {pipeline.get_device()}")
    print(f"Average epoch time: {summary['average_epoch_time']:.2f} seconds")
    
    return pipeline


def checkpoint_loading_example():
    """Example of loading and resuming from checkpoints."""
    print("\n" + "=" * 60)
    print("Checkpoint Loading Example")
    print("=" * 60)
    
    # First, train a model briefly and save it
    model1 = AutoencoderMLP(latent_dim=200)
    
    config = {
        'epochs': 5,
        'batch_size': 64,
        'save_best_model': True
    }
    
    pipeline1 = TrainingPipeline(model1, config)
    summary1 = pipeline1.train('./data')
    
    best_checkpoint_path = pipeline1.checkpoint.get_best_checkpoint_path()
    print(f"Saved best checkpoint: {best_checkpoint_path}")
    
    # Now load the checkpoint into a new model
    model2 = AutoencoderMLP(latent_dim=200)  # Same architecture
    pipeline2 = TrainingPipeline(model2, config)
    
    if best_checkpoint_path:
        # Load the checkpoint
        checkpoint_data = pipeline2.checkpoint.load_checkpoint(
            best_checkpoint_path, model2, pipeline2.optimizer, pipeline2.scheduler
        )
        
        print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
        print(f"Checkpoint validation loss: {checkpoint_data['val_loss']:.6f}")
        
        # Continue training from the checkpoint
        continue_config = {
            'epochs': 10,  # Train for additional epochs
            'batch_size': 64,
            'learning_rate': 5e-4  # Lower learning rate for fine-tuning
        }
        
        pipeline2.config.update(continue_config)
        summary2 = pipeline2.train('./data')
        
        print("Resumed training from checkpoint successfully!")
    
    return pipeline2


def main():
    """Run all training examples."""
    print("Autoencoder Training Pipeline Examples")
    print("=" * 60)
    
    import os
    
    # Create directories if they don't exist
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    examples = [
        basic_training_example,
        advanced_training_example,
        custom_dataloader_example,
        gpu_training_example,
        checkpoint_loading_example
    ]
    
    # Check if MNIST data is available
    if not os.path.exists('./data'):
        print("\n⚠ MNIST data not found in './data' directory")
        print("Please ensure MNIST binary files are available:")
        print("- train-images.idx3-ubyte")
        print("- train-labels.idx1-ubyte") 
        print("- t10k-images.idx3-ubyte")
        print("- t10k-labels.idx1-ubyte")
        return
    
    # Run examples
    results = []
    for example_func in examples:
        try:
            pipeline = example_func()
            results.append(True)
            print(f"✓ {example_func.__name__} completed successfully")
        except Exception as e:
            print(f"✗ {example_func.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Examples Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Completed: {passed}/{total} examples")
    
    if passed == total:
        print("✓ All examples ran successfully!")
    else:
        print("⚠ Some examples had issues. Check output above.")
    
    print("\nFiles created:")
    print("- ./models/ - Saved model checkpoints")
    print("- ./checkpoints/ - Training checkpoints")  
    print("- ./logs/ - Training logs")
    print("- ./plots/ - Training curve plots")


if __name__ == "__main__":
    main()
