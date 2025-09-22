#!/usr/bin/env python3
"""
Autoencoder End-to-End Demonstration Script

This script demonstrates the migrated autoencoder working end-to-end on synthetic data.
It shows:
1. Synthetic pattern generation
2. Autoencoder model creation and training
3. Evaluation and reconstruction visualization
4. Performance metrics tracking

Usage:
    python demo_autoencoder.py [--samples N] [--epochs N] [--hidden-dim N]
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List, Optional

# Add the directory containing this script to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our modules
try:
    from synthetic_demo_data import SyntheticPatternGenerator, visualize_samples
    from model import MNISTAutoencoder
    from data_utils import train_validation_split
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    print(f"Python path: {sys.path}")
    print("\nMake sure to run this script from the autoencoder directory or ensure all dependencies are accessible.")
    sys.exit(1)


class SimpleAutoencoderDemo:
    """
    A simplified autoencoder demonstration class.
    
    This creates a streamlined version of the autoencoder framework
    specifically for demonstration purposes with synthetic data.
    """
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, 
                 seed: int = 42, device: Optional[str] = None):
        """
        Initialize the demonstration.
        
        Args:
            input_dim: Input dimension (flattened image size)
            hidden_dim: Hidden layer dimension
            seed: Random seed for reproducibility
            device: Device to use ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.training_history = {'losses': [], 'epochs': []}
        
        print(f"Demo initialized on device: {self.device}")
    
    def create_model(self) -> MNISTAutoencoder:
        """Create and initialize the autoencoder model."""
        self.model = MNISTAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=0.0,
            seed=self.seed
        )
        self.model.to(self.device)
        
        # Use Adam optimizer for simplicity (faster than L-BFGS for demo)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print("Model Architecture:")
        arch_info = self.model.get_architecture_info()
        for key, value in arch_info.items():
            print(f"  {key}: {value}")
        
        return self.model
    
    def train_on_synthetic_data(self, train_images: np.ndarray, 
                               val_images: np.ndarray,
                               epochs: int = 50,
                               batch_size: int = 64,
                               print_interval: int = 10) -> Dict:
        """
        Train the autoencoder on synthetic data.
        
        Args:
            train_images: Training images (N, input_dim)
            val_images: Validation images (N, input_dim)  
            epochs: Number of training epochs
            batch_size: Batch size for training
            print_interval: Print progress every N epochs
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.create_model()
        
        # Convert to tensors
        train_tensor = torch.FloatTensor(train_images).to(self.device)
        val_tensor = torch.FloatTensor(val_images).to(self.device)
        
        print(f"\nTraining autoencoder...")
        print(f"Training samples: {len(train_images)}")
        print(f"Validation samples: {len(val_images)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            # Create batches
            indices = torch.randperm(len(train_tensor))
            for i in range(0, len(train_tensor), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_data = train_tensor[batch_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                loss = self.model.reconstruction_loss(batch_data, reconstructed)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_reconstructed = self.model(val_tensor)
                val_loss = self.model.reconstruction_loss(val_tensor, val_reconstructed)
            
            # Record metrics
            avg_train_loss = np.mean(train_losses)
            self.training_history['losses'].append((avg_train_loss, val_loss.item()))
            self.training_history['epochs'].append(epoch + 1)
            
            # Print progress
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss.item():.6f} | "
                      f"Time: {elapsed_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Final validation loss: {self.training_history['losses'][-1][1]:.6f}")
        
        return self.training_history
    
    def evaluate_reconstruction(self, test_images: np.ndarray, 
                               test_labels: np.ndarray,
                               pattern_names: List[str]) -> Dict:
        """
        Evaluate reconstruction quality and create visualizations.
        
        Args:
            test_images: Test images for evaluation
            test_labels: Test labels for evaluation
            pattern_names: Names of pattern types
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        print("\nEvaluating reconstruction quality...")
        
        # Convert to tensor
        test_tensor = torch.FloatTensor(test_images).to(self.device)
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(test_tensor)
            mse_loss = self.model.reconstruction_loss(test_tensor, reconstructed)
        
        # Convert back to numpy
        reconstructed_np = reconstructed.cpu().numpy()
        
        # Calculate metrics
        pixel_errors = np.abs(test_images - reconstructed_np)
        pixel_accuracy = np.mean(pixel_errors < 0.1) * 100
        rmse_loss = np.sqrt(mse_loss.item())
        
        results = {
            'mse_loss': mse_loss.item(),
            'rmse_loss': rmse_loss,
            'pixel_accuracy': pixel_accuracy,
            'mean_pixel_error': np.mean(pixel_errors),
            'max_pixel_error': np.max(pixel_errors)
        }
        
        print("Reconstruction Metrics:")
        print(f"  MSE Loss: {results['mse_loss']:.8f}")
        print(f"  RMSE Loss: {results['rmse_loss']:.8f}")
        print(f"  Pixel Accuracy: {results['pixel_accuracy']:.2f}%")
        print(f"  Mean Pixel Error: {results['mean_pixel_error']:.6f}")
        
        # Create visualizations
        self._plot_training_curves()
        self._plot_reconstruction_samples(
            test_images, reconstructed_np, test_labels, pattern_names
        )
        
        return results
    
    def _plot_training_curves(self) -> None:
        """Plot training and validation loss curves."""
        if not self.training_history['losses']:
            return
        
        epochs = self.training_history['epochs']
        train_losses = [loss[0] for loss in self.training_history['losses']]
        val_losses = [loss[1] for loss in self.training_history['losses']]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Progress (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_reconstruction_samples(self, originals: np.ndarray, 
                                   reconstructed: np.ndarray,
                                   labels: np.ndarray,
                                   pattern_names: List[str],
                                   num_samples: int = 8,
                                   image_size: int = 28) -> None:
        """Plot original vs reconstructed samples."""
        # Select diverse samples
        unique_labels = np.unique(labels)
        sample_indices = []
        
        for label in unique_labels[:num_samples//2]:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 0:
                sample_indices.append(np.random.choice(label_indices))
        
        # Fill with random samples if needed
        while len(sample_indices) < num_samples:
            sample_indices.append(np.random.randint(0, len(originals)))
        
        sample_indices = sample_indices[:num_samples]
        
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        
        for i, idx in enumerate(sample_indices):
            # Original
            orig_img = originals[idx].reshape(image_size, image_size)
            axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=12)
            axes[0, i].set_title(f'{pattern_names[labels[idx]]}')
            axes[0, i].axis('off')
            
            # Reconstructed  
            recon_img = reconstructed[idx].reshape(image_size, image_size)
            axes[1, i].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=12)
            axes[1, i].axis('off')
            
            # Error map
            error_img = np.abs(orig_img - recon_img)
            im = axes[2, i].imshow(error_img, cmap='hot', vmin=0, vmax=np.max(error_img))
            if i == 0:
                axes[2, i].set_ylabel('Error', fontsize=12)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Original vs Reconstructed Patterns', y=1.02, fontsize=14)
        plt.show()
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        if self.model is None:
            return {}
        
        return {
            'architecture': self.model.get_architecture_info(),
            'device': str(self.device),
            'training_epochs': len(self.training_history['epochs']),
            'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else None
        }


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='Autoencoder Synthetic Data Demo')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of synthetic samples to generate (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden layer dimension (default: 128)')
    parser.add_argument('--image-size', type=int, default=28,
                       help='Size of generated images (default: 28)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test with minimal parameters')
    
    args = parser.parse_args()
    
    # Test mode with smaller parameters
    if args.test:
        args.samples = 100
        args.epochs = 10
        args.hidden_dim = 32
        args.image_size = 16
        print("🧪 Running in test mode with reduced parameters...")
    
    try:
        run_demo(args)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_demo(args):
    """Run the actual demonstration."""
    print("="*60)
    print("🤖 AUTOENCODER SYNTHETIC DEMONSTRATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Samples: {args.samples}")
    print(f"  Epochs: {args.epochs}")  
    print(f"  Hidden Dimension: {args.hidden_dim}")
    print(f"  Image Size: {args.image_size}")
    print(f"  Random Seed: {args.seed}")
    print()
    
    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic patterns...")
    generator = SyntheticPatternGenerator(image_size=args.image_size, seed=args.seed)
    pattern_types = ['circle', 'rectangle', 'line', 'cross', 'dots']
    
    images, labels = generator.generate_dataset(
        num_samples=args.samples,
        pattern_types=pattern_types
    )
    
    print(f"Generated {len(images)} samples")
    print(f"Pattern distribution: {dict(zip(pattern_types, np.bincount(labels)))}")
    
    # Show sample patterns
    print("\n🎨 Sample patterns:")
    visualize_samples(images, labels, pattern_types, 
                     image_size=args.image_size, num_samples=10)
    
    # Step 2: Split data
    print("\n🔄 Step 2: Splitting data...")
    train_images, val_images, train_labels, val_labels = train_validation_split(
        images, labels, validation_ratio=0.2, random_seed=args.seed
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Step 3: Create and train autoencoder
    print("\n🏗️ Step 3: Creating and training autoencoder...")
    demo = SimpleAutoencoderDemo(
        input_dim=args.image_size * args.image_size,
        hidden_dim=args.hidden_dim,
        seed=args.seed
    )
    
    # Train the model
    history = demo.train_on_synthetic_data(
        train_images, val_images,
        epochs=args.epochs,
        batch_size=64,
        print_interval=max(1, args.epochs // 10)
    )
    
    # Step 4: Evaluate and visualize results
    print("\n📈 Step 4: Evaluating reconstruction quality...")
    results = demo.evaluate_reconstruction(
        val_images, val_labels, pattern_types
    )
    
    # Step 5: Model summary
    print("\n📋 Step 5: Model Summary")
    print("-" * 30)
    model_info = demo.get_model_info()
    
    if 'architecture' in model_info:
        arch = model_info['architecture']
        print(f"Input Dimension: {arch['input_dim']}")
        print(f"Hidden Dimension: {arch['hidden_dim']}")
        print(f"Total Parameters: {arch['total_parameters']:,}")
        print(f"Training Device: {model_info['device']}")
        print(f"Training Epochs: {model_info['training_epochs']}")
        
        if model_info['final_loss']:
            final_train, final_val = model_info['final_loss']
            print(f"Final Training Loss: {final_train:.6f}")
            print(f"Final Validation Loss: {final_val:.6f}")
    
    print("\n✅ Demonstration completed successfully!")
    print("\nKey Results:")
    print(f"  • MSE Loss: {results['mse_loss']:.8f}")
    print(f"  • Pixel Accuracy: {results['pixel_accuracy']:.1f}%")
    print(f"  • Mean Pixel Error: {results['mean_pixel_error']:.6f}")
    
    print("\nThis demonstrates that the migrated autoencoder can:")
    print("  ✓ Learn to compress 2D patterns into lower-dimensional representations")
    print("  ✓ Successfully reconstruct the original patterns from compressed form")
    print("  ✓ Handle different geometric shapes and structures")
    print("  ✓ Achieve good reconstruction quality with reasonable training time")


if __name__ == "__main__":
    import sys
    sys.exit(main())
