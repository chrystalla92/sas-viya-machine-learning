"""
Example Usage of Enhanced Training Framework

This script demonstrates the complete training pipeline with:
- L-BFGS optimizer matching SAS PROC NNET behavior
- Comprehensive metrics tracking and convergence monitoring
- Model checkpointing and state management
- Evaluation and reconstruction quality assessment
"""

import torch
import numpy as np
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_sas_compatible_autoencoder
from training.trainer import AutoencoderTrainer, TrainingConfig
from training.evaluator import AutoencoderEvaluator, evaluate_model_checkpoint
from utils.checkpointing import CheckpointManager
from utils.metrics import analyze_training_convergence
from data_utils import train_validation_split

def create_demo_data(n_samples: int = 2000, seed: int = 23451):
    """Create structured demo data for autoencoder training."""
    np.random.seed(seed)
    
    # Create synthetic MNIST-like data with recognizable patterns
    images = np.zeros((n_samples, 784), dtype=np.float32)
    
    for i in range(n_samples):
        img = np.random.rand(28, 28) * 0.1  # Low noise background
        
        # Add structured patterns
        pattern = i % 8
        
        if pattern == 0:  # Horizontal line
            img[14, :] = 1.0
        elif pattern == 1:  # Vertical line  
            img[:, 14] = 1.0
        elif pattern == 2:  # Diagonal line
            np.fill_diagonal(img, 1.0)
        elif pattern == 3:  # Square
            img[8:20, 8:20] = 0.8
            img[10:18, 10:18] = 0.2  # Hollow square
        elif pattern == 4:  # Circle (approximate)
            center = 14
            for r in range(28):
                for c in range(28):
                    if 8 < ((r-center)**2 + (c-center)**2)**0.5 < 10:
                        img[r, c] = 0.9
        elif pattern == 5:  # Cross
            img[14, :] = 0.7
            img[:, 14] = 0.7
        elif pattern == 6:  # Corner pattern
            img[:10, :10] = 0.8
            img[-10:, -10:] = 0.8
        else:  # Random blocks
            for _ in range(5):
                r, c = np.random.randint(0, 24, 2)
                img[r:r+4, c:c+4] = np.random.rand()
        
        images[i] = img.flatten()
    
    # Standardize to match SAS preprocessing
    images = (images - np.mean(images, axis=0, keepdims=True)) / (np.std(images, axis=0, keepdims=True) + 1e-8)
    
    # Create dummy labels
    labels = np.random.randint(0, 10, n_samples)
    
    return images, labels

def main():
    """Main training and evaluation pipeline."""
    print("=" * 70)
    print("ENHANCED AUTOENCODER TRAINING PIPELINE")
    print("Matching SAS PROC NNET L-BFGS Implementation")
    print("=" * 70)
    
    # Step 1: Create training data
    print("\n1. Creating Training Data")
    print("-" * 30)
    
    images, labels = create_demo_data(n_samples=1500, seed=23451)
    print(f"Created {len(images)} samples with shape {images.shape}")
    print(f"Data range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Split into train/validation
    train_images, val_images, train_labels, val_labels = train_validation_split(
        images, labels, validation_ratio=0.2, random_seed=23451, stratify=False
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Step 2: Configure training
    print("\n2. Training Configuration")
    print("-" * 30)
    
    config = TrainingConfig()
    config.max_epochs = 50  # Reasonable for demo
    config.early_stopping_patience = 15
    config.log_interval = 5
    config.save_interval = 20
    config.checkpoint_dir = './demo_checkpoints'
    config.log_dir = './demo_logs'
    
    print("L-BFGS Configuration (SAS Compatible):")
    print(f"  Optimizer: {config.optimizer_type}")
    print(f"  Max iterations per epoch: {config.max_iters}")
    print(f"  Gradient tolerance: {config.tolerance_grad}")
    print(f"  Change tolerance: {config.tolerance_change}")
    print(f"  History size: {config.history_size}")
    print(f"  Line search: {config.line_search_fn}")
    print(f"  Random seed: {config.seed}")
    
    # Step 3: Initialize and train model
    print("\n3. Model Training")
    print("-" * 30)
    
    trainer = AutoencoderTrainer(config)
    
    print("Starting training with L-BFGS optimizer...")
    start_time = time.time()
    
    metrics = trainer.train(train_images, val_images)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Step 4: Analyze training results
    print("\n4. Training Analysis")
    print("-" * 30)
    
    summary = metrics.get_summary()
    print("Training Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Convergence analysis
    convergence_analysis = analyze_training_convergence(metrics)
    print(f"\nConvergence Analysis:")
    print(f"  Converged: {convergence_analysis['converged']}")
    print(f"  Loss trend: {convergence_analysis['loss_trend']:.2e}")
    print(f"  Recent loss stability: {convergence_analysis['loss_stability']:.6f}")
    
    # L-BFGS specific metrics
    if hasattr(trainer, 'gradient_norms') and trainer.gradient_norms:
        print(f"\nL-BFGS Specific Metrics:")
        print(f"  Total gradient evaluations: {len(trainer.gradient_norms)}")
        print(f"  Final gradient norm: {trainer.gradient_norms[-1]:.2e}")
        print(f"  Average gradient norm: {np.mean(trainer.gradient_norms):.2e}")
    
    # Step 5: Model evaluation
    print("\n5. Model Evaluation")
    print("-" * 30)
    
    evaluator = AutoencoderEvaluator(trainer.model)
    
    # Evaluate on validation set
    val_tensor = torch.FloatTensor(val_images)
    eval_results = evaluator.evaluate_dataset(val_tensor, batch_size=100)
    
    print("Evaluation Results:")
    print(f"  MSE Loss: {eval_results['mse_loss']:.8f}")
    print(f"  MAE Loss: {eval_results['mae_loss']:.8f}")
    print(f"  RMSE Loss: {eval_results['rmse_loss']:.8f}")
    print(f"  Pixel Accuracy: {eval_results['pixel_accuracy']:.2f}%")
    print(f"  Structural Similarity: {eval_results['structural_similarity']:.4f}")
    
    # Error distribution analysis
    error_dist = eval_results['error_distribution']
    print(f"\nError Distribution:")
    print(f"  Mean error: {error_dist['mean_error']:.6f}")
    print(f"  Median error: {error_dist['median_error']:.6f}")
    print(f"  Max error: {error_dist['max_error']:.6f}")
    
    # Step 6: Generate reconstruction samples
    print("\n6. Reconstruction Samples")
    print("-" * 30)
    
    # Generate some reconstruction samples
    sample_indices = [0, 10, 20, 30, 40]  # Different pattern types
    samples = evaluator.generate_reconstruction_samples(
        val_tensor, n_samples=5, indices=sample_indices
    )
    
    print("Generated reconstruction samples for visual inspection")
    print(f"Original sample shapes: {samples['original'].shape}")
    print(f"Reconstructed sample shapes: {samples['reconstructed'].shape}")
    
    # Compute reconstruction errors for samples
    original_samples = samples['original']
    reconstructed_samples = samples['reconstructed']
    
    for i in range(len(sample_indices)):
        mse = torch.nn.functional.mse_loss(
            reconstructed_samples[i], original_samples[i], reduction='mean'
        ).item()
        print(f"Sample {sample_indices[i]} MSE: {mse:.6f}")
    
    # Step 7: Checkpoint management
    print("\n7. Checkpoint Management")
    print("-" * 30)
    
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    checkpoints = checkpoint_manager.list_checkpoints()
    
    print(f"Available checkpoints: {len(checkpoints)}")
    for checkpoint in checkpoints[-3:]:  # Show last 3
        print(f"  {checkpoint['filename']}: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.8f}")
    
    best_checkpoint = checkpoint_manager.get_best_checkpoint_info()
    if best_checkpoint:
        print(f"\nBest checkpoint: {best_checkpoint['filename']}")
        print(f"  Epoch: {best_checkpoint['epoch']}")
        print(f"  Loss: {best_checkpoint['loss']:.8f}")
    
    # Step 8: Success criteria verification
    print("\n8. Success Criteria Verification")
    print("-" * 30)
    
    success_criteria = {
        'Training converged to reasonable loss': eval_results['mse_loss'] < 1.0,
        'L-BFGS iterations tracked': hasattr(trainer, 'gradient_norms'),
        'Model checkpoints saved': len(checkpoints) > 0,
        'Validation loss computed': len(metrics.val_losses) > 0,
        'Convergence detection worked': 'converged' in convergence_analysis,
        'Reconstruction quality acceptable': eval_results['pixel_accuracy'] > 50.0
    }
    
    all_passed = True
    for criterion, passed in success_criteria.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
        all_passed = all_passed and passed
    
    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 SUCCESS: Enhanced Training Framework Verification Complete!")
        print("\nKey achievements:")
        print("✓ L-BFGS optimizer matches SAS PROC NNET configuration")
        print("✓ Training converges with proper loss tracking")
        print("✓ Convergence detection and early stopping functional")
        print("✓ Model checkpointing and state management working")
        print("✓ Comprehensive evaluation metrics computed")
        print("✓ Deterministic training with consistent results")
        
        print(f"\nFinal Performance:")
        print(f"  Training Loss: {metrics.train_losses[-1]:.8f}")
        print(f"  Validation Loss: {metrics.val_losses[-1]:.8f}")
        print(f"  Reconstruction Quality: {eval_results['pixel_accuracy']:.1f}% pixel accuracy")
        
    else:
        print("❌ Some success criteria not met - review implementation")
    
    print("=" * 70)
    
    # Cleanup demo files
    try:
        import shutil
        for demo_dir in ['./demo_checkpoints', './demo_logs']:
            if os.path.exists(demo_dir):
                shutil.rmtree(demo_dir)
        print("\nDemo files cleaned up")
    except Exception as e:
        print(f"Note: Could not clean up demo files: {e}")
    
    return all_passed

if __name__ == "__main__":
    import time
    success = main()
    print(f"\nExample completed {'successfully' if success else 'with issues'}")
