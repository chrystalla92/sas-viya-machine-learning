"""
Comprehensive Test for Enhanced Training Framework

This script tests the complete training pipeline with L-BFGS optimizer,
convergence detection, checkpointing, and evaluation capabilities.
Designed to verify that the implementation meets SAS PROC NNET behavior.
"""

import torch
import numpy as np
import os
import sys
import time
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_sas_compatible_autoencoder
from training.trainer import AutoencoderTrainer, TrainingConfig
from training.evaluator import AutoencoderEvaluator
from utils.checkpointing import CheckpointManager
from utils.metrics import TrainingMetrics, analyze_training_convergence
from data_utils import train_validation_split

# Global variables to store test results
test_trainer = None
test_metrics = None

def create_test_data(n_samples: int = 1000, seed: int = 23451) -> tuple:
    """Create test data that mimics MNIST structure."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate synthetic MNIST-like data with some structure
    images = np.random.rand(n_samples, 784).astype(np.float32)
    
    # Add some pattern to make reconstruction meaningful
    for i in range(n_samples):
        # Create simple patterns (diagonal lines, squares, etc.)
        img = images[i].reshape(28, 28)
        pattern_type = i % 4
        
        if pattern_type == 0:  # Diagonal line
            np.fill_diagonal(img, 1.0)
        elif pattern_type == 1:  # Square in center
            img[10:18, 10:18] = 1.0
        elif pattern_type == 2:  # Horizontal lines
            img[7::7, :] = 1.0
        else:  # Vertical lines
            img[:, 7::7] = 1.0
        
        images[i] = img.flatten()
    
    # Standardize data (matching SAS preprocessing)
    images = (images - np.mean(images, axis=0)) / (np.std(images, axis=0) + 1e-8)
    
    labels = np.random.randint(0, 10, n_samples)
    
    return images, labels

def test_training_configuration():
    """Test training configuration matches SAS settings."""
    print("=== Testing Training Configuration ===")
    
    config = TrainingConfig()
    
    # Verify SAS-compatible settings
    assert config.optimizer_type == 'lbfgs', "Should use L-BFGS optimizer"
    assert config.max_iters == 500, "Should have max 500 iterations"
    assert config.tolerance_grad == 1e-10, "Should match SAS fConv=1E-10"
    assert config.seed == 23451, "Should use SAS-compatible seed"
    assert config.batch_size is None, "Should use full batch training like SAS"
    
    print("✓ Training configuration matches SAS requirements")

def test_model_architecture():
    """Test model architecture matches SAS specification."""
    print("=== Testing Model Architecture ===")
    
    model = create_sas_compatible_autoencoder()
    arch_info = model.get_architecture_info()
    
    assert arch_info['input_dim'] == 784, "Should have 784 input neurons"
    assert arch_info['hidden_dim'] == 400, "Should have 400 hidden neurons"
    assert arch_info['output_dim'] == 784, "Should have 784 output neurons"
    assert arch_info['activation'] == 'tanh', "Should use tanh activation"
    
    # Test forward pass
    test_input = torch.randn(10, 784)
    output = model(test_input)
    assert output.shape == test_input.shape, "Output should match input shape"
    
    print("✓ Model architecture matches SAS specification")

def test_lbfgs_training():
    """Test L-BFGS training with convergence tracking."""
    print("=== Testing L-BFGS Training ===")
    
    # Create test data
    images, labels = create_test_data(200, seed=23451)
    train_images, val_images, _, _ = train_validation_split(
        images, labels, validation_ratio=0.2, random_seed=23451, stratify=False
    )
    
    # Configure for quick test
    config = TrainingConfig()
    config.max_epochs = 20  # Shorter for testing
    config.early_stopping_patience = 10
    config.log_interval = 2
    config.save_interval = 10
    config.checkpoint_dir = './test_checkpoints'
    config.log_dir = './test_logs'
    
    # Train model
    trainer = AutoencoderTrainer(config)
    metrics = trainer.train(train_images, val_images)
    
    # Verify training results
    assert len(metrics.train_losses) > 0, "Should have training losses"
    assert metrics.train_losses[-1] < metrics.train_losses[0], "Loss should decrease"
    
    # Check best_loss based on whether validation data was used
    if metrics.val_losses:
        # When validation is used, best_loss should track validation loss
        assert metrics.best_loss <= min(metrics.val_losses), "Best loss should be minimum validation loss"
    else:
        # When only training data is used, best_loss should track training loss
        assert metrics.best_loss <= min(metrics.train_losses), "Best loss should be minimum training loss"
    
    # Verify L-BFGS specific tracking
    if hasattr(trainer, 'gradient_norms'):
        assert len(trainer.gradient_norms) > 0, "Should track gradient norms"
        print(f"Tracked {len(trainer.gradient_norms)} gradient norm values")
    
    print("✓ L-BFGS training completed successfully")
    print(f"✓ Final loss: {metrics.train_losses[-1]:.8f}")
    print(f"✓ Best loss: {metrics.best_loss:.8f} at epoch {metrics.best_epoch}")
    
    # Store results in global variables for use by other functions
    global test_trainer, test_metrics
    test_trainer, test_metrics = trainer, metrics

def test_checkpointing():
    """Test checkpoint saving and loading."""
    print("=== Testing Checkpointing ===")
    
    # Create simple model for testing
    model = create_sas_compatible_autoencoder()
    optimizer = torch.optim.LBFGS(model.parameters())
    
    # Test checkpoint manager
    checkpoint_manager = CheckpointManager('./test_checkpoints', max_checkpoints=3)
    
    # Save a checkpoint
    test_epoch = 5
    test_loss = 0.123456
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model, optimizer, test_epoch, test_loss,
        metrics={'test_metric': 0.5}, is_best=True
    )
    
    assert os.path.exists(checkpoint_path), "Checkpoint file should exist"
    
    # Load the checkpoint
    checkpoint_data, loaded_model = checkpoint_manager.load_checkpoint(load_best=True)
    
    assert checkpoint_data['epoch'] == test_epoch, "Should load correct epoch"
    assert abs(checkpoint_data['loss'] - test_loss) < 1e-6, "Should load correct loss"
    assert loaded_model is not None, "Should load model"
    
    # Test training resumption
    model, optimizer, start_epoch, best_loss = checkpoint_manager.resume_training(load_best=True)
    assert start_epoch == test_epoch + 1, "Should resume from next epoch"
    
    print("✓ Checkpointing works correctly")

def test_evaluation_metrics():
    """Test evaluation and metrics computation."""
    print("=== Testing Evaluation Metrics ===")
    
    # Create model and test data
    model = create_sas_compatible_autoencoder()
    test_data = torch.randn(50, 784)
    
    # Test evaluator
    evaluator = AutoencoderEvaluator(model)
    results = evaluator.evaluate_dataset(test_data, batch_size=25)
    
    # Verify results structure
    required_keys = ['mse_loss', 'mae_loss', 'rmse_loss', 'pixel_accuracy', 
                     'structural_similarity', 'error_distribution']
    for key in required_keys:
        assert key in results, f"Results should contain {key}"
    
    # Verify metrics are reasonable
    assert results['mse_loss'] >= 0, "MSE should be non-negative"
    assert 0 <= results['pixel_accuracy'] <= 100, "Pixel accuracy should be percentage"
    
    print("✓ Evaluation metrics computed correctly")
    print(f"✓ MSE Loss: {results['mse_loss']:.6f}")
    print(f"✓ Pixel Accuracy: {results['pixel_accuracy']:.2f}%")

def test_convergence_analysis():
    """Test convergence analysis functionality."""
    print("=== Testing Convergence Analysis ===")
    
    # Create mock training metrics
    metrics = TrainingMetrics()
    
    # Simulate decreasing loss (converging)
    for epoch in range(1, 21):
        loss = 1.0 / (epoch ** 0.5)  # Decreasing loss
        metrics.update(epoch, loss)
    
    # Test convergence analysis
    convergence_info = analyze_training_convergence(metrics, window_size=5)
    
    assert 'converged' in convergence_info, "Should return convergence status"
    assert 'loss_trend' in convergence_info, "Should analyze loss trend"
    assert convergence_info['loss_trend'] < 0, "Loss trend should be negative (decreasing)"
    
    print("✓ Convergence analysis working correctly")
    print(f"✓ Converged: {convergence_info['converged']}")
    print(f"✓ Loss trend: {convergence_info['loss_trend']:.6f}")

def test_sas_compatibility():
    """Test specific SAS compatibility features."""
    print("=== Testing SAS Compatibility ===")
    
    # Test deterministic behavior with seed
    seed = 23451
    
    # Create two identical training setups
    model1 = create_sas_compatible_autoencoder(seed=seed)
    model2 = create_sas_compatible_autoencoder(seed=seed)
    
    # Test input
    torch.manual_seed(seed)
    test_input = torch.randn(5, 784)
    
    # Both models should produce identical results
    with torch.no_grad():
        output1 = model1(test_input)
        output2 = model2(test_input)
        
    difference = torch.abs(output1 - output2).max().item()
    assert difference < 1e-10, f"Models should be identical, difference: {difference}"
    
    # Test MSE loss computation matches expected behavior
    target = test_input  # Autoencoder targets
    loss1 = model1.reconstruction_loss(test_input, output1)
    loss2 = torch.nn.functional.mse_loss(output1, target, reduction='mean')
    
    assert torch.abs(loss1 - loss2).item() < 1e-10, "MSE computation should match PyTorch"
    
    print("✓ SAS compatibility verified")
    print(f"✓ Deterministic behavior confirmed (difference: {difference:.2e})")

def run_comprehensive_test():
    """Run all tests and verify success criteria."""
    global test_trainer, test_metrics
    print("=" * 60)
    print("COMPREHENSIVE TRAINING FRAMEWORK TEST")
    print("=" * 60)
    
    test_results = {}
    
    try:
        # Run all tests
        try:
            test_training_configuration()
            test_results['config'] = True
        except AssertionError:
            test_results['config'] = False
            
        try:
            test_model_architecture()
            test_results['architecture'] = True
        except AssertionError:
            test_results['architecture'] = False
            
        try:
            test_checkpointing()
            test_results['checkpointing'] = True
        except AssertionError:
            test_results['checkpointing'] = False
            
        try:
            test_evaluation_metrics()
            test_results['evaluation'] = True
        except AssertionError:
            test_results['evaluation'] = False
            
        try:
            test_convergence_analysis()
            test_results['convergence'] = True
        except AssertionError:
            test_results['convergence'] = False
            
        try:
            test_sas_compatibility()
            test_results['sas_compatibility'] = True
        except AssertionError:
            test_results['sas_compatibility'] = False
        
        # Run full training test
        print("\n" + "=" * 40)
        try:
            test_lbfgs_training()
            test_results['training'] = True
            trainer, metrics = test_trainer, test_metrics
        except AssertionError:
            test_results['training'] = False
            trainer, metrics = None, None
        
        if trainer and metrics:
            # Analyze final results
            print("\n=== Final Training Analysis ===")
            summary = metrics.get_summary()
            for key, value in summary.items():
                print(f"{key}: {value}")
            
            # Success criteria verification
            success_criteria = {
                'training_completed': len(metrics.train_losses) > 0,
                'loss_decreased': metrics.train_losses[-1] < metrics.train_losses[0],
                'convergence_tracked': hasattr(trainer, 'gradient_norms'),
                'checkpoints_saved': os.path.exists('./test_checkpoints'),
                'metrics_saved': os.path.exists('./test_logs/training_metrics.json')
            }
            
            print("\n=== Success Criteria Verification ===")
            all_passed = True
            for criterion, passed in success_criteria.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{criterion}: {status}")
                all_passed = all_passed and passed
        else:
            all_passed = False
        
        # Overall result
        print("\n" + "=" * 60)
        if all_passed and all(test_results.values()):
            print("🎉 ALL TESTS PASSED - Training Framework Ready!")
            print("✓ L-BFGS optimizer configured correctly")
            print("✓ SAS compatibility verified")
            print("✓ Convergence detection working")
            print("✓ Checkpointing functional")
            print("✓ Evaluation metrics comprehensive")
            print("✓ Training pipeline complete")
        else:
            print("❌ SOME TESTS FAILED - Review Implementation")
            failed_tests = [k for k, v in test_results.items() if not v]
            if failed_tests:
                print(f"Failed tests: {failed_tests}")
        
        print("=" * 60)
        
        return all_passed and all(test_results.values())
        
    except Exception as e:
        print(f"❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test files
        import shutil
        for test_dir in ['./test_checkpoints', './test_logs']:
            if os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir)
                except:
                    pass

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
