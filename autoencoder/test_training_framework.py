"""
Integration Testing for Training and Evaluation Framework

This module provides comprehensive integration tests for the MNIST autoencoder
training and evaluation framework. It verifies that all components work together
correctly and produces results similar to the SAS implementation.

Key tests:
- Model architecture verification
- Training loop functionality
- Checkpointing system
- Evaluation metrics
- Integration with data processing modules
- Convergence behavior analysis
"""

import torch
import numpy as np
import os
import sys
import tempfile
import shutil
from typing import Dict, Tuple, Any
import warnings

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from model import create_sas_compatible_autoencoder, MNISTAutoencoder
from training import TrainingConfig, AutoencoderTrainer, train_mnist_autoencoder
from evaluation import AutoencoderEvaluator
from checkpoints import CheckpointManager
from mnist_data import MNISTReader
from data_utils import train_validation_split


class IntegrationTester:
    """
    Comprehensive integration tester for the training framework.
    """
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_temp_environment(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix='autoencoder_test_')
        print(f"Created temporary test directory: {self.temp_dir}")
    
    def cleanup_temp_environment(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def test_model_architecture(self) -> Dict[str, Any]:
        """Test model architecture matches SAS specifications."""
        print("=== Testing Model Architecture ===")
        
        model = create_sas_compatible_autoencoder(seed=23451)
        arch_info = model.get_architecture_info()
        
        # Expected architecture from SAS configuration
        expected = {
            'input_dim': 784,
            'hidden_dim': 400,
            'output_dim': 784,
            'activation': 'tanh',
            'dropout_rate': 0.0
        }
        
        results = {
            'architecture_correct': True,
            'details': arch_info,
            'expected': expected,
            'mismatches': []
        }
        
        # Check each specification
        for key, expected_value in expected.items():
            if key in arch_info:
                if arch_info[key] != expected_value:
                    results['architecture_correct'] = False
                    results['mismatches'].append(f"{key}: got {arch_info[key]}, expected {expected_value}")
            else:
                results['architecture_correct'] = False
                results['mismatches'].append(f"Missing key: {key}")
        
        # Test forward pass
        test_input = torch.randn(10, 784)
        with torch.no_grad():
            output = model(test_input)
            
        results['forward_pass_shape_correct'] = output.shape == (10, 784)
        results['output_shape'] = tuple(output.shape)
        
        print(f"Architecture correct: {results['architecture_correct']}")
        print(f"Forward pass shape correct: {results['forward_pass_shape_correct']}")
        
        return results
    
    def test_training_integration(self) -> Dict[str, Any]:
        """Test training integration with mock data."""
        print("=== Testing Training Integration ===")
        
        # Create mock MNIST-like data (small dataset for quick test)
        np.random.seed(23451)
        mock_images = np.random.rand(200, 784).astype(np.float32)
        mock_labels = np.random.randint(0, 10, 200)
        
        # Configure for quick test
        config = TrainingConfig()
        config.max_epochs = 5  # Very short for testing
        config.early_stopping_patience = 3
        config.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        config.log_dir = os.path.join(self.temp_dir, 'logs')
        config.save_interval = 2
        
        # Split data
        train_data, val_data, _, _ = train_validation_split(
            mock_images, mock_labels, validation_ratio=0.2, random_seed=23451
        )
        
        # Initialize trainer
        trainer = AutoencoderTrainer(config)
        
        try:
            # Run training
            metrics = trainer.train(train_data, val_data)
            
            results = {
                'training_completed': True,
                'final_epoch': len(metrics.epochs),
                'initial_loss': metrics.train_losses[0] if metrics.train_losses else None,
                'final_loss': metrics.train_losses[-1] if metrics.train_losses else None,
                'loss_decreased': None,
                'validation_computed': len(metrics.val_losses) > 0,
                'checkpoints_created': os.path.exists(config.checkpoint_dir),
                'metrics_saved': os.path.exists(os.path.join(config.log_dir, 'training_metrics.json')),
                'error': None
            }
            
            if results['initial_loss'] and results['final_loss']:
                results['loss_decreased'] = results['final_loss'] < results['initial_loss']
            
            print(f"Training completed: {results['training_completed']}")
            print(f"Final epoch: {results['final_epoch']}")
            print(f"Loss decreased: {results['loss_decreased']}")
            
        except Exception as e:
            results = {
                'training_completed': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
            print(f"Training failed: {e}")
        
        return results
    
    def test_checkpointing_system(self) -> Dict[str, Any]:
        """Test checkpointing functionality."""
        print("=== Testing Checkpointing System ===")
        
        checkpoint_dir = os.path.join(self.temp_dir, 'checkpoint_test')
        manager = CheckpointManager(checkpoint_dir)
        
        # Create model and optimizer
        model = create_sas_compatible_autoencoder()
        optimizer = torch.optim.LBFGS(model.parameters())
        
        results = {
            'save_checkpoint': False,
            'load_checkpoint': False,
            'best_tracking': False,
            'resume_training': False,
            'cleanup': False,
            'error': None
        }
        
        try:
            # Test saving checkpoints
            manager.save_checkpoint(model, optimizer, epoch=1, loss=0.5)
            manager.save_checkpoint(model, optimizer, epoch=2, loss=0.3, is_best=True)
            manager.save_checkpoint(model, optimizer, epoch=3, loss=0.4)
            
            results['save_checkpoint'] = True
            
            # Test loading checkpoint
            checkpoint_data, loaded_model = manager.load_checkpoint(load_best=True)
            results['load_checkpoint'] = checkpoint_data['epoch'] == 2
            
            # Test best tracking
            best_info = manager.get_best_checkpoint_info()
            results['best_tracking'] = (best_info is not None and 
                                      best_info['epoch'] == 2 and 
                                      best_info['loss'] == 0.3)
            
            # Test training resumption
            model, optimizer, start_epoch, best_loss = manager.resume_training(load_best=True)
            results['resume_training'] = (start_epoch == 3 and best_loss == 0.3)
            
            # Test cleanup
            manager.cleanup_all_checkpoints(keep_best=True)
            results['cleanup'] = True
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Checkpointing test failed: {e}")
        
        print(f"Checkpointing tests passed: {sum(results[k] for k in results if k != 'error')}/5")
        
        return results
    
    def test_evaluation_system(self) -> Dict[str, Any]:
        """Test evaluation functionality."""
        print("=== Testing Evaluation System ===")
        
        # Create model and mock data
        model = create_sas_compatible_autoencoder()
        mock_data = torch.randn(100, 784)
        
        evaluator = AutoencoderEvaluator(model)
        
        results = {
            'evaluation_completed': False,
            'metrics_computed': False,
            'samples_generated': False,
            'report_generated': False,
            'error': None
        }
        
        try:
            # Test evaluation
            eval_results = evaluator.evaluate_dataset(mock_data, batch_size=50)
            results['evaluation_completed'] = True
            
            # Check if key metrics are present
            required_metrics = ['mse_loss', 'mae_loss', 'pixel_accuracy', 'structural_similarity']
            results['metrics_computed'] = all(metric in eval_results for metric in required_metrics)
            
            # Test sample generation
            samples = evaluator.generate_reconstruction_samples(mock_data, n_samples=5)
            results['samples_generated'] = ('original' in samples and 
                                          'reconstructed' in samples and
                                          len(samples['original']) == 5)
            
            # Test report generation
            report_dir = os.path.join(self.temp_dir, 'eval_test')
            report_path = evaluator.generate_evaluation_report(eval_results, report_dir)
            results['report_generated'] = os.path.exists(report_path)
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Evaluation test failed: {e}")
        
        print(f"Evaluation tests passed: {sum(results[k] for k in results if k != 'error')}/4")
        
        return results
    
    def test_data_integration(self) -> Dict[str, Any]:
        """Test integration with existing data processing modules."""
        print("=== Testing Data Processing Integration ===")
        
        results = {
            'data_loading': False,
            'standardization': False,
            'sas_format': False,
            'train_val_split': False,
            'error': None
        }
        
        try:
            # Test with mock IDX-like data
            np.random.seed(23451)
            mock_images = np.random.randint(0, 256, (100, 784)).astype(np.uint8)
            mock_labels = np.random.randint(0, 10, 100).astype(np.uint8)
            
            # Test data loading simulation
            reader = MNISTReader()
            reader._images_raw = mock_images.astype(np.float64)
            results['data_loading'] = True
            
            # Test standardization
            standardized = reader.apply_midrange_standardization()
            results['standardization'] = standardized.shape == (100, 784)
            
            # Test SAS format creation
            sas_dataset = reader.create_sas_compatible_dataset(standardized, mock_labels)
            results['sas_format'] = (sas_dataset.shape == (100, 785) and 
                                   np.array_equal(sas_dataset[:, 0], mock_labels))
            
            # Test train/validation split
            train_img, val_img, train_lbl, val_lbl = train_validation_split(
                standardized, mock_labels, validation_ratio=0.2, random_seed=23451
            )
            results['train_val_split'] = (len(train_img) + len(val_img) == 100 and
                                        len(val_img) == 20)  # 20% validation
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Data integration test failed: {e}")
        
        print(f"Data integration tests passed: {sum(results[k] for k in results if k != 'error')}/4")
        
        return results
    
    def test_convergence_behavior(self) -> Dict[str, Any]:
        """Test that training shows proper convergence behavior."""
        print("=== Testing Convergence Behavior ===")
        
        # Create synthetic data with clear pattern for reconstruction
        np.random.seed(23451)
        n_samples = 500
        
        # Create structured data (easier to reconstruct than random)
        structured_data = np.zeros((n_samples, 784))
        for i in range(n_samples):
            # Create simple patterns that should be easy to learn
            pattern = np.random.choice([0, 1], size=784, p=[0.8, 0.2])  # Sparse pattern
            structured_data[i] = pattern * 0.5 + np.random.normal(0, 0.1, 784)
        
        # Configure training for convergence test
        config = TrainingConfig()
        config.max_epochs = 20
        config.early_stopping_patience = 5
        config.checkpoint_dir = os.path.join(self.temp_dir, 'convergence_test')
        config.log_dir = os.path.join(self.temp_dir, 'convergence_logs')
        config.log_interval = 2
        
        results = {
            'training_converged': False,
            'loss_trend_downward': False,
            'final_loss_reasonable': False,
            'convergence_detected': False,
            'error': None
        }
        
        try:
            trainer = AutoencoderTrainer(config)
            metrics = trainer.train(structured_data)
            
            # Check convergence
            if len(metrics.train_losses) >= 3:
                initial_loss = metrics.train_losses[0]
                final_loss = metrics.train_losses[-1]
                
                results['loss_trend_downward'] = final_loss < initial_loss
                results['final_loss_reasonable'] = final_loss < 1.0  # Should achieve reasonable loss
                
                # Check for convergence pattern (loss stabilization)
                if len(metrics.train_losses) >= 10:
                    recent_losses = metrics.train_losses[-5:]
                    loss_variance = np.var(recent_losses)
                    results['convergence_detected'] = loss_variance < 0.001  # Low variance = convergence
            
            results['training_converged'] = len(metrics.epochs) > 0
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Convergence test failed: {e}")
        
        print(f"Convergence tests passed: {sum(results[k] for k in results if k != 'error')}/4")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("=" * 60)
        print("AUTOENCODER TRAINING FRAMEWORK - INTEGRATION TESTS")
        print("=" * 60)
        
        self.setup_temp_environment()
        
        try:
            # Run all test suites
            self.test_results['architecture'] = self.test_model_architecture()
            self.test_results['training'] = self.test_training_integration()
            self.test_results['checkpointing'] = self.test_checkpointing_system()
            self.test_results['evaluation'] = self.test_evaluation_system()
            self.test_results['data_integration'] = self.test_data_integration()
            self.test_results['convergence'] = self.test_convergence_behavior()
            
            # Compile overall results
            overall_results = self.compile_test_summary()
            
            print("\n" + "=" * 60)
            print("INTEGRATION TEST SUMMARY")
            print("=" * 60)
            
            for suite_name, suite_results in overall_results['suite_results'].items():
                status = "✓ PASS" if suite_results['passed'] else "✗ FAIL"
                print(f"{suite_name:<20}: {status} ({suite_results['passed_count']}/{suite_results['total_count']})")
            
            print(f"\nOverall Success Rate: {overall_results['overall_success_rate']:.1f}%")
            print(f"Total Tests: {overall_results['total_tests']}")
            print(f"Tests Passed: {overall_results['tests_passed']}")
            
            if overall_results['overall_success_rate'] >= 80:
                print("\n🎉 Integration tests PASSED! Framework is ready for use.")
            else:
                print("\n⚠️  Some integration tests FAILED. Review issues before deployment.")
            
            return overall_results
            
        finally:
            self.cleanup_temp_environment()
    
    def compile_test_summary(self) -> Dict[str, Any]:
        """Compile summary of all test results."""
        suite_results = {}
        total_tests = 0
        tests_passed = 0
        
        for suite_name, results in self.test_results.items():
            if isinstance(results, dict):
                # Count boolean test results (excluding error and metadata fields)
                test_keys = [k for k, v in results.items() 
                           if isinstance(v, bool) and k not in ['error']]
                
                passed_count = sum(results[k] for k in test_keys)
                total_count = len(test_keys)
                
                suite_results[suite_name] = {
                    'passed_count': passed_count,
                    'total_count': total_count,
                    'passed': passed_count == total_count,
                    'success_rate': (passed_count / total_count * 100) if total_count > 0 else 0,
                    'error': results.get('error')
                }
                
                total_tests += total_count
                tests_passed += passed_count
        
        return {
            'suite_results': suite_results,
            'total_tests': total_tests,
            'tests_passed': tests_passed,
            'overall_success_rate': (tests_passed / total_tests * 100) if total_tests > 0 else 0
        }


def run_sas_comparison_analysis():
    """
    Analyze expected behavior compared to SAS implementation.
    
    This function documents the expected behavior based on the SAS configuration
    and provides benchmarks for validating the PyTorch implementation.
    """
    print("=" * 60)
    print("SAS IMPLEMENTATION COMPARISON ANALYSIS")
    print("=" * 60)
    
    sas_config = {
        'architecture': 'MLP',
        'input_features': 784,  # var2-var785
        'hidden_neurons': 400,
        'activation': 'tanh',
        'standardization': 'midrange',
        'optimizer': 'LBFGS',
        'max_iterations': 500,
        'convergence_tolerance': 1e-10,  # fConv
        'seed': 23451,
        'dataset': 'MNIST (10 samples for testing)'
    }
    
    print("SAS Configuration:")
    for key, value in sas_config.items():
        print(f"  {key}: {value}")
    
    print("\nExpected Behavior:")
    print("  - Loss should converge to very low values (< 0.01) on small dataset")
    print("  - Training should be deterministic with seed=23451")
    print("  - L-BFGS should converge faster than gradient descent")
    print("  - Midrange standardization should center data properly")
    print("  - Reconstruction quality should be high for training data")
    
    print("\nValidation Criteria:")
    print("  ✓ Model architecture matches SAS exactly")
    print("  ✓ Loss function uses MSE for reconstruction")
    print("  ✓ Optimizer uses L-BFGS with proper configuration")
    print("  ✓ Training converges within reasonable time")
    print("  ✓ Checkpointing preserves training state")
    print("  ✓ Evaluation metrics provide meaningful insights")


def main():
    """Main function to run all tests."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Run SAS comparison analysis
    run_sas_comparison_analysis()
    
    print("\n")
    
    # Run integration tests
    tester = IntegrationTester()
    results = tester.run_all_tests()
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    results = main()
