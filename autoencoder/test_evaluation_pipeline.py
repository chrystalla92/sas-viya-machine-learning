#!/usr/bin/env python3
"""
Test suite for autoencoder evaluation and inference pipeline.

This script tests all the new evaluation functionality including metrics
calculation, batch processing, model I/O, and SAS output generation.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
import torch
from pathlib import Path

from autoencoder import (
    AutoencoderMLP,
    ModelEvaluator,
    BatchInferenceProcessor,
    ModelComparator,
    PerformanceBenchmark,
    ReconstructionMetrics,
    LatentSpaceAnalyzer,
    ModelSaver,
    ModelLoader,
    SASOutputFormatter,
    calculate_reconstruction_errors,
    calculate_per_sample_errors,
    compute_latent_statistics,
    prepare_latent_visualization,
    create_sas_compatible_outputs
)


class TestReconstructionMetrics(unittest.TestCase):
    """Test reconstruction metrics calculations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.batch_size = 10
        self.feature_dim = 784
        
        self.original = torch.randn(self.batch_size, self.feature_dim)
        # Add some noise to create reconstructed data
        self.reconstructed = self.original + torch.randn_like(self.original) * 0.1
        
    def test_mse_calculation(self):
        """Test MSE calculation."""
        metrics = ReconstructionMetrics()
        
        # Test aggregate MSE
        mse_agg = metrics.mse(self.original, self.reconstructed, per_sample=False)
        self.assertIsInstance(mse_agg, float)
        self.assertGreater(mse_agg, 0)
        
        # Test per-sample MSE
        mse_per_sample = metrics.mse(self.original, self.reconstructed, per_sample=True)
        self.assertEqual(mse_per_sample.shape, (self.batch_size,))
        self.assertTrue(torch.all(mse_per_sample > 0))
        
    def test_mae_calculation(self):
        """Test MAE calculation."""
        metrics = ReconstructionMetrics()
        
        # Test aggregate MAE
        mae_agg = metrics.mae(self.original, self.reconstructed, per_sample=False)
        self.assertIsInstance(mae_agg, float)
        self.assertGreater(mae_agg, 0)
        
        # Test per-sample MAE
        mae_per_sample = metrics.mae(self.original, self.reconstructed, per_sample=True)
        self.assertEqual(mae_per_sample.shape, (self.batch_size,))
        self.assertTrue(torch.all(mae_per_sample > 0))
        
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        metrics = ReconstructionMetrics()
        
        rmse = metrics.rmse(self.original, self.reconstructed, per_sample=False)
        mse = metrics.mse(self.original, self.reconstructed, per_sample=False)
        
        self.assertAlmostEqual(rmse, np.sqrt(mse), places=6)
        
    def test_reconstruction_errors_function(self):
        """Test convenience function for reconstruction errors."""
        errors = calculate_reconstruction_errors(self.original, self.reconstructed)
        
        required_keys = ['mse', 'mae', 'rmse', 'ssim_approx']
        for key in required_keys:
            self.assertIn(key, errors)
            self.assertIsInstance(errors[key], float)
            self.assertGreater(errors[key], 0)


class TestLatentSpaceAnalyzer(unittest.TestCase):
    """Test latent space analysis functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.n_samples = 100
        self.latent_dim = 50
        self.latent_data = torch.randn(self.n_samples, self.latent_dim)
        
    def test_compute_statistics(self):
        """Test latent space statistics computation."""
        analyzer = LatentSpaceAnalyzer()
        stats = analyzer.compute_statistics(self.latent_data)
        
        required_keys = ['mean', 'std', 'var', 'min', 'max', 'mean_per_dim', 
                        'std_per_dim', 'var_per_dim', 'effective_dimensionality']
        
        for key in required_keys:
            self.assertIn(key, stats)
            
        # Check dimensions
        self.assertEqual(len(stats['mean_per_dim']), self.latent_dim)
        self.assertEqual(len(stats['std_per_dim']), self.latent_dim)
        
        # Check effective dimensionality is reasonable
        self.assertGreater(stats['effective_dimensionality'], 0)
        self.assertLessEqual(stats['effective_dimensionality'], self.latent_dim)
        
    def test_prepare_pca(self):
        """Test PCA preparation."""
        analyzer = LatentSpaceAnalyzer()
        pca_data, pca_obj = analyzer.prepare_pca(self.latent_data, n_components=10)
        
        self.assertEqual(pca_data.shape, (self.n_samples, 10))
        self.assertEqual(len(pca_obj.explained_variance_ratio_), 10)
        
    def test_prepare_tsne(self):
        """Test t-SNE preparation."""
        analyzer = LatentSpaceAnalyzer()
        # Use smaller dataset for t-SNE speed
        small_data = self.latent_data[:30]  
        tsne_data = analyzer.prepare_tsne(small_data, n_components=2)
        
        self.assertEqual(tsne_data.shape, (30, 2))
        
    def test_compute_latent_statistics_function(self):
        """Test convenience function for latent statistics."""
        stats = compute_latent_statistics(self.latent_data)
        
        self.assertIn('effective_dimensionality', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)


class TestBatchInferenceProcessor(unittest.TestCase):
    """Test batch inference processing."""
    
    def setUp(self):
        """Set up test model and data."""
        self.model = AutoencoderMLP(input_dim=784, latent_dim=400, device='cpu')
        self.model.eval()
        
        # Create synthetic test data
        self.test_data = np.random.randn(200, 784).astype(np.float32)
        
    def test_process_data_arrays(self):
        """Test processing numpy arrays."""
        processor = BatchInferenceProcessor(self.model, batch_size=32)
        
        original, reconstructed, latent = processor.process_data_arrays(
            self.test_data, return_latent=True
        )
        
        # Check shapes
        self.assertEqual(original.shape, self.test_data.shape)
        self.assertEqual(reconstructed.shape, self.test_data.shape)
        self.assertEqual(latent.shape, (200, 400))
        
        # Check types
        self.assertIsInstance(original, np.ndarray)
        self.assertIsInstance(reconstructed, np.ndarray)
        self.assertIsInstance(latent, np.ndarray)
        
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        processor = BatchInferenceProcessor(self.model, batch_size=32)
        
        # Process data with performance tracking
        processor.process_data_arrays(self.test_data, track_performance=True)
        
        # Get performance summary
        perf_summary = processor.get_performance_summary()
        
        required_keys = ['total_samples', 'total_time', 'avg_throughput_samples_per_sec']
        for key in required_keys:
            self.assertIn(key, perf_summary)
            
        self.assertEqual(perf_summary['total_samples'], 200)
        self.assertGreater(perf_summary['total_time'], 0)


class TestModelIO(unittest.TestCase):
    """Test model saving and loading functionality."""
    
    def setUp(self):
        """Set up test model and temporary directory."""
        self.model = AutoencoderMLP(input_dim=784, latent_dim=400, device='cpu')
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
        
    def test_model_save_and_load(self):
        """Test basic model saving and loading."""
        saver = ModelSaver(base_dir=self.temp_dir)
        
        # Save model
        save_path = saver.save_model(self.model, filepath="test_model.pth")
        self.assertTrue(os.path.exists(save_path))
        
        # Load model
        loader = ModelLoader()
        loaded_model, metadata = loader.load_model(save_path, device='cpu')
        
        # Check model configuration matches
        self.assertEqual(loaded_model.get_config(), self.model.get_config())
        self.assertIn('model_config', metadata)
        
    def test_weights_only_save_load(self):
        """Test saving and loading weights only."""
        saver = ModelSaver(base_dir=self.temp_dir)
        loader = ModelLoader()
        
        # Save weights only
        weights_path = saver.save_weights_only(self.model, "test_weights.pth")
        self.assertTrue(os.path.exists(weights_path))
        
        # Create new model and load weights
        new_model = AutoencoderMLP(input_dim=784, latent_dim=400, device='cpu')
        loader.load_weights_only(new_model, weights_path)
        
        # Test that weights are the same
        for (name1, param1), (name2, param2) in zip(
            self.model.named_parameters(), new_model.named_parameters()
        ):
            self.assertEqual(name1, name2)
            self.assertTrue(torch.allclose(param1, param2))


class TestSASOutputFormatter(unittest.TestCase):
    """Test SAS output formatting functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.n_samples = 50
        self.n_features = 784
        self.n_latent = 400
        
        self.original = np.random.randn(self.n_samples, self.n_features)
        self.reconstructed = self.original + np.random.randn(self.n_samples, self.n_features) * 0.1
        self.latent = np.random.randn(self.n_samples, self.n_latent)
        self.labels = np.random.randint(0, 10, self.n_samples)
        
    def test_format_model_predictions(self):
        """Test formatting model predictions for SAS."""
        formatter = SASOutputFormatter()
        
        df = formatter.format_model_predictions(
            self.original, self.reconstructed, self.latent,
            labels=self.labels
        )
        
        # Check dataframe structure
        self.assertEqual(len(df), self.n_samples)
        
        # Check required columns exist
        required_cols = ['sample_id', 'label', 'reconstruction_mse', 'reconstruction_mae']
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Check original and reconstructed columns
        original_cols = [col for col in df.columns if col.startswith('original_')]
        reconstructed_cols = [col for col in df.columns if col.startswith('reconstructed_')]
        latent_cols = [col for col in df.columns if col.startswith('latent_')]
        
        self.assertEqual(len(original_cols), self.n_features)
        self.assertEqual(len(reconstructed_cols), self.n_features)
        self.assertEqual(len(latent_cols), self.n_latent)
        
    def test_create_sas_compatible_outputs(self):
        """Test creation of SAS-compatible output files."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            output_paths = create_sas_compatible_outputs(
                self.original, self.reconstructed, self.latent,
                output_dir=temp_dir,
                base_filename="test_output"
            )
            
            # Check files were created
            self.assertIn('csv', output_paths)
            self.assertIn('metadata', output_paths)
            
            self.assertTrue(os.path.exists(output_paths['csv']))
            self.assertTrue(os.path.exists(output_paths['metadata']))
            
        finally:
            shutil.rmtree(temp_dir)


class TestModelEvaluator(unittest.TestCase):
    """Test the main ModelEvaluator class."""
    
    def setUp(self):
        """Set up test model and data."""
        self.model = AutoencoderMLP(input_dim=784, latent_dim=400, device='cpu')
        self.test_data = np.random.randn(100, 784).astype(np.float32)
        self.test_labels = np.random.randint(0, 10, 100)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
        
    def test_evaluate_dataset(self):
        """Test comprehensive dataset evaluation."""
        evaluator = ModelEvaluator(self.model, output_dir=self.temp_dir)
        
        results = evaluator.evaluate_dataset(
            dataset=self.test_data,
            labels=self.test_labels,
            save_results=True,
            include_visualization=True
        )
        
        # Check result structure
        required_sections = [
            'evaluation_info',
            'reconstruction_metrics',
            'latent_statistics',
            'performance_metrics',
            'raw_data'
        ]
        
        for section in required_sections:
            self.assertIn(section, results)
            
        # Check evaluation info
        eval_info = results['evaluation_info']
        self.assertEqual(eval_info['n_samples'], 100)
        self.assertEqual(eval_info['n_features'], 784)
        
        # Check reconstruction metrics
        recon_metrics = results['reconstruction_metrics']
        self.assertIn('aggregate', recon_metrics)
        self.assertIn('per_sample_stats', recon_metrics)
        
    def test_benchmark_inference_speed(self):
        """Test inference speed benchmarking."""
        evaluator = ModelEvaluator(self.model, output_dir=self.temp_dir)
        
        benchmark_results = evaluator.benchmark_inference_speed(
            data=self.test_data,
            batch_sizes=[32, 64],
            n_runs=2
        )
        
        # Check benchmark structure
        self.assertIn('benchmark_results', benchmark_results)
        self.assertIn('optimal_batch_size', benchmark_results)
        
        # Check batch size results
        for batch_size in [32, 64]:
            self.assertIn(batch_size, benchmark_results['benchmark_results'])
            batch_results = benchmark_results['benchmark_results'][batch_size]
            self.assertIn('avg_throughput', batch_results)
            self.assertIn('avg_time', batch_results)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test performance benchmarking utilities."""
    
    def test_basic_measurement(self):
        """Test basic performance measurement."""
        benchmark = PerformanceBenchmark()
        
        # Simulate some processing
        benchmark.start_measurement()
        import time
        time.sleep(0.01)  # Short sleep
        benchmark.end_measurement(batch_size=32)
        
        # Get summary
        summary = benchmark.get_summary()
        
        self.assertEqual(summary['total_batches'], 1)
        self.assertEqual(summary['total_samples'], 32)
        self.assertGreater(summary['total_time'], 0)
        self.assertGreater(summary['avg_throughput_samples_per_sec'], 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestReconstructionMetrics,
        TestLatentSpaceAnalyzer,
        TestBatchInferenceProcessor,
        TestModelIO,
        TestSASOutputFormatter,
        TestModelEvaluator,
        TestPerformanceBenchmark
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Autoencoder Evaluation Pipeline Tests")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("The evaluation and inference pipeline is working correctly.")
    else:
        print("\n" + "=" * 60)
        print("✗ Some tests failed. Please check the output above.")
        exit(1)
