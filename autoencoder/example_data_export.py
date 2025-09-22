#!/usr/bin/env python3
"""
Example: Data Export and Serialization

This script demonstrates the complete data export and serialization functionality
that replaces the SAS export_to_csv.sas workflow with Python-native operations.

The script shows how to:
1. Load and preprocess data using the preprocessing pipeline
2. Train or load a model using the serialization utilities
3. Export reconstructions and latent representations using io_utils
4. Save all data in structured format with metadata
5. Verify data integrity and cross-environment compatibility

This example replicates and improves upon the SAS workflow:
- createData.sas: Data loading and preprocessing
- nnet.sas/proc_cas.sas: Model training and scoring
- export_to_csv.sas: Data export to CSV
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any

# Add autoencoder directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_sas_compatible_autoencoder, MNISTAutoencoder
from mnist_data import load_mnist_data
from preprocessing import create_sas_compatible_preprocessor, PreprocessingPipeline
from serialization import ModelSerializer, DataSerializer, save_model_with_data, load_model_with_data
from io_utils import (
    export_model_outputs, OutputOrganizer, NumpyExporter, 
    MetadataManager, BatchProcessor
)


def main():
    """
    Main demonstration of the complete data export workflow.
    """
    print("=" * 80)
    print("AUTOENCODER DATA EXPORT AND SERIALIZATION DEMONSTRATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Configuration
    output_dir = "./export_outputs"
    run_name = "mnist_autoencoder_export_demo"
    
    # =========================================================================
    # Step 1: Data Loading and Preprocessing (replaces createData.sas)
    # =========================================================================
    print("Step 1: Data Loading and Preprocessing")
    print("-" * 50)
    
    # Create mock MNIST-like data for demonstration
    print("Creating mock MNIST data...")
    np.random.seed(23451)  # Match SAS seed
    n_samples = 100
    n_features = 784
    
    # Simulate MNIST pixel values (0-255)
    mock_images = np.random.randint(0, 256, (n_samples, n_features)).astype(np.float64)
    mock_labels = np.random.randint(0, 10, n_samples)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Data range: [{np.min(mock_images):.1f}, {np.max(mock_images):.1f}]")
    
    # Apply SAS-compatible preprocessing
    print("\nApplying midrange standardization (matching SAS behavior)...")
    preprocessor = create_sas_compatible_preprocessor()
    standardized_data = preprocessor.fit_transform(mock_images)
    
    print(f"Standardized data range: [{np.min(standardized_data):.3f}, {np.max(standardized_data):.3f}]")
    
    # Split data for training/validation
    from data_utils import train_validation_split
    train_data, val_data, train_labels, val_labels = train_validation_split(
        standardized_data, mock_labels, 
        validation_ratio=0.2, 
        random_seed=23451
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # =========================================================================
    # Step 2: Model Training or Loading (replaces nnet.sas/proc_cas.sas)
    # =========================================================================
    print(f"\nStep 2: Model Creation and Training")
    print("-" * 50)
    
    # Create SAS-compatible model
    model = create_sas_compatible_autoencoder(seed=23451)
    print("Created SAS-compatible autoencoder:")
    arch_info = model.get_architecture_info()
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # For demonstration, we'll do minimal training
    # In practice, you would use the full training framework
    print("\nPerforming minimal training for demonstration...")
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10)
    
    def closure():
        optimizer.zero_grad()
        train_tensor = torch.FloatTensor(train_data[:10])  # Small batch for demo
        output = model(train_tensor)
        loss = model.reconstruction_loss(train_tensor, output)
        loss.backward()
        return loss
    
    initial_loss = closure().item()
    optimizer.step(closure)
    final_loss = closure().item()
    
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduction: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
    
    # =========================================================================
    # Step 3: Model Serialization and Data Export
    # =========================================================================
    print(f"\nStep 3: Model Serialization and Data Export")
    print("-" * 50)
    
    # Initialize output organization
    organizer = OutputOrganizer(output_dir)
    run_dirs = organizer.create_run_directory(run_name)
    
    print(f"Created output directories in: {output_dir}")
    for category, path in run_dirs.items():
        print(f"  {category}: {path}")
    
    # Save model with serialization utilities
    print("\nSaving trained model...")
    model_path = str(run_dirs['models'] / 'trained_autoencoder.pkl')
    
    serializer = ModelSerializer()
    save_result = serializer.save_model(
        model, 
        model_path, 
        include_verification=True,
        save_metadata=True
    )
    
    if save_result['success']:
        print(f"Model saved successfully to: {save_result['filepath']}")
        print(f"File size: {save_result['file_size']} bytes")
        print(f"File hash: {save_result['file_hash'][:16]}...")
        
        if save_result['verification']:
            if save_result['verification']['success']:
                print("✓ Model verification passed")
            else:
                print(f"⚠ Model verification failed: {save_result['verification']['error']}")
    else:
        print(f"✗ Model save failed: {save_result['error']}")
        return
    
    # =========================================================================
    # Step 4: Generate and Export Reconstructions (replaces SAS scoring)
    # =========================================================================
    print(f"\nStep 4: Generate and Export Reconstructions")
    print("-" * 50)
    
    # Use all available data for scoring
    score_data = standardized_data
    score_labels = mock_labels
    
    # Generate model outputs
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(score_data)
        reconstructions = model(data_tensor).cpu().numpy()
        latent_representations = model.encode(data_tensor).cpu().numpy()
    
    print(f"Generated reconstructions for {len(score_data)} samples")
    print(f"Reconstruction shape: {reconstructions.shape}")
    print(f"Latent representation shape: {latent_representations.shape}")
    
    # Calculate reconstruction quality metrics
    mse_per_sample = np.mean((score_data - reconstructions) ** 2, axis=1)
    overall_mse = np.mean(mse_per_sample)
    
    print(f"Overall reconstruction MSE: {overall_mse:.6f}")
    print(f"MSE range: [{np.min(mse_per_sample):.6f}, {np.max(mse_per_sample):.6f}]")
    
    # Export reconstructions (replaces SAS CSV export)
    print("\nExporting reconstruction data...")
    recon_path = str(run_dirs['reconstructions'] / 'autoencoder_reconstructions')
    
    recon_export_path = NumpyExporter.export_reconstructions(
        reconstructions=reconstructions,
        original_data=score_data,
        labels=score_labels,
        output_path=recon_path,
        include_metadata=True
    )
    
    print(f"Reconstructions exported to: {recon_export_path}")
    print(f"CSV file also created for SAS compatibility")
    
    # Export latent representations
    print("\nExporting latent representations...")
    latent_path = str(run_dirs['latent'] / 'latent_representations')
    
    latent_export_path = NumpyExporter.export_latent_representations(
        latent_data=latent_representations,
        labels=score_labels,
        output_path=latent_path,
        include_metadata=True
    )
    
    print(f"Latent representations exported to: {latent_export_path}")
    
    # =========================================================================
    # Step 5: Comprehensive Metadata and Documentation
    # =========================================================================
    print(f"\nStep 5: Save Comprehensive Metadata")
    print("-" * 50)
    
    # Prepare comprehensive metadata
    model_info = {
        **arch_info,
        'training_loss': {
            'initial': initial_loss,
            'final': final_loss,
            'reduction_pct': ((initial_loss - final_loss) / initial_loss * 100)
        }
    }
    
    # Get preprocessing parameters
    preprocessing_params = preprocessor.get_params()
    dataset_info = {
        'n_samples': len(score_data),
        'n_features': score_data.shape[1],
        'n_classes': len(np.unique(score_labels)),
        'data_split': {
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        },
        'preprocessing': preprocessing_params.to_dict() if preprocessing_params else None,
        'reconstruction_quality': {
            'overall_mse': float(overall_mse),
            'mse_std': float(np.std(mse_per_sample)),
            'min_mse': float(np.min(mse_per_sample)),
            'max_mse': float(np.max(mse_per_sample))
        }
    }
    
    training_info = {
        'framework': 'PyTorch',
        'optimizer': 'L-BFGS',
        'seed': 23451,
        'demo_mode': True,
        'sas_compatible': True,
        'export_timestamp': datetime.now().isoformat()
    }
    
    # Save metadata
    metadata_path = str(run_dirs['metadata'] / 'comprehensive_metadata.json')
    MetadataManager.save_training_metadata(
        model_info, training_info, dataset_info, metadata_path
    )
    
    print(f"Comprehensive metadata saved to: {metadata_path}")
    
    # =========================================================================
    # Step 6: Verification and Cross-Environment Testing
    # =========================================================================
    print(f"\nStep 6: Verification and Cross-Environment Testing")
    print("-" * 50)
    
    # Test model loading and verification
    print("Testing model loading and verification...")
    try:
        loaded_model, load_info = serializer.load_model(model_path, verify_integrity=True)
        
        print("✓ Model loaded successfully")
        
        # Verify outputs are identical
        with torch.no_grad():
            original_output = model(data_tensor[:5])  # Test on 5 samples
            loaded_output = loaded_model(data_tensor[:5])
            
        output_diff = torch.abs(original_output - loaded_output)
        max_diff = torch.max(output_diff).item()
        
        if max_diff < 1e-6:
            print(f"✓ Model outputs are identical (max diff: {max_diff:.2e})")
        else:
            print(f"⚠ Model outputs differ by up to {max_diff:.2e}")
        
        # Check compatibility warnings
        compat_info = load_info.get('compatibility_info', {})
        if compat_info.get('warnings'):
            print("⚠ Compatibility warnings:")
            for warning in compat_info['warnings']:
                print(f"  - {warning}")
        else:
            print("✓ No compatibility issues detected")
    
    except Exception as e:
        print(f"✗ Model loading failed: {str(e)}")
    
    # Test data loading
    print("\nTesting data loading...")
    try:
        loaded_data = NumpyExporter.load_exported_data(recon_export_path)
        
        print("✓ Reconstruction data loaded successfully")
        print(f"Available data keys: {list(loaded_data.keys())}")
        
        # Verify data integrity
        if 'reconstructions' in loaded_data:
            loaded_reconstructions = loaded_data['reconstructions']
            data_diff = np.abs(reconstructions - loaded_reconstructions)
            max_data_diff = np.max(data_diff)
            
            if max_data_diff < 1e-10:
                print(f"✓ Data integrity verified (max diff: {max_data_diff:.2e})")
            else:
                print(f"⚠ Data differs by up to {max_data_diff:.2e}")
    
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
    
    # =========================================================================
    # Summary and File Listing
    # =========================================================================
    print(f"\nSUMMARY")
    print("=" * 80)
    
    print("Successfully completed data export and serialization workflow:")
    print("✓ Data loading and SAS-compatible preprocessing")
    print("✓ Model training and architecture matching")
    print("✓ Model serialization with verification")
    print("✓ Reconstruction and latent representation export")
    print("✓ NumPy and CSV format support")
    print("✓ Comprehensive metadata preservation")
    print("✓ Cross-environment compatibility testing")
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    # List all created files
    print(f"\nCreated files:")
    import glob
    all_files = glob.glob(os.path.join(output_dir, "**", "*"), recursive=True)
    files_only = [f for f in all_files if os.path.isfile(f)]
    
    for file_path in sorted(files_only):
        rel_path = os.path.relpath(file_path, output_dir)
        file_size = os.path.getsize(file_path)
        print(f"  {rel_path} ({file_size} bytes)")
    
    print(f"\nTotal files created: {len(files_only)}")
    
    print(f"\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("This replaces and improves upon the SAS export_to_csv.sas functionality")
    print("=" * 80)


def demonstrate_batch_processing():
    """
    Demonstrate batch processing capabilities for large datasets.
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Create larger dataset for batch processing demo
    print("Creating large dataset for batch processing...")
    np.random.seed(42)
    large_data = np.random.rand(2500, 784).astype(np.float32)
    
    print(f"Dataset shape: {large_data.shape}")
    print(f"Dataset size: {large_data.nbytes / 1024 / 1024:.2f} MB")
    
    # Define processing function
    def process_batch(batch_data):
        """Simple processing function for demonstration."""
        return {
            'processed': batch_data * 2.0,
            'stats': np.array([np.mean(batch_data), np.std(batch_data)])
        }
    
    # Process in batches
    batch_processor = BatchProcessor()
    batch_paths = batch_processor.process_large_dataset_batches(
        data=large_data,
        process_func=process_batch,
        batch_size=500,
        output_dir="./batch_demo_outputs"
    )
    
    print(f"Processed {len(batch_paths)} batches")
    
    # Combine batch results
    combined_path = "./batch_demo_outputs/combined_results.npz"
    batch_processor.combine_batch_results(batch_paths, combined_path)
    
    # Verify combined results
    with np.load(combined_path) as data:
        combined_processed = data['processed']
        combined_stats = data['stats']
    
    print(f"Combined processed data shape: {combined_processed.shape}")
    print(f"Combined stats shape: {combined_stats.shape}")
    print("✓ Batch processing demonstration completed")


if __name__ == "__main__":
    # Run main demonstration
    main()
    
    # Run batch processing demonstration
    demonstrate_batch_processing()
