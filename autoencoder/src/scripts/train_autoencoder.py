#!/usr/bin/env python3
"""
Autoencoder Training Script

This script demonstrates how to use the L-BFGS autoencoder training system
with all the implemented features: convergence checking, metrics tracking,
checkpointing, and visualization.

Usage:
    python train_autoencoder.py [--data_path PATH] [--checkpoint_dir PATH]
                                [--max_iterations INT] [--batch_size INT]
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import create_autoencoder
from data.mnist_loader import create_mnist_loader
from training import (
    create_trainer,
    create_checkpoint_manager,
    plot_training_progress,
    create_training_report,
    DEFAULT_CONFIG
)
from utils import get_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PyTorch Autoencoder with L-BFGS')
    parser.add_argument('--data_path', type=str, default='./data', 
                       help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--max_iterations', type=int, default=DEFAULT_CONFIG['max_iterations'],
                       help='Maximum training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--convergence_tolerance', type=float, 
                       default=DEFAULT_CONFIG['convergence_tolerance'],
                       help='Convergence tolerance (fConv)')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if None)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--report_dir', type=str, default='./reports',
                       help='Directory to save training reports')
    
    args = parser.parse_args()
    
    logger.info("Starting autoencoder training...")
    logger.info(f"Configuration: {vars(args)}")
    
    # Setup device
    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Using device: {device}")
    
    # Create data loader
    logger.info("Loading MNIST dataset...")
    data_loader_manager = create_mnist_loader(
        data_root=args.data_path,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
    
    train_loader, test_loader, val_loader = data_loader_manager.create_data_loaders()
    logger.info(f"Data loaded: {len(train_loader)} training batches")
    
    # Create model
    logger.info("Creating autoencoder model...")
    model = create_autoencoder(
        input_dim=784,
        hidden_dim=400,
        seed=args.seed,
        device=str(device)
    )
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        args.checkpoint_dir,
        max_checkpoints=5,
        save_best=True
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        max_iterations=args.max_iterations,
        convergence_tolerance=args.convergence_tolerance,
        seed=args.seed,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume, model, trainer.optimizer, device)
    
    # Train the model
    logger.info("Starting training...")
    try:
        results = trainer.train(train_loader, verbose=True)
        
        # Log final results
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Final Loss: {results['final_loss']:.6e}")
        logger.info(f"Best Loss: {results['best_loss']:.6e}")
        logger.info(f"Total Iterations: {results['total_iterations']}")
        logger.info(f"Converged: {results['converged']}")
        logger.info(f"Training Time: {results['total_training_time']:.2f}s")
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_results = trainer.evaluate(test_loader)
        logger.info(f"Test Reconstruction Loss: {test_results['reconstruction_loss']:.6e}")
        
        # Create training report with visualizations
        logger.info(f"\nCreating training report in {args.report_dir}...")
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sample data for reconstruction visualization
        sample_batch, _ = next(iter(test_loader))
        sample_data = sample_batch[:10]  # First 10 samples
        
        # Generate comprehensive report
        report_files = create_training_report(
            model=model,
            metrics_data=results,
            sample_data=sample_data,
            save_dir=str(report_dir),
            report_name="autoencoder_training"
        )
        
        logger.info(f"Training report generated with {len(report_files)} visualizations:")
        for file_path in report_files:
            logger.info(f"  - {file_path}")
        
        # Save training metrics
        metrics_file = report_dir / "training_metrics.json"
        trainer.metrics.save_metrics(str(metrics_file))
        logger.info(f"Training metrics saved to: {metrics_file}")
        
        # Save final model state
        final_checkpoint = checkpoint_manager.save_checkpoint(
            model=model,
            iteration=results['total_iterations'],
            loss=results['final_loss'],
            optimizer=trainer.optimizer,
            metrics=results,
            is_best=False  # Best is already saved during training
        )
        logger.info(f"Final checkpoint saved: {final_checkpoint}")
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*60)
        
        return results
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Save current state
        interrupt_checkpoint = checkpoint_manager.save_checkpoint(
            model=model,
            iteration=trainer.current_iteration,
            loss=trainer.loss_history[-1] if trainer.loss_history else float('inf'),
            optimizer=trainer.optimizer,
            metadata={'interrupted': True}
        )
        logger.info(f"Interrupted training state saved: {interrupt_checkpoint}")
        return None
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
