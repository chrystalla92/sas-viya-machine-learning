#!/usr/bin/env python3
"""
Example script demonstrating MLP Autoencoder training.

This script shows how to:
1. Set up training environment
2. Create and configure the autoencoder model
3. Prepare MNIST data
4. Train the model with comprehensive monitoring
5. Evaluate and visualize results

Usage:
    python train_autoencoder.py [--config CONFIG_FILE] [--resume CHECKPOINT]
"""

import argparse
import logging
from pathlib import Path

import torch

from mnist_autoencoder.models.autoencoder import MLPAutoencoder
from mnist_autoencoder.training import (
    Trainer,
    TrainingConfig,
    create_training_config,
    create_sas_compatible_config,
    prepare_mnist_data,
    setup_training_environment,
    plot_training_history,
    plot_reconstruction_examples,
    evaluate_reconstruction_quality
)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MLP Autoencoder on MNIST")
    parser.add_argument("--config", type=str, help="Path to training configuration JSON file")
    parser.add_argument("--sas-compatible", action="store_true", 
                       help="Use SAS PROC NNET compatible configuration")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--data-dir", type=str, default="./data", 
                       help="Directory for MNIST data")
    parser.add_argument("--output-dir", type=str, default="./training_output",
                       help="Output directory for training artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", 
                       choices=["adam", "lbfgs", "sgd"], help="Optimizer type")
    parser.add_argument("--device", type=str, help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Set up training environment
    directories = setup_training_environment(
        save_dir=args.output_dir,
        log_level="INFO",
        seed=args.seed
    )
    
    logger = logging.getLogger("TrainingExample")
    logger.info("=" * 60)
    logger.info("MLP AUTOENCODER TRAINING")
    logger.info("=" * 60)
    
    # Create training configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        from mnist_autoencoder.training import load_training_config
        config = load_training_config(args.config)
    elif args.sas_compatible:
        logger.info("Using SAS PROC NNET compatible configuration")
        config = create_sas_compatible_config()
    else:
        logger.info("Using custom configuration")
        config = create_training_config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            device=args.device,
            seed=args.seed
        )
    
    # Override paths with command line arguments
    config.save_dir = directories["checkpoints"]
    config.log_dir = directories["logs"]
    
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Optimizer: {config.optimizer}")
    logger.info(f"  Early stopping: {config.early_stopping}")
    logger.info(f"  Patience: {config.patience}")
    
    # Create model
    logger.info("Creating MLP Autoencoder model...")
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MLPAutoencoder(
        input_size=784,
        hidden_size=400,
        device=device,
        seed=config.seed
    )
    
    model_info = model.get_model_info()
    logger.info(f"Model created:")
    logger.info(f"  Architecture: {model_info['architecture']}")
    logger.info(f"  Parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Device: {model_info['device']}")
    
    # Prepare data
    logger.info("Preparing MNIST data...")
    train_loader, val_loader = prepare_mnist_data(
        data_dir=args.data_dir,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        num_workers=config.num_workers,
        download=True
    )
    
    logger.info(f"Data prepared:")
    logger.info(f"  Training samples: {len(train_loader.dataset):,}")
    logger.info(f"  Validation samples: {len(val_loader.dataset):,}")
    logger.info(f"  Batches per epoch: {len(train_loader):,}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(model, config)
    
    # Train model
    logger.info("Starting training...")
    try:
        results = trainer.train(train_loader, val_loader, resume_from=args.resume)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {results['final_train_loss']:.6f}")
        logger.info(f"Final validation loss: {results['final_val_loss']:.6f}")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f} (epoch {results['best_epoch']})")
        logger.info(f"Total training time: {results['total_training_time']:.1f}s")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        results = {"training_completed": False}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Evaluate model
    if results.get("training_completed", False):
        logger.info("Evaluating final model...")
        
        # Comprehensive evaluation
        eval_metrics = trainer.evaluate_model(val_loader)
        logger.info("Evaluation metrics:")
        logger.info(f"  MSE: {eval_metrics['mse']:.6f}")
        logger.info(f"  MAE: {eval_metrics['mae']:.6f}")
        logger.info(f"  RMSE: {eval_metrics['rmse']:.6f}")
        
        # Reconstruction quality assessment
        recon_metrics = evaluate_reconstruction_quality(
            model=trainer.model,
            data_loader=val_loader,
            num_samples=100
        )
        logger.info("Reconstruction quality:")
        logger.info(f"  Mean MSE: {recon_metrics['mse_mean']:.6f} ± {recon_metrics['mse_std']:.6f}")
        logger.info(f"  Mean MAE: {recon_metrics['mae_mean']:.6f} ± {recon_metrics['mae_std']:.6f}")
        logger.info(f"  Mean Similarity: {recon_metrics['similarity_mean']:.3f} ± {recon_metrics['similarity_std']:.3f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        try:
            # Plot training history
            plot_path = directories["plots"] / "training_history.png"
            plot_training_history(
                metrics=trainer.metrics,
                save_path=plot_path,
                show=False
            )
            logger.info(f"Training history plot saved: {plot_path}")
            
            # Plot reconstruction examples
            examples_path = directories["plots"] / "reconstruction_examples.png"
            plot_reconstruction_examples(
                model=trainer.model,
                data_loader=val_loader,
                num_examples=8,
                save_path=examples_path,
                show=False
            )
            logger.info(f"Reconstruction examples saved: {examples_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
        
        # Save training summary
        summary = trainer.get_training_summary()
        summary_path = directories["results"] / "training_summary.json"
        
        import json
        with open(summary_path, 'w') as f:
            # Convert non-serializable objects
            serializable_summary = {}
            for key, value in summary.items():
                if isinstance(value, dict):
                    serializable_summary[key] = {k: v for k, v in value.items() 
                                               if isinstance(v, (int, float, str, bool, list))}
                else:
                    serializable_summary[key] = value
            
            json.dump(serializable_summary, f, indent=2)
        
        logger.info(f"Training summary saved: {summary_path}")
    
    logger.info("=" * 60)
    logger.info("TRAINING EXAMPLE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info(f"  Checkpoints: {directories['checkpoints']}")
    logger.info(f"  Logs: {directories['logs']}")
    logger.info(f"  Plots: {directories['plots']}")
    logger.info(f"  Results: {directories['results']}")


if __name__ == "__main__":
    main()
