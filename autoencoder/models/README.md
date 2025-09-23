# Models Directory  

This directory stores trained model files, checkpoints, and model-related artifacts.

## Structure

```
models/
├── checkpoints/      # Training checkpoints and intermediate models
├── saved_models/     # Final trained models ready for deployment
├── configs/          # Model configuration files
└── logs/             # Training logs and metrics
```

## File Types

- **`.pth` files**: PyTorch model state dictionaries
- **`.pt` files**: Complete PyTorch model files  
- **`.yaml/.json` files**: Configuration files
- **`.log` files**: Training and evaluation logs

## Usage

```python
# Loading a trained model
model = torch.load('models/saved_models/mnist_autoencoder.pth')

# Loading model state dict
model = MyAutoencoder()
model.load_state_dict(torch.load('models/checkpoints/epoch_100.pth'))
```

## Naming Convention

- Use descriptive names: `dataset_architecture_version.pth`
- Example: `mnist_conv_autoencoder_v1.pth`
- Include date for experiments: `mnist_vae_20241201.pth`
