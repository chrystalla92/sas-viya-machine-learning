# Autoencoder Synthetic Data Demonstration

This demonstration shows the migrated autoencoder working end-to-end on simple synthetic patterns, providing an easy way to understand how the autoencoder learns to compress and reconstruct data.

## Quick Start

Run the demonstration with default settings:

```bash
cd autoencoder
python demo_autoencoder.py
```

This will:
1. Generate 1000 synthetic 2D patterns (circles, rectangles, lines, crosses, dots)
2. Train an autoencoder for 50 epochs
3. Show training progress and final reconstruction quality
4. Display visualizations comparing original vs reconstructed patterns

## Customization Options

You can customize the demonstration with command-line arguments:

```bash
# Generate more samples and train longer
python demo_autoencoder.py --samples 2000 --epochs 100

# Use a larger hidden layer
python demo_autoencoder.py --hidden-dim 256

# Generate smaller images for faster training
python demo_autoencoder.py --image-size 16 --epochs 30

# Use different random seed
python demo_autoencoder.py --seed 123
```

### Available Arguments

- `--samples N`: Number of synthetic patterns to generate (default: 1000)
- `--epochs N`: Number of training epochs (default: 50)  
- `--hidden-dim N`: Hidden layer size (default: 128)
- `--image-size N`: Size of generated square images (default: 28)
- `--seed N`: Random seed for reproducible results (default: 42)

## What You'll See

### 1. Sample Pattern Generation
The script generates 5 types of geometric patterns:
- **Circles**: Various sizes and positions
- **Rectangles**: Filled and outlined rectangles
- **Lines**: Diagonal lines with different orientations
- **Crosses**: Cross shapes with varying arm lengths
- **Dots**: Random dot patterns

### 2. Training Progress
Real-time training output showing:
- Training and validation loss per epoch
- Time elapsed and final performance metrics
- Model architecture details

### 3. Visualizations
- **Training Curves**: Loss progression over epochs (linear and log scale)
- **Reconstruction Comparison**: Original patterns vs autoencoder reconstructions
- **Error Maps**: Heat maps showing pixel-level reconstruction errors

### 4. Performance Metrics
- **MSE Loss**: Mean squared error between original and reconstructed patterns
- **Pixel Accuracy**: Percentage of pixels reconstructed within threshold
- **RMSE**: Root mean squared error for reconstruction quality

## Example Output

```
🤖 AUTOENCODER SYNTHETIC DEMONSTRATION
============================================================
Configuration:
  Samples: 1000
  Epochs: 50
  Hidden Dimension: 128
  Image Size: 28
  Random Seed: 42

📊 Step 1: Generating synthetic patterns...
Generated 1000 samples
Pattern distribution: {'circle': 206, 'rectangle': 195, 'line': 203, 'cross': 198, 'dots': 198}

🔄 Step 2: Splitting data...
Training samples: 800
Validation samples: 200

🏗️ Step 3: Creating and training autoencoder...
Model Architecture:
  type: MLP Autoencoder
  input_dim: 784
  hidden_dim: 128
  total_parameters: 101,136

Training autoencoder...
Epoch   1/50 | Train Loss: 0.089234 | Val Loss: 0.078456 | Time: 2.1s
Epoch  10/50 | Train Loss: 0.034567 | Val Loss: 0.032198 | Time: 12.5s
Epoch  20/50 | Train Loss: 0.018923 | Val Loss: 0.017654 | Time: 23.1s
...

✅ Demonstration completed successfully!

Key Results:
  • MSE Loss: 0.01234567
  • Pixel Accuracy: 94.2%
  • Mean Pixel Error: 0.076543
```

## Understanding the Results

### Good Performance Indicators:
- **Low MSE Loss** (< 0.05): The autoencoder successfully learned to compress and reconstruct patterns
- **High Pixel Accuracy** (> 90%): Most pixels are reconstructed accurately
- **Converging Loss Curves**: Training and validation losses decrease and stabilize

### What the Autoencoder Learns:
- **Compression**: The 784-dimensional input patterns are compressed to 128 dimensions
- **Reconstruction**: The compressed representation is expanded back to recreate the original pattern
- **Feature Extraction**: The hidden layer learns to represent key geometric features
- **Noise Robustness**: The autoencoder can handle and filter out noise in the patterns

## Technical Details

### Model Architecture:
- **Input Layer**: 784 neurons (28×28 flattened images)
- **Hidden Layer**: 128 neurons with tanh activation (configurable)
- **Output Layer**: 784 neurons for reconstruction
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001

### Training Process:
- Data is split 80% training / 20% validation
- Batch size of 64 samples
- Early visualization of sample patterns
- Real-time loss monitoring
- Automatic device selection (GPU if available)

## Dependencies

The demonstration requires:
- PyTorch
- NumPy  
- Matplotlib
- The existing autoencoder framework modules (`model.py`, `data_utils.py`)

## Files Created

- `synthetic_demo_data.py`: Synthetic pattern generator
- `demo_autoencoder.py`: Main demonstration script  
- `DEMO_README.md`: This documentation

## Next Steps

After running the demonstration, you can:

1. **Experiment with parameters**: Try different hidden dimensions, training epochs, or pattern types
2. **Explore the trained model**: The model object is available for further analysis
3. **Extend patterns**: Add new pattern types to the `SyntheticPatternGenerator`
4. **Compare with real data**: Use the same framework with actual image datasets like MNIST

This demonstration proves that the migrated autoencoder can successfully learn meaningful representations of structured data and reconstruct them with high fidelity.
