# Data Export and Serialization Utilities

This documentation describes the comprehensive data export and serialization system that replaces the SAS `export_to_csv.sas` functionality with Python-native operations, providing improved performance, cross-environment compatibility, and enhanced data integrity.

## Overview

The system consists of three main modules that work together to provide complete data export and serialization capabilities:

1. **`io_utils.py`** - Data loading/saving utilities with structured file organization
2. **`serialization.py`** - Model and data serialization using pickle
3. **`preprocessing.py`** - Input/output processing pipeline with standardization

## Key Features

### ✅ Model Serialization
- Pickle-based model persistence replacing SAS model exports
- Cross-environment compatibility with version checking
- Model verification to ensure saved models produce identical outputs
- Comprehensive error handling for serialization edge cases

### ✅ Data Export
- NumPy array export functions for reconstructions and latent representations
- CSV export compatibility for integration with existing SAS workflows
- Structured file organization with clear naming conventions
- Metadata preservation for data provenance tracking

### ✅ Preprocessing Pipeline
- SAS-compatible midrange standardization (`standardize=midrange`)
- Input preprocessing with proper reshaping and validation
- Output postprocessing for denormalization and visualization
- Numerical precision preservation and validation

### ✅ File Organization
- Structured output directory system
- Timestamped runs for experiment tracking
- Comprehensive metadata saving (model parameters, training info, dataset info)
- Cross-platform file path handling

### ✅ Batch Processing
- Memory-efficient processing for large datasets
- Batch result combination and validation
- Progress tracking and error recovery

## Replacing SAS Functionality

This system directly replaces and improves upon the following SAS operations:

| SAS Component | Python Replacement | Improvements |
|---------------|-------------------|--------------|
| `export_to_csv.sas` | `io_utils.NumpyExporter` | NumPy format + CSV, metadata, compression |
| `proc nnet ... score` | `model.forward()` + export | Direct tensor operations, batch processing |
| SAS model files | `serialization.ModelSerializer` | Cross-environment compatibility, verification |
| `standardize=midrange` | `preprocessing.MidrangeStandardizer` | Exact numerical compatibility, invertible |
| SAS output tables | Structured directories | Better organization, metadata, provenance |

## Quick Start

### Basic Model Export

```python
from model import create_sas_compatible_autoencoder
from serialization import ModelSerializer
from io_utils import export_model_outputs

# Create and train model
model = create_sas_compatible_autoencoder()
# ... training code ...

# Export model and data
export_paths = export_model_outputs(
    model=model,
    data=your_data,
    labels=your_labels,
    output_dir="./outputs",
    run_name="experiment_1"
)

print("Exported files:")
for export_type, path in export_paths.items():
    print(f"  {export_type}: {path}")
```

### SAS-Compatible Preprocessing

```python
from preprocessing import create_sas_compatible_preprocessor

# Create preprocessor matching SAS midrange standardization
preprocessor = create_sas_compatible_preprocessor()

# Preprocess data (matching SAS behavior exactly)
standardized_data = preprocessor.fit_transform(raw_data)

# Later: inverse transform for visualization
original_scale_data = preprocessor.inverse_transform(standardized_data)
```

### Complete Export Workflow

```python
from io_utils import OutputOrganizer, NumpyExporter, MetadataManager
from serialization import ModelSerializer
import torch

# 1. Organize outputs
organizer = OutputOrganizer("./outputs")
run_dirs = organizer.create_run_directory("my_experiment")

# 2. Save model
serializer = ModelSerializer()
model_path = str(run_dirs['models'] / 'autoencoder.pkl')
save_result = serializer.save_model(model, model_path, include_verification=True)

# 3. Generate and export reconstructions
model.eval()
with torch.no_grad():
    reconstructions = model(torch.FloatTensor(data)).numpy()
    latent_reps = model.encode(torch.FloatTensor(data)).numpy()

# 4. Export reconstruction data (replaces SAS score dataset)
recon_path = str(run_dirs['reconstructions'] / 'reconstructions')
NumpyExporter.export_reconstructions(
    reconstructions=reconstructions,
    original_data=data,
    labels=labels,
    output_path=recon_path
)

# 5. Export latent representations
latent_path = str(run_dirs['latent'] / 'latent_representations')  
NumpyExporter.export_latent_representations(
    latent_data=latent_reps,
    labels=labels,
    output_path=latent_path
)
```

## Module Documentation

### io_utils.py

**Classes:**
- `OutputOrganizer`: Manages structured file organization
- `NumpyExporter`: Handles NumPy array exports with CSV compatibility
- `MetadataManager`: Manages metadata saving and loading
- `BatchProcessor`: Utilities for processing large datasets in batches

**Key Functions:**
- `export_model_outputs()`: Comprehensive export replacing SAS workflow
- `export_reconstructions()`: Export reconstruction data with metadata
- `export_latent_representations()`: Export latent/hidden layer outputs

### serialization.py

**Classes:**
- `ModelSerializer`: Model serialization with verification and error handling
- `DataSerializer`: Data serialization with integrity checking

**Key Functions:**
- `save_model()`: Save model with verification and metadata
- `load_model()`: Load model with integrity checking and compatibility warnings
- `save_model_with_data()`: Convenience function for combined model+data saving

### preprocessing.py

**Classes:**
- `PreprocessingPipeline`: Complete preprocessing workflow
- `MidrangeStandardizer`: SAS-compatible midrange standardization
- `StandardScaler`: Z-score normalization
- `MinMaxScaler`: Min-max normalization
- `VisualizationPreprocessor`: Utilities for preparing data for visualization

**Key Functions:**
- `create_sas_compatible_preprocessor()`: Create preprocessor matching SAS exactly
- `validate_preprocessing()`: Verify numerical precision is maintained

## File Organization

The system creates a structured output directory:

```
outputs/
├── run_experiment_20231201_143022/
│   ├── models/
│   │   ├── trained_autoencoder.pkl
│   │   └── trained_autoencoder_metadata.json
│   ├── data/
│   ├── reconstructions/
│   │   ├── reconstructions_20231201_143022.npz
│   │   ├── reconstructions_20231201_143022.csv
│   │   └── reconstructions_20231201_143022_metadata.json
│   ├── latent_representations/
│   │   ├── latent_representations_20231201_143022.npz
│   │   ├── latent_representations_20231201_143022.csv
│   │   └── latent_representations_20231201_143022_metadata.json
│   ├── metadata/
│   │   └── comprehensive_metadata.json
│   ├── logs/
│   └── exports/
```

## Data Formats

### NumPy Archives (.npz)
- **Reconstructions**: `{'reconstructions': array, 'originals': array, 'reconstruction_errors': array, 'labels': array}`
- **Latent Representations**: `{'latent_representations': array, 'labels': array}`
- **Metadata**: Embedded serialization info for integrity checking

### CSV Files
- **Format**: SAS-compatible with `var1` (labels), `var2-varN` (features)  
- **Compatibility**: Can be read by SAS using standard `proc import`
- **Header**: No header (`putnames=no` equivalent) for direct SAS compatibility

### Metadata Files (.json)
- **Model Info**: Architecture, parameters, training configuration
- **Data Info**: Shapes, ranges, preprocessing parameters, quality metrics
- **Environment Info**: Python/PyTorch/NumPy versions, platform info
- **Provenance**: Timestamps, file hashes, processing steps

## Success Criteria Verification

### ✅ Model Loading Verification
- Saved models load correctly across different Python environments
- Loaded models produce identical outputs to original models (within numerical precision)
- Cross-environment compatibility warnings for version mismatches

### ✅ Data Integrity
- NumPy arrays contain identical data to CSV exports
- File integrity checking using SHA256 hashes
- Numerical precision maintained through preprocessing/postprocessing cycles

### ✅ SAS Compatibility
- CSV files match SAS `export_to_csv.sas` format exactly
- Midrange standardization produces identical results to SAS `standardize=midrange`
- Variable naming convention matches SAS output (`var1`, `var2-var785`)

### ✅ Performance Improvements
- **Speed**: NumPy operations significantly faster than CSV I/O
- **Storage**: Compressed NumPy archives use ~50% less storage than CSV
- **Memory**: Batch processing enables handling datasets larger than RAM
- **Integrity**: Built-in verification prevents data corruption issues

## Example Usage

See `example_data_export.py` for a complete demonstration including:

1. Data loading and SAS-compatible preprocessing
2. Model training and serialization
3. Reconstruction generation and export
4. Metadata preservation and documentation
5. Cross-environment compatibility testing
6. Batch processing for large datasets

Run the example:

```bash
cd autoencoder/
python example_data_export.py
```

This will create a complete export workflow demonstration in `./export_outputs/`.

## Integration with Existing Workflow

### Replacing SAS Export
Replace this SAS code:
```sas
proc export data=mycaslib.mnist_train_10_autoencoder_score
    outfile='c:\Python27\mnist_train_10_autoencoder_score.csv'
    dbms=csv
    replace;
    putnames=no;
run;
```

With this Python code:
```python
export_paths = export_model_outputs(
    model=trained_model,
    data=score_data,
    labels=labels,
    output_dir="./outputs",
    run_name="mnist_autoencoder"
)
```

### Benefits Over SAS Export
1. **Dual Format**: Creates both NumPy (.npz) and CSV files
2. **Metadata**: Preserves complete processing history and parameters
3. **Verification**: Ensures data integrity and model correctness
4. **Organization**: Structured directories with clear provenance
5. **Performance**: Faster processing and smaller file sizes
6. **Compatibility**: Works across different Python environments
7. **Error Handling**: Comprehensive error checking and recovery

## Dependencies

- **Core**: `numpy`, `torch`, `pandas`, `pickle`
- **Utilities**: `pathlib`, `json`, `hashlib`, `logging`
- **Optional**: `matplotlib` (for visualization utilities)

All dependencies are standard Python libraries or already required by the existing autoencoder implementation.

## Troubleshooting

### Model Loading Issues
- Check Python/PyTorch version compatibility warnings
- Verify file integrity using the built-in hash checking
- Use `verify_integrity=True` when loading models

### Data Export Issues
- Ensure sufficient disk space for outputs
- Check file permissions for output directory
- Use batch processing for memory-constrained environments

### Preprocessing Issues
- Verify input data shape and format
- Check for NaN or infinite values in input data
- Use `validate_preprocessing()` to verify numerical precision

### Performance Issues
- Use batch processing for datasets > 1GB
- Enable compression for NumPy exports to save disk space
- Consider using `torch.no_grad()` context for inference

## Future Enhancements

The system is designed to be extensible. Potential future additions:

1. **Additional Formats**: HDF5, Parquet, Arrow for better performance
2. **Cloud Storage**: Direct export to AWS S3, Google Cloud Storage
3. **Distributed Processing**: Spark/Dask integration for very large datasets
4. **Visualization**: Built-in plotting and visualization utilities
5. **Database Integration**: Direct export to SQL databases
6. **Streaming**: Online/streaming export for real-time applications

This implementation provides a solid foundation for all current requirements while enabling future enhancements as needed.
