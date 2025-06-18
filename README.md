# CAMELS-Diffusion-Model

# Cosmological Hydrogen Intensity Map Generation with Diffusion Models

This repository implements a context-aware diffusion model for generating cosmological hydrogen intensity maps conditioned on astrophysical and cosmological parameters using the CAMELS dataset.

## Overview

Diffusion models offer promising approaches for generating cosmological hydrogen intensity maps with controllable parameter conditioning. This implementation uses a context-aware U-Net architecture that generates CAMELS dataset maps conditioned on up to 6 astrophysical and cosmological parameters.

## Features

- **Parameter-Conditional Generation**: Generate maps conditioned on 1-6 cosmological parameters
- **Comprehensive Evaluation**: ELBO, BPD, and negative log-likelihood metrics
- **Classifier-Free Guidance**: Support for guided generation with adjustable strength
- **Parameter Sensitivity Analysis**: Systematic evaluation of parameter effects
- **GPU Acceleration**: Optimized for CUDA-enabled training

## Requirements

### Environment Setup

```bash
# Create conda environment
conda create -n ddpm_env python=3.9
conda activate ddpm_env

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install additional dependencies
pip install numpy matplotlib tqdm
```

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM)
- **RAM**: 32GB+ recommended for large datasets
- **Storage**: ~10GB for datasets and model outputs

## Data Preparation

### Expected Data Structure

```
data/
├── Maps_HI_IllustrisTNG_LH_z=0.00.npy  # HI intensity maps [N, H, W]
└── params.npy                          # Cosmological parameters [N_sets, 6]
```

### Data Format

- **Maps**: Shape `[N_total_maps, height, width]` where `N_total_maps = N_parameter_sets × 15`
- **Parameters**: Shape `[N_parameter_sets, 6]` containing:
  1. Ωm (matter density)
  2. σ8 (matter fluctuation amplitude)  
  3. ASN1 (supernova feedback parameter 1)
  4. AAGN1 (AGN feedback parameter 1)
  5. ASN2 (supernova feedback parameter 2)
  6. AAGN2 (AGN feedback parameter 2)

## Usage

### Local Training

```bash
# Basic training with all 6 parameters
python train_diffusion.py 1e-5 100 1500 6

# Training with fewer parameters
python train_diffusion.py 1e-5 50 1000 3

# Arguments: learning_rate epochs timesteps num_params
```

### SLURM Cluster Training

1. **Modify the job script** (`submit_job.sh`):
   ```bash
   # Set your email
   #SBATCH --mail-user=your.email@institution.edu
   
   # Adjust resources as needed
   #SBATCH --mem=32G
   #SBATCH --time=24:00:00
   ```

2. **Submit the job**:
   ```bash
   sbatch submit_job.sh
   ```

3. **Monitor progress**:
   ```bash
   # Check job status
   squeue -u $USER
   
   # View output logs
   tail -f diffusion_[JOBID].log
   ```

### Command Line Arguments

| Argument | Description | Example Values |
|----------|-------------|----------------|
| `learning_rate` | Adam optimizer learning rate | `1e-5`, `5e-5`, `1e-4` |
| `n_epochs` | Number of training epochs | `50`, `100`, `200` |
| `timesteps` | Diffusion process timesteps | `1000`, `1500`, `2000` |
| `num_params` | Number of conditioning parameters (1-6) | `1`, `3`, `6` |

## Output Structure

Training generates organized outputs in `outputs/paper_lr_{lr}_epochs_{epochs}_timesteps_{timesteps}_params_{params}/`:

```
outputs/
└── paper_lr_1e-05_epochs_100_timesteps_1500_params_6/
    ├── weights/
    │   ├── model_epoch_25.pth
    │   ├── model_epoch_50.pth
    │   └── model_epoch_100.pth
    ├── training_metrics.png
    ├── reconstruction_comparison_viridis.png
    ├── parameter_grid_samples_6params.png
    ├── distribution_comparison.png
    ├── guidance_metrics.png
    ├── parameter_sensitivity.png
    ├── timing_and_performance.log
    └── dataset_info.txt
```

## Key Features

### 1. Parameter Conditioning
- Support for 1-6 cosmological parameters
- Automatic parameter normalization to [0,1] range
- Parameter sensitivity analysis with visualization

### 2. Evaluation Metrics
- **ELBO**: Evidence Lower Bound for model quality
- **BPD**: Bits Per Dimension for compression efficiency
- **NLL**: Negative Log-Likelihood for data fit

### 3. Generation Modes
- **Reconstruction**: Denoise images with original parameters
- **Parameter Sweep**: Generate maps across parameter ranges
- **Classifier-Free Guidance**: Enhanced generation quality

### 4. Visualization
- Training loss curves and metrics
- Parameter sensitivity grids
- Reconstruction comparisons
- Distribution analyses

## Model Architecture

**ContextUnet**: U-Net with parameter conditioning
- **Input**: 64×64 single-channel images
- **Conditioning**: 1-6 parameter embedding
- **Features**: 128-dimensional base features
- **Context Integration**: FiLM-style parameter injection

## Performance Tips

### Memory Optimization
```python
# Reduce batch size for limited GPU memory
batch_size = 16  # Instead of 32

# Use gradient checkpointing (add to model)
model.gradient_checkpointing = True
```

### Training Acceleration
```python
# Use mixed precision training
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# Enable cudNN benchmarking
torch.backends.cudnn.benchmark = True
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in script
   batch_size = 16  # or 8
   ```

2. **Import Error: diffusion_utilities**
   ```bash
   # Ensure utilities module is in same directory
   ls diffusion_utilities.py
   ```

3. **Data Loading Issues**
   ```bash
   # Check data paths and shapes
   python -c "import numpy as np; print(np.load('data/params.npy').shape)"
   ```

### Performance Monitoring

Monitor training via logs:
```bash
# Real-time monitoring
tail -f outputs/*/timing_and_performance.log

# Check GPU utilization
nvidia-smi -l 1
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tengis2025camels,
  title={Evaluating the Efficacy of Diffusion Models on High-Resolution Large-Scale Maps in Cosmology},
  author={Tengis Temuulen},
  year={2025},
  howpublished = {\url{https://www.overleaf.com/read/68386a84341ed65c20c51f02}},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Support

For issues and questions:
- **GitHub Issues**: Open an issue on this repository
- **Email**: [tt2273@nyu.edu]
- **Documentation**: Check the inline code documentation

## Acknowledgments

- CAMELS Collaboration for the simulation dataset
- PyTorch team for the deep learning framework
- Original diffusion model implementations that inspired this work
- This research was carried out on the High Performance Computing resources at New York University Abu Dhabi.