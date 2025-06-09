import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from diffusion_utilities import *  # Assuming this contains the necessary utilities
import random

# Define the same ContextUnet class (must match the trained model)
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=128, n_cfeat=6, height=64):  # n_cfeat=6 for 6 parameters
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d((self.h // 4)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def denoise_add_noise(x, t, pred_noise, z=None, b_t=None, a_t=None, ab_t=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(model, n_sample=1, size=64, device=None, params=None, guide_w=0.0, 
                timesteps=1000, b_t=None, a_t=None, ab_t=None):
    """
    Generate samples using the reverse diffusion process.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Start from random noise
    x = torch.randn(n_sample, 1, size, size).to(device)
    
    if params is None:
        # If no parameters provided, use random parameter values
        params = torch.rand(n_sample, 6).to(device)  # 6 parameters
    else:
        params = params.to(device)
        
    # Setup for classifier-free guidance
    uncond_params = torch.zeros_like(params).to(device)
    
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps]).to(device)
        z = torch.randn_like(x) if i > 1 else 0
        
        # Classifier-free guidance
        if guide_w > 0:
            # Predict noise with conditioning
            pred_noise_cond = model(x, t, params)
            # Predict noise without conditioning
            pred_noise_uncond = model(x, t, uncond_params)
            # Combine predictions with guidance weight
            eps = pred_noise_uncond + guide_w * (pred_noise_cond - pred_noise_uncond)
        else:
            # Standard conditional generation
            eps = model(x, t, params)
            
        x = denoise_add_noise(x, i, eps, z, b_t, a_t, ab_t)
    
    return x

def calculate_power_spectrum_2d(image, dl=1.0):
    """
    Calculate the 2D power spectrum of a 2D image.
    
    Args:
        image: 2D numpy array
        dl: grid spacing (default=1.0)
    
    Returns:
        k: k values (wavenumber)
        pk: power spectrum P(k)
    """
    # Get image dimensions
    nx, ny = image.shape
    
    # Calculate 2D FFT
    fft_image = np.fft.fft2(image)
    fft_image = np.fft.fftshift(fft_image)
    
    # Calculate power spectrum
    power_2d = np.abs(fft_image)**2
    
    # Create k-space coordinates
    kx = np.fft.fftfreq(nx, dl)
    ky = np.fft.fftfreq(ny, dl)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    
    # Create 2D k magnitude array
    kx_2d, ky_2d = np.meshgrid(kx, ky, indexing='ij')
    k_2d = np.sqrt(kx_2d**2 + ky_2d**2)
    
    # Flatten arrays for binning
    k_flat = k_2d.flatten()
    power_flat = power_2d.flatten()
    
    # Define k bins (logarithmic spacing)
    k_min = 2*np.pi/(nx*dl)  # Fundamental frequency
    k_max = np.pi/dl  # Nyquist frequency
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 20)
    
    # Bin the power spectrum
    k_centers = []
    pk_values = []
    
    for i in range(len(k_bins)-1):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        if np.sum(mask) > 0:
            k_center = np.mean(k_flat[mask])
            pk_mean = np.mean(power_flat[mask])
            k_centers.append(k_center)
            pk_values.append(pk_mean)
    
    return np.array(k_centers), np.array(pk_values)

def generate_comparison_plot(model_path, camels_data_path, params_path, output_dir, 
                           selected_params_dict, n_maps=15, timesteps=1000):
    """
    Generate the power spectrum comparison plot between HI-CDM and CAMELS.
    
    Args:
        model_path: Path to the trained .pth model
        camels_data_path: Path to CAMELS data (.npy file)
        params_path: Path to parameter data (.npy file)  
        output_dir: Directory to save outputs
        selected_params_dict: Dictionary with cosmological parameters for subtitle
        n_maps: Number of maps to generate for statistics (default=15)
        timesteps: Number of diffusion timesteps used during training
    """
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    
    # Load the trained model
    print("Loading trained model...")
    model = ContextUnet(in_channels=1, n_feat=128, n_cfeat=6, height=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Setup diffusion parameters (should match training)
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    
    # Load CAMELS data
    print("Loading CAMELS data...")
    camels_data = np.load(camels_data_path)
    param_data = np.load(params_path)
    
    # Preprocess CAMELS data (same as in training)
    min_value = np.min(camels_data)
    if min_value <= 0:
        camels_data = camels_data - min_value + 1e-8
    camels_data = camels_data / np.max(camels_data)
    camels_data = np.log10(camels_data)
    camels_data = (camels_data - camels_data.min()) / (camels_data.max() - camels_data.min())
    
    # Convert to tensor and resize
    camels_data_tensor = torch.tensor(camels_data, dtype=torch.float32).unsqueeze(1)
    camels_data_resized = F.interpolate(camels_data_tensor, size=(64, 64), mode='bilinear')
    
    # Normalize parameters to [0,1] range (same as training)
    param_min = param_data.min(axis=0)
    param_max = param_data.max(axis=0)
    param_data_normalized = (param_data - param_min) / (param_max - param_min + 1e-8)
    
    # Randomly select a parameter set from the available data
    random_param_idx = random.randint(0, len(param_data_normalized)-1)
    selected_params = param_data_normalized[random_param_idx:random_param_idx+1]  # Shape: [1, 6]
    
    print(f"Selected parameter set {random_param_idx}:")
    for i, (key, value) in enumerate(selected_params_dict.items()):
        print(f"  {key}: {value}")
    
    # Get corresponding CAMELS maps (15 maps per parameter set)
    camels_start_idx = random_param_idx * 15
    camels_maps = camels_data_resized[camels_start_idx:camels_start_idx + n_maps]
    
    print(f"Using CAMELS maps {camels_start_idx} to {camels_start_idx + n_maps - 1}")
    
    # Generate HI-CDM maps with the same parameters
    print(f"Generating {n_maps} HI-CDM maps...")
    selected_params_tensor = torch.tensor(selected_params, dtype=torch.float32).repeat(n_maps, 1)
    
    hicdm_maps = sample_ddpm(
        model, 
        n_sample=n_maps, 
        size=64, 
        device=device, 
        params=selected_params_tensor,
        guide_w=0.0,  # No guidance
        timesteps=timesteps,
        b_t=b_t,
        a_t=a_t,
        ab_t=ab_t
    )
    
    # Convert to numpy for analysis
    camels_maps_np = camels_maps.squeeze(1).cpu().numpy()  # Shape: [n_maps, 64, 64]
    hicdm_maps_np = hicdm_maps.squeeze(1).cpu().numpy()    # Shape: [n_maps, 64, 64]
    
    print("Calculating power spectra...")
    
    # Calculate power spectra for all maps
    camels_power_spectra = []
    hicdm_power_spectra = []
    
    dl = 1.0  # Grid spacing - adjust based on your physical units
    
    for i in range(n_maps):
        # CAMELS maps
        k_camels, pk_camels = calculate_power_spectrum_2d(camels_maps_np[i], dl=dl)
        camels_power_spectra.append(pk_camels)
        
        # HI-CDM maps
        k_hicdm, pk_hicdm = calculate_power_spectrum_2d(hicdm_maps_np[i], dl=dl)
        hicdm_power_spectra.append(pk_hicdm)
    
    # Convert to arrays for statistics
    camels_power_spectra = np.array(camels_power_spectra)  # Shape: [n_maps, n_k_bins]
    hicdm_power_spectra = np.array(hicdm_power_spectra)    # Shape: [n_maps, n_k_bins]
    
    # Calculate mean and standard deviation
    camels_pk_mean = np.mean(camels_power_spectra, axis=0)
    camels_pk_std = np.std(camels_power_spectra, axis=0)
    hicdm_pk_mean = np.mean(hicdm_power_spectra, axis=0)
    hicdm_pk_std = np.std(hicdm_power_spectra, axis=0)
    
    # Use k values from one of the calculations (they should be the same)
    k = k_camels
    
    # Create the comparison plot
    plt.figure(figsize=(10, 8))
    
    # Plot CAMELS (red)
    plt.plot(k, camels_pk_mean, 'r-', linewidth=2, label='CAMELS', alpha=0.8)
    plt.fill_between(k, camels_pk_mean - camels_pk_std, camels_pk_mean + camels_pk_std, 
                     color='red', alpha=0.3)
    
    # Plot HI-CDM (blue)
    plt.plot(k, hicdm_pk_mean, 'b-', linewidth=2, label='Model', alpha=0.8)
    plt.fill_between(k, hicdm_pk_mean - hicdm_pk_std, hicdm_pk_mean + hicdm_pk_std, 
                     color='blue', alpha=0.3)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k', fontsize=14)
    plt.ylabel('P(k)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create subtitle with cosmological parameters
    param_text = ", ".join([f"{key}={value}" for key, value in selected_params_dict.items()])
    plt.title(f'Power Spectrum Comparison\n{param_text}', fontsize=12)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'power_spectrum_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Power spectrum comparison plot saved to: {plot_path}")
    
    # Save some example maps for visual inspection
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Show first 5 CAMELS maps
    for i in range(5):
        axes[0, i].imshow(camels_maps_np[i], cmap='viridis')
        axes[0, i].set_title(f'CAMELS {i+1}')
        axes[0, i].axis('off')
    
    # Show first 5 HI-CDM maps
    for i in range(5):
        axes[1, i].imshow(hicdm_maps_np[i], cmap='viridis')
        axes[1, i].set_title(f'HI-CDM {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    maps_path = os.path.join(output_dir, 'example_maps_comparison.png')
    plt.savefig(maps_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Example maps saved to: {maps_path}")
    
    # Calculate and print some statistics
    pk_ratio_mean = np.mean(hicdm_pk_mean / camels_pk_mean)
    pk_ratio_std = np.std(hicdm_pk_mean / camels_pk_mean)
    
    print(f"\nPower Spectrum Statistics:")
    print(f"Mean P(k) ratio (HI-CDM/CAMELS): {pk_ratio_mean:.4f} ± {pk_ratio_std:.4f}")
    
    # Save numerical results
    results = {
        'k': k,
        'camels_pk_mean': camels_pk_mean,
        'camels_pk_std': camels_pk_std,
        'hicdm_pk_mean': hicdm_pk_mean,
        'hicdm_pk_std': hicdm_pk_std,
        'parameters': selected_params_dict
    }
    
    np.save(os.path.join(output_dir, 'power_spectrum_results.npy'), results)
    print(f"Numerical results saved to: {os.path.join(output_dir, 'power_spectrum_results.npy')}")

# Example usage
if __name__ == "__main__":
    # Configuration
    model_path = "../code/outputs/paper_lr_1e-05_epochs_100_timesteps_1500_params_6/weights/model_epoch_100.pth"  # Path to your trained 6-parameter model
    camels_data_path = "../data/Maps_HI_IllustrisTNG_LH_z=0.00.npy"
    params_path = "../data/params.npy"
    output_dir = "power_spectrum_comparison_output"
    
    # Example cosmological parameters for the subtitle
    # You should replace these with the actual values from your selected parameter set
    selected_params_dict = {
        'Ωm': 0.21940,
        'σ8': 0.90020,
        'ASN1': 3.88523,
        'AAGN1': 0.29895,
        'ASN2': 1.61664,
        'AAGN2': 1.48968
    }
    
    # Generate the comparison plot
    generate_comparison_plot(
        model_path=model_path,
        camels_data_path=camels_data_path,
        params_path=params_path,
        output_dir=output_dir,
        selected_params_dict=selected_params_dict,
        n_maps=15,
        timesteps=1500  # Should match the timesteps used during training
    )