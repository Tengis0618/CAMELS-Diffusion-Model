import os
import sys
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
from diffusion_utilities import *

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=128, n_cfeat=10, height=64):  # adjusted for 64x64
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

def log_device_used(output_file="output.log"):
    device_used = "GPU" if torch.cuda.is_available() else "CPU"
    with open(output_file, "a") as f:
        f.write(f"Device used: {device_used}\n")

# Accept command-line arguments for learning rate, epochs, and timesteps
lrate = float(sys.argv[1])
n_epoch = int(sys.argv[2])
timesteps = int(sys.argv[3])
num_params = int(sys.argv[4])  # New argument for number of parameters to condition on


# Define a unique directory to save outputs for this parameter combination
output_dir = f"outputs/conditional_lr_{lrate}_epochs_{n_epoch}_timesteps_{timesteps}_params_{num_params}"
save_dir = os.path.join(output_dir, "weights")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Diffusion hyperparameters
beta1 = 1e-4
beta2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 128  # Feature size
n_cfeat = num_params  # Number of conditioning features
height = 64
batch_size = 32

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# Load the dataset
file_path = '../data/Maps_HI_IllustrisTNG_LH_z=0.00.npy'
camels_data = np.load(file_path)

# Load corresponding parameter data 
# The format should match your actual parameter data file
param_file_path = '../data/params.npy'
param_data = np.load(param_file_path)  # Shape should be [n_samples, n_params]

# Expand parameter data to match 15 images per parameter set
# Each row in param_data corresponds to 15 images in camels_data
expanded_param_data = np.repeat(param_data, 15, axis=0)
assert expanded_param_data.shape[0] == camels_data.shape[0], "Parameter expansion doesn't match image count"

# Normalize parameters to [0,1] range for better conditioning
param_min = expanded_param_data.min(axis=0, keepdims=True)
param_max = expanded_param_data.max(axis=0, keepdims=True)
param_data_normalized = (expanded_param_data - param_min) / (param_max - param_min + 1e-8)

# Save parameter normalization values for later use during generation
np.save(os.path.join(output_dir, "param_min.npy"), param_min)
np.save(os.path.join(output_dir, "param_max.npy"), param_max)

# Use only the specified number of parameters for conditioning
if param_data_normalized.shape[1] > num_params:
    param_data_normalized = param_data_normalized[:, :num_params]
elif param_data_normalized.shape[1] < num_params:
    # This shouldn't happen with the dataset described, but included for robustness
    padding = np.zeros((param_data_normalized.shape[0], num_params - param_data_normalized.shape[1]))
    param_data_normalized = np.concatenate([param_data_normalized, padding], axis=1)

param_data_tensor = torch.tensor(param_data_normalized, dtype=torch.float32)

# Normalize and preprocess data
min_value = np.min(camels_data)
if min_value <= 0:
    camels_data = camels_data - min_value + 1e-8
camels_data = camels_data / np.max(camels_data)
camels_data = np.log10(camels_data)
camels_data = (camels_data - camels_data.min()) / (camels_data.max() - camels_data.min())
camels_data_tensor = torch.tensor(camels_data, dtype=torch.float32).unsqueeze(1)
camels_data_resized = F.interpolate(camels_data_tensor, size=(64, 64), mode='bilinear')

# Create a dataset with both image and parameter data
full_dataset = TensorDataset(camels_data_resized, param_data_tensor)

# Split into train and test sets (1500 images for test, rest for training)
test_size = 1500
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Log dataset information
with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
    f.write(f"Total dataset size: {len(full_dataset)}\n")
    f.write(f"Train dataset size: {len(train_dataset)}\n")
    f.write(f"Test dataset size: {len(test_dataset)}\n")
    f.write(f"Number of parameters used for conditioning: {num_params}\n")
    f.write(f"Original parameter data shape: {param_data.shape}\n")
    f.write(f"Expanded parameter data shape: {expanded_param_data.shape}\n")
    f.write(f"Final normalized parameter data shape: {param_data_normalized.shape}\n")

# Initialize the model with the correct number of conditioning features
nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# Select 10 random images and their corresponding parameters from the test set for evaluation
random_indices = random.sample(range(len(test_dataset)), 10)
selected_images = []
selected_params = []

for idx in random_indices:
    img, param = test_dataset[idx]
    selected_images.append(img)
    selected_params.append(param)

selected_images = torch.stack(selected_images)
selected_params = torch.stack(selected_params)

# Save the processed images and compute their mean
save_image(selected_images, os.path.join(output_dir, "test_images.png"))
processed_images_mean = selected_images.mean().item()
print(f"Processed test images mean: {processed_images_mean}")

# Save parameter information for the selected images
param_info = ""
for i, params in enumerate(selected_params):
    param_info += f"Image {i+1}: {[f'{p.item():.4f}' for p in params]}\n"
with open(os.path.join(output_dir, "selected_params.txt"), "w") as f:
    f.write(param_info)

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

# Training loop
loss_log = []
val_loss_log = []
nn_model.train()

for ep in range(n_epoch):
    log_device_used()
    epoch_loss = 0
    optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
    pbar = tqdm(train_dataloader, mininterval=2)
    
    for x, param in pbar:
        optim.zero_grad()
        x = x.to(device)
        param = param.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)
        
        # Pass conditioning parameters to the model
        pred_noise = nn_model(x_pert, t / timesteps, param)
        
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
        
    loss_log.append(epoch_loss / len(train_dataloader))
    
    # Evaluate on validation set every 5 epochs
    if ep % 5 == 0 or ep == n_epoch - 1:
        nn_model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, param in test_dataloader:
                x = x.to(device)
                param = param.to(device)
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
                x_pert = perturb_input(x, t, noise)
                
                pred_noise = nn_model(x_pert, t / timesteps, param)
                val_loss += F.mse_loss(pred_noise, noise).item()
        
        val_loss_log.append(val_loss / len(test_dataloader))
        print(f"Epoch {ep+1}/{n_epoch}, Train Loss: {loss_log[-1]:.6f}, Val Loss: {val_loss_log[-1]:.6f}")
        nn_model.train()

    # Save model checkpoints
    if (ep + 1) % 25 == 0 or ep == n_epoch - 1:
        torch.save(nn_model.state_dict(), os.path.join(save_dir, f"model_epoch_{ep+1}.pth"))

# Modify the loss plotting section
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, n_epoch+1), np.log(loss_log), label='Training Loss')

# Create x-axis for validation loss that matches its actual logged points
val_epochs = list(range(0, n_epoch, 5)) + ([n_epoch-1] if (n_epoch-1) % 5 != 0 else [])
val_loss_x = [e+1 for e in val_epochs]  # +1 because epochs are 0-indexed in the code but 1-indexed in the plot
plt.plot(val_loss_x, np.log(val_loss_log), 'o-', label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.title(f"Loss Evolution with {num_params} conditioning parameters")
plt.savefig(os.path.join(output_dir, "loss_evolution.png"))
print("Saved loss evolution plot as loss_evolution.png")

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample=1, size=64, device=None, params=None, guide_w=0.0):
    """
    Generate samples using the reverse diffusion process.
    
    Args:
        n_sample (int): Number of samples to generate
        size (int): Image size
        device (torch.device): Device to use
        params (torch.Tensor): Conditioning parameters 
        guide_w (float): Classifier-free guidance weight
        
    Returns:
        samples (torch.Tensor): Generated samples
        intermediate (list): Intermediate states
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Start from random noise
    x = torch.randn(n_sample, 1, size, size).to(device)
    
    if params is None:
        # If no parameters provided, use random parameters
        params = torch.rand(n_sample, n_cfeat).to(device)
    else:
        params = params.to(device)
        
    # Setup for classifier-free guidance
    uncond_params = torch.zeros_like(params).to(device)
    
    intermediate = []
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps]).to(device)
        z = torch.randn_like(x) if i > 1 else 0
        
        # Classifier-free guidance
        if guide_w > 0:
            # Predict noise with conditioning
            pred_noise_cond = nn_model(x, t, params)
            # Predict noise without conditioning
            pred_noise_uncond = nn_model(x, t, uncond_params)
            # Combine predictions with guidance weight
            eps = pred_noise_uncond + guide_w * (pred_noise_cond - pred_noise_uncond)
        else:
            # Standard conditional generation
            eps = nn_model(x, t, params)
            
        x = denoise_add_noise(x, i, eps, z)
        
        if i % 20 == 0 or i == timesteps or i < 8:
            intermediate.append(x.detach().cpu().numpy())
            
    intermediate = np.stack(intermediate)
    return x, intermediate

@torch.no_grad()
def sample_ddpm_from_noise(noise_images, params=None, save_rate=20, guide_w=0.0):
    """
    Reverse diffusion process starting from the given noise images.
    
    Args:
        noise_images (torch.Tensor): Starting noisy images
        params (torch.Tensor): Conditioning parameters
        save_rate (int): Frequency of saving intermediate images
        guide_w (float): Classifier-free guidance weight
        
    Returns:
        samples (torch.Tensor): Generated samples
        intermediate (list): Intermediate states
    """
    samples = noise_images.clone().to(device)
    
    # Setup for classifier-free guidance
    if params is not None:
        params = params.to(device)
        uncond_params = torch.zeros_like(params).to(device)
    else:
        uncond_params = None
        
    intermediate = []
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps]).to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        
        # Classifier-free guidance
        if guide_w > 0 and params is not None:
            # Predict noise with conditioning
            pred_noise_cond = nn_model(samples, t, params)
            # Predict noise without conditioning
            pred_noise_uncond = nn_model(samples, t, uncond_params)
            # Combine predictions with guidance weight
            eps = pred_noise_uncond + guide_w * (pred_noise_cond - pred_noise_uncond)
        else:
            # Standard conditional or unconditional generation
            eps = nn_model(samples, t, params)
            
        samples = denoise_add_noise(samples, i, eps, z)
        
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())
            
    intermediate = np.stack(intermediate)
    return samples, intermediate


# Evaluation after training
nn_model.eval()

# Test model on selected test images
# 1. First, reconstruct the original images with their original parameters
x = selected_images.to(device)
params = selected_params.to(device)
noise = torch.randn_like(x)
# Forward diffusion to the last timestep (timesteps)
x_forward_diffused = perturb_input(x, timesteps, noise)

# Now sample (reverse diffusion) from the noisy images with conditioning
samples, intermediate = sample_ddpm_from_noise(x_forward_diffused, params)
for idx, img in enumerate(intermediate):
    if idx % 5 == 0:  # Save every 5th intermediate state to avoid too many images
        save_image(torch.tensor(img), os.path.join(output_dir, f"intermediate_step_{idx}.png"))
save_image(samples, os.path.join(output_dir, "reconstructed_images.png"))

# Function to compare distributions
def compare_distributions(camels_images, diffusion_images, output_dir):
    bin_max = max(camels_images.max(), diffusion_images.max())
    bin_min = min(camels_images.min(), diffusion_images.min())
    bin_delta = 0.01
    bins = np.arange(bin_min, bin_max + bin_delta, bin_delta)

    train_pdf = []
    test_pdf = []
    for i in range(len(camels_images)):
        h = np.histogram(camels_images[i].ravel(), bins, density=True)[0]
        train_pdf.append(h)
        h = np.histogram(diffusion_images[i].ravel(), bins, density=True)[0]
        test_pdf.append(h)
    train_pdf = np.array(train_pdf)
    test_pdf = np.array(test_pdf)

    train_pdf_mean = np.mean(train_pdf, axis=0)
    train_pdf_std = np.std(train_pdf, axis=0)
    test_pdf_mean = np.mean(test_pdf, axis=0)
    test_pdf_std = np.std(test_pdf, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    bin_mid = (bins[:-1] + bins[1:]) / 2.0

    ax[0].plot(bin_mid, train_pdf_mean, "k-", label="Original")
    ax[0].plot(bin_mid, test_pdf_mean, "k--", label="Diffusion model")
    ax[0].set_ylabel(r"$\mu(\rm PDF)$", fontsize=14)

    ax[1].plot(bin_mid, train_pdf_std, "k-")
    ax[1].plot(bin_mid, test_pdf_std, "k--")
    ax[1].set_ylabel(r"$\sigma(\rm PDF)$", fontsize=14)

    for i in range(2):
        ax[i].set_xlabel(r"$N_{\rm HI}$", fontsize=14)
    ax[0].legend(fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "distribution_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved distribution comparison plot as {plot_path}")

# Compare distributions
compare_distributions(selected_images.numpy(), samples.cpu().numpy(), output_dir)
reconstructed_mean = samples.mean().item()
print(f"Reconstructed images mean: {reconstructed_mean}")
print(f"Original images mean: {processed_images_mean}")

# 2. Generate samples with varied parameters (focusing on parameters we conditioned on)
# Generate a grid by varying the conditioned parameters
if num_params >= 2:
    # If we have at least 2 parameters, vary the first two
    param1_values = torch.linspace(0.0, 1.0, 5)
    param2_values = torch.linspace(0.0, 1.0, 5)
    grid_params = []

    for p1 in param1_values:
        for p2 in param2_values:
            # Start with a base parameter set (using the first sample's parameters)
            new_param = selected_params[0].clone()
            # Modify the first two parameters
            new_param[0] = p1
            new_param[1] = p2
            grid_params.append(new_param)
else:
    # If we only have 1 parameter, vary just that one
    param1_values = torch.linspace(0.0, 1.0, 25)
    grid_params = []
    
    for p1 in param1_values:
        new_param = selected_params[0].clone()
        new_param[0] = p1
        grid_params.append(new_param)

grid_params = torch.stack(grid_params)

# Generate samples for each parameter combination
grid_samples, _ = sample_ddpm(n_sample=len(grid_params), size=height, device=device, params=grid_params)

# Save grid of generated images
grid_size = int(np.sqrt(len(grid_samples)))
save_image(grid_samples, os.path.join(output_dir, f"parameter_grid_samples_{num_params}params.png"), nrow=grid_size)

# 3. Test classifier-free guidance at different strengths
guidance_strengths = [0.0, 1.0, 2.0, 3.0, 5.0]
guided_samples = []

for w in guidance_strengths:
    # Use the same parameter set for all guidance strengths (first sample's parameters)
    params = selected_params[0].unsqueeze(0).repeat(5, 1)
    samples, _ = sample_ddpm(n_sample=5, size=height, device=device, params=params, guide_w=w)
    guided_samples.append(samples)

# Combine and save samples from different guidance strengths
guided_samples = torch.cat(guided_samples)
save_image(guided_samples, os.path.join(output_dir, "guidance_strength_samples.png"), nrow=5)
print(f"Saved guidance strength comparison to {output_dir}/guidance_strength_samples.png")

# 4. Create visualization summarizing parameter sensitivity
# This helps understand how each conditioned parameter affects the output
if num_params > 0:
    fig, axs = plt.subplots(num_params, 5, figsize=(15, 3*num_params))
    if num_params == 1:
        axs = axs.reshape(1, -1)  # Make sure axs is 2D for single parameter case
    
    for param_idx in range(num_params):
        # For each parameter, generate samples with increasing values
        param_values = torch.linspace(0.0, 1.0, 5)
        for i, val in enumerate(param_values):
            # Start with base parameters and modify just one parameter
            base_param = selected_params[0].clone().unsqueeze(0)
            base_param[0, param_idx] = val
            
            # Generate a sample with this parameter value
            sample, _ = sample_ddpm(n_sample=1, size=height, device=device, params=base_param)
            img = sample[0, 0].cpu().numpy()
            
            # Display the parameter value and sample
            axs[param_idx, i].imshow(img, cmap='viridis')
            axs[param_idx, i].set_title(f"Param {param_idx+1} = {val:.2f}")
            axs[param_idx, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_sensitivity.png"))
    plt.close()
    print(f"Saved parameter sensitivity analysis to {output_dir}/parameter_sensitivity.png")

print(f"Training and evaluation completed with {num_params} conditioning parameters.")