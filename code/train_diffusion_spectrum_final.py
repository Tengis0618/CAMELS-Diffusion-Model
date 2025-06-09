import os
import sys
import time
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
        else:
            # Make sure c is on the same device as x
            c = c.to(x.device)
            
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

def calculate_elbo_and_bpd(model, dataloader, timesteps, device, ab_t, b_t, a_t):
    """
    Calculate ELBO and BPD for diffusion models over the full dataset.
    
    Args:
        model: The diffusion model
        dataloader: DataLoader containing the dataset
        timesteps: Number of diffusion steps
        device: Device to run calculations on
        ab_t, b_t, a_t: Diffusion schedule tensors
        
    Returns:
        elbo: Evidence Lower Bound
        bpd: Bits Per Dimension
    """
    model.eval()
    total_elbo = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for x, param in dataloader:
            batch_size = x.shape[0]
            x = x.to(device)
            param = param.to(device)
            
            # Initialize batch elbo
            batch_elbo = torch.zeros(batch_size).to(device)
            
            # Sample multiple timesteps for more stable estimates
            # Use evenly spaced timesteps to cover the diffusion process
            sampled_t = torch.linspace(1, timesteps, 10).long().to(device)
            
            for t in sampled_t:
                # Add noise according to timestep
                noise = torch.randn_like(x)
                x_t = ab_t.sqrt()[t] * x + torch.sqrt(1 - ab_t[t]) * noise
                time_tensor = torch.tensor([t / timesteps]).to(device)
                
                # Predict noise
                pred_noise = model(x_t, time_tensor, param)
                
                # Mean squared error between predicted and target noise
                noise_mse = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
                
                # Use the correct SNR-related weighting for ELBO
                # This is the denoising score matching objective
                # We use a more numerically stable formula
                if t > 1:  # Skip the first timestep weight which can be unstable
                    weight = 0.5 * (b_t[t] / (1.0 - ab_t[t]))
                    # Contribution to ELBO
                    batch_elbo += weight * noise_mse / 10.0  # Average over sampled timesteps
            
            total_elbo += batch_elbo.sum().item()
            num_samples += batch_size
    
    # Calculate average ELBO
    avg_elbo = total_elbo / num_samples
    
    # Convert to bits per dimension (using log base 2)
    dims = 64 * 64  # For 64x64 images
    bpd = avg_elbo / (dims * np.log(2))
    
    return avg_elbo, bpd

# Function to calculate the likelihood of the data given the model
def calculate_likelihood(model, dataloader, timesteps, device, ab_t, b_t, a_t):
    """
    Calculate the approximate likelihood of the data given the model.
    This uses importance sampling to approximate the marginal likelihood.
    
    Returns the mean negative log likelihood over the dataset.
    """
    model.eval()
    total_nll = 0
    num_samples = 0
    
    with torch.no_grad():
        for x, param in dataloader:
            batch_size = x.shape[0]
            x = x.to(device)
            param = param.to(device)
            
            # Initialize log likelihood for batch
            batch_nll = torch.zeros(batch_size).to(device)
            
            # For each timestep, compute likelihood contribution
            for t in range(1, timesteps + 1):
                # Perturb data to timestep t
                noise = torch.randn_like(x)
                x_t = ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
                time_tensor = torch.tensor([t / timesteps]).to(device)
                
                # Predict noise
                pred_noise = model(x_t, time_tensor, param)
                
                # Calculate MSE between predicted and actual noise
                mse = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
                
                # Contribution to negative log likelihood
                # This is a simplified approximation
                nll_contribution = mse / (2 * b_t[t])
                batch_nll += nll_contribution
            
            total_nll += batch_nll.sum().item()
            num_samples += batch_size
    
    return total_nll / num_samples

# Accept command-line arguments for learning rate, epochs, and timesteps
lrate = float(sys.argv[1])
n_epoch = int(sys.argv[2])
timesteps = int(sys.argv[3])
num_params = int(sys.argv[4])  # New argument for number of parameters to condition on


# Define a unique directory to save outputs for this parameter combination
output_dir = f"outputs/elbo_bpd_lr_{lrate}_epochs_{n_epoch}_timesteps_{timesteps}_params_{num_params}"
save_dir = os.path.join(output_dir, "weights")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Create a timing and performance log file
timing_log_path = os.path.join(output_dir, "timing_and_performance.log")
with open(timing_log_path, "w") as f:
    f.write("=== Diffusion Model Training and Sampling Timing Log ===\n\n")
    f.write(f"Parameters: learning_rate={lrate}, epochs={n_epoch}, timesteps={timesteps}, num_params={num_params}\n\n")

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

# Record total training time
training_start_time = time.time()

# Initialize training logs
loss_log = []
val_loss_log = []
likelihood_log = []
elbo_log = []
bpd_log = []
val_elbo_log = []
val_bpd_log = []
epoch_times = []

nn_model.train()

for ep in range(n_epoch):
    epoch_start_time = time.time()
    log_device_used()
    epoch_loss = 0
    optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
    pbar = tqdm(train_dataloader, mininterval=2)
    
    # Training mode
    nn_model.train()
    
    for x, param in pbar:
        optim.zero_grad()
        x = x.to(device)
        param = param.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)
        
        # Pass conditioning parameters to the model
        pred_noise = nn_model(x_pert, t / timesteps, param)
        
        # MSE loss for training updates
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        optim.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar with only MSE loss during training
        pbar.set_description(f"Epoch {ep+1}, Loss: {loss.item():.4f}")
    
    # Average over batches for MSE loss
    epoch_loss /= len(train_dataloader)
    loss_log.append(epoch_loss)
    
    # Record epoch time
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)
    
    # Log the epoch time
    with open(timing_log_path, "a") as f:
        f.write(f"Epoch {ep+1}/{n_epoch} completed in {epoch_duration:.2f} seconds\n")
        f.write(f"  Training Loss: {epoch_loss:.6f}\n")
    
    # Evaluate ELBO/BPD and other metrics every 5 epochs or at the end
    if ep % 5 == 0 or ep == n_epoch - 1:
        # Evaluation mode
        nn_model.eval()
        
        # Calculate validation MSE loss
        val_loss = 0
        with torch.no_grad():
            for x, param in test_dataloader:
                x = x.to(device)
                param = param.to(device)
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
                x_pert = perturb_input(x, t, noise)
                
                pred_noise = nn_model(x_pert, t / timesteps, param)
                
                # Standard MSE loss
                batch_loss = F.mse_loss(pred_noise, noise).item()
                val_loss += batch_loss
        
        # Average over validation batches
        val_loss /= len(test_dataloader)
        val_loss_log.append(val_loss)
        
        # Calculate ELBO and BPD on training set (using a subset for efficiency if needed)
        if len(train_dataset) > 2000:
            # Use a subset of training data for efficiency
            subset_indices = random.sample(range(len(train_dataset)), 2000)
            train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
            train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            train_elbo, train_bpd = calculate_elbo_and_bpd(
                nn_model, train_subset_loader, timesteps, device, ab_t, b_t, a_t
            )
        else:
            train_elbo, train_bpd = calculate_elbo_and_bpd(
                nn_model, train_dataloader, timesteps, device, ab_t, b_t, a_t
            )
        
        # Calculate ELBO and BPD on validation set
        val_elbo, val_bpd = calculate_elbo_and_bpd(
            nn_model, test_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        
        # Store results
        elbo_log.append(train_elbo)
        bpd_log.append(train_bpd)
        val_elbo_log.append(val_elbo)
        val_bpd_log.append(val_bpd)
        
        # Calculate likelihood on a subset of test set
        subset_size = min(len(test_dataset), 200)
        subset_indices = random.sample(range(len(test_dataset)), subset_size)
        subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
        subset_dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)
        
        likelihood_calculation_start = time.time()
        neg_log_likelihood = calculate_likelihood(
            nn_model, subset_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        likelihood_calculation_time = time.time() - likelihood_calculation_start
        likelihood_log.append(neg_log_likelihood)
        
        # Log all metrics
        with open(timing_log_path, "a") as f:
            f.write(f"  Validation Loss: {val_loss:.6f}\n")
            f.write(f"  Train ELBO: {train_elbo:.6f}, Train BPD: {train_bpd:.6f}\n")
            f.write(f"  Val ELBO: {val_elbo:.6f}, Val BPD: {val_bpd:.6f}\n")
            f.write(f"  Negative Log Likelihood: {neg_log_likelihood:.6f}\n")
            f.write(f"  Likelihood calculation took {likelihood_calculation_time:.2f} seconds\n")
        
        print(f"Epoch {ep+1}/{n_epoch}, Train Loss: {loss_log[-1]:.6f}, Val Loss: {val_loss_log[-1]:.6f}")
        print(f"Train BPD: {train_bpd:.6f}, Val BPD: {val_bpd:.6f}")
        print(f"Negative Log Likelihood: {neg_log_likelihood:.6f}")
        
        # Return to training mode
        nn_model.train()

    # Save model checkpoints
    if (ep + 1) % 25 == 0 or ep == n_epoch - 1:
        torch.save(nn_model.state_dict(), os.path.join(save_dir, f"model_epoch_{ep+1}.pth"))

# Record total training time
total_training_time = time.time() - training_start_time

# Log overall training statistics
with open(timing_log_path, "a") as f:
    f.write(f"\n=== Training Complete ===\n")
    f.write(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)\n")
    f.write(f"Average time per epoch: {np.mean(epoch_times):.2f} seconds\n")
    f.write(f"Final training loss: {loss_log[-1]:.6f}\n")
    f.write(f"Final validation loss: {val_loss_log[-1]:.6f}\n")
    f.write(f"Final training BPD: {bpd_log[-1]:.6f}\n")
    f.write(f"Final validation BPD: {val_bpd_log[-1]:.6f}\n")
    f.write(f"Final negative log likelihood: {likelihood_log[-1]:.6f}\n\n")

# Plot training metrics
plt.figure(figsize=(15, 10))

# Plot 1: Training and validation loss
plt.subplot(2, 2, 1)
plt.plot(range(1, n_epoch+1), np.log(loss_log), label='Training Loss')

# Create correct x-axis for validation metrics
eval_epochs = list(range(0, n_epoch, 5))
if (n_epoch-1) % 5 != 0:
    eval_epochs.append(n_epoch-1)
eval_x = [e+1 for e in eval_epochs]  # +1 for 1-indexing in plots

plt.plot(eval_x, np.log(val_loss_log), 'o-', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.title(f"Loss Evolution with {num_params} conditioning parameters")

# Plot 2: Negative log likelihood
plt.subplot(2, 2, 2)
plt.plot(eval_x, likelihood_log, 'o-r', label='Negative Log Likelihood')
plt.xlabel("Epoch")
plt.ylabel("NLL")
plt.legend()
plt.grid(True)
plt.title("Negative Log Likelihood Evolution")

# Plot 3: ELBO
plt.subplot(2, 2, 3)
plt.plot(eval_x, elbo_log, label='Training ELBO')
plt.plot(eval_x, val_elbo_log, 'o-', label='Validation ELBO')
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.legend()
plt.grid(True)
plt.title("ELBO Evolution")

# Plot 4: BPD
plt.subplot(2, 2, 4)
plt.plot(eval_x, bpd_log, label='Training BPD')
plt.plot(eval_x, val_bpd_log, 'o-', label='Validation BPD')
plt.xlabel("Epoch")
plt.ylabel("Bits Per Dimension (BPD)")
plt.legend()
plt.grid(True)
plt.title("BPD Evolution")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_metrics.png"))
print("Saved training metrics plot as training_metrics.png")

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
        sampling_time (float): Time taken for sampling
    """
    sampling_start_time = time.time()
    
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
    
    # Track time for each timestep
    timestep_times = []
    
    for i in range(timesteps, 0, -1):
        step_start_time = time.time()
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
        
        # Calculate step time
        step_time = time.time() - step_start_time
        timestep_times.append(step_time)
        
        if i % 20 == 0 or i == timesteps or i < 8:
            intermediate.append(x.detach().cpu().numpy())
            
    intermediate = np.stack(intermediate)
    sampling_time = time.time() - sampling_start_time
    
    return x, intermediate, sampling_time, timestep_times

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
        sampling_time (float): Time taken for sampling
    """
    sampling_start_time = time.time()
    
    samples = noise_images.clone().to(device)
    
    # Setup for classifier-free guidance
    if params is not None:
        params = params.to(device)
        uncond_params = torch.zeros_like(params).to(device)
    else:
        uncond_params = None
        
    intermediate = []
    
    # Track time for each timestep
    timestep_times = []
    
    for i in range(timesteps, 0, -1):
        step_start_time = time.time()
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
        
        # Calculate step time
        step_time = time.time() - step_start_time
        timestep_times.append(step_time)
        
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())
            
    intermediate = np.stack(intermediate)
    sampling_time = time.time() - sampling_start_time
    
    return samples, intermediate, sampling_time, timestep_times

def visualize_viridis_style(samples, output_path, nrow=5, title="CAMELS"):
    """
    Visualize samples using the viridis colormap
    
    Args:
        samples (torch.Tensor): Samples to visualize [B, 1, H, W]
        output_path (str): Path to save the visualization
        nrow (int): Number of images per row
        title (str): Title label for the visualization
    """
    # Convert to numpy array if needed
    if isinstance(samples, torch.Tensor):
        samples = samples.squeeze(1).cpu().numpy()
    elif isinstance(samples, np.ndarray) and samples.ndim == 4:
        samples = samples.squeeze(1)
    
    num_images = min(len(samples), 25)  # Limit to 25 images max
    num_cols = nrow
    num_rows = (num_images + num_cols - 1) // num_cols
    
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    
    # Add title on the left side
    plt.figtext(0.05, 0.5, title, rotation=90, fontsize=16, 
                fontweight='bold', va='center')
    
    # Adjust subplot parameters to make room for the title
    plt.subplots_adjust(left=0.1)
    
    # Plot each image
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        
        # Plot with the viridis colormap
        plt.imshow(samples[i], cmap='viridis')
        plt.axis('off')
    
    plt.tight_layout(rect=[0.1, 0, 1, 1])  # Adjust for the title
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved viridis-style visualization to {output_path}")

# Evaluation after training
nn_model.eval()

with open(timing_log_path, "a") as f:
    f.write("\n=== Sampling Performance ===\n")

# Test model on selected test images
# 1. First, reconstruct the original images with their original parameters
x = selected_images.to(device)
params = selected_params.to(device)
noise = torch.randn_like(x)
# Forward diffusion to the last timestep (timesteps)
x_forward_diffused = perturb_input(x, timesteps, noise)

# Now sample (reverse diffusion) from the noisy images with conditioning
samples, intermediate, sampling_time, timestep_times = sample_ddpm_from_noise(x_forward_diffused, params)

# Log sampling performance
with open(timing_log_path, "a") as f:
    f.write(f"Reconstructing {len(selected_images)} test images took {sampling_time:.2f} seconds\n")
    f.write(f"Average time per timestep: {np.mean(timestep_times):.4f} seconds\n")
    f.write(f"Total timesteps: {timesteps}\n")

for idx, img in enumerate(intermediate):
    if idx % 5 == 0:  # Save every 5th intermediate state to avoid too many images
        save_image(torch.tensor(img), os.path.join(output_dir, f"intermediate_step_{idx}.png"))
save_image(samples, os.path.join(output_dir, "reconstructed_images.png"))
visualize_viridis_style(samples, os.path.join(output_dir, "reconstructed_images_viridis.png"))

# Calculate ELBO and BPD for the reconstructed images
recon_dataset = TensorDataset(samples, params)
recon_dataloader = DataLoader(recon_dataset, batch_size=batch_size, shuffle=False)

# Calculate likelihood for the reconstructed images
nn_model.eval()
with torch.no_grad():
    # Calculate ELBO and BPD on reconstructed images
    total_elbo = 0
    total_bpd = 0
    for x, param in recon_dataloader:
        x = x.to(device)
        param = param.to(device)
        
        # Sample random timesteps
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        
        # Add noise according to timestep
        noise = torch.randn_like(x)
        x_noisy = perturb_input(x, t, noise)
        
        # Predict noise
        pred_noise = nn_model(x_noisy, t / timesteps, param)
        
        # Calculate ELBO and BPD
        dims = x.shape[2] * x.shape[3]  # height * width
        # Create a temporary dataset for this batch
        temp_dataset = TensorDataset(x, param)
        temp_dataloader = DataLoader(temp_dataset, batch_size=x.shape[0], shuffle=False)
                
        # Calculate ELBO and BPD using the main function
        elbo, bpd = calculate_elbo_and_bpd(
            nn_model, temp_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        
        total_elbo += elbo * x.shape[0]
        total_bpd += bpd * x.shape[0]
    
    # Calculate average
    recon_elbo = total_elbo / len(recon_dataset)
    recon_bpd = total_bpd / len(recon_dataset)

# Calculate NLL using our existing function
recon_likelihood = calculate_likelihood(
    nn_model, recon_dataloader, timesteps, device, ab_t, b_t, a_t
)

with open(timing_log_path, "a") as f:
    f.write(f"ELBO of reconstructed images: {recon_elbo:.6f}\n")
    f.write(f"BPD of reconstructed images: {recon_bpd:.6f}\n")
    f.write(f"Negative log likelihood of reconstructed images: {recon_likelihood:.6f}\n")

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
parameter_grid_start_time = time.time()

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
grid_samples, grid_intermediate, grid_sampling_time, _ = sample_ddpm(
    n_sample=len(grid_params), size=height, device=device, params=grid_params
)

# Log parameter grid generation time
with open(timing_log_path, "a") as f:
    f.write(f"Generating {len(grid_params)} parameter grid samples took {grid_sampling_time:.2f} seconds\n")

# Save grid of generated images
grid_size = int(np.sqrt(len(grid_samples)))
save_image(grid_samples, os.path.join(output_dir, f"parameter_grid_samples_{num_params}params.png"), nrow=grid_size)

# Calculate ELBO and BPD for the grid samples
grid_dataset = TensorDataset(grid_samples, grid_params)
grid_dataloader = DataLoader(grid_dataset, batch_size=batch_size, shuffle=False)

# Calculate metrics for parameter grid samples
nn_model.eval()
with torch.no_grad():
    # Calculate ELBO and BPD
    total_elbo = 0
    total_bpd = 0
    for x, param in grid_dataloader:
        x = x.to(device)
        param = param.to(device)
        
        # Sample random timesteps
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        
        # Add noise according to timestep
        noise = torch.randn_like(x)
        x_noisy = perturb_input(x, t, noise)
        
        # Predict noise
        pred_noise = nn_model(x_noisy, t / timesteps, param)
        
        # Calculate ELBO and BPD
        dims = x.shape[2] * x.shape[3]  # height * width
        # Create a temporary dataset for this batch
        temp_dataset = TensorDataset(x, param)
        temp_dataloader = DataLoader(temp_dataset, batch_size=x.shape[0], shuffle=False)
                
        # Calculate ELBO and BPD using the main function
        elbo, bpd = calculate_elbo_and_bpd(
            nn_model, temp_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        
        total_elbo += elbo * x.shape[0]
        total_bpd += bpd * x.shape[0]
    
    # Calculate average
    grid_elbo = total_elbo / len(grid_dataset)
    grid_bpd = total_bpd / len(grid_dataset)

# Calculate NLL for grid samples
grid_likelihood = calculate_likelihood(
    nn_model, grid_dataloader, timesteps, device, ab_t, b_t, a_t
)

with open(timing_log_path, "a") as f:
    f.write(f"ELBO of parameter grid samples: {grid_elbo:.6f}\n")
    f.write(f"BPD of parameter grid samples: {grid_bpd:.6f}\n")
    f.write(f"Negative log likelihood of parameter grid samples: {grid_likelihood:.6f}\n")

# 3. Test classifier-free guidance at different strengths
guidance_strengths = [0.0, 1.0, 2.0, 3.0, 5.0]
guided_samples = []
guided_metrics = []

for w in guidance_strengths:
    # Use the same parameter set for all guidance strengths (first sample's parameters)
    params = selected_params[0].unsqueeze(0).repeat(5, 1)
    
    # Generate samples with this guidance strength
    samples, _, sampling_time, _ = sample_ddpm(n_sample=5, size=height, device=device, params=params, guide_w=w)
    guided_samples.append(samples)
    
    # Calculate metrics for this guidance strength
    with torch.no_grad():
        # Create a dataset from these samples
        guidance_dataset = TensorDataset(samples, params)
        guidance_dataloader = DataLoader(guidance_dataset, batch_size=5, shuffle=False)
        
        # Calculate ELBO and BPD
        total_elbo = 0
        total_bpd = 0
        for x, param in guidance_dataloader:
            # Sample random timesteps
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            
            # Add noise according to timestep
            noise = torch.randn_like(x)
            x_noisy = perturb_input(x, t, noise)
            
            # Predict noise
            pred_noise = nn_model(x_noisy, t / timesteps, param)
            
            # Calculate ELBO and BPD
            dims = x.shape[2] * x.shape[3]  # height * width
            # Create a temporary dataset for this batch
            temp_dataset = TensorDataset(x, param)
            temp_dataloader = DataLoader(temp_dataset, batch_size=x.shape[0], shuffle=False)
                    
            # Calculate ELBO and BPD using the main function
            elbo, bpd = calculate_elbo_and_bpd(
                nn_model, temp_dataloader, timesteps, device, ab_t, b_t, a_t
            )
            
            total_elbo += elbo * x.shape[0]
            total_bpd += bpd * x.shape[0]
        
        # Calculate NLL
        guidance_likelihood = calculate_likelihood(
            nn_model, guidance_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        
        # Store metrics
        guided_metrics.append({
            'guidance': w,
            'elbo': total_elbo / len(guidance_dataset),
            'bpd': total_bpd / len(guidance_dataset),
            'nll': guidance_likelihood,
        })
    
    with open(timing_log_path, "a") as f:
        f.write(f"Guidance strength {w} - ELBO: {guided_metrics[-1]['elbo']:.6f}, ")
        f.write(f"BPD: {guided_metrics[-1]['bpd']:.6f}, NLL: {guided_metrics[-1]['nll']:.6f}\n")

# Combine and save samples from different guidance strengths
guided_samples = torch.cat(guided_samples)
save_image(guided_samples, os.path.join(output_dir, "guidance_strength_samples.png"), nrow=5)
print(f"Saved guidance strength comparison to {output_dir}/guidance_strength_samples.png")

# Plot guidance strength vs. metrics (ELBO, BPD, NLL)
plt.figure(figsize=(15, 5))

# Plot guidance strength vs. metrics
plt.subplot(1, 3, 1)
plt.plot([m['guidance'] for m in guided_metrics], [m['elbo'] for m in guided_metrics], 'o-')
plt.xlabel("Guidance Strength")
plt.ylabel("ELBO")
plt.grid(True)
plt.title("Guidance Strength vs. ELBO")

plt.subplot(1, 3, 2)
plt.plot([m['guidance'] for m in guided_metrics], [m['bpd'] for m in guided_metrics], 'o-')
plt.xlabel("Guidance Strength")
plt.ylabel("Bits Per Dimension (BPD)")
plt.grid(True)
plt.title("Guidance Strength vs. BPD")

plt.subplot(1, 3, 3)
plt.plot([m['guidance'] for m in guided_metrics], [m['nll'] for m in guided_metrics], 'o-')
plt.xlabel("Guidance Strength")
plt.ylabel("Negative Log Likelihood (NLL)")
plt.grid(True)
plt.title("Guidance Strength vs. NLL")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "guidance_metrics.png"))
plt.close()
print(f"Saved guidance metrics plot to {output_dir}/guidance_metrics.png")

# 4. Create visualization summarizing parameter sensitivity
# This helps understand how each conditioned parameter affects the output
if num_params > 0:
    fig, axs = plt.subplots(num_params, 5, figsize=(15, 3*num_params))
    if num_params == 1:
        axs = axs.reshape(1, -1)  # Make sure axs is 2D for single parameter case
    
    for param_idx in range(num_params):
        # For each parameter, generate samples with increasing values
        param_values = torch.linspace(0.0, 1.0, 5)
        
        # Store metrics for each parameter value
        param_metrics = []
        
        for i, val in enumerate(param_values):
            # Start with base parameters and modify just one parameter
            base_param = selected_params[0].clone().unsqueeze(0)
            base_param[0, param_idx] = val
            
            # Generate a sample with this parameter value
            sample, _, _, _ = sample_ddpm(n_sample=1, size=height, device=device, params=base_param)
            img = sample[0, 0].cpu().numpy()
            
            # Calculate metrics for this parameter value
            with torch.no_grad():
                # Create a dataset from this sample
                param_dataset = TensorDataset(sample, base_param)
                param_dataloader = DataLoader(param_dataset, batch_size=1, shuffle=False)
                
                # Calculate ELBO and BPD
                t = torch.randint(1, timesteps + 1, (1,)).to(device)
                noise = torch.randn_like(sample)
                x_noisy = perturb_input(sample, t, noise)
                pred_noise = nn_model(x_noisy, t / timesteps, base_param)
                dims = sample.shape[2] * sample.shape[3]  # height * width
                # Create a temporary dataset for this batch
                temp_dataset = TensorDataset(sample, base_param)
                temp_dataloader = DataLoader(temp_dataset, batch_size=sample.shape[0], shuffle=False)
                        
                # Calculate ELBO and BPD using the main function
                elbo, bpd = calculate_elbo_and_bpd(
                    nn_model, temp_dataloader, timesteps, device, ab_t, b_t, a_t
                )
                
                # Calculate NLL
                param_likelihood = calculate_likelihood(
                    nn_model, param_dataloader, timesteps, device, ab_t, b_t, a_t
                )
                
                # Store metrics
                param_metrics.append({
                    'param_idx': param_idx,
                    'param_value': val.item(),
                    'elbo': elbo,
                    'bpd': bpd,
                    'nll': param_likelihood,
                })
            
            # Display the parameter value and sample
            axs[param_idx, i].imshow(img, cmap='viridis')
            axs[param_idx, i].set_title(f"Param {param_idx+1} = {val:.2f}")
            axs[param_idx, i].axis('off')
        
        # Log parameter sensitivity metrics
        with open(timing_log_path, "a") as f:
            f.write(f"\nParameter {param_idx+1} sensitivity metrics:\n")
            for m in param_metrics:
                f.write(f"  Value {m['param_value']:.2f} - ELBO: {m['elbo']:.6f}, ")
                f.write(f"BPD: {m['bpd']:.6f}, NLL: {m['nll']:.6f}\n")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_sensitivity.png"))
    plt.close()
    print(f"Saved parameter sensitivity analysis to {output_dir}/parameter_sensitivity.png")

    # Plot parameter value vs. metrics for each parameter
    for param_idx in range(num_params):
        # Filter metrics for this parameter
        metrics = [m for m in param_metrics if m['param_idx'] == param_idx]
        
        plt.figure(figsize=(15, 5))
        
        # Plot parameter value vs. metrics
        plt.subplot(1, 3, 1)
        plt.plot([m['param_value'] for m in metrics], [m['elbo'] for m in metrics], 'o-')
        plt.xlabel(f"Parameter {param_idx+1} Value")
        plt.ylabel("ELBO")
        plt.grid(True)
        plt.title(f"Parameter {param_idx+1} Value vs. ELBO")
        
        plt.subplot(1, 3, 2)
        plt.plot([m['param_value'] for m in metrics], [m['bpd'] for m in metrics], 'o-')
        plt.xlabel(f"Parameter {param_idx+1} Value")
        plt.ylabel("Bits Per Dimension (BPD)")
        plt.grid(True)
        plt.title(f"Parameter {param_idx+1} Value vs. BPD")
        
        plt.subplot(1, 3, 3)
        plt.plot([m['param_value'] for m in metrics], [m['nll'] for m in metrics], 'o-')
        plt.xlabel(f"Parameter {param_idx+1} Value")
        plt.ylabel("Negative Log Likelihood (NLL)")
        plt.grid(True)
        plt.title(f"Parameter {param_idx+1} Value vs. NLL")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parameter_{param_idx+1}_metrics.png"))
        plt.close()

print(f"Training and evaluation completed with {num_params} conditioning parameters.")