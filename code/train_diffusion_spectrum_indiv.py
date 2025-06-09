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
    def __init__(self, in_channels, n_feat=128, n_cfeat=1, height=64):  # Changed to n_cfeat=1 for single parameter
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

def calculate_elbo_and_bpd(x, pred_noise, noise, t, b_t, a_t, ab_t, dims):
    """
    Calculate ELBO and BPD for diffusion models.
    
    Args:
        x: original images
        pred_noise: predicted noise by the model
        noise: target noise
        t: timestep indices
        b_t, a_t, ab_t: diffusion schedule tensors
        dims: dimensions of the data (e.g., 64*64 for a 64x64 image)
        
    Returns:
        elbo: Evidence Lower Bound
        bpd: Bits Per Dimension
    """
    # Mean squared error between predicted and target noise
    noise_mse = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
    
    # Calculate the weight for each timestep
    # This accounts for the scaling of the noise prediction loss in the ELBO
    weight = 0.5 * (1.0 / (1.0 - ab_t[t]) - 1.0)
    
    # Calculate ELBO
    elbo = weight * noise_mse
    elbo = elbo.mean()
    
    # Convert to bits per dimension
    # Factor log(2) converts from nats to bits
    bpd = elbo / (dims * np.log(2))
    
    return elbo, bpd

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

# Accept command-line arguments for learning rate, epochs, timesteps, and which parameter to use
lrate = float(sys.argv[1])
n_epoch = int(sys.argv[2])
timesteps = int(sys.argv[3])
param_index = int(sys.argv[4])  # Which parameter to condition on (0-based index)

# Define a unique directory to save outputs for this parameter combination
output_dir = f"outputs/spectrum_lr_{lrate}_epochs_{n_epoch}_timesteps_{timesteps}_param_{param_index}"
save_dir = os.path.join(output_dir, "weights")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Create a timing and performance log file
timing_log_path = os.path.join(output_dir, "timing_and_performance.log")
with open(timing_log_path, "w") as f:
    f.write("=== Diffusion Model Training and Sampling Timing Log ===\n\n")
    f.write(f"Parameters: learning_rate={lrate}, epochs={n_epoch}, timesteps={timesteps}, param_index={param_index}\n\n")

# Diffusion hyperparameters
beta1 = 1e-4
beta2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 128  # Feature size
n_cfeat = 1  # Always 1 since we're conditioning on a single parameter
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
param_file_path = '../data/params.npy'
param_data = np.load(param_file_path)  # Shape should be [n_samples, n_params]

# Check if the requested parameter index is valid
if param_index >= param_data.shape[1]:
    raise ValueError(f"Parameter index {param_index} is out of bounds. Available parameters: 0-{param_data.shape[1]-1}")

# Expand parameter data to match 15 images per parameter set
# Each row in param_data corresponds to 15 images in camels_data
expanded_param_data = np.repeat(param_data, 15, axis=0)
assert expanded_param_data.shape[0] == camels_data.shape[0], "Parameter expansion doesn't match image count"

# Extract only the specified parameter (single column)
selected_param_data = expanded_param_data[:, param_index:param_index+1]  # Keep as 2D array

# Normalize the selected parameter to [0,1] range for better conditioning
param_min = selected_param_data.min()
param_max = selected_param_data.max()
param_data_normalized = (selected_param_data - param_min) / (param_max - param_min + 1e-8)

# Save parameter normalization values for later use during generation
np.save(os.path.join(output_dir, "param_min.npy"), param_min)
np.save(os.path.join(output_dir, "param_max.npy"), param_max)
np.save(os.path.join(output_dir, "param_index.npy"), param_index)

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
    f.write(f"Parameter index used for conditioning: {param_index}\n")
    f.write(f"Original parameter data shape: {param_data.shape}\n")
    f.write(f"Selected parameter range: [{param_min:.6f}, {param_max:.6f}]\n")
    f.write(f"Final parameter data shape: {param_data_normalized.shape}\n")

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
    param_info += f"Image {i+1}: Parameter {param_index} = {params.item():.6f}\n"
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
    epoch_elbo = 0
    epoch_bpd = 0
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
        
        # MSE loss (already in your code)
        loss = F.mse_loss(pred_noise, noise)
        
        # Calculate ELBO and BPD
        dims = x.shape[2] * x.shape[3]  # height * width (64*64)
        elbo, bpd = calculate_elbo_and_bpd(x, pred_noise, noise, t, b_t, a_t, ab_t, dims)
        
        loss.backward()
        optim.step()
        
        epoch_loss += loss.item()
        epoch_elbo += elbo.item()
        epoch_bpd += bpd.item()
        
        # Update progress bar to include BPD and parameter info
        pbar.set_description(f"Epoch {ep+1}, Loss: {loss.item():.4f}, BPD: {bpd.item():.4f}, Param{param_index}")
    
    # Average over batches
    epoch_loss /= len(train_dataloader)
    epoch_elbo /= len(train_dataloader)
    epoch_bpd /= len(train_dataloader)
    
    loss_log.append(epoch_loss)
    elbo_log.append(epoch_elbo)
    bpd_log.append(epoch_bpd)
    
    # Record epoch time
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)
    
    # Log the epoch time
    with open(timing_log_path, "a") as f:
        f.write(f"Epoch {ep+1}/{n_epoch} completed in {epoch_duration:.2f} seconds\n")
        f.write(f"  Training Loss: {epoch_loss:.6f}, ELBO: {epoch_elbo:.6f}, BPD: {epoch_bpd:.6f}\n")
    
    # Evaluate on validation set and calculate likelihood every 5 epochs
    if ep % 5 == 0 or ep == n_epoch - 1:
        nn_model.eval()
        val_loss = 0
        val_elbo = 0
        val_bpd = 0
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
                
                # Calculate ELBO and BPD
                dims = x.shape[2] * x.shape[3]  # height * width
                elbo, bpd = calculate_elbo_and_bpd(x, pred_noise, noise, t, b_t, a_t, ab_t, dims)
                val_elbo += elbo.item()
                val_bpd += bpd.item()
        
        # Average over validation batches
        val_loss /= len(test_dataloader)
        val_elbo /= len(test_dataloader)
        val_bpd /= len(test_dataloader)
        
        val_loss_log.append(val_loss)
        val_elbo_log.append(val_elbo)
        val_bpd_log.append(val_bpd)
        
        # Calculate likelihood on a subset of test set to save time
        subset_size = min(len(test_dataset), 200)  # Use a smaller subset for likelihood calculation
        subset_indices = random.sample(range(len(test_dataset)), subset_size)
        subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
        subset_dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)
        
        likelihood_calculation_start = time.time()
        neg_log_likelihood = calculate_likelihood(
            nn_model, subset_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        likelihood_calculation_time = time.time() - likelihood_calculation_start
        likelihood_log.append(neg_log_likelihood)
        
        # Log likelihood metrics
        with open(timing_log_path, "a") as f:
            f.write(f"  Validation Loss: {val_loss:.6f}, Val ELBO: {val_elbo:.6f}, Val BPD: {val_bpd:.6f}\n")
            f.write(f"  Negative Log Likelihood: {neg_log_likelihood:.6f}\n")
            f.write(f"  Likelihood calculation took {likelihood_calculation_time:.2f} seconds\n")
        
        print(f"Epoch {ep+1}/{n_epoch}, Train Loss: {loss_log[-1]:.6f}, Val Loss: {val_loss_log[-1]:.6f}")
        print(f"Train BPD: {bpd_log[-1]:.6f}, Val BPD: {val_bpd_log[-1]:.6f}")
        print(f"Negative Log Likelihood: {neg_log_likelihood:.6f}")
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

# Modify the plotting section to include parameter information
plt.figure(figsize=(15, 10))

# Plot 1: Training and validation loss
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, n_epoch+1), np.log(loss_log), label='Training Loss')

# Create x-axis for validation loss that matches its actual logged points
val_epochs = list(range(0, n_epoch, 5)) + ([n_epoch-1] if (n_epoch-1) % 5 != 0 else [])
val_loss_x = [e+1 for e in val_epochs]  # +1 because epochs are 0-indexed in the code but 1-indexed in the plot
plt.plot(val_loss_x, np.log(val_loss_log), 'o-', label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.title(f"Loss Evolution conditioning on Parameter {param_index}")

# Plot 2: Negative log likelihood
plt.subplot(2, 2, 2)
plt.plot(val_loss_x, likelihood_log, 'o-r', label='Negative Log Likelihood')
plt.xlabel("Epoch")
plt.ylabel("NLL")
plt.legend()
plt.grid(True)
plt.title("Negative Log Likelihood Evolution")

# Plot 3: ELBO
plt.subplot(2, 2, 3)
plt.plot(np.arange(1, n_epoch+1), elbo_log, label='Training ELBO')
plt.plot(val_loss_x, val_elbo_log, 'o-', label='Validation ELBO')
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.legend()
plt.grid(True)
plt.title("ELBO Evolution")

# Plot 4: BPD
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, n_epoch+1), bpd_log, label='Training BPD')
plt.plot(val_loss_x, val_bpd_log, 'o-', label='Validation BPD')
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
        params (torch.Tensor): Conditioning parameters (single parameter values)
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
        # If no parameters provided, use random parameter values
        params = torch.rand(n_sample, 1).to(device)
    else:
        params = params.to(device)
        if params.dim() == 1:
            params = params.unsqueeze(1)  # Ensure it's [batch_size, 1]
        
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
        params (torch.Tensor): Conditioning parameters (single parameter values)
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
        if params.dim() == 1:
            params = params.unsqueeze(1)  # Ensure it's [batch_size, 1]
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
        elbo, bpd = calculate_elbo_and_bpd(x, pred_noise, noise, t, b_t, a_t, ab_t, dims)
        
        total_elbo += elbo.item() * x.shape[0]
        total_bpd += bpd.item() * x.shape[0]
    
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

# Calculate and compare power spectra
print("Calculating power spectra...")
k, orig_pk, gen_pk = compare_power_spectra(
    selected_images, 
    samples, 
    output_dir,
    dl=1.0,  # You may need to adjust this based on your physical grid spacing
    title=f"Power Spectrum conditioning on Parameter {param_index}"
)

# Log the power spectrum ratio (to assess how well the model preserves spectral properties)
pk_ratio = gen_pk / orig_pk
pk_ratio_mean = np.mean(pk_ratio[1:])  # Skip the k=0 bin
pk_ratio_std = np.std(pk_ratio[1:])

with open(timing_log_path, "a") as f:
    f.write(f"Power spectrum analysis:\n")
    f.write(f"  Mean P(k) ratio (generated/original): {pk_ratio_mean:.4f} ± {pk_ratio_std:.4f}\n")
    
    # Calculate the k range where the ratio is within 20% of unity (good match)
    good_match = np.where((pk_ratio > 0.8) & (pk_ratio < 1.2) & (k > 0))[0]
    if len(good_match) > 0:
        k_min = k[good_match[0]]
        k_max = k[good_match[-1]]
        f.write(f"  Good spectral match (within 20%) from k={k_min:.4f} to k={k_max:.4f}\n")
    else:
        f.write(f"  No k range with spectral match within 20%\n")

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

# 2. Generate samples with varied parameter values (focusing on the single parameter we conditioned on)
# Generate a range of values for the selected parameter
parameter_sweep_start_time = time.time()

param_values = torch.linspace(0.0, 1.0, 25)  # Generate 25 samples across the parameter range
sweep_params = param_values.unsqueeze(1)  # Shape: [25, 1]

# Generate samples for each parameter value
sweep_samples, sweep_intermediate, sweep_sampling_time, _ = sample_ddpm(
    n_sample=len(sweep_params), size=height, device=device, params=sweep_params
)

# Log parameter sweep generation time
with open(timing_log_path, "a") as f:
    f.write(f"Generating {len(sweep_params)} parameter sweep samples took {sweep_sampling_time:.2f} seconds\n")

# Save sweep of generated images
save_image(sweep_samples, os.path.join(output_dir, f"parameter_sweep_param_{param_index}.png"), nrow=5)
visualize_viridis_style(sweep_samples, os.path.join(output_dir, f"parameter_sweep_param_{param_index}_viridis.png"), nrow=5, title=f"Parameter {param_index} Sweep")

# Calculate ELBO and BPD for the parameter sweep samples
sweep_dataset = TensorDataset(sweep_samples, sweep_params)
sweep_dataloader = DataLoader(sweep_dataset, batch_size=batch_size, shuffle=False)

# Calculate metrics for parameter sweep samples
nn_model.eval()
with torch.no_grad():
    # Calculate ELBO and BPD
    total_elbo = 0
    total_bpd = 0
    for x, param in sweep_dataloader:
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
        elbo, bpd = calculate_elbo_and_bpd(x, pred_noise, noise, t, b_t, a_t, ab_t, dims)
        
        total_elbo += elbo.item() * x.shape[0]
        total_bpd += bpd.item() * x.shape[0]
    
    # Calculate average
    sweep_elbo = total_elbo / len(sweep_dataset)
    sweep_bpd = total_bpd / len(sweep_dataset)

# Calculate NLL for sweep samples
sweep_likelihood = calculate_likelihood(
    nn_model, sweep_dataloader, timesteps, device, ab_t, b_t, a_t
)

with open(timing_log_path, "a") as f:
    f.write(f"ELBO of parameter sweep samples: {sweep_elbo:.6f}\n")
    f.write(f"BPD of parameter sweep samples: {sweep_bpd:.6f}\n")
    f.write(f"Negative log likelihood of parameter sweep samples: {sweep_likelihood:.6f}\n")

# 3. Test classifier-free guidance at different strengths
guidance_strengths = [0.0, 1.0, 2.0, 3.0, 5.0]
guided_samples = []
guided_metrics = []

for w in guidance_strengths:
    # Use a fixed parameter value for all guidance strengths (middle of the range)
    params = torch.tensor([[0.5]]).repeat(5, 1)  # Shape: [5, 1]
    
    # Generate samples with this guidance strength
    samples_guided, _, sampling_time, _ = sample_ddpm(n_sample=5, size=height, device=device, params=params, guide_w=w)
    guided_samples.append(samples_guided)
    
    # Calculate metrics for this guidance strength
    with torch.no_grad():
        # Create a dataset from these samples
        guidance_dataset = TensorDataset(samples_guided, params)
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
            elbo, bpd = calculate_elbo_and_bpd(x, pred_noise, noise, t, b_t, a_t, ab_t, dims)
            
            total_elbo += elbo.item() * x.shape[0]
            total_bpd += bpd.item() * x.shape[0]
        
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

# 4. Create detailed parameter sensitivity analysis
# This shows how the single conditioned parameter affects the output across its full range
fig, axs = plt.subplots(5, 5, figsize=(15, 15))
axs = axs.flatten()

param_sensitivity_metrics = []

for i, val in enumerate(param_values):
    if i >= 25:  # Limit to 25 samples for visualization
        break
        
    # Generate a sample with this parameter value
    param_tensor = torch.tensor([[val.item()]])  # Shape: [1, 1]
    sample, _, _, _ = sample_ddpm(n_sample=1, size=height, device=device, params=param_tensor)
    img = sample[0, 0].cpu().numpy()
    
    # Calculate metrics for this parameter value
    with torch.no_grad():
        # Create a dataset from this sample
        param_dataset = TensorDataset(sample, param_tensor)
        param_dataloader = DataLoader(param_dataset, batch_size=1, shuffle=False)
        
        # Calculate ELBO and BPD
        t = torch.randint(1, timesteps + 1, (1,)).to(device)
        noise = torch.randn_like(sample)
        x_noisy = perturb_input(sample, t, noise)
        pred_noise = nn_model(x_noisy, t / timesteps, param_tensor)
        dims = sample.shape[2] * sample.shape[3]  # height * width
        elbo, bpd = calculate_elbo_and_bpd(sample, pred_noise, noise, t, b_t, a_t, ab_t, dims)
        
        # Calculate NLL
        param_likelihood = calculate_likelihood(
            nn_model, param_dataloader, timesteps, device, ab_t, b_t, a_t
        )
        
        # Store metrics
        param_sensitivity_metrics.append({
            'param_value': val.item(),
            'elbo': elbo.item(),
            'bpd': bpd.item(),
            'nll': param_likelihood,
        })
    
    # Display the parameter value and sample
    axs[i].imshow(img, cmap='viridis')
    axs[i].set_title(f"Param {param_index} = {val:.3f}")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"parameter_{param_index}_sensitivity.png"))
plt.close()
print(f"Saved parameter sensitivity analysis to {output_dir}/parameter_{param_index}_sensitivity.png")

# Plot parameter value vs. metrics
plt.figure(figsize=(15, 5))

# Plot parameter value vs. metrics
plt.subplot(1, 3, 1)
plt.plot([m['param_value'] for m in param_sensitivity_metrics], [m['elbo'] for m in param_sensitivity_metrics], 'o-')
plt.xlabel(f"Parameter {param_index} Value")
plt.ylabel("ELBO")
plt.grid(True)
plt.title(f"Parameter {param_index} Value vs. ELBO")

plt.subplot(1, 3, 2)
plt.plot([m['param_value'] for m in param_sensitivity_metrics], [m['bpd'] for m in param_sensitivity_metrics], 'o-')
plt.xlabel(f"Parameter {param_index} Value")
plt.ylabel("Bits Per Dimension (BPD)")
plt.grid(True)
plt.title(f"Parameter {param_index} Value vs. BPD")

plt.subplot(1, 3, 3)
plt.plot([m['param_value'] for m in param_sensitivity_metrics], [m['nll'] for m in param_sensitivity_metrics], 'o-')
plt.xlabel(f"Parameter {param_index} Value")
plt.ylabel("Negative Log Likelihood (NLL)")
plt.grid(True)
plt.title(f"Parameter {param_index} Value vs. NLL")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"parameter_{param_index}_metrics.png"))
plt.close()

# Log parameter sensitivity metrics
with open(timing_log_path, "a") as f:
    f.write(f"\nParameter {param_index} sensitivity metrics:\n")
    for m in param_sensitivity_metrics:
        f.write(f"  Value {m['param_value']:.3f} - ELBO: {m['elbo']:.6f}, ")
        f.write(f"BPD: {m['bpd']:.6f}, NLL: {m['nll']:.6f}\n")

print(f"Training and evaluation completed conditioning on Parameter {param_index}.")

# Final summary
with open(timing_log_path, "a") as f:
    f.write(f"\n=== Final Summary ===\n")
    f.write(f"Model conditioned on parameter index: {param_index}\n")
    f.write(f"Parameter range: [{param_min:.6f}, {param_max:.6f}]\n")
    f.write(f"Total parameter sweep samples generated: {len(param_values)}\n")
    f.write(f"Guidance strengths tested: {guidance_strengths}\n")
    f.write(f"All outputs saved to: {output_dir}\n")