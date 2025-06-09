import os
import sys
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

# Define a unique directory to save outputs for this parameter combination
output_dir = f"outputs/BIGmassnoiselr_{lrate}_epochs_{n_epoch}_timesteps_{timesteps}"
save_dir = os.path.join(output_dir, "weights")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Diffusion hyperparameters
beta1 = 1e-4
beta2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 128  # Feature size
n_cfeat = 5
height = 64
batch_size = 32

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# Load the dataset
file_path = '/scratch/mr6174/Maps_HI_IllustrisTNG_LH_z=0.00.npy'
camels_data = np.load(file_path)

# Normalize and preprocess data
min_value = np.min(camels_data)
if min_value <= 0:
    camels_data = camels_data - min_value + 1e-8
camels_data = camels_data / np.max(camels_data)
camels_data = np.log10(camels_data)
camels_data = (camels_data - camels_data.min()) / (camels_data.max() - camels_data.min())
camels_data_tensor = torch.tensor(camels_data, dtype=torch.float32).unsqueeze(1)
camels_data_resized = F.interpolate(camels_data_tensor, size=(64, 64), mode='bilinear')

# Create DataLoader for the entire dataset
training_data = TensorDataset(camels_data_resized)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# Select 10 random images for reconstruction
random_indices = random.sample(range(camels_data_tensor.shape[0]), 10)
selected_images = camels_data_tensor[random_indices]
selected_images_resized = F.interpolate(selected_images, size=(64, 64), mode='bilinear')

# Save the processed images and compute their mean
save_image(selected_images_resized, os.path.join(output_dir, "processed_images.png"))
processed_images_mean = selected_images_resized.mean().item()
print(f"Processed images mean: {processed_images_mean}")

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

loss_log = []
nn_model.train()
for ep in range(n_epoch):
    log_device_used()
    epoch_loss = 0
    optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
    pbar = tqdm(dataloader, mininterval=2)
    for x, in pbar:
        optim.zero_grad()
        x = x.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)
        pred_noise = nn_model(x_pert, t / timesteps)
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    loss_log.append(epoch_loss / len(dataloader))

    if ep % 4 == 0 or ep == n_epoch - 1:
        torch.save(nn_model.state_dict(), os.path.join(save_dir, f"model_epoch_{ep}.pth"))

plt.plot(np.log(loss_log))
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.savefig(os.path.join(output_dir, "loss_evolution.png"))
print("Saved loss evolution plot as loss_evolution.png")

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(noisy_images, save_rate=20):
    """
    Reverse diffusion process starting from the given processed images.

    Args:
        noisy_images (torch.Tensor): Tensor of images to reconstruct (e.g., selected images).
        save_rate (int): Frequency of saving intermediate images.

    Returns:
        samples (torch.Tensor): Reconstructed images.
        intermediate (list): Intermediate images at each step.
    """
    samples = noisy_images.clone().to(device)  # Use the selected images instead of random noise
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = nn_model(samples, t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())
    intermediate = np.stack(intermediate)
    return samples, intermediate

# Compare distributions using PDFs
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

    ax[0].plot(bin_mid, train_pdf_mean, "k-", label="CAMELS")
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

nn_model.eval()
# Forward diffuse the selected images to the final timestep
x = selected_images_resized.to(device)
noise = torch.randn_like(x)
# Forward diffusion to the last timestep (timesteps)
x_forward_diffused = perturb_input(x, timesteps, noise)

# Now sample (reverse diffusion) from the noisy images
samples, intermediate = sample_ddpm(x_forward_diffused)
for idx, img in enumerate(intermediate):
    save_image(torch.tensor(img), os.path.join(output_dir, f"intermediate_step_{idx}.png"))
save_image(samples, os.path.join(output_dir, "reconstructed_images.png"))
compare_distributions(selected_images_resized.numpy(), samples.cpu().numpy(), output_dir)
reconstructed_mean = samples.mean().item()
print(f"Reconstructed images mean: {reconstructed_mean}")

with open(os.path.join(output_dir, "means.txt"), "w") as f:
    f.write(f"Processed Images Mean: {processed_images_mean}\n")
    f.write(f"Reconstructed Images Mean: {reconstructed_mean}\n")

plt.figure(figsize=(8, 6))
plt.bar(["Processed Images", "Reconstructed Images"], [processed_images_mean, reconstructed_mean], color=['blue', 'orange'])
plt.ylabel("Mean")
plt.title("Comparison of Processed and Reconstructed Image Means")
plt.savefig(os.path.join(output_dir, "mean_histogram.png"))
plt.close()
print("Saved histogram of means as mean_histogram.png")

# Compute the mean ratio
mean_ratio = processed_images_mean / reconstructed_mean
print(f"Mean ratio: {mean_ratio}")

# Adjust reconstructed images
corrected_samples = samples * mean_ratio

# Save the corrected reconstructed images
save_image(corrected_samples, os.path.join(output_dir, "corrected_reconstructed_images.png"))
print(f"Saved corrected reconstructed images to {output_dir}/corrected_reconstructed_images.png")

# Update the PDFs with corrected samples
compare_distributions(selected_images_resized.numpy(), corrected_samples.cpu().numpy(), output_dir)

# Save corrected mean histogram
corrected_reconstructed_mean = corrected_samples.mean().item()
plt.figure(figsize=(8, 6))
plt.bar(["Processed Images", "Corrected Reconstructed Images"],
        [processed_images_mean, corrected_reconstructed_mean],
        color=['blue', 'orange'])
plt.ylabel("Mean")
plt.title("Comparison of Processed and Corrected Reconstructed Image Means")
plt.savefig(os.path.join(output_dir, "corrected_mean_histogram.png"))
plt.close()
print("Saved corrected histogram of means as corrected_mean_histogram.png")

# Log corrected means
with open(os.path.join(output_dir, "corrected_means.txt"), "w") as f:
    f.write(f"Processed Images Mean: {processed_images_mean}\n")
    f.write(f"Corrected Reconstructed Images Mean: {corrected_reconstructed_mean}\n")
