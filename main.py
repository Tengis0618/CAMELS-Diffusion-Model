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
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=128):  # Increased n_feat and n_cfeat
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  

        # Downsampling path
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)  # Additional downsampling layer
        self.to_vec = nn.Sequential(nn.AvgPool2d((self.h // 8)), nn.GELU())

        # Embedding layers
        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 4 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 2 * n_feat)

        # Upsampling path
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, self.h // 8, self.h // 8),
            nn.GroupNorm(8, 4 * n_feat),
            nn.GELU(),  # Changed activation to GELU
        )
        self.up1 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)  # Additional upsampling layer

        # Modified output layer with additional convolutions
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),  # Additional convolution
            nn.GroupNorm(8, n_feat),
            nn.GELU(),  # Changed activation to GELU
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
            nn.Tanh()  # Output activation remains Tanh
        )

    def forward(self, x, t, c=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 4, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down3)
        up3 = self.up2(cemb2 * up2 + temb2, down2)
        up4 = self.up3(up3, down1)
        out = self.out(torch.cat((up4, x), 1))
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
output_dir = f"outputs/lr_{lrate}_epochs_{n_epoch}_timesteps_{timesteps}"
save_dir = os.path.join(output_dir, "weights")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Diffusion hyperparameters
beta1 = 1e-4
beta2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 256  # Increased feature size
n_cfeat = 10  # Increased context feature size
height = 128
batch_size = 32

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumprod(a_t, dim=0)
ab_t[0] = 1

nn_model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

# Load the dataset
file_path = '/scratch/mr6174/Maps_HI_IllustrisTNG_LH_z=0.00.npy'
camels_data = np.load(file_path)

# Ensure positive values for log10
min_value = np.min(camels_data)
if min_value <= 0:
    camels_data = camels_data - min_value + 1e-8

# Apply log10 scaling
camels_data = np.log10(camels_data)

# Normalize to zero mean and unit variance
camels_data = (camels_data - camels_data.mean()) / camels_data.std()

# Scale to [-1, 1] using min-max scaling
camels_min = camels_data.min()
camels_max = camels_data.max()
camels_data = 2 * (camels_data - camels_min) / (camels_max - camels_min) - 1


print("Min value:", camels_data.min())
print("Max value:", camels_data.max())

# Convert to tensor and resize to 128x128
camels_data_tensor = torch.tensor(camels_data, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
camels_data_resized = F.interpolate(camels_data_tensor, size=(height, height), mode='bilinear')

# Create DataLoader for the entire dataset
training_data = TensorDataset(camels_data_resized)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# Select 10 random images for reconstruction
random_indices = random.sample(range(camels_data_tensor.shape[0]), 10)
selected_images = camels_data_tensor[random_indices]
selected_images_resized = F.interpolate(selected_images, size=(height, height), mode='bilinear')

# Save the processed images and compute their mean
save_image((selected_images_resized + 1) / 2, os.path.join(output_dir, "processed_images.png"))
processed_images_mean = selected_images_resized.mean().item()
print(f"Processed images mean: {processed_images_mean}")

optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise

loss_log = []
nn_model.train()
for ep in range(n_epoch):
    log_device_used()
    epoch_loss = 0
    optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
    pbar = tqdm(dataloader, mininterval=2)
    for x_batch, in pbar:
        optim.zero_grad()
        x = x_batch.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
        x_pert = perturb_input(x, t, noise)
        pred_noise = nn_model(x_pert, t / timesteps)
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    loss_log.append(avg_loss)
    print(f"Epoch {ep+1}/{n_epoch}, Loss: {avg_loss:.6f}")

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
    noise = b_t.sqrt()[t, None, None, None] * z
    mean = (x - pred_noise * ((1 - a_t[t, None, None, None]) / (1 - ab_t[t, None, None, None]).sqrt())) / a_t[t, None, None, None].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    samples = torch.randn(n_sample, 1, height, height).to(device)
    intermediate = [] 
    for i in range(timesteps, 0, -1):
        t = torch.full((n_sample,), i, device=device, dtype=torch.long)
        eps = nn_model(samples, t / timesteps)
        z = torch.randn_like(samples) if i > 1 else torch.zeros_like(samples)
        samples = denoise_add_noise(samples, t, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu())
    return samples, intermediate

# Compare distributions using PDFs
def compare_distributions(camels_images, diffusion_images, output_dir):
    bin_max = 1.0  # Because normalized range is [-1, 1]
    bin_min = -1.0
    bin_delta = 0.01  # Smaller bin size for better resolution
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
samples, _ = sample_ddpm(10)
save_image((samples + 1) / 2, os.path.join(output_dir, "reconstructed_images.png"))
compare_distributions(selected_images_resized.numpy(), samples.cpu().numpy(), output_dir)
reconstructed_mean = samples.mean().item()
print(f"Reconstructed images mean: {reconstructed_mean}")

with open(os.path.join(output_dir, "means.txt"), "w") as f:
    f.write(f"Processed Images Mean: {processed_images_mean}\n")
    f.write(f"Reconstructed Images Mean: {reconstructed_mean}\n")

plt.bar(["Processed Images", "Reconstructed Images"], [processed_images_mean, reconstructed_mean], color=['blue', 'orange'])
plt.ylabel("Mean")
plt.title("Comparison of Processed and Reconstructed Image Means")
plt.savefig(os.path.join(output_dir, "mean_histogram.png"))
plt.close()
print("Saved histogram of means as mean_histogram.png")

# Additional code to visualize individual images and their histograms
def plot_image(tensor_image, title='Image'):
    image = tensor_image.cpu().numpy().squeeze()
    plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_image_histogram(tensor_image, title='Histogram'):
    image = tensor_image.cpu().numpy().flatten()
    plt.hist(image, bins=100, range=(-1, 1), density=True)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

# Plot individual processed and reconstructed images
plot_image(selected_images_resized[0], title='Processed Image')
plot_image(samples[0], title='Generated Image')

# Plot their histograms
plot_image_histogram(selected_images_resized[0], title='Processed Image Histogram')
plot_image_histogram(samples[0], title='Generated Image Histogram')
