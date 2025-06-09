import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.ReLU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.ReLU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out #/ 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

        

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        
        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

    
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]
        
        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        device = next(self.parameters()).device
    
        # Move input to the same device
        x = x.to(device)
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)
    
def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)

def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t,s] = unorm(store[t,s])
    return nstore

def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2,3))
    xmin = x.min((2,3))
    xmax = np.expand_dims(xmax,(2,3)) 
    xmin = np.expand_dims(xmin,(2,3))
    nstore = (x - xmin)/(xmax - xmin)
    return torch.from_numpy(nstore)

def gen_tst_context(n_cfeat):
    """
    Generate test context vectors
    """
    vec = torch.tensor([
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0]]      # human, non-human, food, spell, side-facing
    )
    return len(vec), vec

def plot_grid(x,n_sample,n_rows,save_dir,w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('saved image at ' + save_dir + f"run_image_w{w}.png")
    return grid

def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani


class CustomDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.maps = np.load(filename)

        # Calculate global statistics ONCE during initialization                                                                            
        self.global_min = np.min(self.maps)
        self.global_max = np.max(self.maps)

        # Log transformation global statistics                                                                                              
        self.log_maps = np.log1p(self.maps + 1e-6)
        self.log_global_min = np.min(self.log_maps)
        self.log_global_max = np.max(self.log_maps)

        print(f"Original maps shape: {self.maps.shape}")
        print(f"Data statistics before normalization:")
        print(f"Min: {np.min(self.maps):.2f}")
        print(f"Max: {np.max(self.maps):.2f}")
        print(f"Mean: {np.mean(self.maps):.2f}")
        print(f"Std: {np.std(self.maps):.2f}")
        
        # Ensure correct shape (N, H, W, C) before transforming
        if len(self.maps.shape) == 3:
            self.maps = self.maps[..., np.newaxis]
        
        self.transform = transform
        self.maps_shape = self.maps.shape
    
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        image = self.maps[idx].astype(np.float32)  # Convert to float32

        # Apply log normalization
        eps = 1e-6
        image = np.log1p(image + eps)
        
        # Min-max normalization to [-1, 1]
        #image = 2 * (image - global_min) / (global_max - global_min) - 1
        image = 2 * (image - self.log_global_min) / (self.log_global_max - self.log_global_min) - 1
        
        # Convert to tensor and move channels first (NHWC -> NCHW)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
        
        if self.transform:
            image = self.transform(image)
            
        return image
    
    def getshapes(self):
        return self.maps_shape

# Simplified transform - we're doing the normalization in __getitem__
transform = transforms.Compose([
    transforms.Lambda(lambda x: x)  # Identity transform as we handle normalization in dataset
])

# Diagnostic function
def verify_dataset(dataset, num_samples=5):
    """Verify the dataset outputs correct tensor shapes and values"""
    print("\nDataset Verification:")
    
    sample = dataset[0]
    print(f"Single sample shape: {sample.shape}")
    print(f"Sample min: {sample.min():.4f}")
    print(f"Sample max: {sample.max():.4f}")
    
    # Check multiple samples
    for i in range(num_samples):
        sample = dataset[i]
        if not (sample.shape[0] == 1 and len(sample.shape) == 3):
            print(f"Warning: Sample {i} has incorrect shape: {sample.shape}")
        if torch.isnan(sample).any():
            print(f"Warning: Sample {i} contains NaN values")
        if torch.isinf(sample).any():
            print(f"Warning: Sample {i} contains infinite values")

# Usage example:
# dataset = CustomDataset("./sampled_dataset_64x64_60000.npy", transform)
# verify_dataset(dataset)

def power_spectrum(box, dl=1.0):
    """
    Calculate the power spectrum of a box (2D or 3D).
    
    Args:
        box (np.ndarray): Input box/grid (2D or 3D)
        dl (float): Physical spacing between grid points
        
    Returns:
        k_bins (np.ndarray): Frequency bins (k values)
        pk (np.ndarray): Power spectrum
    """
    # Get dimensions of the box
    dims = box.shape
    ndims = len(dims)
    
    if ndims not in [2, 3]:
        raise ValueError("Input box must be 2D or 3D")
    
    # Compute FFT with orthogonal normalization
    FT_box = np.fft.fftn(box, norm="ortho")
    
    # Create frequency grid
    k_components = []
    for i in range(ndims):
        k_components.append(2*np.pi*np.fft.fftfreq(dims[i], dl))
    
    # For 2D
    if ndims == 2:
        kx, ky = np.meshgrid(k_components[0], k_components[1], indexing='ij')
        kgrid = np.sqrt(kx**2 + ky**2)
    # For 3D
    else:
        kx, ky, kz = np.meshgrid(k_components[0], k_components[1], k_components[2], indexing='ij')
        kgrid = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Determine k-bins
    dk = 2*np.pi/(np.min(dims)*dl)  # Bin size
    k_max = np.max(kgrid)
    n_bins = int(np.ceil(k_max/dk)) + 1
    
    # Initialize arrays for binning
    pk = np.zeros(n_bins)
    count = np.zeros(n_bins)
    
    # Flatten arrays for easier binning
    kgrid_flat = kgrid.flatten()
    power_flat = (np.abs(FT_box)**2).flatten()
    
    # Bin the power
    for i in range(len(kgrid_flat)):
        bin_idx = int(round(kgrid_flat[i]/dk))
        if bin_idx < n_bins:  # Guard against out-of-bounds
            pk[bin_idx] += power_flat[i]
            count[bin_idx] += 1
    
    # Avoid division by zero
    valid_bins = count > 0
    pk[valid_bins] /= count[valid_bins]
    
    # Apply physical scaling
    pk *= dl**ndims
    
    # Create k-axis
    k_bins = np.arange(n_bins) * dk
    
    return k_bins, pk

def compare_power_spectra(original_images, generated_images, output_dir, dl=1.0, title="Power Spectrum Comparison"):
    """
    Compare power spectra of original and generated images
    
    Args:
        original_images (torch.Tensor): Original images [B, 1, H, W]
        generated_images (torch.Tensor): Generated images [B, 1, H, W]
        output_dir (str): Directory to save the comparison plot
        dl (float): Physical spacing between grid points
        title (str): Title for the plot
    """
    # Convert to numpy if needed
    if isinstance(original_images, torch.Tensor):
        original_images = original_images.squeeze(1).cpu().numpy()
    if isinstance(generated_images, torch.Tensor):
        generated_images = generated_images.squeeze(1).cpu().numpy()
    
    n_samples = min(len(original_images), len(generated_images))
    
    # Calculate power spectra
    orig_k_all = []
    orig_pk_all = []
    gen_k_all = []
    gen_pk_all = []
    
    for i in range(n_samples):
        # Original image
        k, pk = power_spectrum(original_images[i], dl)
        orig_k_all.append(k)
        orig_pk_all.append(pk)
        
        # Generated image
        k, pk = power_spectrum(generated_images[i], dl)
        gen_k_all.append(k)
        gen_pk_all.append(pk)
    
    # Calculate mean and standard deviation
    # Use the shortest k range for all samples
    min_len = min([len(k) for k in orig_k_all + gen_k_all])
    
    orig_pk_array = np.array([pk[:min_len] for pk in orig_pk_all])
    gen_pk_array = np.array([pk[:min_len] for pk in gen_pk_all])
    
    orig_pk_mean = np.mean(orig_pk_array, axis=0)
    orig_pk_std = np.std(orig_pk_array, axis=0)
    
    gen_pk_mean = np.mean(gen_pk_array, axis=0)
    gen_pk_std = np.std(gen_pk_array, axis=0)
    
    # Common k axis (use the first one, they should be the same)
    k = orig_k_all[0][:min_len]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot original power spectrum with error band
    plt.loglog(k[1:], orig_pk_mean[1:], 'b-', label='Original')
    plt.fill_between(k[1:], orig_pk_mean[1:] - orig_pk_std[1:], 
                    orig_pk_mean[1:] + orig_pk_std[1:], alpha=0.3, color='b')
    
    # Plot generated power spectrum with error band
    plt.loglog(k[1:], gen_pk_mean[1:], 'r-', label='Diffusion Model')
    plt.fill_between(k[1:], gen_pk_mean[1:] - gen_pk_std[1:], 
                    gen_pk_mean[1:] + gen_pk_std[1:], alpha=0.3, color='r')
    
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_spectrum_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Saved power spectrum comparison to {output_dir}/power_spectrum_comparison.png")
    
    return k, orig_pk_mean, gen_pk_mean
'''
def calculate_elbo_and_bpd(x, model, param, b_t, a_t, ab_t, dims, device, timesteps, is_training=True):
    """
    Calculate ELBO and BPD for diffusion models using the full trajectory approach.
    
    Args:
        x: original images
        model: the diffusion model
        param: conditioning parameters
        b_t, a_t, ab_t: diffusion schedule tensors
        dims: dimensions of the data (e.g., 64*64 for a 64x64 image)
        device: computation device
        timesteps: total number of diffusion timesteps
        is_training: whether in training mode (random timesteps) or evaluation mode (full trajectory)
        
    Returns:
        elbo: Evidence Lower Bound
        bpd: Bits Per Dimension
    """
    batch_size = x.shape[0]
    
    if is_training:
        # For training, use random timesteps (original approach)
        t = torch.randint(1, timesteps + 1, (batch_size,)).to(device)
        
        # Add noise according to timestep
        noise = torch.randn_like(x)
        x_t = ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
        
        # Predict noise
        pred_noise = model(x_t, t / timesteps, param)
        
        # Mean squared error between predicted and target noise
        noise_mse = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
        
        # Calculate the weight for each timestep
        weight = 0.5 * (1.0 / (1.0 - ab_t[t]) - 1.0)
        
        # Calculate ELBO
        elbo = (weight * noise_mse).mean()
        
    else:
        # For evaluation, compute full trajectory ELBO
        total_elbo = torch.zeros(batch_size).to(device)
        
        # Iterate through all timesteps for a stable evaluation
        for t in range(1, timesteps + 1):
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Add noise for current timestep
            noise = torch.randn_like(x)
            x_t = ab_t.sqrt()[t] * x + (1 - ab_t[t]) * noise
            
            # Predict noise
            pred_noise = model(x_t, t_tensor / timesteps, param)
            
            # Mean squared error for this timestep
            noise_mse = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
            
            # Weight for this timestep
            weight = 0.5 * (1.0 / (1.0 - ab_t[t]) - 1.0)
            
            # Add contribution to ELBO
            total_elbo += weight * noise_mse
            
        # Average over timesteps to get per-sample ELBO
        total_elbo /= timesteps
        
        # Average over batch
        elbo = total_elbo.mean()
    
    # Convert to bits per dimension
    bpd = elbo / (dims * np.log(2))
    
    return elbo, bpd
'''