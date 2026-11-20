import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
import numpy as np
import random
import math
import pdb
from tqdm import tqdm
from typing import List
from torch import device
from PIL import Image
from torchsummary import summary
from torchviz import make_dot
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import Counter

# MODELS
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
    
    def forward(self, x, embeddings):
        # Fix: Ensure embeddings match x dimensions
        if embeddings.shape[1] > x.shape[1]:
            embeddings = embeddings[:, :x.shape[1], :, :]
        elif embeddings.shape[1] < x.shape[1]:
            # Pad embeddings if needed
            pad_size = x.shape[1] - embeddings.shape[1]
            padding = torch.zeros(embeddings.shape[0], pad_size, embeddings.shape[2], embeddings.shape[3], device=embeddings.device)
            embeddings = torch.cat([embeddings, padding], dim=1)
        
        x = x + embeddings
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
    
    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)
    
    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int, device: device):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings.to(device)
    
    def forward(self, t):
        # Fix: Handle batch indexing properly
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]

class UNET(nn.Module):
    def __init__(self,
                 Channels: List = [64, 128, 256, 512, 512, 384],
                 Attentions: List = [False, True, False, False, False, True],
                 Upscales: List = [False, False, False, True, True, True],
                 num_groups: int = 32,
                 dropout_prob: float = 0.0,
                 num_heads: int = 8,
                 input_channels: int = 3, 
                 output_channels: int = 3, 
                 device: str = 'cuda',
                 time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels), device=device)
        
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)
    
    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        embeddings = self.embeddings(t)  # Move this outside the loop
        
        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x, embeddings)
            residuals.append(r)
        
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x, _ = layer(x, embeddings)  # Fix: Get both outputs
            # Fix: Concatenate with correct residual
            residual_idx = self.num_layers - i - 1
            if residual_idx < len(residuals):
                x = torch.cat((x, residuals[residual_idx]), dim=1)
        
        x = self.relu(self.late_conv(x))
        return self.output_conv(x)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)
    
    def forward(self, t):
        return self.beta[t], self.alpha[t]

# Test scheduler
scheduler = DDPM_Scheduler(num_time_steps=1000)
print(scheduler(499))

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image_path, _ = self.samples[index]
        image_name = os.path.basename(image_path)
        # Get class name from the path
        class_name = os.path.basename(os.path.dirname(image_path))
        return image, label, image_name, class_name

def load_data(data_path, batch_size=16, image_size=(128, 128)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    train_dataset = CustomImageFolder(root=data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    class_to_idx = train_dataset.class_to_idx
    return train_loader, class_to_idx

def map_labels(class_to_idx, test_labels):
    idx_to_test_labels = {}
    for class_name, idx in class_to_idx.items():
        if class_name in test_labels:
            idx_to_test_labels[idx] = test_labels[class_name]
        else:
            raise ValueError(f"Class {class_name} not found in test_labels.")
    return idx_to_test_labels

test_labels = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'mel': 3,
    'nv': 4,
    'df': 5,
    'vasc': 6
}

train_data_path = '/home/pratyushprasad/DPO_PROJECT/1. data/ham_10000_images_all/ham_10000_train'
train_loader, train_class_to_idx = load_data(train_data_path, batch_size=8, image_size=(128, 128))
train_idx_to_test_labels = map_labels(train_class_to_idx, test_labels)

print("Class to Index Mapping (from dataset):", train_class_to_idx)
print("Index to Test Labels Mapping:", train_idx_to_test_labels)

for images, labels, image_names, class_names in train_loader:
    print(f"Image Batch Shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Image Names: {image_names}")
    print(f"Class Names: {class_names}")
    break

def denoise_val_images(checkpoint_path: str,
                       train_loader,
                       num_time_steps: int = 1000,
                       ema_decay: float = 0.9999,
                       save_path: str = "/home/pratyushprasad/DPO_PROJECT/2. vanilla_ddpm_model/skin_lesions/dataset/generated_images_for_annotation",
                       num_generated_per_image: int = 5):
    
    checkpoint = torch.load(checkpoint_path)
    model = UNET(input_channels=3, output_channels=3).cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    
    scheduler.alpha = scheduler.alpha.cuda()
    scheduler.beta = scheduler.beta.cuda()
    
    # Create main save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Create class folders based on the class names from the dataset
    for class_name in train_class_to_idx.keys():
        class_folder = os.path.join(save_path, class_name)
        os.makedirs(class_folder, exist_ok=True)
        print(f"Created class folder: {class_folder}")
    
    image_paths = []
    with torch.no_grad():
        model = ema.module.eval()
        for batch_idx, (images, labels, image_names, class_names) in enumerate(train_loader):
            images = images.cuda()
            batch_size = images.size(0)
            t_values = torch.ones(batch_size, dtype=torch.long, device=images.device) * (num_time_steps // 2)
            
            for i in range(batch_size):
                t = t_values[i]
                alpha_t = scheduler.alpha[t].view(1, 1, 1)
                
                # Get class name for this image
                class_name = class_names[i]
                class_folder = os.path.join(save_path, class_name)
                
                # Generate multiple images from the same input
                for gen_idx in range(num_generated_per_image):
                    # Create noisy image with different random noise each time
                    noisy_image = torch.sqrt(alpha_t) * images[i] + torch.sqrt(1 - alpha_t) * torch.randn_like(images[i])
                    
                    z = noisy_image.unsqueeze(0)
                    
                    # Denoising process
                    for t_step in reversed(range(1, t.item() + 1)):
                        t_tensor = torch.tensor([t_step], device=z.device)
                        beta_t = scheduler.beta[t_step]
                        alpha_t = scheduler.alpha[t_step]
                        
                        pred_noise = model(z, t_tensor)
                        
                        z = (1 / torch.sqrt(1 - beta_t)) * (z - (beta_t / torch.sqrt(1 - alpha_t)) * pred_noise)
                        
                        if t_step > 1:
                            noise = torch.randn_like(z)
                            z = z + noise * torch.sqrt(beta_t)
                    
                    # Final generated image
                    x = z
                    x = (x - x.min()) / (x.max() - x.min())  
                    
                    # Create filename with generation index
                    image_name = image_names[i] if isinstance(image_names[i], str) else str(image_names[i])
                    base_name = os.path.splitext(image_name)[0]  # Remove extension
                    generated_image_name = f"generated_{base_name}_{gen_idx + 1}_0.png"
                    image_path = os.path.join(class_folder, generated_image_name)
                    
                    # Save the generated image
                    save_image(x, image_path)
                    image_paths.append(image_path)
                    print(f"Saved: {image_path}")
    
    print(f"\nGeneration complete! Generated {len(image_paths)} images total.")
    print(f"Images saved in class folders under: {save_path}")

# Run the generation
denoise_val_images(
    checkpoint_path="/home/pratyushprasad/DPO_PROJECT/2. vanilla_ddpm_model/skin_lesions/saved_model/ddpm_skin_lesions_first_model.pt",
    train_loader=train_loader,
    num_time_steps=1000,
    ema_decay=0.9999,
    save_path="/home/pratyushprasad/DPO_PROJECT/2. vanilla_ddpm_model/skin_lesions/dataset/generated_images_for_annotation",
    num_generated_per_image=1
)