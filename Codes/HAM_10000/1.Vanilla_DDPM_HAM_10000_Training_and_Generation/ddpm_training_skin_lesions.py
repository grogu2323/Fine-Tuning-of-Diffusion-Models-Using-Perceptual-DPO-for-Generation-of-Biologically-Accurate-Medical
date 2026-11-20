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

rotation_augment = transforms.RandomRotation(degrees=(-90, 90))

def get_transforms(image_size=(128, 128)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

class AugmentedRotationDataset(Dataset):
    def __init__(self, image_paths, label, transform, augment_count):
        self.image_paths = image_paths
        self.label = label
        self.transform = transform
        self.augment_count = augment_count
    
    def __len__(self):
        return self.augment_count
    
    def __getitem__(self, idx):
        path = random.choice(self.image_paths)
        image = Image.open(path).convert("RGB")
        image = rotation_augment(image)
        if self.transform:
            image = self.transform(image)
        return image, self.label

def load_balanced_dataset(data_path, batch_size=8, image_size=(128, 128)):
    base_transform = get_transforms(image_size)
    dataset = ImageFolder(root=data_path, transform=base_transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label_counts = Counter(dataset.targets)
    
    synthetic_datasets = []
    for class_name in ['df', 'vasc']:
        class_idx = class_to_idx[class_name]
        current_count = label_counts[class_idx]
        if current_count < 100:
            needed = 100 - current_count
            print(f"Augmenting {needed} for class '{class_name}'")
            class_folder = os.path.join(data_path, class_name)
            image_paths = [os.path.join(class_folder, f) for f in os.listdir(class_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            synthetic_datasets.append(AugmentedRotationDataset(
                image_paths=image_paths,
                label=class_idx,
                transform=base_transform,
                augment_count=needed
            ))
        else:
            print(f"No augmentation needed for '{class_name}'")
    
    if synthetic_datasets:
        full_dataset = ConcatDataset([dataset] + synthetic_datasets)
    else:
        full_dataset = dataset
    
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    return loader, full_dataset, class_to_idx

# Data loading with your paths
train_data_path = '/home/pratyushprasad/DPO_PROJECT/1. data/ham_10000_images_all/ham_10000_train'
val_data_path = '/home/pratyushprasad/DPO_PROJECT/1. data/ham_10000_images_all/ham_10000_train'

train_loader, train_dataset, train_class_to_idx = load_balanced_dataset(train_data_path)
val_loader, val_dataset, val_class_to_idx = load_balanced_dataset(val_data_path)

def print_class_distribution(dataset, class_to_idx, name=""):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label_counts = Counter()
    for _, label in dataset:
        label_counts[label] += 1
    print(f"\n{name} Set Class Distribution:")
    for idx, count in sorted(label_counts.items()):
        print(f"{idx_to_class[idx]} ({idx}): {count}")

print_class_distribution(train_dataset, train_class_to_idx, "Train")
print_class_distribution(val_dataset, val_class_to_idx, "Validation")

for images, labels in train_loader:
    print("\nSample Train Batch:")
    print("Images:", images.shape)
    print("Labels:", labels)
    break

for images, labels in val_loader:
    print("\nSample Validation Batch:")
    print("Images:", images.shape)
    print("Labels:", labels)
    break

def train_and_validate(train_loader, val_loader, num_time_steps=1000, num_epochs=2000, seed=100,
                       ema_decay=0.9999, lr=2e-5, checkpoint_path=None,
                       save_path="/home/pratyushprasad/DPO_PROJECT/2. vanilla_ddpm_model/skin_lesions/saved_model/ddpm_skin_lesions_first_model.pt",
                       early_stopping_patience=200, lr_plateau_patience=50):
    
    # Fix: Set seed properly
    if seed == -1:
        set_seed(100)
    else:
        set_seed(seed)
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.cuda()
    scheduler.beta = scheduler.beta.cuda()  # Also move beta to CUDA
    
    model = UNET(device='cuda').cuda()  # Pass device parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    criterion = nn.MSELoss(reduction='mean')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.90, patience=lr_plateau_patience)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for bidx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):  # Fix: unpack properly
            actual_batch_size = x.size(0)
            x = x.cuda()
            
            # Fix: Ensure t is within valid range
            t = torch.randint(0, num_time_steps, (actual_batch_size,), device='cuda')
            e = torch.randn_like(x, requires_grad=False)
            
            # Fix: Proper indexing for alpha
            a = scheduler.alpha[t].view(actual_batch_size, 1, 1, 1)
            x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            
            output = model(x_noisy, t)
            
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc="Validating"):  # Fix: unpack properly
                actual_batch_size = x.size(0)
                x = x.cuda()
                
                t = torch.randint(0, num_time_steps, (actual_batch_size,), device='cuda')
                e = torch.randn_like(x, requires_grad=False)
                a = scheduler.alpha[t].view(actual_batch_size, 1, 1, 1)
                noisy_x = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
                
                output = model(noisy_x, t)
                loss = criterion(output, e)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        
        print(f'Epoch {epoch+1} | Training Loss: {avg_train_loss:.9f}, Validation Loss: {avg_val_loss:.9f}')
        
        lr_scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"Model improved and saved to {save_path}")
        else:
            early_stopping_counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Stopping training.")
            break
    
    return checkpoint

# Start training
checkpoint = train_and_validate(train_loader, val_loader)