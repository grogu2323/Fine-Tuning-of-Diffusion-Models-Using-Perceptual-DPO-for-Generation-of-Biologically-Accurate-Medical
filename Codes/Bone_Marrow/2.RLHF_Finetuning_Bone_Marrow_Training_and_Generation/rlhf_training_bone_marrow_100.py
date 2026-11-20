#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(0)


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
from collections import defaultdict
import glob
from torch.distributions import Normal
from torchvision import models
import timm

from collections import deque
import gc
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


DATA_PATH = "/home/pratyushprasad/DPO_PROJECT/0. format_dataset/bone_marrow"
CLASSES = ["ART", "BAS", "BLA", "EBO", "EOS", "HAC", "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO"]


# In[4]:


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# In[5]:


def load_images_by_class(class_name, image_type, num_images=100):
    class_folder = os.path.join(DATA_PATH, class_name)
    if not os.path.exists(class_folder):
        print(f"Warning: {class_folder} does not exist!")
        return []

    available_images = []
    for i in range(1, 26):
        folder_path = os.path.join(class_folder, f"image_{i}")
        if not os.path.exists(folder_path):
            continue

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            folder_images = glob.glob(os.path.join(folder_path, ext))
            for img_path in folder_images:
                img_name = os.path.basename(img_path).lower()
                if img_name.startswith(image_type.lower()):
                    available_images.append(img_path)
                    break  

    if len(available_images) == 0:
        print(f"Warning: No {image_type} images found for class {class_name}")
        return []

    if len(available_images) >= num_images:
        selected_files = random.sample(available_images, num_images)
    else:
        selected_files = available_images * (num_images // len(available_images) + 1)
        selected_files = selected_files[:num_images]

    transform = get_transforms()
    images = []
    for img_path in selected_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return images


# In[6]:


def create_full_dataset():
    all_data = {'train': [], 'val': []}

    for class_name in CLASSES:
        print(f"Processing class: {class_name}")

        real_images = load_images_by_class(class_name, 'real', 100)
        plausible_images = load_images_by_class(class_name, 'plausible', 100)
        implausible_images = load_images_by_class(class_name, 'implausible', 100)

        for image_type, images in [('real', real_images), ('plausible', plausible_images), ('implausible', implausible_images)]:
            if images:
                split_idx = int(0.8 * len(images))
                train_images = images[:split_idx]
                val_images = images[split_idx:]

                all_data['train'].extend([(img, class_name, image_type) for img in train_images])
                all_data['val'].extend([(img, class_name, image_type) for img in val_images])

        print(f"  Loaded - Real: {len(real_images)}, Plausible: {len(plausible_images)}, Implausible: {len(implausible_images)}")

    print(f"Total - Train: {len(all_data['train'])}, Val: {len(all_data['val'])}")
    return all_data


# In[7]:


def create_preference_pairs(data):
    train_preferences = []
    val_preferences = []

    for split in ['train', 'val']:
        class_data = {}
        for img, class_name, img_type in data[split]:
            if class_name not in class_data:
                class_data[class_name] = {'real': [], 'plausible': [], 'implausible': []}
            class_data[class_name][img_type].append(img)

        preferences = []
        for class_name, types in class_data.items():
            for _ in range(100):  
                # if types['real'] and types['plausible']:
                    # preferences.append((random.choice(types['real']), random.choice(types['plausible']), 1.0))
                if types['plausible'] and types['implausible']:
                    preferences.append((random.choice(types['plausible']), random.choice(types['implausible']), 1.0))
                if types['real'] and types['implausible']:
                    preferences.append((random.choice(types['real']), random.choice(types['implausible']), 1.0))

        if split == 'train':
            train_preferences = preferences
        else:
            val_preferences = preferences

    return train_preferences, val_preferences


# In[8]:


class PreferenceDataset(Dataset):
    def __init__(self, preferences):
        self.preferences = preferences

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        preferred, non_preferred, label = self.preferences[idx]
        return preferred, non_preferred, label


# In[9]:


class RLDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


# In[10]:


def set_seed(seed: int = 100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# In[11]:


set_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_data = create_full_dataset()
train_pref, val_pref = create_preference_pairs(full_data)

# Create datasets and loaders
train_pref_dataset = PreferenceDataset(train_pref)
val_pref_dataset = PreferenceDataset(val_pref)

train_pref_loader = DataLoader(train_pref_dataset, batch_size=64, shuffle=True)
val_pref_loader = DataLoader(val_pref_dataset, batch_size=64, shuffle=False)


# In[12]:


set_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_data = create_full_dataset()
train_pref, val_pref = create_preference_pairs(full_data)

# Create datasets and loaders
train_pref_dataset = PreferenceDataset(train_pref)
val_pref_dataset = PreferenceDataset(val_pref)

train_pref_loader_1 = DataLoader(train_pref_dataset, batch_size=10, shuffle=True)
val_pref_loader_1 = DataLoader(val_pref_dataset, batch_size=10, shuffle=False)


# In[13]:


CLASSES = ["ART", "BAS", "BLA", "EBO", "EOS", "HAC", "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO"]


# In[14]:


class RewardModel(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        self.backbone.classifier = nn.Identity()

        self.reward_head = nn.Sequential(
            nn.Linear(1280, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        reward = self.reward_head(features)
        return reward


# In[15]:


def train_reward_model(reward_model, train_dataloader, val_dataloader, epochs=10, lr=5e-5):
    print("Training reward model...")

    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        reward_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for preferred, non_preferred, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            preferred = preferred.cuda()
            non_preferred = non_preferred.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            reward_preferred = reward_model(preferred)
            reward_non_preferred = reward_model(non_preferred)

            reward_diff = reward_preferred - reward_non_preferred
            loss = criterion(reward_diff.squeeze(), labels.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = (torch.sigmoid(reward_diff.squeeze()) > 0.5).float()
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)

        reward_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for preferred, non_preferred, labels in val_dataloader:
                preferred = preferred.cuda()
                non_preferred = non_preferred.cuda()
                labels = labels.cuda()

                reward_preferred = reward_model(preferred)
                reward_non_preferred = reward_model(non_preferred)
                reward_diff = reward_preferred - reward_non_preferred

                loss = criterion(reward_diff.squeeze(), labels.float())
                val_loss += loss.item()

                pred = (torch.sigmoid(reward_diff.squeeze()) > 0.5).float()
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_dataloader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_dataloader):.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(reward_model.state_dict(), '/home/pratyushprasad/DPO_PROJECT/3. rlhf/bone_marrow/saved_models/best_reward_model_100.pt')
            print(f"New best validation accuracy: {best_val_acc:.4f}")

        print("-" * 10)

    if best_val_acc > 0:
        reward_model.load_state_dict(torch.load(
            '/home/pratyushprasad/DPO_PROJECT/3. rlhf/bone_marrow/saved_models/best_reward_model_100.pt'
        ))
    return reward_model


# In[16]:


reward_model = RewardModel().to(device)

reward_model = train_reward_model(
    reward_model,
    train_pref_loader,
    val_pref_loader,
    epochs=50,
    lr=5e-6
)


# In[17]:


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
        if embeddings.shape[1] > x.shape[1]:
            embeddings = embeddings[:, :x.shape[1], :, :]
        elif embeddings.shape[1] < x.shape[1]:
            pad_size = x.shape[1] - embeddings.shape[1]
            padding = torch.zeros(embeddings.shape[0], pad_size, embeddings.shape[2], embeddings.shape[3], device=embeddings.device)
            embeddings = torch.cat([embeddings, padding], dim=1)

        x = x + embeddings
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


# In[18]:


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


# In[19]:


class UnetLayer(nn.Module):
    def __init__(self, upscale: bool, attention: bool, num_groups: int, 
                 dropout_prob: float, num_heads: int, C: int):
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


# In[20]:


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int, device):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings.to(device)

    def forward(self, t):
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]


# In[21]:


class UNET(nn.Module):
    def __init__(self, Channels=[64, 128, 256, 512, 512, 384],
                 Attentions=[False, True, False, False, False, True],
                 Upscales=[False, False, False, True, True, True],
                 num_groups=32, dropout_prob=0.0, num_heads=8,
                 input_channels=3, output_channels=3, device='cuda', time_steps=1000):
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
                upscale=Upscales[i], attention=Attentions[i],
                num_groups=num_groups, dropout_prob=dropout_prob,
                C=Channels[i], num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        embeddings = self.embeddings(t)

        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x, embeddings)
            residuals.append(r)

        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x, _ = layer(x, embeddings)
            residual_idx = self.num_layers - i - 1
            if residual_idx < len(residuals):
                x = torch.cat((x, residuals[residual_idx]), dim=1)

        x = self.relu(self.late_conv(x))
        return self.output_conv(x)


# In[22]:


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps=1000, device='cuda'):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False, device=device)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]

    def to(self, device):
        super().to(device)
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        return self


# In[23]:


def safe_log_prob(pred, target, epsilon=1e-8):
    mse_loss = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    mse_loss = torch.clamp(mse_loss, min=epsilon, max=100.0)
    log_prob = -mse_loss
    return log_prob

def check_tensors_for_nan(*tensors, names=None):
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    for tensor, name in zip(tensors, names):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}: {tensor}")
            return True
        if torch.isinf(tensor).any():
            print(f"Inf detected in {name}: {tensor}")
            return True
        if tensor.std() > 100:
            print(f"Extremely large values in {name}: mean={tensor.mean():.6f}, std={tensor.std():.6f}")
    return False

def ppo_loss_safe(current_log_probs, old_log_probs, advantages, epsilon=0.2):
    if check_tensors_for_nan(current_log_probs, old_log_probs, advantages, names=['current_log_probs', 'old_log_probs', 'advantages']):
        print("Input tensors contain NaN/Inf values!")
        return torch.tensor(0.0, device=current_log_probs.device, requires_grad=True)

    log_ratio = torch.clamp(current_log_probs - old_log_probs, min=-20, max=20)
    ratio = torch.exp(log_ratio)

    if check_tensors_for_nan(ratio, names=['ratio']):
        print("Ratio contains NaN/Inf!")
        return torch.tensor(0.0, device=current_log_probs.device, requires_grad=True)

    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    if check_tensors_for_nan(surrogate1, surrogate2, names=['surrogate1', 'surrogate2']):
        print("Surrogate objectives contain NaN/Inf!")
        return torch.tensor(0.0, device=current_log_probs.device, requires_grad=True)

    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        print(f"Final policy loss is NaN/Inf: {policy_loss}")
        return torch.tensor(0.0, device=current_log_probs.device, requires_grad=True)

    return policy_loss


# In[24]:


def rlhf_finetuning(model=None,
                    reward_model=None,
                    scheduler=None,
                    train_dataloader=None,
                    val_dataloader=None,
                    checkpoint_path="/home/pratyushprasad/DPO_PROJECT/2. vanilla_ddpm_model/bone_marrow/saved_model/ddpm_bone_marrow_first_model.pt",
                    save_path='/home/pratyushprasad/DPO_PROJECT/3. rlhf/bone_marrow/saved_models/ddpm_bone_marrow_rlhf_model_100.pt',
                    lr=1e-6,
                    epochs=20,
                    num_time_steps=1000,
                    ema_decay=0.9999,
                    ppo_epochs=2,
                    batch_size=2,
                    kl_penalty=0.01,
                    device='cuda',
                    accumulation_steps=1,
                    save_every=5,
                    early_stopping_patience=5,
                    use_amp=False,
                    max_grad_norm=0.1):

    if model is None or reward_model is None or scheduler is None:
        raise ValueError("Model, reward_model, and scheduler must be provided")

    device = torch.device(device)
    model = model.to(device)

    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("Reference model created and frozen")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        print("Model and reference model checkpoints loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, 0.0

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6, eps=1e-8)
    ema = ModelEmaV3(model, decay=ema_decay)

    if 'ema' in checkpoint:
        try:
            ema.load_state_dict(checkpoint['ema'])
        except:
            print("Could not load EMA state, initializing fresh")

    scheduler.alpha = scheduler.alpha.to(device)
    scheduler.beta = scheduler.beta.to(device)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    best_preference_accuracy = 0.0
    patience_counter = 0
    train_losses = deque(maxlen=100)

    print(f"Starting RLHF training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        reference_model.eval()  
        epoch_metrics = {'policy_losses': [], 'preference_accuracies': [], 'kl_divergences': []}
        optimizer.zero_grad()
        accumulated_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) >= 2:
                preferred_imgs, non_preferred_imgs = batch_data[0], batch_data[1]
            else:
                continue

            preferred_imgs = preferred_imgs.to(device, non_blocking=True)
            non_preferred_imgs = non_preferred_imgs.to(device, non_blocking=True)
            current_batch_size = preferred_imgs.size(0)

            if torch.isnan(preferred_imgs).any() or torch.isnan(non_preferred_imgs).any():
                print(f"Skipping batch {batch_idx} due to NaN in input images")
                continue

            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    t = torch.randint(50, min(500, num_time_steps), (current_batch_size,), device=device)
                    noise_pref = torch.randn_like(preferred_imgs) * 0.1
                    noise_non_pref = torch.randn_like(non_preferred_imgs) * 0.1

                    alpha_t = scheduler.alpha[t].view(-1, 1, 1, 1)
                    sqrt_alpha = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
                    sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-8))

                    noisy_pref = sqrt_alpha * preferred_imgs + sqrt_one_minus_alpha * noise_pref
                    noisy_non_pref = sqrt_alpha * non_preferred_imgs + sqrt_one_minus_alpha * noise_non_pref

                    with torch.no_grad():
                        old_pred_pref = reference_model(noisy_pref, t)
                        old_pred_non_pref = reference_model(noisy_non_pref, t)

                        old_log_prob_pref = safe_log_prob(old_pred_pref, noise_pref)
                        old_log_prob_non_pref = safe_log_prob(old_pred_non_pref, noise_non_pref)

                        if torch.isnan(old_log_prob_pref).any() or torch.isnan(old_log_prob_non_pref).any():
                            print(f"NaN in old log probs at batch {batch_idx}, skipping")
                            continue

                    with torch.no_grad():
                        try:
                            generated_pref_batch = generate_samples_batch_safe(model, scheduler, current_batch_size, num_time_steps=300, device=device)
                            generated_non_pref_batch = generate_samples_batch_safe(model, scheduler, current_batch_size, num_time_steps=300, device=device)

                            if generated_pref_batch is None or generated_non_pref_batch is None:
                                print(f"Sample generation failed at batch {batch_idx}, skipping")
                                continue

                            rewards_pref = reward_model(generated_pref_batch).squeeze()
                            rewards_non_pref = reward_model(generated_non_pref_batch).squeeze()

                            if rewards_pref.dim() == 0:
                                rewards_pref = rewards_pref.unsqueeze(0)
                            if rewards_non_pref.dim() == 0:
                                rewards_non_pref = rewards_non_pref.unsqueeze(0)

                            if torch.isnan(rewards_pref).any() or torch.isnan(rewards_non_pref).any():
                                print(f"NaN in rewards at batch {batch_idx}, skipping")
                                continue

                            preference_correct = (rewards_pref > rewards_non_pref).float().mean()
                            epoch_metrics['preference_accuracies'].append(preference_correct.item())

                        except Exception as e:
                            print(f"Error in sample generation/reward computation: {e}")
                            continue

                    reward_diff = torch.clamp(rewards_pref - rewards_non_pref, min=-10, max=10)

                    reward_diff_mean = reward_diff.mean()
                    reward_diff_std = torch.clamp(reward_diff.std(), min=1e-4)
                    normalized_reward_diff = (reward_diff - reward_diff_mean) / reward_diff_std

                    advantages_pref = torch.clamp(normalized_reward_diff, min=-3, max=3)
                    advantages_non_pref = torch.clamp(-normalized_reward_diff, min=-3, max=3)

                    total_ppo_loss = 0.0
                    valid_ppo_updates = 0

                    for ppo_epoch in range(ppo_epochs):
                        current_pred_pref = model(noisy_pref, t)
                        current_pred_non_pref = model(noisy_non_pref, t)

                        current_log_prob_pref = safe_log_prob(current_pred_pref, noise_pref)
                        current_log_prob_non_pref = safe_log_prob(current_pred_non_pref, noise_non_pref)

                        if (torch.isnan(current_log_prob_pref).any() or 
                            torch.isnan(current_log_prob_non_pref).any()):
                            print(f"NaN in current log probs, skipping PPO update {ppo_epoch}")
                            continue

                        # print(f"PPO Epoch {ppo_epoch}: Log prob diff (pref): {(current_log_prob_pref - old_log_prob_pref).mean().item():.6f}")
                        # print(f"PPO Epoch {ppo_epoch}: Log prob diff (non_pref): {(current_log_prob_non_pref - old_log_prob_non_pref).mean().item():.6f}")

                        policy_loss_pref = ppo_loss_safe(current_log_prob_pref, old_log_prob_pref, advantages_pref, epsilon=0.2)
                        policy_loss_non_pref = ppo_loss_safe(current_log_prob_non_pref, old_log_prob_non_pref, advantages_non_pref, epsilon=0.2)

                        policy_loss = (policy_loss_pref + policy_loss_non_pref) / 2
                        # print(f"PPO Epoch {ppo_epoch}: Policy losses - Pref: {policy_loss_pref.item():.6f}, Non-pref: {policy_loss_non_pref.item():.6f}, Combined: {policy_loss.item():.6f}")

                        log_ratio_pref = torch.clamp(current_log_prob_pref - old_log_prob_pref, min=-10, max=10)
                        log_ratio_non_pref = torch.clamp(current_log_prob_non_pref - old_log_prob_non_pref, min=-10, max=10)
                        kl_div = 0.5 * (log_ratio_pref.mean() + log_ratio_non_pref.mean())

                        total_loss = policy_loss + kl_penalty * torch.clamp(kl_div, min=-1, max=1)

                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            print(f"NaN/Inf in total loss at PPO epoch {ppo_epoch}, skipping")
                            continue

                        scaled_loss = total_loss / accumulation_steps

                        if torch.isnan(scaled_loss):
                            print(f"NaN in scaled loss at PPO epoch {ppo_epoch}, skipping backward pass")
                            continue

                        if use_amp:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

                        grad_nan = False
                        total_grad_norm = 0.0
                        for param in model.parameters():
                            if param.grad is not None:
                                param_norm = param.grad.data.norm(2)
                                total_grad_norm += param_norm.item() ** 2
                                if torch.isnan(param.grad).any():
                                    grad_nan = True
                                    break
                        total_grad_norm = total_grad_norm ** (1. / 2)

                        # print(f"PPO Epoch {ppo_epoch}: Total gradient norm: {total_grad_norm:.6f}")

                        if grad_nan:
                            print(f"NaN in gradients at PPO epoch {ppo_epoch}, skipping optimizer step")
                            optimizer.zero_grad()
                            continue

                        if total_grad_norm < 1e-8:
                            print(f"Extremely small gradients ({total_grad_norm:.2e}), potential convergence issue")

                        if use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()
                            print(f"PPO Epoch {ppo_epoch}: Clipped gradient norm: {grad_norm:.6f}")

                        optimizer.zero_grad()
                        ema.update(model)

                        total_ppo_loss += total_loss
                        valid_ppo_updates += 1

                        accumulated_loss += scaled_loss.item()
                        epoch_metrics['kl_divergences'].append(kl_div.item())

                    if valid_ppo_updates > 0:
                        total_ppo_loss = total_ppo_loss / valid_ppo_updates
                        epoch_metrics['policy_losses'].append(total_ppo_loss.item())

                        pbar.set_postfix({'Loss': f"{accumulated_loss:.4f}", 'Pref_Acc': f"{preference_correct.item():.3f}", 'Valid_PPO': valid_ppo_updates})
                        accumulated_loss = 0.0
                    else:
                        print(f"No valid PPO updates for batch {batch_idx}")
                        continue

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    continue
                else:
                    print(f"Runtime error at batch {batch_idx}: {e}")
                    continue

        if (epoch + 1) % 2== 0:  
            print(f"Updating reference model at epoch {epoch+1}")
            reference_model.load_state_dict(model.state_dict())

        try:
            avg_val_pref_acc = validate_model_safe(model, reward_model, scheduler, val_dataloader, device, use_amp)
        except Exception as e:
            print(f"Validation failed: {e}")
            avg_val_pref_acc = 0.0

        avg_train_pref_acc = np.mean(epoch_metrics['preference_accuracies']) if epoch_metrics['preference_accuracies'] else 0.0
        avg_policy_loss = np.mean(epoch_metrics['policy_losses']) if epoch_metrics['policy_losses'] else float('inf')
        avg_kl_div = np.mean(epoch_metrics['kl_divergences']) if epoch_metrics['kl_divergences'] else 0.0

        train_losses.append(avg_policy_loss)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Preference Accuracy: {avg_train_pref_acc:.4f}")
        print(f"Val Preference Accuracy: {avg_val_pref_acc:.4f}")
        print(f"Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"Avg KL Divergence: {avg_kl_div:.4f}")

        if avg_val_pref_acc > best_preference_accuracy and not np.isnan(avg_policy_loss):
            best_preference_accuracy = avg_val_pref_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'weights': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'preference_accuracy': avg_val_pref_acc,
                'best_preference_accuracy': best_preference_accuracy,
                'lr': lr,
                'kl_penalty': kl_penalty
            }
            torch.save(checkpoint, save_path)
            print(f"Model saved! New best preference accuracy: {best_preference_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break

        if (epoch + 1) % save_every == 0:
            periodic_save_path = save_path.replace('.pt', f'_epoch_{epoch+1}.pt')
            try:
                torch.save(checkpoint, periodic_save_path)
            except:
                print(f"Failed to save periodic checkpoint at epoch {epoch+1}")

        torch.cuda.empty_cache()
        gc.collect()

    print("Training completed!")
    return model, best_preference_accuracy

def generate_samples_batch_safe(model, scheduler, batch_size, num_time_steps=100, device='cuda'):
    """Safe sample generation with NaN checking."""
    model.eval()
    try:
        with torch.no_grad():
            # FIX 7: Generate different noise for each sample in batch
            samples = torch.randn(batch_size, 3, 128, 128, device=device) * 0.5

            # Reverse diffusion process with fewer steps
            step_size = max(1, num_time_steps // 50)  # Reduce number of denoising steps
            time_steps = list(range(0, num_time_steps, step_size))

            for t in reversed(time_steps):
                if torch.isnan(samples).any():
                    print(f"NaN detected in samples at timestep {t}")
                    return None

                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

                # Model prediction with error handling
                try:
                    predicted_noise = model(samples, t_tensor)
                    if torch.isnan(predicted_noise).any():
                        print(f"NaN in predicted noise at timestep {t}")
                        return None
                except Exception as e:
                    print(f"Model prediction failed at timestep {t}: {e}")
                    return None

                # Denoise step with clamping
                alpha_t = torch.clamp(scheduler.alpha[t], min=1e-8, max=0.9999)
                beta_t = torch.clamp(scheduler.beta[t], min=1e-8, max=0.9999)

                if t > 0:
                    # FIX 8: Different noise for each sample
                    z = torch.randn_like(samples) * 0.1  # Each sample gets different noise
                else:
                    z = torch.zeros_like(samples)

                # DDPM sampling step with numerical stability
                sqrt_alpha = torch.sqrt(alpha_t)
                coeff = beta_t / torch.sqrt(torch.clamp(1 - alpha_t, min=1e-8))

                samples = (1 / sqrt_alpha) * (samples - coeff * predicted_noise) + torch.sqrt(beta_t) * z

                # Clamp samples to reasonable range
                samples = torch.clamp(samples, min=-5, max=5)

            # Final check
            if torch.isnan(samples).any():
                print("NaN in final samples")
                return None

            model.train()
            return samples

    except Exception as e:
        print(f"Sample generation failed: {e}")
        model.train()
        return None

def validate_model_safe(model, reward_model, scheduler, val_dataloader, device, use_amp=True):
    """Safe validation with error handling."""
    model.eval()
    val_preference_accuracies = []

    try:
        with torch.no_grad():
            for batch_data in val_dataloader:
                if len(batch_data) < 2:
                    continue

                val_pref, val_non_pref = batch_data[0], batch_data[1]
                val_pref = val_pref.to(device, non_blocking=True)
                val_non_pref = val_non_pref.to(device, non_blocking=True)

                # Skip batch if contains NaN
                if torch.isnan(val_pref).any() or torch.isnan(val_non_pref).any():
                    continue

                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # Generate samples
                        val_samples_pref = generate_samples_batch_safe(
                            model, scheduler, val_pref.size(0), num_time_steps=100, device=device
                        )
                        val_samples_non_pref = generate_samples_batch_safe(
                            model, scheduler, val_non_pref.size(0), num_time_steps=100, device=device
                        )

                        if val_samples_pref is None or val_samples_non_pref is None:
                            continue

                        # Get rewards
                        val_rewards_pref = reward_model(val_samples_pref).squeeze()
                        val_rewards_non_pref = reward_model(val_samples_non_pref).squeeze()

                        # Handle scalar rewards
                        if val_rewards_pref.dim() == 0:
                            val_rewards_pref = val_rewards_pref.unsqueeze(0)
                        if val_rewards_non_pref.dim() == 0:
                            val_rewards_non_pref = val_rewards_non_pref.unsqueeze(0)

                        # Check for NaN in rewards
                        if torch.isnan(val_rewards_pref).any() or torch.isnan(val_rewards_non_pref).any():
                            continue

                        # Preference accuracy
                        val_pref_correct = (val_rewards_pref > val_rewards_non_pref).float().mean()
                        if not torch.isnan(val_pref_correct):
                            val_preference_accuracies.append(val_pref_correct.item())

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        continue

        return np.mean(val_preference_accuracies) if val_preference_accuracies else 0.0

    except Exception as e:
        print(f"Validation error: {e}")
        return 0.0
    finally:
        model.train()


# In[ ]:


rlhf_finetuning(model=UNET(device=device).to(device),
                reward_model=reward_model,
                scheduler=DDPM_Scheduler(num_time_steps=1000).to(device),
                train_dataloader=train_pref_loader_1,
                val_dataloader=val_pref_loader_1,
                checkpoint_path="/home/pratyushprasad/DPO_PROJECT/2. vanilla_ddpm_model/bone_marrow/saved_model/ddpm_bone_marrow_first_model.pt",
                save_path='/home/pratyushprasad/DPO_PROJECT/3. rlhf/bone_marrow/saved_models/ddpm_bone_marrow_rlhf_model_100.pt',
                lr=1e-5, 
                epochs=10,
                num_time_steps=1000,
                ema_decay=0.9999,
                ppo_epochs=4,
                batch_size=8,
                kl_penalty=0.1,
                max_grad_norm=1.5,
                use_amp = True)