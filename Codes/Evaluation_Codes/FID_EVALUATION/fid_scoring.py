#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(0)


# In[2]:


import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


def process_dataset(root_dir):
    class_fid = {}
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        # Collect FID scores for individual comparisons to calculate std
        plausible_scores = []
        implausible_scores = []
        for img_folder in os.listdir(class_path):
            img_folder_path = os.path.join(class_path, img_folder)
            if not os.path.isdir(img_folder_path):
                continue
            real_imgs = [f for f in os.listdir(img_folder_path) if f.startswith("image")]
            plausible_imgs = [f for f in os.listdir(img_folder_path) if f.startswith("plausible")]
            implausible_imgs = [f for f in os.listdir(img_folder_path) if f.startswith("implausible")]
            if not real_imgs:
                continue
            # For each folder, calculate FID between real images and plausible/implausible images
            real_img_paths = [os.path.join(img_folder_path, img) for img in real_imgs]
            
            # Calculate FID for each plausible image against all real images in this folder
            for p_img in plausible_imgs:
                plausible_img_paths = [os.path.join(img_folder_path, p_img)]
                
                # Need at least 2 samples for FID, so duplicate if necessary
                if len(real_img_paths) == 1:
                    real_batch = real_img_paths * 2
                else:
                    real_batch = real_img_paths
                
                if len(plausible_img_paths) == 1:
                    plausible_batch = plausible_img_paths * 2
                else:
                    plausible_batch = plausible_img_paths
                fid_metric = FrechetInceptionDistance(feature=64).to(device)
                calculate_fid_batch(real_batch, plausible_batch, fid_metric)
                plausible_scores.append(fid_metric.compute().item())
            # Calculate FID for each implausible image against all real images in this folder
            for imp_img in implausible_imgs:
                implausible_img_paths = [os.path.join(img_folder_path, imp_img)]
                
                # Need at least 2 samples for FID, so duplicate if necessary
                if len(real_img_paths) == 1:
                    real_batch = real_img_paths * 2
                else:
                    real_batch = real_img_paths
                
                if len(implausible_img_paths) == 1:
                    implausible_batch = implausible_img_paths * 2
                else:
                    implausible_batch = implausible_img_paths
                fid_metric = FrechetInceptionDistance(feature=64).to(device)
                calculate_fid_batch(real_batch, implausible_batch, fid_metric)
                implausible_scores.append(fid_metric.compute().item())
        # Calculate statistics (removed median)
        avg_plausible = np.mean(plausible_scores) if plausible_scores else None
        std_plausible = np.std(plausible_scores, ddof=1) if len(plausible_scores) > 1 else None
        
        avg_implausible = np.mean(implausible_scores) if implausible_scores else None
        std_implausible = np.std(implausible_scores, ddof=1) if len(implausible_scores) > 1 else None
        
        class_fid[class_name] = {
            "Average FID (plausible)": avg_plausible,
            "Std FID (plausible)": std_plausible,
            "Average FID (implausible)": avg_implausible,
            "Std FID (implausible)": std_implausible
        }
    df = pd.DataFrame.from_dict(class_fid, orient="index").reset_index()
    df.columns = ["Class", "Average FID (plausible)", "Std FID (plausible)",
                  "Average FID (implausible)", "Std FID (implausible)"]
    return df

def calculate_fid_batch(real_img_paths, gen_img_paths, fid_metric):
    """Calculate FID using batches of images"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor()
    ])
    # Process real images
    for img_path in real_img_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0) * 255
        tensor = tensor.to(torch.uint8).to(device)
        fid_metric.update(tensor, real=True)
    # Process generated images
    for img_path in gen_img_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0) * 255
        tensor = tensor.to(torch.uint8).to(device)
        fid_metric.update(tensor, real=False)

def process_dataset_with_overall_stats(root_dir):
    """Process dataset and return both per-class and overall statistics"""
    # Get per-class statistics
    class_df = process_dataset(root_dir)
    
    # Calculate overall dataset statistics from raw scores
    all_plausible_scores = []
    all_implausible_scores = []
    
    # Recalculate all FID scores to get raw scores for overall statistics
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_folder in os.listdir(class_path):
            img_folder_path = os.path.join(class_path, img_folder)
            if not os.path.isdir(img_folder_path):
                continue
            real_imgs = [f for f in os.listdir(img_folder_path) if f.startswith("image")]
            plausible_imgs = [f for f in os.listdir(img_folder_path) if f.startswith("plausible")]
            implausible_imgs = [f for f in os.listdir(img_folder_path) if f.startswith("implausible")]
            if not real_imgs:
                continue
            real_img_paths = [os.path.join(img_folder_path, img) for img in real_imgs]
            
            # Calculate FID for each plausible image
            for p_img in plausible_imgs:
                plausible_img_paths = [os.path.join(img_folder_path, p_img)]
                
                if len(real_img_paths) == 1:
                    real_batch = real_img_paths * 2
                else:
                    real_batch = real_img_paths
                
                if len(plausible_img_paths) == 1:
                    plausible_batch = plausible_img_paths * 2
                else:
                    plausible_batch = plausible_img_paths
                fid_metric = FrechetInceptionDistance(feature=64).to(device)
                calculate_fid_batch(real_batch, plausible_batch, fid_metric)
                all_plausible_scores.append(fid_metric.compute().item())
                
            # Calculate FID for each implausible image
            for imp_img in implausible_imgs:
                implausible_img_paths = [os.path.join(img_folder_path, imp_img)]
                
                if len(real_img_paths) == 1:
                    real_batch = real_img_paths * 2
                else:
                    real_batch = real_img_paths
                
                if len(implausible_img_paths) == 1:
                    implausible_batch = implausible_img_paths * 2
                else:
                    implausible_batch = implausible_img_paths
                fid_metric = FrechetInceptionDistance(feature=64).to(device)
                calculate_fid_batch(real_batch, implausible_batch, fid_metric)
                all_implausible_scores.append(fid_metric.compute().item())
    
    # Create overall statistics row using all raw scores
    overall_stats = {
        "OVERALL DATASET": {
            "Average FID (plausible)": np.mean(all_plausible_scores) if all_plausible_scores else None,
            "Std FID (plausible)": np.std(all_plausible_scores, ddof=1) if len(all_plausible_scores) > 1 else None,
            "Average FID (implausible)": np.mean(all_implausible_scores) if all_implausible_scores else None,
            "Std FID (implausible)": np.std(all_implausible_scores, ddof=1) if len(all_implausible_scores) > 1 else None
        }
    }
    
    overall_df = pd.DataFrame.from_dict(overall_stats, orient="index").reset_index()
    overall_df.columns = ["Class", "Average FID (plausible)", "Std FID (plausible)",
                         "Average FID (implausible)", "Std FID (implausible)"]
    
    combined_df = pd.concat([class_df, overall_df], ignore_index=True)
    return combined_df


# In[ ]:


dataset1 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/bone_dpo_2"
dataset2 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/bone_dpo_4"
dataset3 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/bone_rlhf_2"
dataset4 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/bone_rlhf_4"
dataset5 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/skin_dpo_2"
dataset6 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/skin_dpo_4"
dataset7 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/skin_rlhf_2"
dataset8 = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/paired_data/skin_rlhf_4"

df1_with_overall = process_dataset_with_overall_stats(dataset1)
df2_with_overall = process_dataset_with_overall_stats(dataset2)
df3_with_overall = process_dataset_with_overall_stats(dataset3)
df4_with_overall = process_dataset_with_overall_stats(dataset4)
df5_with_overall = process_dataset_with_overall_stats(dataset5)
df6_with_overall = process_dataset_with_overall_stats(dataset6)
df7_with_overall = process_dataset_with_overall_stats(dataset7)
df8_with_overall = process_dataset_with_overall_stats(dataset8)


# In[ ]:


import os
output_dir = "/home/pratyushprasad/DPO_PROJECT/8.evaluation/fid_scores_final/results"
os.makedirs(output_dir, exist_ok=True)

dfs = {
    "fid_bone_dpo_2.csv": df1_with_overall,
    "fid_bone_dpo_4.csv": df2_with_overall,
    "fid_bone_rlhf_2.csv": df3_with_overall,
    "fid_bone_rlhf_4.csv": df4_with_overall,
    "fid_skin_dpo_2.csv": df5_with_overall,
    "fid_skin_dpo_4.csv": df6_with_overall,
    "fid_skin_rlhf_2.csv": df7_with_overall,
    "fid_skin_rlhf_4.csv": df8_with_overall,
}

for filename, df in dfs.items():
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved: {save_path}")

print("\nAll DataFrames have been saved successfully.")


# In[ ]:


df1_with_overall


# In[ ]:


df2_with_overall


# In[ ]:


df3_with_overall


# In[ ]:


df4_with_overall


# In[ ]:


df5_with_overall


# In[ ]:


df6_with_overall


# In[ ]:


df7_with_overall


# In[ ]:


df8_with_overall


# In[ ]:




