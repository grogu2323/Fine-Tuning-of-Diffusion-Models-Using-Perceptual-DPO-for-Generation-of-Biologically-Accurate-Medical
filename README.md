# Perceptual-Direct-Preference-Optimization-P-DPO

This directory contains the **complete training, fine-tuning, generation, and evaluation code** used in the *Perceptual-DPO* research paper. The codebase is organized by **dataset** and **training paradigm**, and is designed to support reproducible experiments across **vanilla diffusion**, **RLHF**, and **Direct Preference Optimization (DPO)** settings.

The code supports experiments on two medical imaging domains:

* **Bone Marrow Cytomorphology**
* **Dermatology (HAM10000)**

It also includes standardized **evaluation pipelines** for generative model assessment.

---

## Directory Structure

```
Codes/
├── Bone_Marrow/
│   ├── 1.Vanilla_DDPM_Bone_Marrow_Training_and_Generation/
│   ├── 2.RLHF_Finetuning_Bone_Marrow_Training_and_Generation/
│   └── 3.DPO_Finetuning_Bone_Marrow_Training_and_Generation/
│
├── HAM_10000/
│   ├── 1.Vanilla_DDPM_HAM_10000_Training_and_Generation/
│   ├── 2.RLHF_Finetuning_HAM_10000_Training_and_Generation/
│   └── 3.DPO_Finetuning_HAM_10000_Training_and_Generation/
│
└── Evaluation_Codes/
    ├── FID_EVALUATION/
    └── VIT_MODEL_EVALUATION/
```

---

## 1. Bone Marrow Codes (`Bone_Marrow/`)

This folder contains all code related to **bone marrow cytomorphology image generation and fine-tuning** using diffusion models.

### 1.1 Vanilla DDPM Training and Generation

**Path:**

```
Bone_Marrow/1.Vanilla_DDPM_Bone_Marrow_Training_and_Generation/
```

Includes:

* Training a **baseline DDPM** model on real bone marrow images
* Class-conditional diffusion training (where applicable)
* Image generation scripts for producing synthetic samples

This serves as the **baseline generative model** for all subsequent alignment experiments.

---

### 1.2 RLHF Fine-tuning on DDPM

**Path:**

```
Bone_Marrow/2.RLHF_Finetuning_Bone_Marrow_Training_and_Generation/
```

Includes:

* Reward model–based fine-tuning of a pretrained DDPM
* Incorporation of **expert preference annotations**
* Partial-data and full-data RLHF training setups
* Generation scripts for RLHF-aligned diffusion models

This stage improves **clinical plausibility and perceptual correctness** of generated images.

---

### 1.3 DPO Fine-tuning on DDPM

**Path:**

```
Bone_Marrow/3.DPO_Finetuning_Bone_Marrow_Training_and_Generation/
```

Includes:

* **Direct Preference Optimization (DPO)** fine-tuning of diffusion models
* Preference-based loss functions without explicit reward modeling
* Training configurations using different proportions of annotated data
* Generation scripts for DPO-aligned diffusion outputs

This represents the **core contribution** of the Perceptual-DPO framework.

---

## 2. HAM10000 Codes (`HAM_10000/`)

This folder mirrors the bone marrow setup but is applied to **dermatology images** from the HAM10000 dataset.

### 2.1 Vanilla DDPM Training and Generation

**Path:**

```
HAM_10000/1.Vanilla_DDPM_HAM_10000_Training_and_Generation/
```

Includes:

* Baseline DDPM training on real HAM10000 images
* Class-wise image generation for skin lesion types

---

### 2.2 RLHF Fine-tuning on DDPM

**Path:**

```
HAM_10000/2.RLHF_Finetuning_HAM_10000_Training_and_Generation/
```

Includes:

* RLHF-based diffusion fine-tuning using expert preference annotations
* Generation scripts for RLHF-aligned dermatology images

---

### 2.3 DPO Fine-tuning on DDPM

**Path:**

```
HAM_10000/3.DPO_Finetuning_HAM_10000_Training_and_Generation/
```

Includes:

* Perceptual-DPO fine-tuning of diffusion models
* Preference-optimized generation of dermatology images

---

## 3. Evaluation Codes (`Evaluation_Codes/`)

This folder contains **quantitative evaluation pipelines** used to assess the quality and usefulness of generated images.

---

### 3.1 FID Evaluation

**Path:**

```
Evaluation_Codes/FID_EVALUATION/
```

Includes:

* Scripts to compute **Fréchet Inception Distance (FID)**
* Comparisons between:

  * Real images
  * Vanilla diffusion outputs
  * RLHF-generated images
  * DPO-generated images

FID is used to measure **distributional similarity** between real and synthetic data.

---

### 3.2 ViT Model Training & Evaluation

**Path:**

```
Evaluation_Codes/VIT_MODEL_EVALUATION/
```

Includes:

* Training of a **Vision Transformer (ViT)** classifier using:

  * Real images
  * Annotated synthetic images
* Evaluation on held-out generated datasets
* Extraction of:

  * Classification accuracy
  * Confidence / plausibility scores

This evaluation serves as a **downstream task-based proxy** for generative model quality and clinical relevance.

---

## Experimental Workflow Summary

A typical experimental pipeline follows:

1. Train **vanilla DDPM** on real data
2. Fine-tune the DDPM using:

   * RLHF **or**
   * Direct Preference Optimization (DPO)
3. Generate synthetic images from each model variant
4. Evaluate using:

   * FID scores
   * ViT-based classification performance

---

## Intended Audience

This codebase is intended for:

* Researchers working on **medical image generation**
* Preference learning and **DPO/RLHF** practitioners
* Researchers evaluating **synthetic data fidelity** in medical AI

The code is provided for **research and reproducibility purposes only** and is not intended for clinical deployment.

---

## Notes

* Each subfolder is self-contained and includes dataset-specific configuration files
* Training and generation scripts are designed to be run independently
* Paths and hyperparameters may need adjustment depending on local setup

---

## Citation

If you use this codebase, please cite the corresponding **Perceptual-DPO** paper and the associated dataset.
```bibtex
@inproceedings{prasad2026fine,
  title={Fine-Tuning of Diffusion Models Using Perceptual-DPO for Generation of Biologically Accurate Medical Image},
  author={Prasad, Pratyush and Bhat, Swapnil and Siddiqui, Sadaf and Bhatt, Siddharth and Mallya, Mrinal and Somvanshi, Ankush K and Mehta, Deval and Jadhav, Kshitij},
  booktitle={2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2026},
  organization={IEEE}
}
```
