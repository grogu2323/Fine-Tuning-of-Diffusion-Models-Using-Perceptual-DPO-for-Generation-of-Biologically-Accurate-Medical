#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print(0)


# In[ ]:


import os
import time
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# Try to import timm; if not available raise clear error
try:
    import timm
except Exception as e:
    raise ImportError("timm is required. Install with: pip install timm") from e


# In[ ]:


CONFIG = {
    "dataset_root": "/home/pratyushprasad/DPO_PROJECT/12.classification_task/train_dataset",
    "save_models_root": "/home/pratyushprasad/DPO_PROJECT/12.classification_task/saved_models",
    "save_results_root": "/home/pratyushprasad/DPO_PROJECT/12.classification_task/saved_results",
    "datasets": ["bone", "skin"],
    "n_folds": 5,
    "seed": 42,
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 20,
    "base_lr": 5e-5,           # peak LR for OneCycle
    "weight_decay": 1e-2,
    "accumulation_steps": 1,   # set >1 if you want larger effective batch
    "max_grad_norm": 1.0,
    "model_name": "vit_tiny_patch16_224",  # timm model name; fallback later
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose": True
}


# In[ ]:


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# In[ ]:


class BinaryImageFolder(Dataset):
    def __init__(self, file_paths, labels, img_size=224, is_train=True):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.img_size = img_size
        self.is_train = is_train

        # Basic transforms; can be extended
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        p = self.file_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            # robust handling: return a black image if file unreadable
            img = Image.new("RGB", (self.img_size, self.img_size), (0,0,0))
        img = self.transform(img)
        return img, int(label), str(p)  # include path for debugging


# In[ ]:


def create_model(num_classes=2, model_name="vit_tiny_patch16_224", pretrained=True):
    # Create model via timm and replace head
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        # fallback to a small vit if the exact tiny isn't available
        model = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model


# In[ ]:


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion, accumulation_steps, max_grad_norm, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train {epoch+1}/{total_epochs}", leave=False)
    optimizer.zero_grad()
    for step, (imgs, labels, _) in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            # gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * accumulation_steps

        preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{running_loss/(step+1):.4f}")

    avg_loss = running_loss / len(loader)
    return avg_loss, np.array(all_targets), np.array(all_preds)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
    avg_loss = running_loss / len(loader)
    return avg_loss, np.array(all_targets), np.array(all_preds)


# In[ ]:


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "confusion_matrix": cm}


# In[ ]:


def run_class_cv(dataset_root, dataset_name, class_name, cfg):
    root = Path(dataset_root)
    class_root = root / dataset_name / class_name
    plausible_dir = class_root / "plausible"
    implausible_dir = class_root / "implausible"

    if not plausible_dir.exists() and not implausible_dir.exists():
        print(f"Skipping {dataset_name}/{class_name}: no plausible/implausible folders.")
        return None

    # Gather file paths and labels
    files = []
    labels = []
    # plausible -> label 1
    if plausible_dir.exists():
        for p in plausible_dir.iterdir():
            if p.is_file():
                files.append(str(p))
                labels.append(1)
    if implausible_dir.exists():
        for p in implausible_dir.iterdir():
            if p.is_file():
                files.append(str(p))
                labels.append(0)

    if len(files) < cfg["n_folds"]:
        print(f"Not enough examples for {dataset_name}/{class_name} to run {cfg['n_folds']}-fold CV. Skipping.")
        return None

    files = np.array(files)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])

    # result accumulators
    per_fold_results = []
    fold_idx = 0

    for train_idx, test_idx in skf.split(files, labels):
        fold_idx += 1
        print(f"\n=== {dataset_name}/{class_name} — Fold {fold_idx}/{cfg['n_folds']} ===")
        train_files, test_files = files[train_idx], files[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        # Datasets
        train_ds = BinaryImageFolder(train_files, train_labels, img_size=cfg["img_size"], is_train=True)
        val_ds = BinaryImageFolder(test_files, test_labels, img_size=cfg["img_size"], is_train=False)

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=cfg["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                num_workers=cfg["num_workers"], pin_memory=True)

        # Model
        model = create_model(num_classes=2, model_name=cfg["model_name"], pretrained=True)
        model.to(cfg["device"])

        # Optimizer / scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["base_lr"], weight_decay=cfg["weight_decay"])

        # OneCycleLR — requires total_steps or epochs*iters
        total_steps = cfg["epochs"] * len(train_loader) // max(1, cfg["accumulation_steps"])
        if total_steps <= 0:
            total_steps = cfg["epochs"] * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["base_lr"],
                                                        total_steps=total_steps, pct_start=0.1)

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model_path = None
        history = {"train_loss": [], "val_loss": [], "metrics": []}

        for epoch in range(cfg["epochs"]):
            train_loss, train_targets, train_probs = train_one_epoch(
                model, train_loader, optimizer, scheduler, cfg["device"], criterion,
                cfg["accumulation_steps"], cfg["max_grad_norm"], epoch, cfg["epochs"]
            )

            val_loss, val_targets, val_probs = evaluate(model, val_loader, cfg["device"], criterion)

            train_metrics = compute_metrics(train_targets, train_probs)
            val_metrics = compute_metrics(val_targets, val_probs)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["metrics"].append({"epoch": epoch+1, "train": train_metrics, "val": val_metrics})

            print(f"Epoch {epoch+1}/{cfg['epochs']} — train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_f1: {val_metrics['f1']:.4f}")

            # Checkpoint on best val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model
                timestamp = int(time.time())
                model_dir = Path(cfg["save_models_root"])
                model_dir.mkdir(parents=True, exist_ok=True)
                model_filename = f"{dataset_name}_{class_name}_fold{fold_idx}.pth"
                candidate = model_dir / model_filename
                if candidate.exists():
                    candidate = model_dir / f"{dataset_name}_{class_name}_fold{fold_idx}_{timestamp}.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch+1,
                    "cfg": cfg
                }, str(candidate))
                best_model_path = str(candidate)
                print(f"Saved best model -> {best_model_path}")

        # After fold training, evaluate final best model on val set (reload for safety)
        if best_model_path is not None:
            ckpt = torch.load(best_model_path, map_location=cfg["device"])
            model.load_state_dict(ckpt["model_state_dict"])
            val_loss, val_targets, val_probs = evaluate(model, val_loader, cfg["device"], criterion)
            val_metrics = compute_metrics(val_targets, val_probs)
        else:
            # No best model found (shouldn't happen) — evaluate current model
            val_loss, val_targets, val_probs = evaluate(model, val_loader, cfg["device"], criterion)
            val_metrics = compute_metrics(val_targets, val_probs)

        # Save fold-level metrics
        fold_result = {
            "dataset": dataset_name,
            "class": class_name,
            "fold": fold_idx,
            "n_train": int(len(train_files)),
            "n_val": int(len(test_files)),
            "val_loss": float(val_loss),
            "metrics": val_metrics,
            "model_path": best_model_path
        }
        per_fold_results.append(fold_result)

        # Save fold history JSON
        results_dir = Path(cfg["save_results_root"])
        results_dir.mkdir(parents=True, exist_ok=True)
        hist_file = results_dir / f"{dataset_name}_{class_name}_fold{fold_idx}_history.json"
        with open(hist_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved fold history -> {hist_file}")

    # Aggregate fold metrics
    if len(per_fold_results) == 0:
        return None

    metrics_df_records = []
    for fr in per_fold_results:
        m = fr["metrics"]
        record = {
            "dataset": fr["dataset"],
            "class": fr["class"],
            "fold": fr["fold"],
            "n_train": fr["n_train"],
            "n_val": fr["n_val"],
            "val_loss": fr["val_loss"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
            "model_path": fr["model_path"]
        }
        metrics_df_records.append(record)

    metrics_df = pd.DataFrame(metrics_df_records)
    # compute aggregate stats
    agg = metrics_df[["accuracy", "precision", "recall", "f1", "auc"]].agg(["mean", "std"]).T
    agg = agg.rename(columns={"mean": "mean", "std": "std"})
    agg = agg.reset_index().rename(columns={"index": "metric"})

    # Save per-fold and aggregate
    results_dir = Path(cfg["save_results_root"])
    results_dir.mkdir(parents=True, exist_ok=True)
    per_fold_csv = results_dir / f"{dataset_name}_{class_name}_per_fold.csv"
    metrics_df.to_csv(per_fold_csv, index=False)

    agg_csv = results_dir / f"{dataset_name}_{class_name}_aggregate.csv"
    agg.to_csv(agg_csv, index=False)

    # Also save a consolidated JSON
    summary = {
        "dataset": dataset_name,
        "class": class_name,
        "n_examples": int(len(files)),
        "per_fold": per_fold_results,
        "aggregate": agg.to_dict(orient="records")
    }
    summary_file = results_dir / f"{dataset_name}_{class_name}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-fold CSV -> {per_fold_csv}")
    print(f"Saved aggregate CSV -> {agg_csv}")
    print(f"Saved summary JSON -> {summary_file}")

    return {
        "per_fold_csv": str(per_fold_csv),
        "aggregate_csv": str(agg_csv),
        "summary_json": str(summary_file)
    }


# In[ ]:


def main(cfg):
    set_seed(cfg["seed"])
    dataset_root = Path(cfg["dataset_root"])
    save_models_root = Path(cfg["save_models_root"])
    save_results_root = Path(cfg["save_results_root"])
    save_models_root.mkdir(parents=True, exist_ok=True)
    save_results_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in cfg["datasets"]:
        ds_path = dataset_root / dataset_name
        if not ds_path.exists():
            print(f"Dataset folder missing: {ds_path}. Skipping.")
            continue

        class_dirs = [d for d in ds_path.iterdir() if d.is_dir()]
        if len(class_dirs) == 0:
            print(f"No class folders found in {ds_path}. Skipping.")
            continue

        for class_dir in class_dirs:
            class_name = class_dir.name
            try:
                print(f"\n--- Starting training for {dataset_name}/{class_name} ---")
                res = run_class_cv(cfg["dataset_root"], dataset_name, class_name, cfg)
            except Exception as e:
                print(f"Error while processing {dataset_name}/{class_name}: {e}")
                continue


# In[ ]:


if __name__ == "__main__":
    main(CONFIG)

