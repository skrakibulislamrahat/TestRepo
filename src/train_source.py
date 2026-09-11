#!/usr/bin/env python3
"""Train a DenseNet-121 source classifier for semantic-transport experiments.

Input CSV requirements:
    image_id, image_path, label, split, split_group_id

The split file must already be leakage-safe. This script trains one source task
across a configurable list of seeds and writes validation predictions that can
later be used to select operating points without looking at target labels.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--source-task", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025, 7, 99])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if set(np.unique(y_true)) == {0, 1}:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auprc"] = float(average_precision_score(y_true, prob))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "image_id": str(row["image_id"]),
            "split_group_id": str(row["split_group_id"]),
        }


def build_model() -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def make_transforms(size: int):
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


@torch.no_grad()
def predict(model, loader, device, source_task: str):
    model.eval()
    ids, groups, labels, logits = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        z = model(x).view(-1)
        ids.extend(batch["image_id"])
        groups.extend(batch["split_group_id"])
        labels.extend(batch["label"].numpy().astype(int).tolist())
        logits.extend(z.cpu().numpy().tolist())
    logits = np.asarray(logits, dtype=float)
    prob = 1.0 / (1.0 + np.exp(-logits))
    frame = pd.DataFrame({
        "source_task": source_task,
        "target_task": source_task,
        "image_id": ids,
        "split_group_id": groups,
        "true_label": labels,
        "logit": logits,
        "pred_prob": prob,
        "pred_label_0p5": (prob >= 0.5).astype(int),
    })
    return frame, metrics(labels, prob)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    required = {"image_id", "image_path", "label", "split", "split_group_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    missing_files = ~df["image_path"].map(lambda p: Path(str(p)).exists())
    if missing_files.any():
        raise FileNotFoundError(
            f"{int(missing_files.sum())} image files in the split CSV do not exist."
        )

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Both train and val subsets must be present.")

    train_tf, eval_tf = make_transforms(args.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in args.seeds:
        set_seed(seed)
        seed_dir = args.output_dir / args.source_task / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            ChestXrayDataset(train_df, train_tf),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        val_loader = DataLoader(
            ChestXrayDataset(val_df, eval_tf),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model = build_model().to(device)
        n_neg = int((train_df["label"] == 0).sum())
        n_pos = int((train_df["label"] == 1).sum())
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32, device=device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        best_auc = -np.inf
        best_state = None
        no_improve = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for batch in train_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                z = model(x).view(-1)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(y)
                seen += len(y)

            val_pred, val_metrics = predict(
                model, val_loader, device, args.source_task
            )
            val_auc = val_metrics["auroc"]
            history.append({
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            score = -np.inf if val_auc is None else val_auc
            if score > best_auc:
                best_auc = score
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError("Training completed without a valid checkpoint.")

        model.load_state_dict(best_state)
        model.to(device)
        val_pred, val_metrics = predict(
            model, val_loader, device, args.source_task
        )

        torch.save(best_state, seed_dir / "best_checkpoint.pt")
        pd.DataFrame(history).to_csv(seed_dir / "training_log.csv", index=False)
        val_pred.to_csv(seed_dir / "val_predictions.csv", index=False)
        with open(seed_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

        config = {
            "source_task": args.source_task,
            "seed": seed,
            "architecture": "densenet121",
            "imagenet_pretrained": True,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
        }
        with open(seed_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(
            f"{args.source_task} seed={seed}: "
            f"val_AUROC={val_metrics['auroc']:.4f} "
            f"balanced_acc={val_metrics['balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
