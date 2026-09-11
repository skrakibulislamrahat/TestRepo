#!/usr/bin/env python3
"""Evaluate a trained DenseNet-121 source model on a target manifest/split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import densenet121

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--target-csv", type=Path, required=True)
    p.add_argument("--source-task", required=True)
    p.add_argument("--target-task", required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--subset", default=None, help="Optional split value, e.g. test")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


class TargetDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        image = Image.open(row["image_path"]).convert("RGB")
        return (
            self.transform(image),
            int(row["label"]),
            str(row["image_id"]),
        )


def expected_calibration_error(y, p, bins=15):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def compute_metrics(y, p, threshold=0.5, high_conf=0.90):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    confidence = np.maximum(p, 1.0 - p)
    high = confidence >= high_conf
    high_error = high & (pred != y)
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if set(np.unique(y)) == {0, 1} else None,
        "auprc": float(average_precision_score(y, p)) if set(np.unique(y)) == {0, 1} else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "ece_15": expected_calibration_error(y, p, 15),
        "brier": float(brier_score_loss(y, p)),
        "high_confidence_threshold": float(high_conf),
        "high_confidence_coverage": float(high.mean()),
        "hcer_at_0p90_all": float(high_error.mean()),
        "hcer_at_0p90_covered": float(high_error.sum() / high.sum()) if high.sum() else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    args = parse_args()
    df = pd.read_csv(args.target_csv)
    required = {"image_id", "image_path", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    if args.subset is not None:
        if "split" not in df.columns:
            raise ValueError("--subset requires a split column.")
        df = df[df["split"] == args.subset].copy()
    if df.empty:
        raise ValueError("Target dataframe is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        TargetDataset(df, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    labels, probs, ids = [], [], []
    with torch.no_grad():
        for x, y, image_ids in loader:
            logits = model(x.to(device)).view(-1)
            prob = torch.sigmoid(logits).cpu().numpy()
            labels.extend(y.numpy().astype(int).tolist())
            probs.extend(prob.tolist())
            ids.extend(image_ids)

    metrics = compute_metrics(labels, probs, args.threshold)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "source_task": args.source_task,
        "target_task": args.target_task,
        "image_id": ids,
        "true_label": labels,
        "pred_prob": probs,
        "pred_label": (np.asarray(probs) >= args.threshold).astype(int),
    }).to_csv(str(args.output_prefix) + "_predictions.csv", index=False)
    with open(str(args.output_prefix) + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
