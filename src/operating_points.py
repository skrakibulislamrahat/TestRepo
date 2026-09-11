#!/usr/bin/env python3
"""Select source-validation thresholds and apply them unchanged to target predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--source-validation",type=Path,required=True,
                   help="CSV with true_label and pred_prob")
    p.add_argument("--target-predictions",type=Path,required=True,
                   help="CSV with true_label and pred_prob")
    p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def at_threshold(y,p,t):
    pred=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else 0.0
    spec=tn/(tn+fp) if tn+fp else 0.0
    return {
        "threshold":float(t),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "sensitivity":float(sens),
        "specificity":float(spec),
        "youden":float(sens+spec-1.0),
        "predicted_positive_rate":float(pred.mean()),
    }


def candidate_thresholds(p):
    vals=np.unique(np.r_[0.0,0.5,1.0,np.asarray(p,dtype=float)])
    return np.clip(vals,0.0,1.0)


def choose_thresholds(y,p):
    rows=pd.DataFrame([at_threshold(y,p,t) for t in candidate_thresholds(p)])
    youden=rows.assign(distance=(rows.threshold-0.5).abs()).sort_values(
        ["youden","balanced_accuracy","distance"],ascending=[False,False,True]
    ).iloc[0]
    best_f1=rows.assign(distance=(rows.threshold-0.5).abs()).sort_values(
        ["f1","youden","distance"],ascending=[False,False,True]
    ).iloc[0]
    sens90=rows[rows.sensitivity>=0.90]
    if len(sens90):
        sens90=sens90.assign(distance=(sens90.threshold-0.5).abs()).sort_values(
            ["specificity","balanced_accuracy","distance"],ascending=[False,False,True]
        ).iloc[0]
        sens90_thr=float(sens90.threshold)
    else:
        sens90_thr=None
    return {
        "fixed_0p5":0.5,
        "max_youden":float(youden.threshold),
        "max_f1":float(best_f1.threshold),
        "sensitivity_ge_0p90_max_specificity":sens90_thr,
    }


def main():
    args=parse_args()
    src=pd.read_csv(args.source_validation)
    tgt=pd.read_csv(args.target_predictions)
    for name,frame in [("source",src),("target",tgt)]:
        missing={"true_label","pred_prob"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing: {sorted(missing)}")

    sy=src.true_label.to_numpy(dtype=int)
    sp=src.pred_prob.to_numpy(dtype=float)
    ty=tgt.true_label.to_numpy(dtype=int)
    tp=tgt.pred_prob.to_numpy(dtype=float)
    thresholds=choose_thresholds(sy,sp)

    rows=[]
    for policy,t in thresholds.items():
        if t is None:
            continue
        source_metrics=at_threshold(sy,sp,t)
        target_metrics=at_threshold(ty,tp,t)
        rows.append({
            "policy":policy,
            "threshold_selected_on_source_validation":t,
            **{f"source_{k}":v for k,v in source_metrics.items() if k!="threshold"},
            **{f"target_{k}":v for k,v in target_metrics.items() if k!="threshold"},
        })

    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    with open(args.output.with_suffix(".json"),"w",encoding="utf-8") as f:
        json.dump({
            "rule":"All thresholds selected from source validation only and applied unchanged to target predictions.",
            "thresholds":thresholds,
        },f,indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":
    main()
