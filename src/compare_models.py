"""Collect per-model held-out metrics and create comparison figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("mlp", "cnn", "lstm", "distilbert")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifact-root",default="artifacts"); p.add_argument("--output-dir",default="results"); a=p.parse_args()
    root,out=Path(a.artifact_root),Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; configs={}; metrics={}; test_fingerprints=set()
    for name in MODELS:
        folder=root/name
        config=json.loads((folder/"run_config.json").read_text(encoding="utf-8")); result=json.loads((folder/"test_metrics.json").read_text(encoding="utf-8"))
        if config["model"] != name or result["model"] != name: raise ValueError(f"Model metadata mismatch for {name}")
        if result.get("positive_label") != "fake (0)": raise ValueError(f"{name} uses a different positive-label convention")
        test_fingerprints.add(result["test_ids_sha256"])
        configs[name],metrics[name]=config,result
        rows.append({"model":name,"accuracy":result["accuracy"],"precision_fake":result["precision_fake"],"recall_fake":result["recall_fake"],"f1_fake":result["f1_fake"],"parameters":result["parameter_count"],"mean_train_seconds_per_epoch":result["mean_training_seconds_per_epoch"],"inference_ms_per_1000_articles":result["inference_ms_per_1000_articles"],"test_examples":result["examples"]})
    if len(test_fingerprints) != 1: raise ValueError("Models were not evaluated on the exact same test articles")
    fields=list(rows[0])
    with (out/"model_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    # Save a compact machine-readable copy for reproducible plots and report tables.
    (out/"model_comparison.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc: raise SystemExit("Install matplotlib to generate the comparison figures") from exc
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name,cfg in configs.items():
        history=cfg["history"]; axes[0].plot([h["epoch"] for h in history],[h["train_loss"] for h in history],label=f"{name} train")
        axes[0].plot([h["epoch"] for h in history],[h["validation_loss"] for h in history],linestyle="--",label=f"{name} validation")
    axes[0].set(title="Training and validation loss",xlabel="Epoch",ylabel="Loss"); axes[0].legend(fontsize=8,ncol=2); axes[0].grid(alpha=.2)
    scores=[metrics[name]["f1_fake"] for name in MODELS]; axes[1].bar(MODELS,scores,color="#457b9d"); axes[1].set(title="Held-out fake-news F1",ylabel="F1",ylim=(0,1)); axes[1].grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(out/"training_and_f1.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for ax,name in zip(axes.flat,MODELS):
        cm=np.asarray(metrics[name]["confusion_matrix_labels_fake0_real1"]); ax.imshow(cm,cmap="Blues")
        ax.set(title=name,xlabel="Predicted",ylabel="True",xticks=[0,1],yticks=[0,1],xticklabels=["fake","real"],yticklabels=["fake","real"])
        threshold=cm.max()/2 if cm.size else 0
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>threshold else "black")
    fig.tight_layout(); fig.savefig(out/"confusion_matrices.png",dpi=180); plt.close(fig)
    print(f"Wrote {out/'model_comparison.csv'}, {out/'training_and_f1.png'}, and {out/'confusion_matrices.png'}")


if __name__=="__main__": main()

