"""Evaluate a saved architecture on the held-out test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data import load_split
from src.neural_models import TextCNN, TextLSTM, encode_text


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=("mlp","cnn","lstm","distilbert"),required=True)
    p.add_argument("--model-dir",default=None); p.add_argument("--split",default="data/processed/test.csv"); p.add_argument("--batch-size",type=int,default=None)
    a=p.parse_args(); model_dir=Path(a.model_dir or f"artifacts/{a.model}"); cfg=json.loads((model_dir/"run_config.json").read_text(encoding="utf-8"))
    frame=load_split(a.split,"test")
    if set(frame.id.astype(str)) & (set(cfg.get("train_ids", [])) | set(cfg.get("validation_ids", []))): raise ValueError("Test split overlaps training or validation IDs")
    batch=a.batch_size or int(cfg["batch_size"]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts=frame.text.tolist(); labels=frame.label.to_numpy(dtype=int); probs=[]
    if a.model=="mlp":
        from torch import nn
        vec=joblib.load(model_dir/"vectorizer.joblib"); x=vec.transform(texts)
        model=nn.Sequential(nn.Linear(x.shape[1],256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1))
        model.load_state_dict(torch.load(model_dir/"best_model.pt",map_location=device,weights_only=True)); model.to(device).eval()
        loader=DataLoader(list(range(len(texts))),batch_size=batch)
        inference_start=time.perf_counter()
        with torch.no_grad():
            for indices in loader:
                arr=np.stack([x[int(i)].toarray().ravel() for i in indices]); logits=model(torch.from_numpy(arr).to(device)).cpu().numpy(); probs.extend((1/(1+np.exp(-logits))).tolist())
    elif a.model in ("cnn","lstm"):
        vocab=json.loads((model_dir/"vocabulary.json").read_text(encoding="utf-8")); ids=[encode_text(text,vocab,int(cfg["max_length"])) for text in texts]
        model=(TextCNN if a.model=="cnn" else TextLSTM)(len(vocab)+2)
        model.load_state_dict(torch.load(model_dir/"best_model.pt",map_location=device,weights_only=True)); model.to(device).eval()
        inference_start=time.perf_counter()
        with torch.no_grad():
            for start in range(0,len(ids),batch):
                logits=model(torch.tensor(ids[start:start+batch],device=device)); probs.extend(torch.sigmoid(logits).cpu().tolist())
    else:
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        root=model_dir/"transformer"; tokenizer=AutoTokenizer.from_pretrained(root); model=AutoModelForSequenceClassification.from_pretrained(root).to(device).eval()
        inference_start=time.perf_counter()
        with torch.no_grad():
            for start in range(0,len(texts),batch):
                encoded=tokenizer(texts[start:start+batch],truncation=True,max_length=int(cfg["max_length"]),padding=True,return_tensors="pt"); encoded={k:v.to(device) for k,v in encoded.items()}
                probs.extend(torch.softmax(model(**encoded).logits,dim=1)[:,1].cpu().tolist())
    probs=np.asarray(probs); pred=(probs>=.5).astype(int)
    # This includes data loading and model startup only after setup, and measures model inference over the held-out set.
    if device.type=="cuda": torch.cuda.synchronize()
    inference_ms_per_1000=float((time.perf_counter()-inference_start)*1000/max(len(texts),1)*1000)
    precision,recall,f1,_=precision_recall_fscore_support(labels,pred,average="binary",pos_label=0,zero_division=0)
    errors=[]
    for i in np.flatnonzero(labels!=pred):
        errors.append({"id":str(frame.iloc[i].id),"true_label":"fake" if labels[i]==0 else "real","predicted_label":"fake" if pred[i]==0 else "real","probability_fake":round(float(1-probs[i]),6),"text_excerpt":texts[i][:500]})
    test_ids_sha256=hashlib.sha256("\n".join(sorted(frame.id.astype(str))).encode()).hexdigest()
    results={"model":a.model,"split":str(a.split),"test_ids_sha256":test_ids_sha256,"examples":len(labels),"positive_label":"fake (0)","accuracy":float(accuracy_score(labels,pred)),"precision_fake":float(precision),"recall_fake":float(recall),"f1_fake":float(f1),"confusion_matrix_labels_fake0_real1":confusion_matrix(labels,pred,labels=[0,1]).tolist(),"classification_report":classification_report(labels,pred,labels=[0,1],target_names=["fake","real"],output_dict=True,zero_division=0),"parameter_count":cfg["parameter_count"],"mean_training_seconds_per_epoch":float(np.mean([h["train_seconds"] for h in cfg["history"][:cfg["epochs_run"]]])),"inference_ms_per_1000_articles":inference_ms_per_1000,"misclassified_articles":errors}
    out=model_dir/"test_metrics.json"; out.write_text(json.dumps(results,indent=2),encoding="utf-8"); print(json.dumps(results,indent=2)); print(f"Saved {out}")

if __name__=="__main__": main()

