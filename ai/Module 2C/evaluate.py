"""Evaluate a saved Module 2B checkpoint and write JSON metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import SpectralDataset, read_manifest
from inference import SpectralSpoofDetector
from metrics import evaluate_scores

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("checkpoint"); parser.add_argument("--manifest", required=True); parser.add_argument("--output", default="output/evaluation.json"); args=parser.parse_args()
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False); spectral_type=payload.get("spectral_type","logmel"); dataset=SpectralDataset(read_manifest(args.manifest),spectral_type); detector=SpectralSpoofDetector(args.checkpoint,spectral_type, payload.get("embedding_dim",64)); labels=[]; logits=[]
    with torch.inference_mode():
        for features,target,_ in DataLoader(dataset,batch_size=32):
            features = features.to(detector.device)
            target = target.to(detector.device, dtype=torch.float32)
            _, output=detector.model(features)
            labels.extend(target.detach().cpu().numpy().reshape(-1).tolist())
            logits.extend(output.detach().cpu().numpy().reshape(-1).tolist())
    result={"checkpoint":str(Path(args.checkpoint).resolve()),"spectral_type":spectral_type,"metrics":evaluate_scores(labels,logits)}; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__ == "__main__": main()
