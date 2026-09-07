"""Reproducible same-CNN representation comparison on fixed manifests."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import SpectralDataset, read_manifest
from model import SpectralCNN, count_parameters
from metrics import evaluate_scores

def evaluate_representation(manifest: str, spectral_type: str, seed: int = 0) -> dict:
    torch.manual_seed(seed); dataset = SpectralDataset(read_manifest(manifest), spectral_type); bins={"stft":257,"logmel":64,"lfcc":20}[spectral_type]; model=SpectralCNN(bins); labels=[]; logits=[]; started=time.perf_counter()
    with torch.inference_mode():
        for features,target,_ in DataLoader(dataset,batch_size=16):
            _, output=model(features); labels.extend(target.numpy()); logits.extend(output.numpy())
    result=evaluate_scores(labels,logits); result.update({"representation":spectral_type,"parameter_count":count_parameters(model),"inference_latency_seconds":time.perf_counter()-started,"note":"Untrained architecture smoke comparison unless checkpoints are supplied."}); return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",required=True); parser.add_argument("--output-dir",default="output/comparisons"); parser.add_argument("--seed",type=int,default=0); args=parser.parse_args(); results=[evaluate_representation(args.manifest,name,args.seed) for name in ("stft","logmel","lfcc")]; directory=Path(args.output_dir); directory.mkdir(parents=True,exist_ok=True); (directory/"spectral_comparison.json").write_text(json.dumps(results,indent=2)+"\n"); fields=sorted({key for row in results for key in row});
    with (directory/"spectral_comparison.csv").open("w",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(results)
    print(json.dumps(results,indent=2))
if __name__ == "__main__": main()
