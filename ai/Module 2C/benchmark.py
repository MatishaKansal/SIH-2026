"""Measure extraction and CNN latency; never prints fabricated measurements."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import torch
from extractor import SpectralExtractor
from input import load_audio,pad_or_trim
from model import SpectralCNN,count_parameters

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("audio"); parser.add_argument("--spectral-type",default="logmel"); parser.add_argument("--checkpoint"); args=parser.parse_args(); audio,_=pad_or_trim(load_audio(args.audio).waveform); extractor=SpectralExtractor(args.spectral_type); features=extractor(torch.from_numpy(audio)).unsqueeze(0); model=SpectralCNN({"stft":257,"logmel":64,"lfcc":20}[args.spectral_type]);
    started=time.perf_counter(); features=extractor(torch.from_numpy(audio)).unsqueeze(0); extraction=time.perf_counter()-started; started=time.perf_counter(); model(features); inference=time.perf_counter()-started
    print({"spectral_extraction_seconds":extraction,"cnn_inference_seconds":inference,"total_seconds":extraction+inference,"parameter_count":count_parameters(model),"checkpoint_size_bytes":Path(args.checkpoint).stat().st_size if args.checkpoint else None,"device":"cpu"})
if __name__ == "__main__": main()
