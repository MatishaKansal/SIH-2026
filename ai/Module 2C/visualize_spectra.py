"""Save diagnostic spectrogram PNGs; training consumes tensors, never images."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from extractor import SpectralExtractor
from input import load_audio

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("audio"); parser.add_argument("--output",default="output/visualizations"); parser.add_argument("--spectral-type",default="logmel",choices=["stft","logmel","lfcc"]); args=parser.parse_args(); features=SpectralExtractor(args.spectral_type)(load_audio(args.audio).waveform); path=Path(args.output); path.mkdir(parents=True,exist_ok=True); output=path/(Path(args.audio).stem+f"_{args.spectral_type}.png"); plt.imsave(output,features.detach().numpy(),origin="lower",cmap="magma"); print(output)
if __name__ == "__main__": main()
