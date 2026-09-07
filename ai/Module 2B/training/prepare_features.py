import argparse,sys,logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd
from tqdm import tqdm
logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prosody.data_loader import load_manifest_row
from prosody.feature_extractor import ProsodyFeatureExtractor

def _one(payload):
    row,root=payload
    try:
        y=load_manifest_row(row,root)
        return ProsodyFeatureExtractor().extract(y)
    except Exception as exc:
        _log.warning("Feature extraction failed for %s: %s — using silence fallback", row.get('sample_id','?'), exc)
        return ProsodyFeatureExtractor().extract(np.zeros(16000, dtype=np.float32))
def main(manifest,output_dir='output/features',workers=2,smoke_test=None):
    root=Path(__file__).resolve().parents[2]; out=root/'Module 2B'/output_dir; out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(manifest)
    name={'validation':'val','calibration':'cal'}
    for split in ('train','validation','calibration','test'):
        part=df[df.split==split].copy()
        if smoke_test: part=part.groupby('label',group_keys=False).head(max(1,int(smoke_test)//2))
        records=part.to_dict('records'); payload=[(r,root) for r in records]
        with ProcessPoolExecutor(max_workers=workers) as ex: feats=list(tqdm(ex.map(_one,payload),total=len(payload),desc=split))
        stem=name.get(split,split); np.save(out/f'{stem}_features.npy',np.asarray(feats,dtype=np.float32)); np.save(out/f'{stem}_labels.npy',part.label.to_numpy(dtype=np.int64)); part.to_csv(out/f'{stem}_metadata.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output-dir',default='output/features');p.add_argument('--workers',type=int,default=2);p.add_argument('--smoke-test',type=int);a=p.parse_args();main(**vars(a))
