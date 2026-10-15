"""Create a disjoint TinyStories validation tail, preserving existing 2 MB validation.

NOT a new dataset: this is an untouched, nonoverlapping continuation of the
official TinyStories validation split, useful as a final test slice.
Requires online access on the Mac on first invocation.
"""
import argparse
import hashlib
from pathlib import Path


def prepare(prefix_file, output_file, holdout_bytes, overwrite=False):
    from datasets import load_dataset
    prefix_file = Path(prefix_file).expanduser().resolve()
    output_file = Path(output_file).expanduser().resolve()
    if not prefix_file.exists():
        raise FileNotFoundError('Original TinyStories validation cache is required for byte-identical prefix verification: ' + str(prefix_file))
    prefix = prefix_file.read_bytes()
    if output_file.exists() and not overwrite:
        raise FileExistsError('Holdout already exists: ' + str(output_file))
    if holdout_bytes <= 0:
        raise ValueError('holdout_bytes must be positive')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp = output_file.with_suffix(output_file.suffix+'.partial')
    prefix_digest = hashlib.sha256()
    prefix_remaining = len(prefix)
    remaining = holdout_bytes
    try:
        stream=load_dataset('roneneldan/TinyStories',split='validation',streaming=True)
        with open(temp,'wb') as out:
            for item in stream:
                raw=(item['text']+'\n\n').encode('utf8')
                offset=0
                if prefix_remaining:
                    count=min(len(raw),prefix_remaining)
                    prefix_digest.update(raw[:count])
                    prefix_remaining-=count
                    offset=count
                if offset < len(raw) and remaining:
                    count=min(len(raw)-offset,remaining)
                    out.write(raw[offset:offset+count])
                    remaining-=count
                if not prefix_remaining and not remaining:
                    break
        if prefix_remaining or remaining:
            raise RuntimeError('Validation dataset ended before generating requested holdout bytes')
        expected=hashlib.sha256(prefix).hexdigest()
        actual=prefix_digest.hexdigest()
        if expected!=actual:
            raise RuntimeError('Validation stream prefix does not match existing cache; HOLDOUT REJECTED. '+f'cached={expected} streamed={actual}')
        temp.replace(output_file)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print('Cached prefix verified:',actual)
    print('New disjoint validation holdout:',output_file)
    print('Holdout SHA256:',hashlib.sha256(output_file.read_bytes()).hexdigest())


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--prefix-file',default='../focal_attention_lab/data/tinystories_validation_2000000.bin')
    ap.add_argument('--out',default='data/tinystories_validation_holdout_4000000.bin')
    ap.add_argument('--bytes',type=int,default=4000000)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    prepare(args.prefix_file,args.out,args.bytes,args.overwrite)
