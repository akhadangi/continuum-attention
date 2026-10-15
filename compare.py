"""Read previous and current experiment JSON reports, show comparable metrics."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reports',nargs='+',help='One or more JSON files from benchmark.py')
    args=parser.parse_args()
    for filename in args.reports:
        report=json.loads(Path(filename).read_text(encoding='utf8'))
        print(f'\n{filename} | {report.get("backend")} | {report.get("device")}')
        hashes=report.get('dataset_sha256') or {}
        if hashes:
            print('train SHA256',hashes.get('train_sha256',''))
            print('val SHA256  ',hashes.get('validation_sha256',''))
        print(f'{"VARIANT":>8} {"PARAMS":>11} {"NLL":>10} {"BPB":>9} '
              f'{"TRAIN tok/s":>14} {"FWD tok/s":>13} {"FUSED":>7} {"MASS-TV":>9}')
        for x in report.get('results',[]):
            att=x.get('attention') or {}
            speed=x.get('speed') or {}
            nll=x.get('validation_nll_nats')
            bpb=x.get('validation_bits_per_byte')
            train=x.get('training_tokens_per_second')
            throughput=speed.get('forward_tokens_per_second')
            mass=att.get('first_layer_mean_abs_mass_difference_vs_softmax')
            print(f'{x.get("variant", "?"):>8} {x.get("parameters", 0):>11,} '
                  f'{nll if nll is not None else "-":>10} '
                  f'{bpb if bpb is not None else "-":>9} '
                  f'{train if train is not None else "-":>14} '
                  f'{throughput if throughput is not None else "-":>13} '
                  f'{str(x.get("fused_sdpa_used",False)):>7} '
                  f'{mass if mass is not None else "-":>9}')


if __name__=='__main__':
    main()
