"""Explicit local/Kaggle launcher: independent category processes, then evaluate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from .protocol import select

MOD = 'external_baselines.destseg_visa'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['train','evaluate','all'], required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--dtd-root', type=Path)
    p.add_argument('--training-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--categories', nargs='+', default=['all'])
    p.add_argument('--devices', nargs=2, default=['cuda:0','cuda:1'])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    categories = select(args.categories)
    if args.phase in ['train','all'] and args.dtd_root is None:
        p.error('--dtd-root is required for training')
    if len(set(args.devices)) != 2:
        p.error('Use two distinct devices')
    logs = Path(str(args.output_dir) + '_logs')
    logs.mkdir(parents=True, exist_ok=True)

    def run(stage, extra, tag):
        command = [sys.executable, '-u', '-m', MOD, stage, *map(str, extra)]
        with (logs / (tag+'.log')).open('a', encoding='utf-8') as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); log.flush()
                    print(f'[{tag}] {line}', end='', flush=True)
                if process.wait():
                    raise RuntimeError(f'{tag} failed; inspect {logs}')

    def stage(name):
        def worker(index):
            group = categories[index::2]
            if not group:
                return
            extra = ['--categories', *group, '--device', args.devices[index],
                     '--output-dir', args.training_dir if name == 'train' else args.output_dir]
            if name in ['train','infer','unified']:
                extra += ['--dataset-root', args.dataset_root]
            if name == 'train':
                extra += ['--dtd-root', args.dtd_root, '--batch-size', args.batch_size,
                          '--workers', args.workers, '--seed', args.seed]
            if name == 'infer':
                extra += ['--training-dir', args.training_dir]
            run(name, extra, f'{name}_gpu{index}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(worker, i) for i in range(2)]
            for job in jobs:
                job.result()

    if args.phase in ['train','all']:
        stage('train')
    if args.phase == 'train':
        print('Training complete. Run --phase evaluate for predictions and metrics.', flush=True)
        return
    stage('infer')
    for kind in ['unified','official']:
        stage(kind)
        run('summarize', ['--output-dir', args.output_dir, '--kind', kind,
                         '--categories', *categories], kind+'_summary')
    archive = Path(str(args.output_dir)+'_report.zip')
    roots = [args.training_dir, args.output_dir, logs]
    # Includes metrics, source/configuration, manifests, status, resources and logs.
    with ZipFile(archive, 'x', ZIP_DEFLATED) as z:
        for root in roots:
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.csv','.log','.txt'}:
                    z.write(path, Path(root.name)/path.relative_to(root))
    print(f'All requested categories evaluated; report (no weights/maps): {archive}', flush=True)


if __name__ == '__main__':
    main()
