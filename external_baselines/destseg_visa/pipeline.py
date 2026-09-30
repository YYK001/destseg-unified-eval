"""Explicit local/Kaggle launcher: independent category processes, then evaluate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
from .protocol import select
from .processes import run_process

MOD = 'external_baselines.destseg_visa'


def can_start_category(now, deadline, estimate):
    return deadline is None or now + estimate < deadline


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
    p.add_argument('--max-hours', type=float, default=None,
                   help='Wall-clock budget from launcher start, including 5 minutes reserved for archive')
    p.add_argument('--category-estimate-hours', type=float, default=2.75,
                   help='Conservative initial duration for one complete category; updated from observations')
    p.add_argument('--idle-timeout', type=int, default=900,
                   help='Abort train/infer after this many seconds without child output')
    p.add_argument('--evaluation-idle-timeout', type=int, default=3600,
                   help='Metrics may be silent while computing; separate longer timeout')
    args = p.parse_args()
    categories = select(args.categories)
    if args.phase in ['train','all'] and args.dtd_root is None:
        p.error('--dtd-root is required for training')
    if len(set(args.devices)) != 2:
        p.error('Use two distinct devices')
    if min(args.idle_timeout,args.evaluation_idle_timeout) <= 0:
        p.error('Timeouts must be positive')
    if args.max_hours is not None and args.max_hours <= 5/60:
        p.error('--max-hours must exceed 5 minutes')
    if args.category_estimate_hours <= 0:
        p.error('--category-estimate-hours must be positive')
    logs = Path(str(args.output_dir) + '_logs')
    logs.mkdir(parents=True, exist_ok=True)
    cancel = threading.Event()
    deadline = None if args.max_hours is None else time.monotonic()+args.max_hours*3600-300
    completed = []
    lock = threading.Lock()

    def run(stage, extra, tag):
        command = [sys.executable, '-u', '-m', MOD, stage, *map(str, extra)]
        timeout = args.evaluation_idle_timeout if stage in ['official','unified','summarize'] else args.idle_timeout
        run_process(command, logs/(tag+'.log'), tag, cancel, idle_timeout=timeout,deadline=deadline)

    def execute_categories():
        def worker(index):
            group = categories[index::2]
            if not group:
                return
            estimate = args.category_estimate_hours*3600
            for category in group:
                if cancel.is_set():
                    return
                if not can_start_category(time.monotonic(),deadline,estimate):
                    print(f'[GPU{index}] Insufficient time for another complete category; pending: {category} and later categories',flush=True)
                    return
                started = time.monotonic()
                stages = (['train'] if args.phase=='train' else
                          ['infer','unified','official'] if args.phase=='evaluate' else
                          ['train','infer','unified','official'])
                for name in stages:
                    extra = ['--categories', category, '--device', args.devices[index],
                             '--output-dir', args.training_dir if name == 'train' else args.output_dir]
                    if name in ['train','infer','unified']:
                        extra += ['--dataset-root', args.dataset_root]
                    if name == 'train':
                        extra += ['--dtd-root', args.dtd_root, '--batch-size', args.batch_size,
                                  '--workers', args.workers, '--seed', args.seed]
                    if name == 'infer':
                        extra += ['--training-dir', args.training_dir]
                    run(name, extra, f'{name}_{category}_gpu{index}')
                duration = time.monotonic()-started
                estimate = max(estimate,duration*1.15)
                with lock:
                    completed.append(category)
                    (logs/'progress.json').write_text(json.dumps(dict(
                        requested=categories,completed=completed,
                        pending=[c for c in categories if c not in completed]),indent=2),encoding='utf-8')
                print(f'[GPU{index}] {category}: requested stages complete in {duration/3600:.2f} hours',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(worker, i) for i in range(2)]
            for job in jobs:
                job.result()

    status = 'failed'
    try:
        execute_categories()
        finished = [c for c in categories if c in completed]
        if args.phase != 'train' and finished:
            for kind in ['unified','official']:
                run('summarize', ['--output-dir', args.output_dir, '--kind', kind,
                                 '--categories', *finished], kind+'_summary')
        status = ('training_complete' if args.phase == 'train' else 'complete') if len(finished)==len(categories) else 'partial_budget'
    except Exception as exc:
        (logs/'pipeline_failure.json').write_text(json.dumps(
            dict(status='failed',type=type(exc).__name__,message=str(exc)),indent=2),encoding='utf-8')
        raise
    finally:
        (logs/'pipeline_status.json').write_text(json.dumps(
            dict(status=status,categories=categories,completed=completed,
                 pending=[c for c in categories if c not in completed]),indent=2),encoding='utf-8')
        suffix = '_report.zip' if status == 'complete' else '_partial_records.zip'
        archive = Path(str(args.output_dir)+suffix)
        with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
            for root in [args.training_dir, args.output_dir, logs]:
                for path in sorted(root.rglob('*')):
                    if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.csv','.log','.txt'}:
                        z.write(path, Path(root.name)/path.relative_to(root))
        print(f'Pipeline status={status}; records: {archive}', flush=True)
        # A separate small set of final weights is useful even when a later class fails.
        weights = [args.training_dir/c/'train'/f'DeSTSeg_VISA_5000_{c}.pckl' for c in categories]
        weights = [p for p in weights if p.is_file() and p.stat().st_size > 0]
        if weights:
            recovery = Path(str(args.output_dir)+'_recovery.zip')
            with ZipFile(recovery, 'w', ZIP_STORED) as z:
                for path in weights:
                    z.write(path,Path(args.training_dir.name)/path.relative_to(args.training_dir))
                z.write(archive,archive.name)
            print(f'Final weights + records for recovery (strict validation required): {recovery}',flush=True)


if __name__ == '__main__':
    main()
