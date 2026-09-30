"""Fresh VisA run: quick exit-path checks, then budgeted train/infer/evaluate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from .processes import run_process


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--dtd-root',type=Path,required=True)
    p.add_argument('--run-base',type=Path,default=Path('/kaggle/working'))
    p.add_argument('--max-hours',type=float,default=10.75)
    args=p.parse_args()
    if args.max_hours <= .25:
        p.error('Budget must exceed 15 minutes')
    start=time.monotonic()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    train=args.run_base/f'destseg_visa_train_fresh_{stamp}'
    out=args.run_base/f'destseg_visa_eval_fresh_{stamp}'
    smoke=args.run_base/f'destseg_visa_smoke_fresh_{stamp}'
    logs=Path(str(out)+'_logs'); logs.mkdir(parents=True,exist_ok=False)
    config=dict(training_dir=str(train),output_dir=str(out),smoke_dir=str(smoke),
                dataset_root=str(args.dataset_root),max_hours=args.max_hours,fresh_initialization=True)
    (logs/'launch.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)
    cancel=threading.Event()
    def run(tag,arguments,timeout=900):
        run_process([sys.executable,'-u',*map(str,arguments)],logs/(tag+'.log'),tag,cancel,
                    idle_timeout=timeout,deadline=start+args.max_hours*3600)
    try:
        run('gpu_check',['-c',
            "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==2; "
            "print('torch:',torch.__version__); "
            "print([(i,torch.cuda.get_device_name(i)) for i in range(2)])"])
        run('data_check',['-m','external_baselines.destseg_visa.check_data','--dataset-root',args.dataset_root])
        for name in ['destseg_visa','destseg_btad','destseg_mvtec_pretrained']:
            run(name+'_tests',['-m','unittest','discover','-s',f'external_baselines/{name}/tests','-v'])
        # Exercises both phases, end-of-category cleanup, and starting the next class.
        run('two_category_smoke',['-m','external_baselines.destseg_visa','train',
            '--dataset-root',args.dataset_root,'--dtd-root',args.dtd_root,'--output-dir',smoke,
            '--categories','candle','cashew','--device','cuda:0','--workers','2','--batch-size','32',
            '--seed','42','--smoke-steps','20'])
        for c in ['candle','cashew']:
            state=json.loads((smoke/c/'train/complete.json').read_text())
            if state['status']!='complete' or state['step']!=40 or not state['smoke']:
                raise RuntimeError(f'{c}: invalid smoke completion')
        remaining=args.max_hours-(time.monotonic()-start)/3600
        if remaining <= .25:
            raise RuntimeError('No remaining formal-run budget')
        run('pipeline',['-m','external_baselines.destseg_visa.pipeline','--phase','all',
            '--dataset-root',args.dataset_root,'--dtd-root',args.dtd_root,
            '--training-dir',train,'--output-dir',out,'--categories','all',
            '--devices','cuda:0','cuda:1','--batch-size','32','--workers','2','--seed','42',
            '--max-hours',remaining,'--category-estimate-hours','2.75'],timeout=1200)
        print((logs/'pipeline_status.json').read_text(),flush=True)
    except Exception as exc:
        (logs/'launch_failure.json').write_text(json.dumps(dict(status='failed',
            type=type(exc).__name__,message=str(exc)),indent=2),encoding='utf-8')
        print(f'RUN FAILED: {exc}. Do not treat as completed. Inspect {logs}',flush=True)
        # End Notebook normally so the platform can publish whatever output exists.
        # Domain success is represented by pipeline_status.json, not this CLI exit code.
    print('Saved artifacts:',flush=True)
    for path in sorted(args.run_base.glob(out.name+'*.zip')):
        print(path,flush=True)


if __name__=='__main__':
    main()
