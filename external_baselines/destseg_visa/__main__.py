import argparse
from pathlib import Path
from .protocol import OFFICIAL_ROOT, protocol, schedule, select


def main():
    parser=argparse.ArgumentParser(description='VisA 1cls normal-only official DeSTSeg training and separate evaluations')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['train','infer','official','unified','summarize']:
        p=commands.add_parser(name)
        p.add_argument('--categories',nargs='+',default=['all'])
        p.add_argument('--output-dir',type=Path,required=True)
        if name!='summarize':
            p.add_argument('--device',default='cuda:0')
        if name in ['train','infer','official']:
            p.add_argument('--official-root',type=Path,default=OFFICIAL_ROOT)
        if name in ['train','infer','unified']:
            p.add_argument('--dataset-root',type=Path,required=name!='unified')
        if name=='train':
            p.add_argument('--dtd-root',type=Path,required=True,help='DTD images/ containing47 texture folders')
            p.add_argument('--batch-size',type=int,default=32)
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--workers',type=int,default=2)
            p.add_argument('--smoke-steps',type=int,default=0,help='positive = this many steps in EACH phase; never formal')
        if name=='infer':
            p.add_argument('--training-dir',type=Path,required=True)
            p.add_argument('--limit',type=int,default=0)
        if name in ['official','unified','summarize']:
            p.add_argument('--name',default='v1')
        if name=='summarize':
            p.add_argument('--kind',choices=['official','unified'],required=True)
    args=parser.parse_args()
    selected=select(args.categories)
    if args.command=='train':
        protocol(args.batch_size,args.seed)
        schedule(args.smoke_steps)
        if args.workers<0 or not 0<=args.seed<2**32:
            parser.error('workers>=0 and seed in [0,2**32) required')
    if getattr(args,'limit',0)<0:
        parser.error('limit must be >=0')
    if hasattr(args,'name') and (args.name in ('','.','..') or '/' in args.name or '\\' in args.name):
        parser.error('name must be a single folder name')
    from . import runner
    if args.command in ['train','infer']:
        getattr(runner,args.command)(args,selected)
    else:
        runner.evaluate_or_summarize(args,selected)


if __name__=='__main__':
    main()
