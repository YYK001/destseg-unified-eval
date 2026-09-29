"""Inventory/help need only Python; runtime imports remain lazy."""
import argparse
from pathlib import Path
from .protocol import OFFICIAL_ROOT, WEIGHTS_ROOT, PROTOCOL, selected_categories
from .checkpoints import inventory, require_weights
from external_baselines.patchcore_official_eval.storage import json_write, csv_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('check-weights', 'infer', 'official', 'unified', 'summarize'):
        p = commands.add_parser(name)
        p.add_argument('--categories', nargs='+', default=['all'])
        p.add_argument('--output-dir', type=Path, required=True)
        if name in ('infer', 'official', 'unified'):
            p.add_argument('--device', default='cuda:0')
            p.add_argument('--shard', choices=['0/2', '1/2'])
        if name in ('check-weights', 'infer', 'official'):
            p.add_argument('--official-root', type=Path, default=OFFICIAL_ROOT)
        if name in ('check-weights', 'infer'):
            p.add_argument('--weights-dir', type=Path, default=WEIGHTS_ROOT)
            p.add_argument('--weight-map', type=Path)
        if name == 'check-weights':
            p.add_argument('--load', action='store_true', help='strict-load selected models on CPU; no forward')
        if name in ('infer', 'unified'):
            p.add_argument('--dataset-root', type=Path, required=name == 'infer')
        if name == 'infer':
            p.add_argument('--limit', type=int, default=0, help='positive count = smoke only, never formal evaluation')
            p.add_argument('--compare-official', action='store_true', help='extra forward vs original dataset RGB path')
        if name in ('official', 'unified', 'summarize'):
            p.add_argument('--name', default='v1')
        if name == 'summarize':
            p.add_argument('--kind', choices=['official', 'unified'], required=True)
    args = parser.parse_args()
    if hasattr(args, 'name') and (not args.name or Path(args.name).name != args.name or args.name in ('.', '..')):
        parser.error('--name must be a single directory name')
    if getattr(args, 'limit', 0) < 0:
        parser.error('--limit must be >=0')
    selected = selected_categories(args.categories, getattr(args, 'shard', None))
    if not selected:
        parser.error('empty shard')
    if args.command == 'check-weights':
        from .official import verify_source
        source = verify_source(args.official_root)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        rows = inventory(args.weights_dir, args.weight_map)
        csv_write(args.output_dir/'weights.csv', rows)
        missing = [r['category'] for r in rows if r['status'] == 'missing']
        report = dict(source=source, protocol=PROTOCOL, missing_categories=missing,
            present_count=15-len(missing), strict_load='not_executed', real_inference='not_executed',
            official_evaluation='not_executed', unified_evaluation='not_executed')
        json_write(args.output_dir/'status.json', report)
        print('Present:', 15-len(missing), '/15; missing:', ', '.join(missing) or 'none')
        try:
            require_weights(rows, selected)
        except FileNotFoundError as exc:
            parser.exit(2, str(exc)+'\n')
        if args.load:
            from .official import load_model
            import gc
            try:
                for row in rows:
                    if row['category'] in selected:
                        model = load_model(args.official_root, row['path'], 'cpu')
                        del model
                        gc.collect()
                        row['status'] = 'strict_loaded'
                        csv_write(args.output_dir/'weights.csv', rows)
            except Exception as exc:
                report['strict_load'] = dict(status='failed', category=row['category'], message=str(exc))
                json_write(args.output_dir/'status.json', report)
                raise
            report['strict_load'] = dict(status='passed', categories=selected)
            json_write(args.output_dir/'status.json', report)
    else:
        from . import runner
        function = runner.official_eval if args.command == 'official' else getattr(runner, args.command)
        function(args, selected)


if __name__ == '__main__':
    main()
