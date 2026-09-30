"""Small stage commands; each category owns its directory for independent T4 processes."""
import contextlib
import gc
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from external_baselines.patchcore_official_eval.storage import (
    read_json, json_write, csv_write, start_stage, finish_stage)
from external_baselines.patchcore_official_eval.resources import Resources, environment
from .protocol import PROTOCOL
from .official import verify_source, dataset, load_model, modules
from .checkpoints import inventory, require_weights
from .adapter import load_rgb, outputs, save_prediction, metadata


def device_setup(name):
    device = torch.device(name)
    if device.type != 'cuda' or device.index is None or not torch.cuda.is_available():
        raise RuntimeError('Explicit available CUDA device required, e.g. --device cuda:0')
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(f'CUDA device unavailable: {device}')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def env(device):
    result = environment([str(device)])
    for package in ('torchmetrics', 'anomalib', 'imgaug', 'kornia', 'tensorboardX', 'opencv-python-headless'):
        try:
            result['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['versions'][package] = 'not installed'
    return result


@contextlib.contextmanager
def stage(path, identity, device, args, *, resource_class=None):
    start_stage(path, identity)
    try:
        json_write(path/'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        json_write(path/'environment.json', env(device))
        if resource_class is None:
            if identity.get('protocol', {}).get('dataset') == 'visa':
                from .resources import AllocatorResources
                resource_class = AllocatorResources
            else:
                resource_class = Resources
        meter = resource_class(path, [str(device)], identity.get('category'))
        yield meter
    except Exception as exc:
        json_write(path/'failure.json', dict(status='failed', type=type(exc).__name__, message=str(exc)))
        raise


def infer(args, categories):
    from external_baselines.glass_mvtec_pretrained.data import records
    device = device_setup(args.device)
    source = verify_source(args.official_root)
    weights = inventory(args.weights_dir, args.weight_map)
    chosen = require_weights(weights, categories)  # preflight before creating prediction directories.
    root, groups, counts = records(args.dataset_root, categories)
    for category in categories:
        identity = dict(protocol=PROTOCOL, category=category, stage='predict', smoke=bool(args.limit))
        dest = args.output_dir/category/'predict'
        with stage(dest, identity, device, args) as meter:
            json_write(dest/'run.json', dict(protocol=PROTOCOL, source=source, checkpoint=chosen[category],
                dataset_root=str(root), device=str(device), batch_size=1,
                official_evaluation='not_run', unified_evaluation='not_run', smoke=bool(args.limit)))
            csv_write(dest/'weight_inventory.csv', weights)
            csv_write(dest/'data_counts.csv', [r for r in counts if r['category'] == category])
            ds = dataset(root, args.official_root, category)
            items = groups[category]
            by_path = {str(Path(r.path).resolve()): r for r in items}
            official_paths = [str(Path(p).resolve()) for p in ds.mvtec_paths]
            if len(set(official_paths)) != len(official_paths) or set(official_paths) != set(by_path):
                raise ValueError('Official PNG test list differs from unified list; inspect dataset')
            items = [by_path[p] for p in official_paths]
            indices = list(range(len(items)))
            if args.limit:
                # Explicit diagnostic subset only; preserve official order and include both labels when possible.
                good = next((i for i, r in enumerate(items) if not r.label), None)
                bad = next((i for i, r in enumerate(items) if r.label), None)
                first = [i for i in (good, bad) if i is not None]
                indices = sorted((first + [i for i in indices if i not in first])[:args.limit])
            with meter.measure('strict_checkpoint_load'):
                model = load_model(args.official_root, chosen[category]['path'], device)
            json_write(dest/'checkpoint_load.json', dict(**chosen[category],
                                                        load_status='strict_loaded', strict=True, dest=True, ed=True))
            rows, comparisons = [], []
            try:
                with meter.measure('inference_and_FP32_save'), torch.no_grad():
                    for number, index in enumerate(indices):
                        record = items[index]
                        # Only RGB is passed to the frozen model. Labels/masks are read afterwards for audit/cache.
                        x = load_rgb(record.path, ds).unsqueeze(0).to(device)
                        (main, score), (aux, auxscore) = outputs(model(x))
                        reference = ds[index]  # exact original mask rule, independent of model prediction.
                        if args.compare_official:
                            torch.testing.assert_close(x.cpu()[0], reference['img'], rtol=0, atol=0)
                            reference_output = outputs(model(reference['img'][None].to(device)))
                            torch.testing.assert_close(main, reference_output[0][0], rtol=0, atol=0)
                            torch.testing.assert_close(score, reference_output[0][1], rtol=0, atol=0)
                            comparisons.append(dict(relative_path=Path(record.path).relative_to(root).as_posix(),
                                preprocessing_exact=True, main_map_exact=True, top100_exact=True))
                            del reference_output
                        row = metadata(record, root, 'test')
                        row['prediction_file'] = f'maps/{number:06d}.npz'
                        row['official_dataset_index'] = index
                        save_prediction(dest, row, main[0].cpu().numpy(), aux[0].cpu().numpy(),
                            score[0].item(), auxscore[0].item(), reference['mask'][0].numpy())
                        rows.append(row)
                        if (number+1) % 25 == 0 or number+1 == len(indices):
                            print(f'predict {category}: {number+1}/{len(indices)}', flush=True)
                json_write(dest/'samples.json', rows)
                csv_write(dest/'sample_scores.csv', rows)
                csv_write(dest/'label_audit.csv', [dict(category=category, relative_path=r['relative_path'],
                    unified_label=r['label'], official_label=r['official_label'], differs=r['image_label_differs']) for r in rows])
                json_write(dest/'official_path_comparison.json', dict(
                    status='passed' if args.compare_official else 'not_requested', samples=comparisons))
                finish_stage(dest, identity, image_count=len(rows), expected_full_count=len(items),
                             strict_weight_load=True, smoke=bool(args.limit))
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def prediction_rows(root, category, allow_smoke=False, *, protocol=PROTOCOL):
    path = root/category/'predict'
    state = read_json(path/'complete.json')
    identity = state['identity']
    if (state['status'] != 'complete' or identity['protocol'] != protocol
            or identity['category'] != category or identity['stage'] != 'predict'):
        raise ValueError('Incomplete/incompatible predictions')
    if state['smoke'] and not allow_smoke:
        raise ValueError('Smoke predictions cannot be used for formal evaluation/summary')
    rows = read_json(path/'samples.json')
    if not rows or len(rows) != state['image_count'] or len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Incomplete/duplicate prediction rows')
    if not state['smoke'] and len(rows) != state['expected_full_count']:
        raise ValueError('Full prediction count mismatch')
    for row in rows:
        if row['category'] != category:
            raise ValueError('Prediction category mismatch')
        file = (path/row['prediction_file']).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError('Prediction path escaped category directory')
        with np.load(file, allow_pickle=False) as data:
            for key in ('input_map', 'auxiliary_map'):
                if data[key].dtype != np.float32 or data[key].shape != (256,256) or not np.isfinite(data[key]).all():
                    raise ValueError(f'Invalid saved {key}')
            mask = data['official_mask']
            if mask.shape != (256,256) or mask.dtype != np.uint8 or not np.isin(mask, [0,1]).all():
                raise ValueError('Invalid official binary mask cache')
            if int(mask.any()) != row['official_label']:
                raise ValueError('Official mask-derived label mismatch')
            if (str(data['relative_path']) != row['relative_path'] or str(data['category']) != category
                    or float(data['image_score']) != row['image_score']
                    or float(data['auxiliary_image_score']) != row['auxiliary_image_score']):
                raise ValueError('Saved identity/score mismatch')
            for field, key in (('image_score', 'input_map'), ('auxiliary_image_score', 'auxiliary_map')):
                score = torch.sort(torch.from_numpy(data[key]).flatten(), descending=True)[0][:100].mean().item()
                if not np.isclose(score, row[field], rtol=1e-6, atol=1e-7):
                    raise ValueError('Saved score differs from official Top-100 formula')
    return rows, state


def unified(args, categories, *, protocol=PROTOCOL):
    from external_baselines.patchcore_official_eval.evaluation import saved_inputs, calculate
    device = device_setup(args.device)
    for category in categories:
        prediction_rows(args.output_dir, category, protocol=protocol)
        dest = args.output_dir/category/'unified'/args.name
        identity = dict(protocol=protocol, category=category, stage='unified', evaluation_name=args.name)
        with stage(dest, identity, device, args) as meter:
            with meter.measure('saved_inputs_and_unified_GT'):
                rows, maps, masks, audits = saved_inputs(args.output_dir/category, args.dataset_root)
            with meter.measure('unified_metrics'):
                result, fixed = calculate(rows, maps, masks, str(device), allow_cpu=False)
            csv_write(dest/'category_metrics.csv', [dict(category=category, **result)])
            csv_write(dest/'fixed_fpr.csv', [dict(category=category, **r) for r in fixed])
            csv_write(dest/'mask_audit.csv', audits)
            json_write(dest/'result.json', dict(category=category, metrics=result, fixed_fpr=fixed))
            finish_stage(dest, identity, image_count=len(rows), official_metrics=False, cuda_fast_aupro=True)
            del maps, masks
            gc.collect()
            torch.cuda.empty_cache()
        print(f'unified {category} complete', flush=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
        return len(text)
    def flush(self):
        for file in self.files:
            file.flush()


def official_eval(args, categories, *, protocol=PROTOCOL):
    """Call ORIGINAL evaluate(), replacing only its IO with saved original maps/masks."""
    device = device_setup(args.device)
    verify_source(args.official_root)
    _, _, upstream = modules(args.official_root, evaluation=True)
    for category in categories:
        rows, _ = prediction_rows(args.output_dir, category, protocol=protocol)
        dest = args.output_dir/category/'official'/args.name
        identity = dict(protocol=protocol, category=category, stage='official', evaluation_name=args.name,
                        mode='original_eval.evaluate_on_cached_FP32_outputs_and_official_masks')
        cache = args.output_dir/category/'predict'

        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, **unused):
                pass
            def __len__(self):
                return len(rows)
            def __getitem__(self, i):
                with np.load(cache/rows[i]['prediction_file'], allow_pickle=False) as value:
                    mask = torch.from_numpy(value['official_mask'].copy())[None]
                # Forward consumes saved map index; no image, label, or mask enters a model.
                return {'img': torch.tensor(i, dtype=torch.int64), 'mask': mask}

        class Replay(torch.nn.Module):
            def forward(self, indices):
                main, aux = [], []
                for i in indices.cpu().tolist():
                    with np.load(cache/rows[i]['prediction_file'], allow_pickle=False) as value:
                        main.append(torch.from_numpy(value['input_map'].copy())[None])
                        aux.append(torch.from_numpy(value['auxiliary_map'].copy())[None])
                return torch.stack(main).to(device), torch.stack(aux).to(device), []

        scalars = {}
        class Scalars:
            def add_scalar(self, name, value, step):
                scalars[name] = float(value)

        with stage(dest, identity, device, args) as meter:
            original_dataset = upstream.MVTecDataset
            upstream.MVTecDataset = CachedDataset
            try:
                with (dest/'official.log').open('w', encoding='utf-8') as log:
                    with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                        print(f'category={category}; original official evaluate(); cached output replay; T=100; batch=1')
                        with meter.measure('official_metrics'):
                            upstream.evaluate(SimpleNamespace(mvtec_path='', bs=1, num_workers=0, T=100),
                                              category, Replay(), Scalars())
            finally:
                upstream.MVTecDataset = original_dataset
            if len(scalars) != 12:
                raise ValueError('Official evaluator did not produce all 12 main/aux scalar results')
            result_rows = []
            for branch in ('DeSTSeg', 'DeST'):
                result_rows.append(dict(category=category, branch=branch, evaluation='official',
                    **{k: scalars[branch+'_'+k] for k in ('IAP','IAP90','AUPRO','AP','AUC','detect_AUC')}))
            csv_write(dest/'official_results.csv', result_rows)
            json_write(dest/'result.json', dict(scalars=scalars, rows=result_rows))
            finish_stage(dest, identity, image_count=len(rows), unified_metrics=False)
            gc.collect()
            torch.cuda.empty_cache()


def summarize(args, categories, *, protocol=PROTOCOL, dataset_name='mvtec'):
    from external_baselines.patchcore_official_eval.evaluation import summarize as unified_summary
    from DINOv3.MADEqual.guided_validation.summary import mean_available
    dest = args.output_dir/'summaries'/args.name/args.kind
    identity = dict(protocol=protocol, categories=categories, stage='summary', kind=args.kind)
    start_stage(dest, identity)
    metrics, fixed, official_rows = [], [], []
    totals = [0, 0, 0]
    try:
        for category in categories:
            rows, _ = prediction_rows(args.output_dir, category, protocol=protocol)
            totals[0] += len(rows)
            totals[1] += sum(r['label'] == 0 for r in rows)
            totals[2] += sum(r['label'] == 1 for r in rows)
            path = args.output_dir/category/args.kind/args.name
            state = read_json(path/'complete.json')
            if (state['status'] != 'complete' or state['identity']['protocol'] != protocol
                or state['identity']['category'] != category or state['identity']['stage'] != args.kind
                or state['identity']['evaluation_name'] != args.name):
                raise ValueError(f'Incomplete/incompatible {category} {args.kind}')
            result = read_json(path/'result.json')
            if args.kind == 'unified':
                metrics.append(dict(category=category, **result['metrics']))
                fixed.extend(dict(category=category, **r) for r in result['fixed_fpr'])
            else:
                official_rows.extend(result['rows'])
        from external_baselines.patchcore_official_eval.protocol import CATEGORIES as DATASETS
        all_categories = DATASETS[dataset_name]
        expected = {'mvtec': [1725,467,1258], 'btad': [741,451,290], 'visa': [2162,962,1200]}[dataset_name]
        if set(categories) == set(all_categories) and totals != expected:
            raise ValueError(f'Full {dataset_name} split totals differ: {totals}')
        if args.kind == 'unified':
            macro, fmacro = unified_summary(metrics, fixed, categories, dataset_name)
            for filename, values in [('category_metrics', metrics), ('fixed_fpr', fixed),
                                     ('macro_metrics', macro), ('fixed_fpr_macro', fmacro)]:
                csv_write(dest/(filename+'.csv'), values)
        else:
            macro = []
            for branch in ('DeSTSeg', 'DeST'):
                group = [r for r in official_rows if r['branch'] == branch]
                if len(group) != len(categories) or {r['category'] for r in group} != set(categories):
                    raise ValueError('Incomplete/duplicate official category rows')
                macro.append(dict(branch=branch, evaluation='official_category_unweighted_macro',
                    scope='dataset_macro' if set(categories)==set(all_categories) else 'selected_categories_NOT_full_dataset',
                    category_count=len(group), **{k: mean_available([r[k] for r in group])[0]
                    for k in ('IAP','IAP90','AUPRO','AP','AUC','detect_AUC')}))
            csv_write(dest/'official_category_metrics.csv', official_rows)
            csv_write(dest/'official_macro_metrics.csv', macro)
        finish_stage(dest, identity)
    except Exception as exc:
        json_write(dest/'failure.json', dict(status='failed', type=type(exc).__name__, message=str(exc)))
        raise
    print(f'{args.kind} macro saved: {dest}', flush=True)
