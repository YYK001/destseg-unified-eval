"""Inventory only unless strict model loading is explicitly requested."""
import json
from pathlib import Path
from .protocol import CATEGORIES, WEIGHTS_URL


def inventory(root, mapping_file=None):
    root = Path(root).resolve()
    mapping = json.loads(Path(mapping_file).read_text(encoding='utf-8')) if mapping_file else {}
    if mapping_file and (not isinstance(mapping, dict) or set(mapping) != set(CATEGORIES)):
        raise ValueError('Explicit mapping must contain exactly all 15 categories')
    result = []
    for category in CATEGORIES:
        expected = 'DeSTSeg_MVTec_5000_' + category + '.pckl'
        if mapping_file:
            candidate = Path(mapping[category])
            paths = [candidate if candidate.is_absolute() else root / candidate]
        else:
            paths = sorted(root.rglob(expected)) if root.exists() else []
        if len(paths) > 1:
            raise ValueError(f'Ambiguous checkpoint for {category}: {paths}; supply --weight-map')
        path = paths[0].resolve() if paths else root / expected
        exists = path.is_file() and path.stat().st_size > 0
        result.append(dict(category=category, path=str(path), expected_filename=expected,
                           status='present_not_loaded' if exists else 'missing',
                           bytes=path.stat().st_size if exists else 0, source=WEIGHTS_URL,
                           training_selection_history='not inferred from filename'))
    existing = [r['path'] for r in result if r['status'] != 'missing']
    if len(existing) != len(set(existing)):
        raise ValueError('The same checkpoint was mapped to multiple categories')
    return result


def require_weights(rows, selected):
    chosen = {r['category']: r for r in rows if r['category'] in selected}
    missing = [c for c in selected if c not in chosen or chosen[c]['status'] == 'missing']
    if missing:
        raise FileNotFoundError('Missing checkpoints: ' + ', '.join(missing))
    return chosen
