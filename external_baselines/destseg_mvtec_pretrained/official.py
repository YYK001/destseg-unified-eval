"""Import the pinned original modules without modifying upstream source."""
import importlib
import subprocess
import sys
from pathlib import Path
from .protocol import SOURCE


def verify_source(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if commit != SOURCE['commit'] or dirty.strip():
        raise RuntimeError('Official checkout must be the pinned commit with unmodified tracked files')
    for file in ('README.md', 'LICENSE', 'constant.py', 'model/destseg.py', 'data/mvtec_dataset.py',
                 'eval.py', 'model/metrics.py', 'requirements.txt'):
        if not (root / file).is_file():
            raise FileNotFoundError(root / file)
    return dict(**SOURCE, actual_commit=commit, tracked_source_modified=False)


def modules(root, evaluation=False):
    root = Path(root).resolve()
    verify_source(root)
    # Upstream imports generic top-level names. Reject collisions rather than replacing host modules.
    for name in ('constant', 'model', 'data', 'eval'):
        module = sys.modules.get(name)
        if module is not None:
            locations = [getattr(module, '__file__', None), *getattr(module, '__path__', [])]
            if not any(p and Path(p).resolve().is_relative_to(root) for p in locations):
                raise RuntimeError(f'Upstream import collision: {name}; run in a fresh process')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    dataset = importlib.import_module('data.mvtec_dataset')
    constants = importlib.import_module('constant')
    if evaluation:
        return dataset, constants, importlib.import_module('eval')
    return dataset, constants, importlib.import_module('model.destseg')


def dataset(root, source, category):
    ds, const, _ = modules(source)
    return ds.MVTecDataset(is_train=False, mvtec_dir=str(Path(root)/category/'test'),
        resize_shape=const.RESIZE_SHAPE, normalize_mean=const.NORMALIZE_MEAN, normalize_std=const.NORMALIZE_STD)


def load_model(source, path, device):
    import torch
    _, _, model_module = modules(source)
    # Do not unwrap nested dictionaries, drop keys, or remove prefixes silently.
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected official raw tensor state_dict, not a training checkpoint wrapper')
    model = model_module.DeSTSeg(dest=True, ed=True)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device=device, dtype=torch.float32).eval()
