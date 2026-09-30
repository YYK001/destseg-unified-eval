from pathlib import Path
import numpy as np
import torch
from PIL import Image
from external_baselines.destseg_mvtec_pretrained.official import modules
from external_baselines.patchcore_official_eval.data import records
from .protocol import protocol
from .split import training_paths, split_records


def training_dataset(root, dtd_root, category, source):
    root, paths = training_paths(root, category)
    ds, const, _ = modules(source)
    dtd_root = Path(dtd_root).resolve()
    textures = sorted(dtd_root.glob('*/*.jpg'))
    if len(textures) != 5640 or len({p.parent.name for p in textures}) != 47:
        raise ValueError('DTD images root must contain 47 folders and 5640 JPEGs')
    obj = ds.MVTecDataset(is_train=True, mvtec_dir='', dtd_dir=str(dtd_root),
        resize_shape=const.RESIZE_SHAPE, normalize_mean=const.NORMALIZE_MEAN,
        normalize_std=const.NORMALIZE_STD, rotate_90=False, random_rotate=0)
    obj.mvtec_paths = [str(p) for p in paths]
    if set(obj.dtd_paths) != {str(p) for p in textures}:
        raise ValueError('Official DTD listing differs')
    return root, obj, paths, textures


def test_listing(root, categories):
    split_records(root, check_test_files=True)
    root, _, test, counts = records('visa', root, categories)
    return root, test, counts


def official_mask(record, transform):
    if record.label == 0:
        return np.zeros((256,256), np.uint8)
    with Image.open(record.mask_path) as im:
        raw = np.asarray(im.convert('L'))
    # VisA masks can use multiple foreground values; unified semantics are >0.
    value = transform.mask_preprocessing(Image.fromarray((raw > 0).astype(np.uint8) * 255))
    return (value[0] >= .5).to(torch.uint8).numpy()
