"""Shared MVTec listing, official RGB transform, nearest binary GT in the same crop."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode

from DINOv3.relation_reliability.datasets import mvtec_ad_records, _mvtec_root
from DINOv3.relation_reliability.protocol import MVTEC_AD_CATEGORIES as CATEGORIES


def records(root, categories):
    root = _mvtec_root(Path(root).resolve())
    _, tests = mvtec_ad_records(root, categories)
    counts = []
    for c, items in tests.items():
        paths = [str(Path(r.path).resolve()) for r in items]
        if len(set(paths)) != len(paths):
            raise ValueError(f'duplicate test paths: {c}')
        counts.append(dict(category=c, test_count=len(items), normal_count=sum(r.label == 0 for r in items),
                           anomaly_count=sum(r.label == 1 for r in items)))
    if set(categories) == set(CATEGORIES):
        actual = tuple(sum(r[k] for r in counts) for k in ('test_count', 'normal_count', 'anomaly_count'))
        if actual != (1725, 467, 1258):
            raise ValueError(f'full MVTec split counts {actual} differ from (1725,467,1258); inspect data, do not trim')
    return root, tests, counts


def official_transform(items):
    from external_baselines.glass_external.official import modules
    upstream = modules().mvtec

    class Listing(upstream.MVTecDataset):
        def get_image_data(self):
            return {self.classname: {'good': [r.path for r in items]}}, [
                [self.classname, 'good', r.path, None] for r in items]

    # Build official TEST transforms; no DTD, foreground masks or __getitem__ synthesis.
    dataset = Listing(source='', anomaly_source_path=str(Path(__file__).parent / '_unused_texture_path'),
        classname=items[0].category, resize=288, imagesize=288, split=upstream.DatasetSplit.TEST,
        distribution=0, fg=0)
    return dataset.transform_img, dataset.resize


def geometry(image, resize):
    w, h = image.size
    rw, rh = TF.resize(image, resize).size
    # Same Python round-to-even rule as torchvision CenterCrop.
    top, left = int(round((rh - 288) / 2.)), int(round((rw - 288) / 2.))
    return dict(original_height=h, original_width=w, resized_height=rh, resized_width=rw,
                crop_top=top, crop_left=left, crop_height=288, crop_width=288,
                output_height=288, output_width=288)


def load_rgb(record, transform, resize):
    with Image.open(record.path) as source:
        image = source.convert('RGB')
        geom = geometry(image, resize)
        x = transform(image)
    if tuple(x.shape) != (3, 288, 288):
        raise ValueError('unexpected official RGB transform shape')
    return x, geom


def crop_gt(record, geom):
    if record.mask_path:
        with Image.open(record.mask_path) as im:
            a = np.asarray(im)
            if a.ndim != 2 or a.shape != (geom['original_height'], geom['original_width']):
                raise ValueError(f'mask/image geometry mismatch: {record.mask_path}')
            raw = a != 0
    else:
        raw = np.zeros((geom['original_height'], geom['original_width']), dtype=bool)
    mask = TF.resize(Image.fromarray(raw.astype(np.uint8)),
        [geom['resized_height'], geom['resized_width']], interpolation=InterpolationMode.NEAREST)
    mask = np.asarray(TF.crop(mask, geom['crop_top'], geom['crop_left'], 288, 288)) != 0
    audit = dict(image_label=record.label, original_empty=not bool(raw.any()),
                 cropped_empty=not bool(mask.any()), original_foreground=int(raw.sum()),
                 cropped_foreground=int(mask.sum()))
    return mask.astype(np.uint8), audit


def reference_crop(array, geom):
    array = np.asarray(array)
    if array.ndim != 2 or array.dtype.kind != 'f' or not np.isfinite(array).all():
        raise ValueError('reference must be a finite continuous full-frame floating map')
    x = torch.from_numpy(array.astype(np.float32))[None, None]
    x = F.interpolate(x, size=(geom['resized_height'], geom['resized_width']), mode='bilinear', align_corners=False)
    top, left = geom['crop_top'], geom['crop_left']
    return x[0, 0, top:top+288, left:left+288].numpy()
