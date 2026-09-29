"""Change file listing only for training; reuse original DTD/Perlin implementation."""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from external_baselines.destseg_mvtec_pretrained.official import modules
from external_baselines.patchcore_official_eval.data import records, metadata
from DINOv3.MADEqual.btad_validation.dataset import find_root, images
from .protocol import TRAIN_COUNTS


def seed_everything(seed):
    import imgaug
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    imgaug.seed(seed)
    # Explicitly seed the upstream module-level augmenter, also inside worker processes.
    from data.data_utils import rot
    rot.seed_(seed)


def seed_worker(worker_id):
    # DataLoader has already assigned a distinct torch.initial_seed to each worker.
    seed=torch.initial_seed() % 2**32
    import imgaug
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)
    from data.data_utils import rot
    rot.seed_(seed)


def training_dataset(root, dtd_root, category, source):
    ds,const,_=modules(source)
    root=find_root(root)
    paths=images(root/category/'train/ok')  # no test listing/GT read in training.
    if len(paths)!=TRAIN_COUNTS[category]:
        raise ValueError(f'{category} normal training count {len(paths)} != {TRAIN_COUNTS[category]}')
    dtd_root=Path(dtd_root).resolve()
    textures=sorted(dtd_root.glob('*/*.jpg'))
    if len(textures)!=5640 or len({p.parent.name for p in textures})!=47:
        raise ValueError(f'Expected DTD images root with 47 folders/5640 JPEGs, got {len(textures)}: {dtd_root}')
    obj=ds.MVTecDataset(is_train=True,mvtec_dir=str(root/category/'train/ok'),
        resize_shape=const.RESIZE_SHAPE,normalize_mean=const.NORMALIZE_MEAN,normalize_std=const.NORMALIZE_STD,
        dtd_dir=str(dtd_root),rotate_90=False,random_rotate=0)
    obj.mvtec_paths=[str(p) for p in paths]  # BTAD also uses BMP, which official *.png misses.
    # __getitem__ is the actual unmodified upstream method, not an augmentation rewrite.
    if set(obj.dtd_paths)!={str(p) for p in textures}:
        raise ValueError('Official DTD listing differs')
    return root,obj,paths,textures


def test_listing(root, categories):
    root,_,test,counts=records('btad',root,categories)
    return root,test,counts


def test_transforms(source):
    ds,const,_=modules(source)
    return ds.MVTecDataset(is_train=False,mvtec_dir='',resize_shape=const.RESIZE_SHAPE,
        normalize_mean=const.NORMALIZE_MEAN,normalize_std=const.NORMALIZE_STD)


def official_mask(record, transform):
    if record.label==0:
        return np.zeros((256,256),np.uint8)
    with Image.open(record.mask_path) as im:
        raw=np.asarray(im.convert('L'))
    if not set(np.unique(raw).tolist())<={0,1,255}:
        raise ValueError(f'Unexpected BTAD mask values: {record.mask_path}')
    # BTAD may encode foreground as 1 or255; preserve >0 semantics before ToTensor /255.
    value=transform.mask_preprocessing(Image.fromarray((raw>0).astype(np.uint8)*255))
    return (value[0]>=.5).to(torch.uint8).numpy()
