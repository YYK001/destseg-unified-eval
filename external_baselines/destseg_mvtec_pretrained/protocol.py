"""Frozen definitions; standard library only, usable without a ML environment."""
from pathlib import Path
from external_baselines.patchcore_official_eval.protocol import CATEGORIES as DATASETS

CATEGORIES = DATASETS['mvtec']
OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / 'destseg_official'
WEIGHTS_ROOT = OFFICIAL_ROOT / 'saved_model'
SOURCE = dict(url='https://github.com/apple-aiml-research/ml-destseg',
              commit='f168ba576e0a917ae5e037b677e7d37b741ea3d8', license='upstream LICENSE (Apple)')
WEIGHTS_URL = 'https://www.icloud.com.cn/iclouddrive/0a3OPg_3wcMs38yDpWNRnRW9Q#saved%5Fmodel'
PROTOCOL = dict(id='destseg_official_mvtec_pretrained_fullframe256_top100_v1', source=SOURCE,
    model='DeSTSeg(dest=True, ed=True); strict state_dict', branch='output_segmentation',
    dtype='float32', amp=False, input_hw=[256, 256], image_T=100,
    rgb='PIL RGB; PIL bilinear resize256; official final_preprocessing (ImageNet mean/std)',
    output='official sigmoid already applied; bilinear256 align_corners=False; no extra postprocessing',
    image_score='torch.sort descending then mean first100 at256; frozen when map resized',
    coordinates='full frame256 -> original H//4,W//4; torch bilinear align_corners=False; no crop/pad',
    unified_mask='original L mask; PIL nearest to H//4,W//4; >0; existing data label',
    official_mask='official ToTensor then bilinear Resize antialias=True; >=0.5; image label=max(mask)',
    unified_metrics='existing PatchCore calculate/summarize; AP=average precision; fast CUDA AUPRO200 FPR0.3',
    official_metrics='unmodified eval.evaluate and model.metrics; DeSTSeg and DeST separate; IAP is instance-level')


def selected_categories(names, shard=None):
    selected = list(CATEGORIES) if names == ['all'] else list(names)
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(CATEGORIES):
        raise ValueError('Use all or unique MVTec AD category names')
    if shard is not None:
        i, n = map(int, shard.split('/'))
        if n != 2 or i not in (0, 1):
            raise ValueError('Dual-T4 shards are 0/2 and 1/2')
        selected = selected[i::n]
    return selected
