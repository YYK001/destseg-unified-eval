from external_baselines.destseg_btad.protocol import protocol as btad_protocol, schedule
from external_baselines.destseg_mvtec_pretrained.protocol import OFFICIAL_ROOT
from DINOv3.MADEqual.visa_task_decoupled.dataset import VISA_CATEGORIES

CATEGORIES = VISA_CATEGORIES
# Same 1cls.csv split used in the archived PatchCore VisA evaluation.
COUNTS = dict(zip(CATEGORIES, [
    (900,100,100), (542,60,100), (450,50,100), (453,50,100),
    (450,50,100), (900,100,100), (900,100,100), (904,100,100),
    (901,100,100), (905,101,100), (904,101,100), (450,50,100),
]))
SHARDS = (CATEGORIES[::2], CATEGORIES[1::2])


def protocol(batch_size=32, seed=42):
    if not 2 <= batch_size <= 450:
        raise ValueError('VisA batch size must be 2..450')
    p = btad_protocol(32, seed)
    p.update(id='destseg_visa_1cls_normal_training_fullframe256_top100_v1', dataset='visa',
        official_mask='VisA original mask >0 -> uint8 0/255; official mask_preprocessing; >=0.5',
        official_metrics='official metric implementation applied to VisA adapted masks; not an official VisA benchmark')
    p['training'].update(batch_size=batch_size,
        train_split='split_csv/1cls.csv train normal rows only; no calibration split; no test image or mask reads',
        rotation='none for all VisA classes; frozen adaptation choice, not official VisA defaults')
    return p


def select(names):
    values = list(CATEGORIES) if names == ['all'] else list(names)
    if not values or len(set(values)) != len(values) or not set(values) <= set(CATEGORIES):
        raise ValueError('Use all or unique VisA category names')
    return values
