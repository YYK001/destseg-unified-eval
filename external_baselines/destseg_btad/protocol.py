"""BTAD adaptation choices, explicitly separate from official MVTec results."""
from copy import deepcopy
from external_baselines.destseg_mvtec_pretrained.protocol import PROTOCOL as INFERENCE
from external_baselines.destseg_mvtec_pretrained.protocol import OFFICIAL_ROOT

CATEGORIES = ('01','02','03')
TRAIN_COUNTS = {'01':400,'02':399,'03':1000}


def protocol(batch_size=32, seed=42):
    if batch_size < 2 or batch_size > min(TRAIN_COUNTS.values()):
        raise ValueError('Batch size must be 2..399; batch=1 breaks official ASPP training BatchNorm')
    p=deepcopy(INFERENCE)
    p.update(id='destseg_btad_normal_training_fullframe256_top100_v1', dataset='btad',
        initialization='official ImageNet ResNet18 teacher; random student/segmentation; no MVTec checkpoint',
        training=dict(steps=5000, student_steps=1000, batch_size=batch_size, seed=seed,
            train_split='all BTAD train/ok; no calibration split; no test images or masks',
            rotation='none for all BTAD classes; explicit adaptation choice, not official BTAD defaults',
            augmentation='unmodified official MVTecDataset training __getitem__; DTD Perlin aug_prob=1.0',
            optimizer='SGD momentum0.9 weight_decay1e-4 nesterov=False',
            lr_student=.4, lr_segmentation_res=.1, lr_segmentation_head=.01, focal_gamma=4,
            student_loss='official cosine_similarity_loss, summed over batch, unchanged',
            segmentation_loss='official focal_loss + l1_loss', amp=False, drop_last=True,
            selection='fixed final step5000; no test-set evaluation during training or checkpoint selection'),
        official_mask='BTAD original binary mask >0 -> uint8 0/255; official mask_preprocessing; >=0.5',
        official_metrics='official metric implementation applied to BTAD adapted masks; not an official BTAD benchmark')
    return p


def schedule(smoke_steps=0):
    if smoke_steps < 0:
        raise ValueError('smoke_steps must be >=0')
    return (smoke_steps,smoke_steps) if smoke_steps else (1000,4000)


def select(names):
    values=list(CATEGORIES) if names==['all'] else list(names)
    if not values or len(set(values))!=len(values) or not set(values)<=set(CATEGORIES):
        raise ValueError('Use all or unique BTAD categories 01 02 03')
    return values
