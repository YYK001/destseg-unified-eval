"""Image-only official preprocessing and output conversion; no metric implementations."""
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from external_baselines.patchcore_official_eval.data import map_to_evaluation, metadata


def load_rgb(path, official_dataset):
    with Image.open(path) as im:
        im = im.convert('RGB').resize(official_dataset.resize_shape, Image.BILINEAR)
        return official_dataset.final_preprocessing(im)


def outputs(model_output):
    segmentation, auxiliary, _ = model_output
    result = []
    for value in (segmentation, auxiliary):
        if value.ndim != 4 or value.shape[1] != 1 or value.dtype != torch.float32:
            raise ValueError('Expected FP32 N1HW output from official model')
        value = F.interpolate(value, size=(256, 256), mode='bilinear', align_corners=False)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite model output')
        score = torch.sort(value.reshape(value.size(0), -1), dim=1, descending=True)[0][:, :100].mean(dim=1)
        result.append((value[:, 0], score))
    return result  # main first, auxiliary second; no sigmoid/smoothing/normalization.


def save_prediction(directory, row, main, auxiliary, score, auxiliary_score, official_mask):
    from external_baselines.patchcore_official_eval.storage import npz_write
    main = np.asarray(main)
    auxiliary = np.asarray(auxiliary)
    if main.dtype != np.float32 or auxiliary.dtype != np.float32 or auxiliary.shape != (256, 256):
        raise ValueError('FP32 original maps required')
    if not np.isfinite(auxiliary).all():
        raise ValueError('Nonfinite auxiliary output')
    official_mask = np.asarray(official_mask)
    if official_mask.shape != (256, 256) or not np.isin(official_mask, [0, 1]).all():
        raise ValueError('Expected original official binary mask at 256x256')
    if not np.isfinite([score, auxiliary_score]).all():
        raise ValueError('Nonfinite image score')
    row.update(image_score=float(score), auxiliary_image_score=float(auxiliary_score),
               official_label=int(np.any(official_mask)),
               image_label_differs=bool(row['label'] != int(np.any(official_mask))))
    npz_write(directory / row['prediction_file'], input_map=main,
        evaluation_map=map_to_evaluation(main, row['original_hw']), auxiliary_map=auxiliary,
        official_mask=np.asarray(official_mask, dtype=np.uint8),
        image_score=np.asarray(score, dtype=np.float64), auxiliary_image_score=np.asarray(auxiliary_score, dtype=np.float64),
        category=np.asarray(row['category']), relative_path=np.asarray(row['relative_path']))
