"""Validate shared CSV metadata; training opens only normal training files."""
from pathlib import Path
from DINOv3.MADEqual.visa_task_decoupled.dataset import visa_one_class_records
from .protocol import COUNTS


def split_records(root, *, check_test_files=False):
    root, train, test = visa_one_class_records(root, formal=check_test_files)
    for category, expected in COUNTS.items():
        actual = (len(train[category]), sum(r.label == 0 for r in test[category]),
                  sum(r.label == 1 for r in test[category]))
        if actual != expected:
            raise ValueError(f'VisA {category} split counts {actual} != {expected}')
    return root, train, test


def training_paths(root, category):
    root, train, _ = split_records(root)
    paths = [Path(r.path) for r in train[category]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return root, paths
