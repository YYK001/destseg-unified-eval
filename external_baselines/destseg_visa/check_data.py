"""Read-only check of the existing VisA 1cls split and image/mask files."""
import argparse
from .split import split_records
from .protocol import COUNTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    args = p.parse_args()
    root, _, _ = split_records(args.dataset_root, check_test_files=True)
    print('VISA_ROOT =', root)
    print('category,train_normal,test_normal,test_anomaly')
    for c, counts in COUNTS.items():
        print(c, *counts, sep=',')
    print('Complete: 12 categories, 8659 training normals, 2162 test images (962 normal / 1200 anomaly).')


if __name__ == '__main__':
    main()
