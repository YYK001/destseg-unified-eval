from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import patch
from external_baselines.destseg_mvtec_pretrained.resources import AllocatorResources


class FakeCUDA:
    # Deliberately has no synchronize or sampling API.
    def reset_peak_memory_stats(self, device): pass
    def max_memory_allocated(self, device): return 123
    def max_memory_reserved(self, device): return 456


class ResourceTests(unittest.TestCase):
    def test_body_save_precedes_non_sampling_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            meter=AllocatorResources(root,['cuda:0'],'candle',cuda=FakeCUDA())
            with meter.measure('segmentation_train_and_checkpoint_save'):
                (root/'model_saved').touch()
                self.assertFalse((root/'resources.csv').exists())
            with (root/'resources.csv').open() as handle:
                row=list(csv.DictReader(handle))[0]
            self.assertEqual(row['cuda:0_allocated_peak_bytes'],'123')
            self.assertEqual(row['status'],'complete')
            self.assertIn('disabled',row['nvml_sampling'])

    def test_auxiliary_statistics_failure_does_not_undo_body(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            meter=AllocatorResources(root,[],'candle',cuda=FakeCUDA())
            with patch('external_baselines.destseg_mvtec_pretrained.resources.csv_write',side_effect=OSError('test')):
                with meter.measure('train'):
                    (root/'model_saved').touch()
            self.assertTrue((root/'model_saved').is_file())

    def test_training_exception_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            meter=AllocatorResources(Path(d),[],'candle',cuda=FakeCUDA())
            with self.assertRaisesRegex(ValueError,'training failed'):
                with meter.measure('train'):
                    raise ValueError('training failed')
            self.assertEqual(meter.rows[0]['status'],'failed')
