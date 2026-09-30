import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(x) for x in ['torch','torchvision','numpy','PIL','timm','imgaug'])


@unittest.skipUnless(AVAILABLE, 'Run in the independent Kaggle DeSTSeg environment')
class RuntimeTests(unittest.TestCase):
    def test_multivalue_mask_matches_official_transform(self):
        import numpy as np
        from PIL import Image
        from external_baselines.destseg_visa.data import official_mask
        from external_baselines.destseg_btad.data import test_transforms
        from external_baselines.destseg_visa.protocol import OFFICIAL_ROOT
        transform = test_transforms(OFFICIAL_ROOT)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'mask.png'
            raw = np.zeros((303,517),np.uint8)
            raw[40:160,70:200] = 2
            raw[160:200,200:300] = 7
            Image.fromarray(raw).save(path)
            record = SimpleNamespace(label=1,mask_path=str(path))
            actual = official_mask(record, transform)
            expected = transform.mask_preprocessing(Image.fromarray((raw>0).astype(np.uint8)*255))
            np.testing.assert_array_equal(actual, (expected[0]>=.5).numpy())
            self.assertTrue(actual.any())
            Image.fromarray(np.zeros_like(raw)).save(path)
            self.assertFalse(official_mask(record,transform).any())
            self.assertEqual(record.label,1)

    def test_checkpoint_dataset_and_smoke_guards(self):
        import json
        from external_baselines.destseg_btad.runner import trained
        from external_baselines.destseg_btad.protocol import protocol as btad_protocol
        from external_baselines.destseg_visa.protocol import protocol
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root/'candle/train'
            folder.mkdir(parents=True)
            (folder/'final.pckl').touch()  # Metadata guard only; no model loading claimed.
            state = dict(status='complete', step=5000, smoke=False, checkpoint='final.pckl',
                         identity=dict(category='candle',protocol=protocol()))
            def save():
                (folder/'complete.json').write_text(json.dumps(state))
            save()
            self.assertEqual(trained(root,'candle',protocol_factory=protocol)[0],folder/'final.pckl')
            state['identity']['protocol'] = btad_protocol(); save()
            with self.assertRaises(ValueError):
                trained(root,'candle',protocol_factory=protocol)
            state['identity']['protocol'] = protocol()
            state.update(smoke=True,step=40); save()
            with self.assertRaises(ValueError):
                trained(root,'candle',protocol_factory=protocol)
            self.assertTrue(trained(root,'candle',allow_smoke=True,protocol_factory=protocol)[2])


if __name__ == '__main__':
    unittest.main()
