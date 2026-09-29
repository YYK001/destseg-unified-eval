import ast
from pathlib import Path
import unittest
import tarfile
from external_baselines.destseg_btad.protocol import protocol,schedule,select


class LocalTests(unittest.TestCase):
    def test_dtd_metadata_ignored_but_image_paths_checked(self):
        from external_baselines.destseg_btad.prepare_dtd import image_members
        names=['dtd/images/waffled/.directory','dtd/images/waffled/image.jpg',
               'dtd/images/waffled/notes.txt','dtd/labels/train1.txt']
        selected=image_members([tarfile.TarInfo(n) for n in names])
        self.assertEqual([m.name for m in selected],['dtd/images/waffled/image.jpg'])
        for name in ['dtd/images/../../escape.jpg','dtd/images/waffled/nested/image.jpg']:
            with self.assertRaises(ValueError):
                image_members([tarfile.TarInfo(name)])

    def test_frozen_schedule_and_protocol_separation(self):
        from external_baselines.destseg_mvtec_pretrained.protocol import PROTOCOL
        self.assertEqual(schedule(),(1000,4000))
        self.assertEqual(schedule(20),(20,20))
        self.assertEqual(protocol()['training']['batch_size'],32)
        self.assertNotEqual(protocol(16),protocol(32))
        self.assertNotIn('training',PROTOCOL)  # deepcopy did not mutate MVTec.
        self.assertEqual(select(['all']),['01','02','03'])
        for values in ([],['01','01'],['bottle']):
            with self.assertRaises(ValueError):select(values)
        with self.assertRaises(ValueError):protocol(1)

    def test_python_and_upstream_reuse_contract(self):
        root=Path(__file__).parents[1]
        for p in root.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'))
        tree=ast.parse((root/'data.py').read_text())
        training=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_dataset')
        calls=[n for n in ast.walk(training) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Attribute) and n.func.attr=='MVTecDataset' for n in calls))
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='__getitem__' for n in ast.walk(training)))


if __name__=='__main__':unittest.main()
