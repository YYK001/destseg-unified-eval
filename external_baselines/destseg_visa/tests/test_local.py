import ast
import csv
from pathlib import Path
import tempfile
import unittest
from external_baselines.destseg_visa.protocol import CATEGORIES, COUNTS, SHARDS, protocol, select
from external_baselines.destseg_visa.split import training_paths, split_records


class LocalTests(unittest.TestCase):
    def make_split(self, root):
        (root/'split_csv').mkdir()
        with (root/'split_csv/1cls.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['object','split','label','image','mask'])
            for c, counts in COUNTS.items():
                for split, label, count in [('train','normal',counts[0]),
                                            ('test','normal',counts[1]),('test','anomaly',counts[2])]:
                    for i in range(count):
                        image = f'{c}/{split}/{label}/{i:04d}.JPG'
                        w.writerow([c,split,label,image,f'{c}/masks/{i:04d}.png' if label=='anomaly' else ''])
                        if split == 'train' and c == 'candle':
                            path = root/image
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.touch()

    def test_csv_training_never_requires_test_images_or_masks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_split(root)
            _, paths = training_paths(root, 'candle')
            self.assertEqual(len(paths), 900)
            self.assertTrue(all('/train/normal/' in p.as_posix() for p in paths))
            with self.assertRaises(FileNotFoundError):
                split_records(root, check_test_files=True)
            paths[0].unlink()
            with self.assertRaises(FileNotFoundError):
                training_paths(root, 'candle')

    def test_incomplete_split_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.make_split(root)
            p = root/'split_csv/1cls.csv'
            p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
            with self.assertRaises(ValueError):
                split_records(root)

    def test_protocol_shards_and_counts(self):
        from external_baselines.destseg_btad.protocol import protocol as btad
        self.assertEqual(sum(v[0] for v in COUNTS.values()),8659)
        self.assertEqual(sum(v[1]+v[2] for v in COUNTS.values()),2162)
        self.assertEqual(set(SHARDS[0])|set(SHARDS[1]),set(CATEGORIES))
        self.assertFalse(set(SHARDS[0])&set(SHARDS[1]))
        self.assertEqual(len(select(['all'])),12)
        self.assertEqual(protocol()['training']['steps'],5000)
        self.assertEqual(btad()['dataset'],'btad')
        self.assertNotEqual(protocol(),btad())
        for names in [[],['01'],['candle','candle']]:
            with self.assertRaises(ValueError): select(names)
        with self.assertRaises(ValueError): protocol(1)

    def test_source_syntax(self):
        for path in Path(__file__).parents[1].rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'))


if __name__ == '__main__':
    unittest.main()
