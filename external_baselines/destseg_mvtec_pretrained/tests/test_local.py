"""Executed with stdlib only. These checks do NOT validate model numerics."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from external_baselines.destseg_mvtec_pretrained.checkpoints import inventory, require_weights
from external_baselines.destseg_mvtec_pretrained.protocol import CATEGORIES, OFFICIAL_ROOT, selected_categories
from external_baselines.destseg_mvtec_pretrained.official import verify_source


class LocalTests(unittest.TestCase):
    def test_source_pin_and_python_syntax(self):
        self.assertFalse(verify_source(OFFICIAL_ROOT)['tracked_source_modified'])
        for path in Path(__file__).parents[1].rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_missing_ambiguous_and_explicit_weight_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = inventory(root)
            self.assertEqual([r['category'] for r in rows if r['status']=='missing'], list(CATEGORIES))
            with self.assertRaises(FileNotFoundError):
                require_weights(rows, ['bottle'])
            # Text placeholders validate filename discovery only, never model loading.
            name = 'DeSTSeg_MVTec_5000_bottle.pckl'
            (root/name).write_text('inventory-only placeholder')
            self.assertEqual(require_weights(inventory(root), ['bottle'])['bottle']['status'], 'present_not_loaded')
            (root/'nested').mkdir()
            (root/'nested'/name).write_text('ambiguous placeholder')
            with self.assertRaises(ValueError):
                inventory(root)
            mapping = {c: 'actual_'+c+'.pckl' for c in CATEGORIES}
            mapping['bottle'] = name
            mapfile = root/'map.json'
            mapfile.write_text(json.dumps(mapping))
            self.assertEqual(inventory(root, mapfile)[0]['status'], 'present_not_loaded')
            mapping['cable'] = name
            mapfile.write_text(json.dumps(mapping))
            with self.assertRaises(ValueError):
                inventory(root, mapfile)

    def test_two_shards_cover_all_once(self):
        a = selected_categories(['all'], '0/2')
        b = selected_categories(['all'], '1/2')
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a+b), set(CATEGORIES))
        self.assertEqual((len(a), len(b)), (8,7))
        for bad in ([], ['visa'], ['bottle','bottle']):
            with self.assertRaises(ValueError):
                selected_categories(bad)

    def test_upstream_output_contract(self):
        source = (OFFICIAL_ROOT/'model/destseg.py').read_text()
        tree = ast.parse(source)
        network = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeSTSeg')
        forward = next(n for n in network.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        ret = next(n for n in forward.body if isinstance(n,ast.Return))
        self.assertEqual([n.id for n in ret.value.elts],
                         ['output_segmentation','output_de_st','output_de_st_list'])
        self.assertIn('x = torch.sigmoid(x)', source)
        self.assertIn('output_segmentation_sample[:, : args.T]', (OFFICIAL_ROOT/'eval.py').read_text())


if __name__ == '__main__':
    unittest.main()
