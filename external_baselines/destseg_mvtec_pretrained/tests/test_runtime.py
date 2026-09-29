"""Small actual tensor/IO tests, skipped when local scientific dependencies are absent."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in
                ('torch','torchvision','numpy','PIL','timm','imgaug','scipy','sklearn'))


@unittest.skipUnless(AVAILABLE, 'Scientific runtime absent: numerical validation NOT executed')
class RuntimeTests(unittest.TestCase):
    def test_preprocessing_matches_original_dataset(self):
        import numpy as np
        import torch
        from PIL import Image
        from external_baselines.destseg_mvtec_pretrained.official import dataset
        from external_baselines.destseg_mvtec_pretrained.protocol import OFFICIAL_ROOT
        from external_baselines.destseg_mvtec_pretrained.adapter import load_rgb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'bottle/test/defect').mkdir(parents=True)
            (root/'bottle/ground_truth/defect').mkdir(parents=True)
            rng = np.random.default_rng(42)
            image = root/'bottle/test/defect/000.png'
            Image.fromarray(rng.integers(0,256,(83,137,3),dtype=np.uint8)).save(image)
            mask = np.zeros((83,137),np.uint8)
            mask[20:40,10:80] = 255
            Image.fromarray(mask).save(root/'bottle/ground_truth/defect/000_mask.png')
            original = dataset(root, OFFICIAL_ROOT, 'bottle')
            torch.testing.assert_close(load_rgb(image, original), original[0]['img'], rtol=0, atol=0)
            self.assertEqual(tuple(original[0]['mask'].shape), (1,256,256))

    def test_segmentation_branch_and_official_sort_formula(self):
        import torch
        import torch.nn.functional as F
        from external_baselines.destseg_mvtec_pretrained.adapter import outputs
        seg = torch.linspace(0.01,0.99,64*64).reshape(1,1,64,64)
        aux = torch.full_like(seg, 2.0)
        (main, score), (other, _) = outputs((seg, aux, []))
        reference = F.interpolate(seg, size=(256,256), mode='bilinear', align_corners=False)
        official_score = torch.sort(reference.view(1,-1), dim=1, descending=True)[0][:,:100].mean(1)
        torch.testing.assert_close(main, reference[:,0], rtol=0, atol=0)
        torch.testing.assert_close(score, official_score, rtol=0, atol=0)
        self.assertLess(float(main.max()), 1.0)
        self.assertEqual(float(other.min()), 2.0)  # no second sigmoid/clipping.

    def test_non_square_full_frame_mapping(self):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from external_baselines.destseg_mvtec_pretrained.adapter import map_to_evaluation
        source = np.arange(256*256,dtype=np.float32).reshape(256,256)
        actual = map_to_evaluation(source, (303,517))
        reference = F.interpolate(torch.from_numpy(source)[None,None],size=(75,129),
                                  mode='bilinear',align_corners=False)[0,0].numpy()
        np.testing.assert_array_equal(actual,reference)
        self.assertEqual(actual.shape,(75,129))

    def test_FP32_save_load_keeps_score_and_evaluation_inputs(self):
        import numpy as np
        from external_baselines.destseg_mvtec_pretrained.adapter import save_prediction, map_to_evaluation
        from external_baselines.patchcore_official_eval.storage import json_write
        from external_baselines.patchcore_official_eval.evaluation import saved_inputs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dest=root/'predict'
            source=np.arange(256*256,dtype=np.float32).reshape(256,256)/np.float32(65536)
            row=dict(category='bottle',relative_path='bottle/test/good/000.png',label=0,mask_path=None,
                     original_hw=[303,517],evaluation_hw=[75,129],prediction_file='maps/000000.npz')
            score=float(np.sort(source.ravel())[-100:].mean())
            save_prediction(dest,row,source,source,score,score,np.zeros((256,256),np.uint8))
            json_write(dest/'samples.json',[row])
            rows,maps,masks,_=saved_inputs(root,root)
            self.assertEqual(rows[0]['image_score'],score)
            np.testing.assert_array_equal(maps[0],map_to_evaluation(source,(303,517)))
            self.assertEqual(masks[0].shape,(75,129))
            self.assertFalse(masks[0].any())
            # Second read uses existing snapshot, requiring no source dataset.
            reloaded=saved_inputs(root)
            np.testing.assert_array_equal(reloaded[1][0],maps[0])

    def test_existing_macro_is_category_equal_and_excludes_NA(self):
        from external_baselines.patchcore_official_eval.evaluation import summarize
        keys=('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO_at_0p3')
        metrics=[dict(category=c,**{k:v for k in keys},sample_count=n)
                 for c,v,n in [('bottle',0.2,2),('cable',0.8,100)]]
        fixed=[]
        for c in ('bottle','cable'):
            for cap in (.01,.05):
                fixed.append(dict(category=c,fpr_cap=cap,actual_fpr=cap,defect_pixel_recall=.5,
                    region_mean_coverage=.5,small_region_mean_coverage='N/A' if c=='bottle' else .7,
                    region_count=2,small_region_count=0 if c=='bottle' else 1,
                    small_region_image_count=0 if c=='bottle' else 1))
        macro,operating=summarize(metrics,fixed,['bottle','cable'],'mvtec')
        for k in keys:
            self.assertAlmostEqual(macro[0][k],.5)
        self.assertEqual(operating[0]['small_region_mean_coverage_valid_category_count'],1)
        self.assertEqual(operating[0]['small_region_count_sum'],1)


if __name__=='__main__':
    unittest.main()
