"""Small tensor checks, not a substitute for actual DeSTSeg training smoke."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

AVAILABLE=all(importlib.util.find_spec(x) for x in ['torch','torchvision','numpy','PIL','timm','imgaug'])


@unittest.skipUnless(AVAILABLE,'Scientific runtime absent; run in Kaggle independent environment')
class RuntimeTests(unittest.TestCase):
    def test_btad_mask_one_and_255_have_identical_foreground(self):
        import numpy as np
        from PIL import Image
        from external_baselines.destseg_btad.data import test_transforms,official_mask
        from external_baselines.destseg_btad.protocol import OFFICIAL_ROOT
        transform=test_transforms(OFFICIAL_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.bmp'
            raw=np.zeros((303,517),np.uint8);raw[50:160,70:200]=1
            Image.fromarray(raw).save(path)
            record=SimpleNamespace(label=1,mask_path=str(path))
            a=official_mask(record,transform)
            Image.fromarray(raw*255).save(path)
            b=official_mask(record,transform)
            np.testing.assert_array_equal(a,b)
            self.assertTrue(a.any())
            Image.fromarray(np.zeros_like(raw)).save(path)
            self.assertFalse(official_mask(record,transform).any())
            self.assertEqual(record.label,1)  # anomalous empty masks never relabel unified image GT.

    def test_stage_optimizer_and_batchnorm_updates_are_separate(self):
        import torch
        from torch import nn
        import importlib
        from external_baselines.destseg_mvtec_pretrained.official import modules
        from external_baselines.destseg_btad.protocol import OFFICIAL_ROOT
        from external_baselines.destseg_btad.runner import optimizers,train_step
        modules(OFFICIAL_ROOT)
        losses=importlib.import_module('model.losses')
        class Seg(nn.Module):
            def __init__(self):
                super().__init__()
                self.res=nn.Sequential(nn.Conv2d(2,2,1),nn.BatchNorm2d(2))
                self.head=nn.Sequential(nn.Conv2d(2,1,1),nn.Sigmoid())
            def forward(self,x):return self.head(self.res(x))
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.student_net=nn.Sequential(nn.Conv2d(3,2,1),nn.BatchNorm2d(2))
                self.segmentation_net=Seg()
            def forward(self,aug,original):
                s=self.student_net(aug)
                discrepancy=(s-original[:,:2]).square().sum(1,keepdim=True)
                return self.segmentation_net(s),discrepancy,[discrepancy]
        torch.manual_seed(42)
        model=Tiny();student,seg=optimizers(model)
        batch=dict(img_aug=torch.rand(2,3,8,8),img_origin=torch.rand(2,3,8,8),mask=torch.ones(2,1,8,8))
        def snap(net):return {k:v.clone() for k,v in net.state_dict().items()}
        initial_seg=snap(model.segmentation_net)
        initial_student=snap(model.student_net)
        train_step(model,batch,student,seg,'student','cpu',losses)
        self.assertTrue(all(torch.equal(v,model.segmentation_net.state_dict()[k]) for k,v in initial_seg.items()))
        self.assertTrue(any(not torch.equal(v,model.student_net.state_dict()[k]) for k,v in initial_student.items()))
        mid=snap(model.student_net)
        train_step(model,batch,student,seg,'segmentation','cpu',losses)
        self.assertTrue(all(torch.equal(v,model.student_net.state_dict()[k]) for k,v in mid.items()))
        self.assertTrue(any(not torch.equal(v,model.segmentation_net.state_dict()[k]) for k,v in initial_seg.items()))


if __name__=='__main__':unittest.main()
