"""Official two-stage optimization on BTAD normals; no evaluation during training."""
import csv
import faulthandler
import gc
import importlib
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from external_baselines.destseg_mvtec_pretrained import runner as shared
from external_baselines.destseg_mvtec_pretrained.official import modules, verify_source, load_model
from external_baselines.destseg_mvtec_pretrained.adapter import load_rgb, outputs, save_prediction
from external_baselines.patchcore_official_eval.storage import json_write, read_json, csv_write, finish_stage
from .data import training_dataset, seed_everything, seed_worker, test_listing, test_transforms, official_mask, metadata
from .protocol import protocol, schedule


def optimizers(model):
    seg=torch.optim.SGD([
        {'params':model.segmentation_net.res.parameters(),'lr':.1},
        {'params':model.segmentation_net.head.parameters(),'lr':.01}],
        lr=.001,momentum=.9,weight_decay=1e-4,nesterov=False)
    student=torch.optim.SGD([{'params':model.student_net.parameters(),'lr':.4}],
        lr=.4,momentum=.9,weight_decay=1e-4,nesterov=False)
    return student,seg


def train_step(model, batch, student_optimizer, seg_optimizer, phase, device, losses):
    if phase not in ('student','segmentation'):
        raise ValueError(phase)
    seg_optimizer.zero_grad()
    student_optimizer.zero_grad()
    model.student_net.train(phase=='student')
    model.segmentation_net.train(phase=='segmentation')
    original=batch['img_origin'].to(device)
    augmented=batch['img_aug'].to(device)
    mask=batch['mask'].to(device)
    segmentation,_,de_st_list=model(augmented,original)
    mask=F.interpolate(mask,size=segmentation.size()[2:],mode='bilinear',align_corners=False)
    mask=torch.where(mask<.5,torch.zeros_like(mask),torch.ones_like(mask))
    cosine=losses.cosine_similarity_loss(de_st_list)
    focal=losses.focal_loss(segmentation,mask,gamma=4)
    l1=losses.l1_loss(segmentation,mask)
    total=cosine if phase=='student' else focal+l1
    if not all(torch.isfinite(v).all() for v in (cosine,focal,l1,total)):
        raise FloatingPointError('Nonfinite training loss; no checkpoint accepted')
    total.backward()
    (student_optimizer if phase=='student' else seg_optimizer).step()
    # No batch normalization of cosine loss, AMP, accumulation or extra freezing.
    return dict(cosine_loss=float(cosine.detach()),focal_loss=float(focal.detach()),
                l1_loss=float(l1.detach()),total_loss=float(total.detach()))


def save_model(path, model):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},temporary)
    os.replace(temporary,path)


def train(args,categories, *, backend=None):
    make_protocol = protocol if backend is None else backend.protocol
    make_dataset = training_dataset if backend is None else backend.training_dataset
    device=shared.device_setup(args.device)
    source=verify_source(args.official_root)
    _,_,network=modules(args.official_root)
    losses=importlib.import_module('model.losses')
    p=make_protocol(args.batch_size,args.seed)
    smoke=args.smoke_steps>0
    n_student,n_seg=schedule(args.smoke_steps)
    for category in categories:
        # Training opens only normal training images and DTD; no test images/masks.
        root,ds,paths,textures=make_dataset(args.dataset_root,args.dtd_root,category,args.official_root)
        seed_everything(args.seed)
        loader=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,shuffle=True,
            num_workers=args.workers,drop_last=True,worker_init_fn=seed_worker,
            multiprocessing_context='spawn' if args.workers else None,
            generator=torch.Generator().manual_seed(args.seed))
        if len(loader)==0:
            raise ValueError('Empty drop_last loader')
        dest=args.output_dir/category/'train'
        identity=dict(protocol=p,category=category,stage='train',smoke=smoke)
        with shared.stage(dest,identity,device,args) as meter:
            json_write(dest/'run.json',dict(protocol=p,source=source,category=category,smoke=smoke,
                dataset_root=str(root),dtd_root=str(args.dtd_root.resolve()),device=str(device),
                workers=args.workers,worker_start_method='spawn' if args.workers else 'single_process',
                actual_student_steps=n_student,actual_segmentation_steps=n_seg,
                evaluation_during_training=False,final_checkpoint_rule='fixed final step; never best test score'))
            json_write(dest/'train_manifest.json',[dict(relative_path=x.relative_to(root).as_posix(),
                label=0,role='train_normal') for x in paths])
            json_write(dest/'dtd_manifest.json',[str(x.relative_to(args.dtd_root.resolve())) for x in textures])
            with meter.measure('official_model_initialization'):
                model=network.DeSTSeg(dest=True,ed=True).to(device)
                student_optimizer,seg_optimizer=optimizers(model)
            step=0
            iterator=None
            try:
                with (dest/'losses.csv').open('w',newline='',encoding='utf-8') as log:
                    writer=csv.DictWriter(log,fieldnames=['step','phase','cosine_loss','focal_loss','l1_loss','total_loss'])
                    writer.writeheader()
                    for phase,count in [('student',n_student),('segmentation',n_seg)]:
                        print(f'train {category} {phase}: {count} steps; batch={args.batch_size}; {device}',flush=True)
                        with meter.measure(phase+'_train'):
                            if iterator is None:
                                iterator=iter(loader)
                            begin=time.perf_counter()
                            for local_step in range(count):
                                try:
                                    batch=next(iterator)
                                except StopIteration:
                                    iterator=iter(loader)
                                    batch=next(iterator)
                                values=train_step(model,batch,student_optimizer,seg_optimizer,phase,device,losses)
                                step+=1
                                writer.writerow(dict(step=step,phase=phase,**values))
                                if local_step==0 or (local_step+1)%50==0 or local_step+1==count:
                                    log.flush()
                                    seconds=(time.perf_counter()-begin)/(local_step+1)
                                    print(f'{category} {phase} step={step}/{n_student+n_seg} '
                                          f'loss={values["total_loss"]:.6f} mean_step={seconds:.3f}s',flush=True)
                            print(f'{category} {phase}: compute finished; collecting resource statistics',flush=True)
                            # Dump once if phase finalization blocks; parent watchdog remains independent.
                            faulthandler.dump_traceback_later(120, repeat=False)
                        print(f'{category} {phase}: resource statistics saved',flush=True)
                        faulthandler.cancel_dump_traceback_later()
                        if phase=='student' and not smoke:
                            print(f'{category}: saving student checkpoint',flush=True)
                            with meter.measure('student_checkpoint_save'):
                                save_model(dest/'student_step1000.pckl',model)
                    filename='smoke_model.pckl' if smoke else f'DeSTSeg_{p["dataset"].upper()}_5000_{category}.pckl'
                    print(f'{category}: saving final checkpoint {filename}',flush=True)
                    faulthandler.dump_traceback_later(120, repeat=False)
                    with meter.measure('final_checkpoint_save'):
                        save_model(dest/filename,model)
                    json_write(dest/'checkpoint.json',dict(filename=filename,step=step,smoke=smoke,
                        protocol=p,selection='fixed_final_step',contains_optimizer_state=False))
                    finish_stage(dest,identity,step=step,smoke=smoke,training_normal_count=len(paths),
                                 dtd_count=len(textures),checkpoint=filename)
                    print(f'{category}: final checkpoint and complete.json saved',flush=True)
            finally:
                print(f'{category}: cleaning DataLoader workers and model resources',flush=True)
                faulthandler.dump_traceback_later(120, repeat=False)
                del model,student_optimizer,seg_optimizer,iterator,loader
                gc.collect()
                torch.cuda.empty_cache()
                faulthandler.cancel_dump_traceback_later()
                print(f'{category}: cleanup finished',flush=True)
        print(f'train {category} complete; smoke={smoke}',flush=True)


def trained(root,category,allow_smoke=False, *, protocol_factory=protocol):
    dest=root/category/'train'
    state=read_json(dest/'complete.json')
    p=state['identity']['protocol']
    expected=protocol_factory(p['training']['batch_size'],p['training']['seed'])
    if state['status']!='complete' or state['identity']['category']!=category or p!=expected:
        raise ValueError('Incomplete/incompatible dataset training')
    if state['smoke'] and not allow_smoke:
        raise ValueError('Smoke-trained checkpoint cannot be used for formal inference')
    if not state['smoke'] and state['step']!=5000:
        raise ValueError('Formal model must be fixed step5000')
    path=dest/state['checkpoint']
    if not path.is_file() or path.parent.resolve()!=dest.resolve():
        raise ValueError('Invalid final checkpoint path')
    return path,p,state['smoke']


def infer(args,categories, *, backend=None):
    listing = test_listing if backend is None else backend.test_listing
    make_mask = official_mask if backend is None else backend.official_mask
    make_protocol = protocol if backend is None else backend.protocol
    device=shared.device_setup(args.device)
    source=verify_source(args.official_root)
    root,groups,counts=listing(args.dataset_root,categories)
    ds=test_transforms(args.official_root)
    for category in categories:
        weight,p,train_smoke=trained(args.training_dir,category,allow_smoke=args.limit>0,protocol_factory=make_protocol)
        smoke=bool(args.limit or train_smoke)
        dest=args.output_dir/category/'predict'
        identity=dict(protocol=p,category=category,stage='predict',smoke=smoke)
        with shared.stage(dest,identity,device,args) as meter:
            json_write(dest/'run.json',dict(protocol=p,source=source,category=category,smoke=smoke,
                dataset_root=str(root),device=str(device),batch_size=1,training_dir=str(args.training_dir.resolve()),
                checkpoint=str(weight.resolve()),official_evaluation='not_run',unified_evaluation='not_run'))
            csv_write(dest/'data_counts.csv',[r for r in counts if r['category']==category])
            items=groups[category]
            indices=list(range(len(items)))
            if args.limit:
                priority=[next(i for i,r in enumerate(items) if r.label==label) for label in [0,1]]
                indices=sorted((priority+[i for i in indices if i not in priority])[:args.limit])
            with meter.measure('strict_checkpoint_load'):
                model=load_model(args.official_root,weight,device)
            json_write(dest/'checkpoint_load.json',dict(path=str(weight),load_status='strict_loaded',strict=True))
            rows=[]
            try:
                with meter.measure('inference_and_FP32_save'),torch.no_grad():
                    for number,index in enumerate(indices):
                        r=items[index]
                        (main,score),(aux,auxscore)=outputs(model(load_rgb(r.path,ds)[None].to(device)))
                        row=metadata(r,root,'test')
                        row['prediction_file']=f'maps/{number:06d}.npz'
                        mask=make_mask(r,ds)
                        save_prediction(dest,row,main[0].cpu().numpy(),aux[0].cpu().numpy(),
                                        score[0].item(),auxscore[0].item(),mask)
                        rows.append(row)
                        if (number+1)%25==0 or number+1==len(indices):
                            print(f'predict {p["dataset"]}/{category}: {number+1}/{len(indices)}',flush=True)
                json_write(dest/'samples.json',rows)
                csv_write(dest/'sample_scores.csv',rows)
                csv_write(dest/'label_audit.csv',[dict(relative_path=r['relative_path'],unified_label=r['label'],
                    official_label=r['official_label'],differs=r['image_label_differs']) for r in rows])
                finish_stage(dest,identity,image_count=len(rows),expected_full_count=len(items),
                             smoke=smoke,strict_weight_load=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def evaluate_or_summarize(args,categories, *, backend=None):
    make_protocol = protocol if backend is None else backend.protocol
    protocols=[read_json(args.output_dir/c/'predict/complete.json')['identity']['protocol'] for c in categories]
    p=protocols[0]
    if p!=make_protocol(p['training']['batch_size'],p['training']['seed']) or any(v!=p for v in protocols):
        raise ValueError('Cannot combine differing training protocols')
    if args.command=='unified':
        shared.unified(args,categories,protocol=p)
    elif args.command=='official':
        shared.official_eval(args,categories,protocol=p)
    else:
        shared.summarize(args,categories,protocol=p,dataset_name=p['dataset'])
