"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import subprocess
import re
import torch
import numpy as np
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

from mit_semseg.utils import intersectionAndUnion as compute_intersection_and_union, accuracy as compute_accuracy
from models.networks.segmenter import UperNet101Segmenter


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

segmenter = UperNet101Segmenter(opt).to(opt.gpu_ids[0])
segmenter.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
results_dir = os.path.join(opt.results_dir, opt.name,
                           '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(results_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
accuracy_gen_sum, intersection_gen_sum, union_gen_sum = 0, 0, 0
accuracy_seg_sum, intersection_seg_sum, union_seg_sum = 0, 0, 0
batch_count = 0


def evaluate_iou_accu(model, images, label):
    segmentation_preds = model(images)
    segmentation_preds = segmentation_preds.argmax(dim=1).unsqueeze(1)
    preds, labels = segmentation_preds.detach().cpu(), label.detach().cpu()
    preds[preds == opt.label_nc], labels[labels == opt.label_nc] = -1, -1  # convert back to original label values
    preds, labels = preds + 1, labels + 1
    accuracy, _ = compute_accuracy(preds, labels)
    intersection, union = compute_intersection_and_union(preds, labels, opt.label_nc)
    return segmentation_preds, accuracy, intersection, union


print('\nGenerating images...')
with torch.no_grad():
    for i, data_i in enumerate(tqdm(dataloader, desc='batch', total=opt.how_many if opt.how_many < np.inf else None)):
        if i * opt.batchSize >= opt.how_many:
            break

        generated, reconstructed_semantics = model(data_i, mode='inference')

        # evaluate generated images by predicting the semantic segmentation given in input to the generator
        gen_segmentation_preds, orig_segmentation_preds = None, None
        if opt.evaluate:
            batch_count += 1
            # Evaluate generator
            gen_segmentation_preds, accuracy, intersection, union = evaluate_iou_accu(segmenter, generated, data_i['label'])
            accuracy_gen_sum, intersection_gen_sum, union_gen_sum = accuracy_gen_sum + accuracy, intersection_gen_sum + intersection, union_gen_sum + union
            # Evaluate netS
            if model.netS is not None:
                orig_segmentation_preds, accuracy, intersection, union = evaluate_iou_accu(model.netS, data_i['image'], data_i['label'])
                accuracy_seg_sum, intersection_seg_sum, union_seg_sum = accuracy_seg_sum + accuracy, intersection_seg_sum + intersection, union_seg_sum + union

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('external_segmenter_from_generated', gen_segmentation_preds[b] if gen_segmentation_preds is not None else None),
                                   ('internal_segmenter_from_orginal', orig_segmentation_preds[b] if orig_segmentation_preds is not None else None),
                                   ('internal_segmenter_from_generated', reconstructed_semantics[b] if reconstructed_semantics is not None else None),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

    webpage.save()

if opt.evaluate:
    # Compute mIoU, accuracy and FID
    gen_accuracy = accuracy_gen_sum / batch_count
    gen_iou = intersection_gen_sum / (union_gen_sum + 1e-10)
    gen_miou = gen_iou.mean()

    if model.netS is not None:
        seg_accuracy = accuracy_seg_sum / batch_count
        seg_iou = intersection_seg_sum / (union_seg_sum + 1e-10)
        seg_miou = seg_iou.mean()

    print('\nComputing FID score...')
    original_dataroot = f'{opt.dataroot}/images/validation'
    generated_dataroot = f'{results_dir}/images/synthesized_image'
    device = 'cpu' if opt.gpu_ids[0] == -1 else f'cuda:{opt.gpu_ids[0]}'
    process = subprocess.run(['python', '-m', 'pytorch_fid', original_dataroot, generated_dataroot, '--num-workers', str(opt.nThreads), '--device', str(device), '--batch-size', '1'], stdout=subprocess.PIPE)
    fid = float(re.search('FID:  ([\d.]*)', process.stdout.decode('utf8').strip()).group(1))

    # Print and save results in file
    result_str = '' + \
        f'Generator accuracy: {gen_accuracy*100:.4f}\n' + \
        f'Generator mIoU: {gen_miou*100:.4f}\n' + \
        f'Generator FID: {fid:.4f}\n'
    if model.netS is not None:
        result_str += '' + \
            f'Segmenter accuracy: {seg_accuracy*100:.4f}\n' + \
            f'Segmenter mIoU: {seg_miou*100:.4f}\n'

    print('\nResults:')
    print(result_str)

    with open(f'{results_dir}/metrics.txt', 'w') as file:
        file.write(result_str)
