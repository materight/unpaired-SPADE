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
accuracy_sum, intersection_sum, union_sum, batch_count = 0, 0, 0, 0

print('\nGenerating images...')
with torch.no_grad():
    for i, data_i in enumerate(tqdm(dataloader, desc='batch', total=opt.how_many if opt.how_many < np.inf else None)):
        if i * opt.batchSize >= opt.how_many:
            break

        generated, reconstructed_semantics = model(data_i, mode='inference')

        # evaluate generated images by predicting the semantic segmentation given in input to the generator
        segmentation_preds = None
        if opt.evaluate:
            segmentation_preds = segmenter(generated)
            segmentation_preds = segmentation_preds.argmax(dim=1).unsqueeze(1)
            preds, labels = segmentation_preds.detach().cpu(), data_i['label'].detach().cpu()
            preds[preds == opt.label_nc], labels[labels == opt.label_nc] = -1, -1  # convert back to original label values
            preds, labels = preds + 1, labels + 1
            # labels = torch.nn.functional.interpolate(labels.float(), size=preds.shape[2:], mode='nearest').long()  # downsample labels to match model output
            accuracy, _ = compute_accuracy(preds, labels)
            intersection, union = compute_intersection_and_union(preds, labels, opt.label_nc)
            accuracy_sum += accuracy
            intersection_sum += intersection
            union_sum += union
            batch_count += 1

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('reconstructed_semantic', reconstructed_semantics[b] if reconstructed_semantics is not None else None),
                                   ('segmentation_evaluation', segmentation_preds[b] if segmentation_preds is not None else None),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

    webpage.save()

if opt.evaluate:
    accuracy = accuracy_sum / batch_count
    iou = intersection_sum / (union_sum + 1e-10)
    miou = iou.mean()

    print('\nComputing FID score...')
    original_dataroot = f'{opt.dataroot}/images/validation'
    generated_dataroot = f'{results_dir}/images/synthesized_image'
    device = 'cpu' if opt.gpu_ids[0] == -1 else f'cuda:{opt.gpu_ids[0]}'
    process = subprocess.run(['python', '-m', 'pytorch_fid', original_dataroot, generated_dataroot, '--num-workers',
                             str(os.cpu_count()), '--device', str(device), '--batch-size', '1'], stdout=subprocess.PIPE)
    fid = float(re.search('FID:  ([\d.]*)', process.stdout.decode('utf8').strip()).group(1))

    print('\nResults:')
    print(f'accuracy: {accuracy*100:.2f}%')
    print(f'mIoU: {miou*100:.2f}')
    print(f'FID: {fid:.2f}')
    with open(f'{results_dir}/metrics.txt', 'w') as file:
        file.write(f'accuracy: {accuracy*100}\nmIoU: {miou*100}\nFID: {fid}')
