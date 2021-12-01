import os
import torch
import torchvision.transforms as T
from models.networks.base_network import BaseNetwork
from mit_semseg.models import ModelBuilder, SegmentationModule


'''
Class added by @materight
Models implementation taken from: https://github.com/CSAILVision/semantic-segmentation-pytorch
'''


def download_weigths_from_url(dst, base_path, model_name, encoder_filename, decoder_filename):
    model_weights_dirpath = f'{dst}/segmenter-{model_name}/'
    os.makedirs(model_weights_dirpath, exist_ok=True)
    encoder_weights_filepath = f'{model_weights_dirpath}/{encoder_filename}.pth'
    decoder_weights_filepath = f'{model_weights_dirpath}/{decoder_filename}.pth'
    if not os.path.exists(encoder_weights_filepath):
        torch.hub.download_url_to_file(url=f'{base_path}/{model_name}/{encoder_filename}.pth', dst=encoder_weights_filepath)
    if not os.path.exists(decoder_weights_filepath):
        torch.hub.download_url_to_file(url=f'{base_path}/{model_name}/{decoder_filename}.pth', dst=decoder_weights_filepath)
    return encoder_weights_filepath, decoder_weights_filepath


class BaseADE20KSegmenter(BaseNetwork):
    def __init__(self, opt, encoder_name, decoder_name, encoder_filname, decoder_filename, fc_dim):
        super().__init__()
        ADE20K_BASEPATH = 'http://sceneparsing.csail.mit.edu/model/pytorch/'
        self.model_name = f'ade20k-{encoder_name}-{decoder_name}'

        encoder_weigths_filepath, decoder_weigths_filepath = '', ''
        if opt.pretrained_seg:
            encoder_weigths_filepath, decoder_weigths_filepath = download_weigths_from_url(opt.checkpoints_dir, ADE20K_BASEPATH, self.model_name, encoder_filname, decoder_filename)

        encoder = ModelBuilder.build_encoder(arch=encoder_name, fc_dim=fc_dim, weights=encoder_weigths_filepath)
        decoder = ModelBuilder.build_decoder(arch=decoder_name, fc_dim=fc_dim, weights=decoder_weigths_filepath, use_softmax=True)
        crit = torch.nn.NLLLoss(ignore_index=opt.label_nc)
        self.segmentation_module = SegmentationModule(encoder, decoder, crit)

        self.normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, data, normalize=False):
        if normalize:
            data = self.normalization(data)
        #self.segmentation_module.use_softmax = self.training
        scores = self.segmentation_module({'img_data': data}, segSize=data.shape[2:])
        if isinstance(scores, tuple):
            scores, _ = scores
        scores = torch.roll(scores, 1, dims=1)
        return scores


class UperNet18Segmenter(BaseADE20KSegmenter):
    def __init__(self, opt):
        super().__init__(opt, 'resnet18dilated', 'ppm_deepsup', 'encoder_epoch_20', 'decoder_epoch_20', fc_dim=512)


class UperNet50Segmenter(BaseADE20KSegmenter):
    def __init__(self, opt):
        super().__init__(opt, 'resnet50', 'upernet', 'encoder_epoch_30', 'decoder_epoch_30', fc_dim=2048)


class UperNet101Segmenter(BaseADE20KSegmenter):
    def __init__(self, opt):
        super().__init__(opt, 'resnet101', 'upernet', 'encoder_epoch_50', 'decoder_epoch_50', fc_dim=2048)
