# -*- coding: utf-8 -*-
"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>

Part of this code is taken and adapted from:
https://github.com/TUI-NICR/ESANet/blob/main/src/build_model.py
"""

import warnings

import torch
from torch import nn

from src.models.model import ESANet
from src.models.model_one_modality import ESANetOneModality
from src.models.resnet import ResNet


def build_model(config, n_classes):
    paths, data, hyperparams, model_param, dataset, other = config
    if paths['LAST_CKPT']:
        pretrained_on_imagenet = False
    else:
        pretrained_on_imagenet = True

    # set the number of channels in the encoder and for the
    # fused encoder features
    channels_decoder = [512, 256, 128, 64, 32]

    if isinstance(model_param['N_DEC_BLOCKS'], int):
        nr_decoder_blocks = [model_param['N_DEC_BLOCKS']] * 5
    elif len(model_param['N_DEC_BLOCKS']) == 1:
        nr_decoder_blocks = model_param['N_DEC_BLOCKS'] * 5
    else:
        nr_decoder_blocks = model_param['N_DEC_BLOCKS']
        assert len(nr_decoder_blocks) == 5

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(other['GPU']))
    else:
        device = torch.device("cpu")

    if model_param['MODALITY'] == 'rgbd':
        # use the same encoder for depth encoder and rgb encoder if no
        # specific depth encoder is provided
        if model_param['BACKBONE_DEPTH'] in [None, 'None']:
            model_param['BACKBONE_DEPTH'] = model_param['BACKBONE']

        model = ESANet(
            height=data['HEIGHT'],
            width=data['WIDTH'],
            num_classes=n_classes,
            pretrained_on_imagenet=pretrained_on_imagenet,
            pretrained_dir=paths['PRETRAINED_DIR'],
            encoder_rgb=model_param['BACKBONE'],
            encoder_depth=model_param['BACKBONE_DEPTH'],
            encoder_block=model_param['ENC_BLOCK'],
            activation=model_param['ACTIVATION'],
            encoder_decoder_fusion=model_param['ENC_DEC_FUSION'],
            patches_size=model_param['PATCH_SIZE'],
            bottleneck_dim=model_param['BOTTLENECK_DIM'],
            context_module=model_param['CONTEXT_MODULE'],
            nr_decoder_blocks=nr_decoder_blocks,
            channels_decoder=channels_decoder,
            fuse_depth_in_rgb_encoder=model_param['ENCODERS_FUSION'],
            upsampling=model_param['UPSAMPLING'],
            device=device
        )

    else:  # just one modality
        print("Not implemented")
        exit()

    print('Device:', device)
    model.to(device)
    print(model)

    if model_param['HE_INIT']:
        module_list = []

        # first filter out the already pretrained encoder(s)
        for c in model.children():
            if pretrained_on_imagenet and isinstance(c, ResNet):
                # already initialized
                continue
            for m in c.modules():
                module_list.append(m)

        # iterate over all the other modules
        # output layers, layers followed by sigmoid (in SE block) and
        # depthwise convolutions (currently only used in learned upsampling)
        # are not initialized with He method
        for i, m in enumerate(module_list):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if m.out_channels == n_classes or \
                        isinstance(module_list[i+1], nn.Sigmoid) or \
                        m.groups == m.in_channels:
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Applied He init.')

    return model, device
