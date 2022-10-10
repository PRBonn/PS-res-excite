# -*- coding: utf-8 -*-

import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.post_processing import *
from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd, ExciteFusionAdd, ResidualExciteFusion, SelfAttentionFusion, ResidualAttentionFusion
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish


class ESANet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 patches_size='global',
                 bottleneck_dim=32,
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear',
                 device='cuda:0'):

        super(ESANet, self).__init__()
        self.device = device

        if channels_decoder is None:
            channels_decoder = [128, 128, 128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        h0, w0 = height // 2, width // 2
        h1, w1 = h0 // 2, w0 // 2
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2
        h4, w4 = h3 // 2, w3 // 2

        if patches_size == 'global':
            patch_sizes = [np.gcd(h4, w4)] * 5
        elif patches_size == 'layerwise':
            patch_sizes = [np.gcd(h0, w0), np.gcd(h1, w1), np.gcd(h2, w2), np.gcd(h3, w3), np.gcd(h4, w4)]

        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        elif self.fuse_depth_in_rgb_encoder == 'SelfAttention':
            self.se_layer0 = SelfAttentionFusion(patch_sizes[0], self.encoder_rgb.down_2_channels_out, bottleneck_dim)
            self.se_layer1 = SelfAttentionFusion(patch_sizes[1], self.encoder_rgb.down_4_channels_out, bottleneck_dim)
            self.se_layer2 = SelfAttentionFusion(patch_sizes[2], self.encoder_rgb.down_8_channels_out, bottleneck_dim)
            self.se_layer3 = SelfAttentionFusion(patch_sizes[3], self.encoder_rgb.down_16_channels_out, bottleneck_dim)
            self.se_layer4 = SelfAttentionFusion(patch_sizes[4], self.encoder_rgb.down_32_channels_out, bottleneck_dim)
        elif self.fuse_depth_in_rgb_encoder == 'ResidualAttention':
            self.se_layer0 = ResidualAttentionFusion(patch_sizes[0], self.encoder_rgb.down_2_channels_out, bottleneck_dim)
            self.se_layer1 = ResidualAttentionFusion(patch_sizes[1], self.encoder_rgb.down_4_channels_out, bottleneck_dim)
            self.se_layer2 = ResidualAttentionFusion(patch_sizes[2], self.encoder_rgb.down_8_channels_out, bottleneck_dim)
            self.se_layer3 = ResidualAttentionFusion(patch_sizes[3], self.encoder_rgb.down_16_channels_out, bottleneck_dim)
            self.se_layer4 = ResidualAttentionFusion(patch_sizes[4], self.encoder_rgb.down_32_channels_out, bottleneck_dim)
        elif self.fuse_depth_in_rgb_encoder == 'excite':
            self.se_layer0 = ExciteFusionAdd(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = ExciteFusionAdd(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = ExciteFusionAdd(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = ExciteFusionAdd(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = ExciteFusionAdd(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        elif self.fuse_depth_in_rgb_encoder == 'ResidualExcite':
            self.se_layer0 = ResidualExciteFusion(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = ResidualExciteFusion(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = ResidualExciteFusion(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = ResidualExciteFusion(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = ResidualExciteFusion(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        else:
            if self.fuse_depth_in_rgb_encoder != 'add':
                print('WARNING! You passed an invalid RGB + D fusion. Sum will be used!')
                self.fuse_depth_in_rgb_encoder == 'add'

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling

        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = SmallDecoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, rgb_input, depth_input):
        if rgb_input is None:
            rgb_input = torch.zeros(depth_input.shape[0], 3, 480, 640).to(self.device)
        if depth_input is None:
            depth_input = torch.zeros(rgb_input.shape[0], 1, 480, 640).to(self.device)

        rgb = self.encoder_rgb.forward_first_conv(rgb_input)
        depth = self.encoder_depth.forward_first_conv(depth_input)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)

        out = self.context_module(fuse)
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        return out


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        # Semantic Segmentation modules
        self.sem_decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.sem_decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.sem_decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        """self.sem_conv_out = nn.Conv2d(channels_decoder[2],
                                      num_classes, kernel_size=3, padding=1)

        self.sem_upsample1 = Upsample(mode=upsampling_mode,
                                      channels=num_classes)
        self.sem_upsample2 = Upsample(mode=upsampling_mode,
                                      channels=num_classes)"""
        self.sem_decoder_module_4 = DecoderModule(
            channels_in=channels_decoder[2],
            channels_dec=channels_decoder[3],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[3],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.sem_decoder_module_5 = DecoderModule(
            channels_in=channels_decoder[3],
            channels_dec=channels_decoder[4],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[4],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        out_channels = channels_decoder[4]

        self.sem_conv_out = nn.Conv2d(out_channels,
                                      num_classes, kernel_size=3, padding=1)


        # Center Prediction modules
        self.obj_decoder_module_1 = ObjDecoderModule(channels_in=channels_in,
                                                     channels_dec=channels_decoder[0],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[0],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_decoder_module_2 = ObjDecoderModule(channels_in=channels_decoder[0],
                                                     channels_dec=channels_decoder[1],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[1],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_decoder_module_3 = ObjDecoderModule(channels_in=channels_decoder[1],
                                                     channels_dec=channels_decoder[2],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[2],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_decoder_module_4 = ObjDecoderModule(channels_in=channels_decoder[2],
                                                     channels_dec=channels_decoder[3],
                                                     activation=activation,
                                                     nr_decoder_blocks=1,
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_decoder_module_5 = ObjDecoderModule(channels_in=channels_decoder[3],
                                                     channels_dec=channels_decoder[4],
                                                     activation=activation,
                                                     nr_decoder_blocks=1,
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_conv_out = nn.Conv2d(channels_decoder[4], 1, 1)

        # Embedding Prediction modules
        self.emb_decoder_module_1 = ObjDecoderModule(channels_in=channels_in,
                                                     channels_dec=channels_decoder[0],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[0],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_decoder_module_2 = ObjDecoderModule(channels_in=channels_decoder[0],
                                                     channels_dec=channels_decoder[1],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[1],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_decoder_module_3 = ObjDecoderModule(channels_in=channels_decoder[1],
                                                     channels_dec=channels_decoder[2],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[2],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_decoder_module_4 = ObjDecoderModule(channels_in=channels_decoder[2],
                                                     channels_dec=channels_decoder[3],
                                                     activation=activation,
                                                     nr_decoder_blocks=1,
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_decoder_module_5 = ObjDecoderModule(channels_in=channels_decoder[3],
                                                     channels_dec=channels_decoder[4],
                                                     activation=activation,
                                                     nr_decoder_blocks=1,
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)
        self.emb_conv_out = nn.Conv2d(channels_decoder[4], 32, 1)

        # dropout module
        # self.dropout = nn.Dropout2d(p=0.1)


    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        sem_out = self.sem_decoder_module_1(enc_out, enc_skip_down_16)
        sem_out = self.sem_decoder_module_2(sem_out, enc_skip_down_8)
        sem_out = self.sem_decoder_module_3(sem_out, enc_skip_down_4)
        sem_out = self.sem_decoder_module_4(sem_out, torch.tensor(0))
        sem_out = self.sem_decoder_module_5(sem_out, torch.tensor(0))
        # sem_out = self.dropout(sem_out)
        sem_out = self.sem_conv_out(sem_out)

        obj_out = self.obj_decoder_module_1(enc_out, enc_skip_down_16)
        obj_out = self.obj_decoder_module_2(obj_out, enc_skip_down_8)
        obj_out = self.obj_decoder_module_3(obj_out, enc_skip_down_4)
        obj_out = self.obj_decoder_module_4(obj_out, torch.tensor(0))
        obj_out = self.obj_decoder_module_5(obj_out, torch.tensor(0))
        # obj_out = self.dropout(obj_out)
        obj_out = self.obj_conv_out(obj_out)
        obj_out = torch.sigmoid(obj_out)

        emb_out = self.emb_decoder_module_1(enc_out, enc_skip_down_16)
        emb_out = self.emb_decoder_module_2(emb_out, enc_skip_down_8)
        emb_out = self.emb_decoder_module_3(emb_out, enc_skip_down_4)
        emb_out = self.emb_decoder_module_4(emb_out, torch.tensor(0))
        emb_out = self.emb_decoder_module_5(emb_out, torch.tensor(0))
        # emb_out = self.dropout(emb_out)
        emb_out = self.emb_conv_out(emb_out)

        # if self.training:
        #     return sem_out, sem_out_down_8, sem_out_down_16, sem_out_down_32, obj_out, emb_out
        return sem_out, obj_out, emb_out


class SmallDecoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        # Semantic Segmentation modules
        self.sem_decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.sem_decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.sem_decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.sem_conv_out = nn.Conv2d(channels_decoder[2],
                                      num_classes, kernel_size=3, padding=1)

        self.sem_upsample1 = Upsample(mode=upsampling_mode,
                                      channels=num_classes)
        self.sem_upsample2 = Upsample(mode=upsampling_mode,
                                      channels=num_classes)


        # Center Prediction modules
        self.obj_decoder_module_1 = ObjDecoderModule(channels_in=channels_in,
                                                     channels_dec=channels_decoder[0],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[0],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_decoder_module_2 = ObjDecoderModule(channels_in=channels_decoder[0],
                                                     channels_dec=channels_decoder[1],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[1],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_decoder_module_3 = ObjDecoderModule(channels_in=channels_decoder[1],
                                                     channels_dec=channels_decoder[2],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[2],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.obj_conv_out = nn.Conv2d(channels_decoder[2],
                                      1, kernel_size=3, padding=1)

        self.obj_upsample1 = Upsample(mode=upsampling_mode,
                                      channels=1)
        self.obj_upsample2 = Upsample(mode=upsampling_mode,
                                      channels=1)

        # Embedding Prediction modules
        self.emb_decoder_module_1 = ObjDecoderModule(channels_in=channels_in,
                                                     channels_dec=channels_decoder[0],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[0],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_decoder_module_2 = ObjDecoderModule(channels_in=channels_decoder[0],
                                                     channels_dec=channels_decoder[1],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[1],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_decoder_module_3 = ObjDecoderModule(channels_in=channels_decoder[1],
                                                     channels_dec=channels_decoder[2],
                                                     activation=activation,
                                                     nr_decoder_blocks=nr_decoder_blocks[2],
                                                     encoder_decoder_fusion=encoder_decoder_fusion,
                                                     upsampling_mode=upsampling_mode)

        self.emb_conv_out = nn.Conv2d(channels_decoder[2],
                                      32, kernel_size=3, padding=1)

        self.emb_upsample1 = Upsample(mode=upsampling_mode,
                                      channels=32)
        self.emb_upsample2 = Upsample(mode=upsampling_mode,
                                      channels=32)

        # dropout module
        # self.dropout = nn.Dropout2d(p=0.1)


    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        sem_out = self.sem_decoder_module_1(enc_out, enc_skip_down_16)
        sem_out = self.sem_decoder_module_2(sem_out, enc_skip_down_8)
        sem_out = self.sem_decoder_module_3(sem_out, enc_skip_down_4)
        sem_out = self.sem_conv_out(sem_out)
        sem_out = self.sem_upsample1(sem_out)
        sem_out = self.sem_upsample2(sem_out)

        obj_out = self.obj_decoder_module_1(enc_out, enc_skip_down_16)
        obj_out = self.obj_decoder_module_2(obj_out, enc_skip_down_8)
        obj_out = self.obj_decoder_module_3(obj_out, enc_skip_down_4)
        obj_out = self.obj_conv_out(obj_out)
        obj_out = self.obj_upsample1(obj_out)
        obj_out = self.obj_upsample2(obj_out)
        obj_out = torch.sigmoid(obj_out)

        emb_out = self.emb_decoder_module_1(enc_out, enc_skip_down_16)
        emb_out = self.emb_decoder_module_2(emb_out, enc_skip_down_8)
        emb_out = self.emb_decoder_module_3(emb_out, enc_skip_down_4)
        emb_out = self.emb_conv_out(emb_out)
        emb_out = self.emb_upsample1(emb_out)
        emb_out = self.emb_upsample2(emb_out)

        # if self.training:
        #     return sem_out, sem_out_down_8, sem_out_down_16, sem_out_down_32, obj_out, emb_out
        return sem_out, obj_out, emb_out


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        # self.side_output = nn.Conv2d(channels_dec,
        #                              num_classes,
        #                              kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        # if self.training:
        #     out_side = self.side_output(out)
        # else:
        #     out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features

        return out #, out_side


class ObjDecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)
        out = self.upsample(out)
        if self.encoder_decoder_fusion == 'add':
            out += encoder_features
        return out


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


def main():
    height = 480
    width = 640

    model = ESANet(
        height=height,
        width=width)

    print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)


if __name__ == '__main__':
    main()
