# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from src.models.model_utils import SqueezeAndExcitation, Excitation


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + depth
        return out


class ExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ExciteFusionAdd, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                           activation=activation)
        self.se_depth = Excitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + depth
        return out


class ResidualExciteFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                           activation=activation)
        self.se_depth = Excitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb_se = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + rgb_se + depth
        return out


class ViTFlattener(nn.Module):

    def __init__(self, patch_dim):
        super(ViTFlattener, self).__init__()
        self.patch_dim = patch_dim
        self.patcher = torch.nn.PixelUnshuffle(self.patch_dim)
        self.flattener = torch.nn.Flatten(-2, -1)

    def forward(self, inp):
        patches = self.patcher(inp)
        flat = self.flattener(patches)
        ViT_out = flat
        return ViT_out


class ViTUnFlattener(nn.Module):

    def __init__(self, patch_dim):
        super(ViTUnFlattener, self).__init__()
        self.patch_dim = patch_dim
        self.unpatcher = torch.nn.PixelShuffle(self.patch_dim)

    def forward(self, inp, out_shape):
        _, C, H, W = out_shape
        x = inp
        x = x.reshape(-1, C * self.patch_dim * self.patch_dim, H // self.patch_dim, W // self.patch_dim)
        x = self.unpatcher(x)
        return x


class SelfAttentionFusion(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(SelfAttentionFusion, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_key_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler_1 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)
        self.upsampler_2 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)


    def forward(self, rgb, depth):
        # Self-Attention for RGB
        query_rgb = self.downsampler_query_1(rgb)
        key_rgb = self.downsampler_key_1(rgb)
        value_rgb = self.downsampler_value_1(rgb)
        flattened_query_rgb = self.vit_flatten(query_rgb)
        flattened_key_rgb = self.vit_flatten(key_rgb)
        flattened_value_rgb = self.vit_flatten(value_rgb)

        QKt_rgb = torch.matmul(flattened_query_rgb, flattened_key_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, flattened_value_rgb)
        output_rgb = self.vit_unflatten(output_rgb, query_rgb.shape)
        output_rgb = self.upsampler_1(output_rgb)

        # Self-Attention for Depth
        query_depth = self.downsampler_query_2(depth)
        key_depth = self.downsampler_key_2(depth)
        value_depth = self.downsampler_value_2(depth)
        flattened_query_depth = self.vit_flatten(query_depth)
        flattened_key_depth = self.vit_flatten(key_depth)
        flattened_value_depth = self.vit_flatten(value_depth)

        QKt_depth = torch.matmul(flattened_query_depth, flattened_key_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, flattened_value_depth)
        output_depth = self.vit_unflatten(output_depth, query_depth.shape)
        output_depth = self.upsampler_2(output_depth)

        # Merging
        output = output_rgb + output_depth
        return output


class ResidualAttentionFusion(nn.Module):
    def __init__(self, patches_size, channels, alpha=1., bottleneck_dim=32):
        super(ResidualAttentionFusion, self).__init__()

        self.alpha = alpha

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_key_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler_1 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)
        self.upsampler_2 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)


    def forward(self, rgb, depth):
        # Self-Attention for RGB
        query_rgb = self.downsampler_query_1(rgb)
        key_rgb = self.downsampler_key_1(rgb)
        value_rgb = self.downsampler_value_1(rgb)
        flattened_query_rgb = self.vit_flatten(query_rgb)
        flattened_key_rgb = self.vit_flatten(key_rgb)
        flattened_value_rgb = self.vit_flatten(value_rgb)

        QKt_rgb = torch.matmul(flattened_query_rgb, flattened_key_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, flattened_value_rgb)
        output_rgb = self.vit_unflatten(output_rgb, query_rgb.shape)
        output_rgb = self.upsampler_1(output_rgb)

        # Self-Attention for Depth
        query_depth = self.downsampler_query_2(depth)
        key_depth = self.downsampler_key_2(depth)
        value_depth = self.downsampler_value_2(depth)
        flattened_query_depth = self.vit_flatten(query_depth)
        flattened_key_depth = self.vit_flatten(key_depth)
        flattened_value_depth = self.vit_flatten(value_depth)

        QKt_depth = torch.matmul(flattened_query_depth, flattened_key_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, flattened_value_depth)
        output_depth = self.vit_unflatten(output_depth, query_depth.shape)
        output_depth = self.upsampler_2(output_depth)

        # Merging
        output = rgb + self.alpha * (output_rgb + output_depth)
        return output


class MHAttentionFusionSecond(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(MHAttentionFusion, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = channels
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim
        
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.linear_rgb_q = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_rgb_k = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_rgb_v = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_depth_q = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_depth_k = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.linear_depth_v = nn.Linear(in_features=self.latent_patch_dim, out_features=bottleneck_dim)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.linear_rgb = nn.Linear(in_features=bottleneck_dim, out_features=self.latent_patch_dim)
        self.linear_depth = nn.Linear(in_features=bottleneck_dim, out_features=self.latent_patch_dim)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)

    def forward(self, rgb, depth):

        # Self-Attention for RGB
        vit_rgb = self.vit_flatten(rgb)
        q_rgb = self.linear_rgb_q(vit_rgb)
        k_rgb = self.linear_rgb_k(vit_rgb)
        v_rgb = self.linear_rgb_v(vit_rgb)
        QKt_rgb = torch.matmul(q_rgb, k_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, v_rgb)
        output_rgb = self.linear_rgb(output_rgb)
        output_rgb = self.vit_unflatten(output_rgb, rgb.shape)

        # Self-Attention for Depth
        vit_depth = self.vit_flatten(depth)
        q_depth = self.linear_depth_q(vit_depth)
        k_depth = self.linear_depth_k(vit_depth)
        v_depth = self.linear_depth_v(vit_depth)
        QKt_depth = torch.matmul(q_depth, k_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, v_depth)
        output_depth = self.linear_depth(output_depth)
        output_depth = self.vit_unflatten(output_depth, depth.shape)

        # Merging
        output = output_rgb + output_depth
        return output


class MHAttentionFusionThird(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(MHAttentionFusion, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)

    def forward(self, rgb, depth):
        # Cross-Attention
        query = self.downsampler_query(depth)
        key = self.downsampler_key(rgb)
        value = self.downsampler_value(depth)

        flattened_query = self.vit_flatten(query)
        flattened_key = self.vit_flatten(key)
        flattened_value = self.vit_flatten(value)

        QKt = torch.matmul(flattened_query, flattened_key.permute(0, 2, 1)) / self.scale
        attention_weight = self.softmax(QKt)
        output = torch.matmul(attention_weight, flattened_value)
        output = self.vit_unflatten(output, query.shape)
        output = self.upsampler(output)
        return output
