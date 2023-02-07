# -*- coding: utf-8 -*-
"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>

Parts of this code are taken and adapted from:
https://github.com/TUI-NICR/ESANet/blob/main/src/utils.py
"""


import os
import sys

import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=.01, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out, target):
        out = out.squeeze()
        target = target.squeeze()
        log_positives = (out[target != 0]).clamp(min=1e-4)
        log_negatives = (1 - out[target == 0]).clamp(min=1e-4)
        positives = torch.sum(-self.alpha * (1 - out[target != 0]) ** self.gamma * torch.log(log_positives))
        negatives = torch.sum(-(1 - self.alpha) * out[target == 0] ** self.gamma * torch.log(log_negatives))
        loss = positives + negatives
        return loss


class ComposedHingedLoss(nn.Module):
    # parameters provided by the paper: 0.1, 1.0, 1.0, 1.0, 0.001
    def __init__(self, delta_a=0.1, delta_r=1., alpha=1., beta=1., gamma=0.001):
        super(ComposedHingedLoss, self).__init__()
        self.delta_a = delta_a
        self.delta_r = delta_r
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # embeddings of pixel belonging to same instance should be close
    def AttractionLoss(self, out_batch, target_batch, centers_batch, batch_size, device):
        attraction_loss = torch.tensor(0.).to(device)
        for i in range(batch_size):
            out = out_batch[i].permute(1, 2, 0)  # size = 640*640*32: embedding for each pixel
            target = target_batch[i]  # size = 640*640: instance id of each pixel!
            centers = centers_batch[i]  # len = K: number of centers in the image. it's a list of tuples
            for center in centers:
                center_mask = target[center]
                points = (target == center_mask).nonzero(as_tuple=True)
                distances = out[center] - out[points]
                norms = torch.linalg.norm(distances, dim=1)
                arg = norms - self.delta_a
                hinged_arg = F.relu(arg)
                number_of_points = len(hinged_arg)
                if number_of_points > 1:    # avoid dividing by zero if center is the only point with that instance mask
                    number_of_points = number_of_points - 1
                attraction_loss += torch.sum(hinged_arg) / number_of_points

            if len(centers) >= 1:   # avoid dividing by zero if image has no centers (ie no instances)
                attraction_loss /= len(centers)

        return attraction_loss

    # embeddings of centers of different instances should be different
    def RepellingLoss(self, out_batch, centers_batch, batch_size, device):
        repelling_loss = torch.tensor(0.).to(device)
        for i in range(batch_size):
            out = out_batch[i].permute(1, 2, 0)
            centers = centers_batch[i]
            center_tensor = (torch.tensor(centers)).T
            if len(centers) > 1:
                for center in centers:
                    distance = out[center] - out[tuple(center_tensor)]
                    norm = torch.linalg.norm(distance, dim=1)
                    arg = self.delta_r - norm
                    hinged_arg = F.relu(arg)
                    repelling_loss += torch.sum(hinged_arg) - self.delta_r
                repelling_loss /= len(centers) * (len(centers) - 1)
        return repelling_loss

    # embeddings should not explode
    def RegularizationLoss(self, out_batch, centers_batch, batch_size, device):
        regularization_loss = torch.tensor(0.).to(device)
        for i in range(batch_size):
            out = out_batch[i].permute(1, 2, 0)
            centers = centers_batch[i]
            for center in centers:
                embedding = out[center]
                regularization_loss += torch.linalg.norm(embedding)
            if len(centers) > 0:
                regularization_loss /= len(centers)
        return regularization_loss

    def forward(self, out, target, centers, batch_size, device):
        attraction_loss = self.AttractionLoss(out, target, centers, batch_size, device)
        repelling_loss = self.RepellingLoss(out, centers, batch_size, device)
        regularization_loss = self.RegularizationLoss(out, centers, batch_size, device)
        loss = self.alpha * attraction_loss + self.beta * repelling_loss + self.gamma * regularization_loss
        return loss, attraction_loss, repelling_loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight)
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction='none')
        self.ce_loss.to(device)

    def forward(self, input, target):
        loss = self.ce_loss(input, target.long())
        number_of_pixels_per_class = torch.bincount(target.flatten().type(self.dtype), minlength=self.num_classes)
        divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class * self.weight).clamp(min=1e-5)
        loss = torch.sum(loss) / divisor_weighted_pixel_sum
        return loss, number_of_pixels_per_class, divisor_weighted_pixel_sum


class PanopticLoss(nn.Module):
    def __init__(self, weighting, device, batch_size):
        super(PanopticLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.semantic_loss = CrossEntropyLoss2d(self.device, weighting)
        self.center_loss = BinaryFocalLoss()
        self.embedding_loss = ComposedHingedLoss()

    def forward(self, output, semantic_target, center_target, instance_target, center_coordinates):
        semantic_output = output[0]
        center_output = output[1]
        embedding_output = output[2]

        sem_losses, number_of_pixels_per_class, divisor_weighted_pixel_sum = self.semantic_loss(semantic_output,
                                                                                                semantic_target[0])
        cen_loss = self.center_loss(center_output, center_target)
        emb_loss, attraction_loss, repelling_loss = self.embedding_loss(embedding_output, instance_target,
                                                                        center_coordinates, self.batch_size, self.device)

        return cen_loss, sem_losses, emb_loss, attraction_loss, repelling_loss, number_of_pixels_per_class, divisor_weighted_pixel_sum


def print_log(epoch, epochs, local_count, count_inter, dataset_size, loss, time_inter,
              learning_rates, sem_loss, cen_loss, emb_loss):
    print_string = 'Train Epoch: {:>3}/{:>3} [{:>4}/{:>4} ({: 5.1f}%)]'.format(
        epoch, epochs, local_count, dataset_size,
        100. * local_count / dataset_size)
    for i, lr in enumerate(learning_rates):
        print_string += '   lr_{}: {:>6}'.format(i, round(lr, 10))
    print_string += '   Loss: {:0.6f}'.format(loss.item())
    print_string += '   Loss_semseg: {:0.4f}'.format(sem_loss.item())
    print_string += '   Loss_center: {:0.4f}'.format(cen_loss.item())
    print_string += '   Loss_embed: {:0.4f}'.format(emb_loss.item())
    print_string += '  [{:0.2f}s every {:>4} data]'.format(time_inter,
                                                          count_inter)
    print(print_string, flush=True)


def save_ckpt(ckpt_dir, model, optimizer, epoch, metric):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "model_best_{}.pth".format(metric)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
                          best_miou_epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou': best_miou,
        'best_miou_epoch': best_miou_epoch
    }
    ckpt_model_filename = "ckpt_latest.pth"
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
            print('Best mIoU:', best_miou)
        else:
            best_miou = 0

        if 'best_miou_epoch' in checkpoint:
            best_miou_epoch = checkpoint['best_miou_epoch']
            print('Best mIoU epoch:', best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)


def get_best_checkpoint(ckpt_dir, key='mIoU_test'):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, 'logs.csv')
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        miou = data[key][idx]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pth')
    assert ckpt_path is not None, f'No trainings found at {ckpt_dir}'
    assert os.path.exists(ckpt_path), \
        f'There is no weights file named {ckpt_path}'
    print(f'Best mIoU: {100*miou:0.2f} at epoch: {epoch}')
    return ckpt_path


class BinaryFocalLossVal(nn.Module):
    def __init__(self, alpha=.01, gamma=2):
        super(BinaryFocalLossVal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out, target):
        out = out.squeeze()
        target = target.squeeze()
        log_positives = (out[target != 0]).clamp(min=1e-4)
        log_negatives = (1 - out[target == 0]).clamp(min=1e-4)
        positives = torch.sum(-self.alpha * (1 - out[target != 0]) ** self.gamma * torch.log(log_positives))
        negatives = torch.sum(-(1 - self.alpha) * out[target == 0] ** self.gamma * torch.log(log_negatives))
        loss = positives + negatives
        return loss


class ComposedHingedLossVal(nn.Module):
    # parameters provided by the paper: 0.1, 1.0, 1.0, 1.0, 0.001
    def __init__(self, delta_a=0.1, delta_r=1., alpha=1., beta=1., gamma=0.001):
        super(ComposedHingedLossVal, self).__init__()
        self.delta_a = delta_a
        self.delta_r = delta_r
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # embeddings of pixel belonging to same instance should be close
    def AttractionLossVal(self, out_batch, target_batch, centers_batch, batch_size, device):
        attraction_loss = torch.tensor(0.).to(device)
        batch_size = out_batch.shape[0]
        for i in range(batch_size):
            out = out_batch[i].permute(1, 2, 0)  # size = 640*640*32: embedding for each pixel
            target = target_batch[i]  # size = 640*640: instance id of each pixel!
            centers = centers_batch[i]  # len = K: number of centers in the image. it's a list of tuples
            for center in centers:
                center_mask = target[center]
                points = (target == center_mask).nonzero(as_tuple=True)
                distances = out[center] - out[points]
                norms = torch.linalg.norm(distances, dim=1)
                arg = norms - self.delta_a
                hinged_arg = F.relu(arg)
                number_of_points = len(hinged_arg)
                if number_of_points > 1:    # avoid dividing by zero if center is the only point with that instance mask
                    number_of_points = number_of_points - 1
                attraction_loss += torch.sum(hinged_arg) / number_of_points

            if len(centers) >= 1:   # avoid dividing by zero if image has no centers (ie no instances)
                attraction_loss /= len(centers)

        return attraction_loss

    # embeddings of centers of different instances should be different
    def RepellingLossVal(self, out_batch, centers_batch, batch_size, device):
        repelling_loss = torch.tensor(0.).to(device)
        batch_size = out_batch.shape[0]
        for i in range(batch_size):
            out = out_batch[i].permute(1, 2, 0)
            centers = centers_batch[i]
            center_tensor = (torch.tensor(centers)).T
            if len(centers) > 1:
                for center in centers:
                    distance = out[center] - out[tuple(center_tensor)]
                    norm = torch.linalg.norm(distance, dim=1)
                    arg = self.delta_r - norm
                    hinged_arg = F.relu(arg)
                    repelling_loss += torch.sum(hinged_arg) - self.delta_r
                repelling_loss /= len(centers) * (len(centers) - 1)
        return repelling_loss

    # embeddings should not explode
    def RegularizationLossVal(self, out_batch, centers_batch, batch_size, device):
        regularization_loss = torch.tensor(0.).to(device)
        batch_size = out_batch.shape[0]
        for i in range(batch_size):
            out = out_batch[i].permute(1, 2, 0)
            centers = centers_batch[i]
            for center in centers:
                embedding = out[center]
                regularization_loss += torch.linalg.norm(embedding)
            if len(centers) > 0:
                regularization_loss /= len(centers)
        return regularization_loss

    def forward(self, out, target, centers, batch_size, device):
        attraction_loss = self.AttractionLossVal(out, target, centers, batch_size, device)
        repelling_loss = self.RepellingLossVal(out, centers, batch_size, device)
        regularization_loss = self.RegularizationLossVal(out, centers, batch_size, device)
        loss = self.alpha * attraction_loss + self.beta * repelling_loss + self.gamma * regularization_loss
        return loss, attraction_loss, repelling_loss


class CrossEntropyLoss2dVal(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2dVal, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight)
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction='none')
        self.ce_loss.to(device)

    def forward(self, input, target):
        loss = self.ce_loss(input, target.long())
        number_of_pixels_per_class = torch.bincount(target.flatten().type(self.dtype), minlength=self.num_classes)
        divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class * self.weight).clamp(min=1e-5)
        loss = torch.sum(loss) / divisor_weighted_pixel_sum
        return loss, number_of_pixels_per_class, divisor_weighted_pixel_sum


class PanopticLossVal(nn.Module):
    def __init__(self, weighting, device, batch_size):
        super(PanopticLossVal, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.semantic_loss = CrossEntropyLoss2dVal(self.device, weighting)
        self.center_loss = BinaryFocalLossVal()
        self.embedding_loss = ComposedHingedLossVal()

    def forward(self, output, semantic_target, center_target, instance_target, center_coordinates):
        semantic_output = output[0]
        center_output = output[1]
        embedding_output = output[2]
        sem_losses, number_of_pixels_per_class, divisor_weighted_pixel_sum = self.semantic_loss(semantic_output,
                                                                                                semantic_target[0])
        cen_loss = self.center_loss(center_output, center_target)
        emb_loss, attraction_loss, repelling_loss = self.embedding_loss(embedding_output, instance_target,
                                                                        center_coordinates, self.batch_size, self.device)

        return cen_loss, sem_losses, emb_loss, attraction_loss, repelling_loss, number_of_pixels_per_class, divisor_weighted_pixel_sum
