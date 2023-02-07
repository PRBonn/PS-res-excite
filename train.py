# -*- coding: utf-8 -*-
"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>
"""

from datetime import datetime
import os
import time
import yaml
import random
import numpy as np

import torch.nn.functional as F
import torch.optim
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import IoU

from save_partials import *
from src.build_model import build_model
from src import utils
from src.post_processing import *
from src.prepare_data import prepare_data
from src.utils import save_ckpt, save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log
from src.logger import CSVLogger
from src.panoptic_quality import PanopticQuality

# torch.autograd.set_detect_anomaly(True)


def get_config():
    with open('src/config.yaml') as config:
        params = yaml.safe_load(config)
    paths = params['PATHS']
    data = params['DATA']
    hyperparams = params['HYPERPARAMS']
    model_param = params['MODEL']
    dataset = params['DATASET']
    other = params['OTHERS']

    return paths, data, hyperparams, model_param, dataset, other


def train_main():
    config = get_config()
    paths, data, hyperparams, model_param, dataset, other = config

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(paths['RESULTS_DIR'], dataset['DATASET'], paths['RESULTS_FOLDER'],
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    # when using multi scale supervision the label needs to be downsampled.
    label_downsampling_rates = [8, 16, 32]

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(config, ckpt_dir)

    train_loader, valid_loader = data_loaders
    
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    n_classes = n_classes_without_void
    if hyperparams['CLASS_WEIGHT'] != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(weight_mode=hyperparams['CLASS_WEIGHT'])
    else:
        class_weighting = np.ones(n_classes)

    # model building -----------------------------------------------------------
    model, device = build_model(config, n_classes=n_classes)

    if other['CHECK_MODEL'] == True:
        from torchsummary import summary
        summary(model)
        exit()

    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions
    loss_function_train = utils.PanopticLoss(class_weighting, device, data['BATCH_SIZE'])
    loss_function_val = utils.PanopticLossVal(class_weighting, device, data['BATCH_SIZE_VALID'])

    # in this script lr_scheduler.step() is only called once per epoch
    opt = hyperparams['OPTIMIZER']
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['LR'], weight_decay=hyperparams['W_DECAY'])
    elif opt == 'radam':
        optimizer = RAdam(model.parameters(), lr=hyperparams['LR'])
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=hyperparams['LR'], momentum=hyperparams['MOMENTUM'],
                              weight_decay=hyperparams['W_DECAY'], nesterov=True)
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=hyperparams['LR'], weight_decay=hyperparams['W_DECAY'])
    else:
        print('Optimizer {} is not implemented. Current choices: adam, radam, adamw, sgd!'.format(opt))
        exit()

    sched = hyperparams['SCHEDULER']
    if sched == 'cosine_annealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    elif sched == 'onecycle':
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=hyperparams['LR'],
            total_steps=hyperparams['EPOCHS'], #len(train_loader) // data['BATCH_SIZE'],
            div_factor=25,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )
    elif sched == 'exponential':
        lr_scheduler = ExponentialLR(optimizer, 0.9)
    elif sched == 'none':
        lr_scheduler = ExponentialLR(optimizer, 1)
    else:
        print('Scheduler {} is not implemented. Current choices: cosine_annealing, onecycle, exponential, none!'.format(sched))
        exit()

    # load checkpoint if parameter last_ckpt is provided
    if paths['LAST_CKPT']:
        ckpt_path = os.path.join(ckpt_dir, paths['LAST_CKPT'])

        epoch_last_ckpt, best_miou, best_miou_epoch = \
            load_ckpt(model, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    valid_split = valid_loader.dataset.split

    # build the log keys for the csv log file and for the web logger
    log_keys = [f'mIoU_{valid_split}']

    log_keys_for_csv = log_keys.copy()

    log_keys_for_csv.append('epoch')
    for i in range(len(lr_scheduler.get_lr())):
        log_keys_for_csv.append('lr_{}'.format(i))
    log_keys_for_csv.extend(['loss_train_total', 'loss_train_full_size'])
    log_keys_for_csv.append('loss_valid')
    log_keys_for_csv.extend(['loss_centers', 'loss_embeddings'])
    log_keys_for_csv.extend(['time_training', 'time_validation',
                             'time_forward',
                             'time_post_processing', 'time_copy_to_gpu'])

    valid_names = [valid_split]

    for valid_name in valid_names:
        # iou for every class
        for i in range(n_classes):
            log_keys_for_csv.append(f'IoU_{valid_name}_class_{i}')
            log_keys_for_csv.append(f'PQ_{valid_name}_class_{i+1}')
        log_keys_for_csv.append(f'PQ_{valid_name}')

    csvlogger = CSVLogger(log_keys_for_csv, os.path.join(ckpt_dir, 'logs.csv'),
                          append=True)

    # panoptic quality init
    best_pq = 0.
    best_pq_epoch = 0

    # start training -----------------------------------------------------------
    monitoring_images = []
    monitoring_images_val = []

    writer = SummaryWriter('runs/'+paths['RESULTS_FOLDER'])

    if data['ROBUST']:
        print('Input invariance is activated. The network will randomly drop either RGB or D at training time!')

    train_losses = [[], [], [], [], [], []]
    val_loss = []

    for epoch in range(int(start_epoch), hyperparams['EPOCHS']):
        # unfreeze
        for param in model.parameters():
            param.requires_grad = True

        logs = train_one_epoch(
            model, train_loader, device, optimizer, loss_function_train, epoch,
            lr_scheduler, model_param['MODALITY'], label_downsampling_rates, hyperparams['EPOCHS'],
            os.path.join(paths['RESULTS_DIR'], 'train', paths['RESULTS_FOLDER'] + '/'),
            monitoring_images, writer, train_losses, data['ROBUST'], config, debug_mode=other['DEBUG'])
        
        # validation after every epoch -----------------------------------------
        miou, logs = validate(
            model, valid_loader, device,
            model_param['MODALITY'], loss_function_val, logs,
            ckpt_dir, epoch, hyperparams['EPOCHS'], os.path.join(paths['RESULTS_DIR'], 'val', paths['RESULTS_FOLDER'] + '/'),
            monitoring_images_val, writer, val_loss, hyperparams['PQ_WAIT'], config, debug_mode=other['DEBUG'])

        writer.flush()

        logs.pop('time', None)
        csvlogger.write_logs(logs)

        # save weights if enabled by yaml
        save_current_checkpoint_miou = False
        save_current_checkpoint_pq = False

        if other['SAVE_CKPT']:
            if miou > best_miou:
                best_miou = miou
                best_miou_epoch = epoch
                save_current_checkpoint_miou = True

            pq_epoch = logs[f'PQ_{valid_split}']
            if pq_epoch > best_pq:
                best_pq = pq_epoch
                best_pq_epoch = epoch
                save_current_checkpoint_pq = True

            # save / overwrite latest weights (useful for resuming training)
            save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
                                  best_miou_epoch)

        # don't save weights for the first 10 epochs as mIoU/PQ are likely getting
        # better anyway
        if epoch >= 100 and save_current_checkpoint_miou is True:
            save_ckpt(ckpt_dir, model, optimizer, epoch, 'miou')
        if epoch >= 100 and save_current_checkpoint_pq is True:
            save_ckpt(ckpt_dir, model, optimizer, epoch, 'pq')

        print('\n')

    # write a finish file with best miou values in order overview
    # training result quickly
    with open(os.path.join(ckpt_dir, 'finished.txt'), 'w') as f:
        f.write('best miou: {}\n'.format(best_miou))
        f.write('best miou epoch: {}\n'.format(best_miou_epoch))
        f.write('best pq: {}\n'.format(best_pq))
        f.write('best pq epoch: {}\n'.format(best_pq_epoch))

    print("Training completed ")


def train_one_epoch(model, train_loader, device, optimizer, loss_function_train,
                    epoch, lr_scheduler, modality, label_downsampling_rates, epochs, out_folder,
                    monitoring_images, writer, train_losses, robust, config, debug_mode=False):
    paths, data, hyperparams, model_param, dataset, other = config

    training_start_time = time.time()
    samples_of_epoch = 0

    # set model to train mode
    model.train()

    # loss for every resolution
    losses_list = []

    # summed loss of all resolutions
    total_loss_list = []

    control_image = 1
    lr_scheduler.step()
    
    for i, sample in enumerate(train_loader):

        start_time_for_one_step = time.time()
       
        if ~robust:
            # load the data and send them to gpu
            if modality in ['rgbd', 'rgb', 'rgbd_single']:
                image = sample['image'].to(device)
                image_copy = torch.clone(image)
                batch_size = image.data.shape[0]
            if modality in ['rgbd', 'depth', 'rgbd_single']:
                depth = sample['depth'].to(device)
                batch_size = depth.data.shape[0]
            if modality in ['rgbd_single']:
                rgbd_data = torch.zeros((batch_size, 4, image.shape[2], image.shape[3])).to(device)
                rgbd_data[:, :3, :, :] = image
                depth = depth.squeeze()
                rgbd_data[:, 3, :, :] = depth
            filenames = sample['name']
            target_scales = [sample['label'].to(device)]
            instance_label = sample['instance'].to(device)
            center_mask = sample['center_gt'].to(device)
            center_coordinates = compute_center_coordinates(center_mask)
        
        # if robustness to input is activated, the network randomly drops either rgb or d at training time
        if robust:
            image_flag, depth_flag = reduce_data()
            image, depth = None, None
            if image_flag:
                image = sample['image'].to(device)
                image_copy = torch.clone(image)
                batch_size = image.data.shape[0]
            if depth_flag:
                depth = sample['depth'].to(device)
                batch_size = depth.data.shape[0]
            filenames = sample['name']
            target_scales = [sample['label'].to(device)]
            instance_label = sample['instance'].to(device)
            center_mask = sample['center_gt'].to(device)
            center_coordinates = compute_center_coordinates(center_mask)

        # if there's any nan depth value, we set it to max range + 1m
        try:
            assert not torch.any(torch.isnan(depth))
        except AssertionError:
            with torch.no_grad():
                depth_meters = torch.unique(depth)
                depth_meters = [x for x in depth_meters if ~torch.isnan(x)]
                max_valid_depth = depth_meters[-1]
                depth[torch.isnan(depth)] = max_valid_depth + 1

        # if there's any nan color value, we set it to 0
        try:
            assert not torch.any(torch.isnan(image))
        except AssertionError:
            image[torch.isnan(image)] = 0

        train = 0
        for item in center_coordinates:
            if not item:
                pass
            else:
                train = 1
                break
        if train:
            filename = sample['name']
            if epoch == 0:
                if i < 2:
                    monitoring_images += filename
                if i == 0:
                    num_samples = int(sample['samples'][0])
                    print('\nNumber of samples available for training:', num_samples)

            if len(label_downsampling_rates) > 0:
                for rate in sample['label_down']:
                    target_scales.append(sample['label_down'][rate].to(device))

            # optimizer.zero_grad()
            # this is more efficient than optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            # forward pass
            if modality == 'rgbd':
                pred_scales = model(image, depth)
            elif modality == 'rgb':
                pred_scales = model(image)
            elif modality == 'depth':
                pred_scales = model(depth)
            elif modality == 'rgbd_single':
                pred_scales = model(rgbd_data)

            # figures
            if epoch > data['SAVE_IMG_TRAIN'] and epoch % data['SAVE_IMG_EVERY'] == 0:
                for j in range(batch_size):
                    file = filename[j]
                    if file in monitoring_images:
                        img = image_copy[j]
                        sem_masks = [mask[j] for mask in target_scales]
                        sem_outputs = [torch.argmax(torch.softmax(output[j], dim=0), dim=0) for output in pred_scales]
                        cen_masks = center_mask[j]
                        cen_output = pred_scales[-2][j]
                        embedding_pred = pred_scales[-1][j]
                        instance_mask = instance_label[j]
                        save_partials(img, sem_outputs, sem_masks[0], cen_output, cen_masks,
                                      embedding_pred, instance_mask, epoch, i, j, out_folder)
                        control_image += 1

            # loss computation center_target, instance_target, center_coordinates
            cen_loss, sem_loss, emb_loss, attraction_loss, repelling_loss, number_of_pixels_per_class, divisor_weighted_pixel_sum = loss_function_train(pred_scales,
                                             target_scales,
                                             center_mask,
                                             instance_label,
                                             center_coordinates)

            loss = cen_loss + sem_loss + emb_loss
            losses = []

            losses.append(sem_loss)
            losses.append(cen_loss)
            losses.append(emb_loss)
            loss.backward()

            optimizer.step()

            for name, param in model.named_parameters():
                # if 'obj_decoder_module' not in name:
                #     continue
                try:
                    grad = param.grad.norm().cpu().detach().numpy()
                except AttributeError:
                    print(name)
                if np.isnan(grad):
                    print('\n\n\n GRADIENT IS NAN!!')
                    print(name, grad, 'nan')
                    print(filenames)
                    import ipdb; ipdb.set_trace()
                    break
            
            train_losses[0].append(loss)
            train_losses[1].append(sem_loss)
            train_losses[2].append(cen_loss)
            train_losses[3].append(emb_loss)
            train_losses[4].append(attraction_loss)
            train_losses[5].append(repelling_loss)
            writer.add_scalars('Loss/train', {'TrainingLoss': train_losses[0][-1]}, epoch)
            writer.add_scalars('Loss/semantic', {'CrossEntropyLoss': train_losses[1][-1]}, epoch)
            writer.add_scalars('Loss/centers', {'BinaryFocalLoss': train_losses[2][-1]}, epoch)
            writer.add_scalars('Loss/embeddings', {'ComposedHingedLoss':train_losses[3][-1],
                                                   'AttractiveLoss': train_losses[4][-1],
                                                   'RepellingLoss': train_losses[5][-1]}, epoch)

            # append loss values to the lists. Later we can calculate the
            # mean training loss of this epoch
            losses_list.append([loss.cpu().detach().numpy() for loss in losses])
            total_loss = loss.cpu().detach().numpy()
            total_loss_list.append(loss.item())

            if np.isnan(total_loss):
                print('Losses: ', losses)
                print('Number of pixels per class: ', number_of_pixels_per_class)
                print('Divisor: ', divisor_weighted_pixel_sum)
                print('Images: ', filenames)
                import ipdb; ipdb.set_trace()
                raise ValueError('Loss is None')

            samples_of_epoch += batch_size
            time_inter = time.time() - start_time_for_one_step

            learning_rates = lr_scheduler.get_lr()

            print_log(epoch, epochs, samples_of_epoch, batch_size,
                      len(train_loader.dataset), total_loss, time_inter,
                      learning_rates, sem_loss, cen_loss, emb_loss)

            if debug_mode:
                # only one batch while debugging
                break

    # fill the logs for csv log file and web logger
    logs = dict()
    logs['time_training'] = time.time() - training_start_time
    logs['loss_train_total'] = np.mean(total_loss_list)
    losses_train = np.mean(losses_list, axis=0)
    logs['loss_train_full_size'] = losses_train[0]
    logs['loss_centers'] = losses_train[-2]
    logs['loss_embeddings'] = losses_train[-1]
    logs['epoch'] = epoch
    for i, lr in enumerate(learning_rates):
        logs['lr_{}'.format(i)] = lr
    return logs


def validate(model, valid_loader, device,
             modality, loss_function_valid, logs, ckpt_dir, epoch,
             epochs, out_folder, monitoring_images_val, writer, val_loss,
             pq_wait, config,
             add_log_key='', debug_mode=False):
    paths, data, hyperparams, model_param, dataset, other = config

    valid_split = valid_loader.dataset.split + add_log_key
    PQ_wait_epochs = pq_wait
    PQ = PanopticQuality(device=device)

    print(f'Validation on {valid_split}')

    # we want to track how long each part of the validation takes
    validation_start_time = time.time()

    forward_time = 0
    copy_to_gpu_time = 0
    times = []

    # set model to eval mode
    model.eval()

    print(f'Validation: {len(valid_loader.dataset)} samples')

    control_image = 1
    pq_val = []

    compute_iou = IoU(num_classes=data['SEM_CLASSES'], reduction='none').to(device)
    camera_iou_with_count = {}
    camera_iou_without_count = {}
    iou_camera = 0.
    valid_labels = 0

    miou_list = []
    pq_list = []

    for i in range(data['SEM_CLASSES']):
        camera_iou_with_count[i] = [0., 0]
        camera_iou_without_count[i] = 0.

    loss_valid = 0

    for i, sample in enumerate(valid_loader):
        # copy the data to gpu
        copy_to_gpu_time_start = time.time()
        if modality in ['rgbd', 'rgb', 'rgbd_single']:
            image = sample['image'].to(device)
            batch_size = image.data.shape[0]
        if modality in ['rgbd', 'depth', 'rgbd_single']:
            depth = sample['depth'].to(device)
            batch_size = image.data.shape[0]
        if modality in ['rgbd_single']:
                rgbd_data = torch.zeros((batch_size, 4, image.shape[2], image.shape[3])).to(device)
                rgbd_data[:, :3, :, :] = image
                depth = depth.squeeze()
                rgbd_data[:, 3, :, :] = depth
        filename = sample['name']

        instance_predictions = torch.zeros((batch_size, image.shape[2], image.shape[3])).to(device)
        computed_instances = False

        if epoch == 0:
            if i < 2:
                monitoring_images_val += filename

        if not device.type == 'cpu':
            torch.cuda.synchronize()
        copy_to_gpu_time += time.time() - copy_to_gpu_time_start

        # forward pass
        with torch.no_grad():
            forward_time_start = time.time()
            if modality == 'rgbd':
                prediction = model(image, depth)
            elif modality == 'rgb':
                prediction = model(image)
            elif modality == 'depth':
                prediction = model(depth)
            elif modality == 'rgbd_single':
                prediction = model(rgbd_data)
            sem_out, obj_out, emb_out = prediction

            if not device.type == 'cpu':
                torch.cuda.synchronize()
            forward_time += time.time() - forward_time_start

            sem_out_batch = torch.argmax(torch.softmax(sem_out, dim=1), dim=1)
            label = sample['label'].int().to(device)
            centers = sample['center_gt'].to(device)
            instance = sample['instance'].to(device)
            inst_semantic = sample['inst_semantic'].to(device)
            center_coordinates = compute_center_coordinates(centers)

            ious = compute_iou(torch.tensor(sem_out_batch).unsqueeze(0), label)
            for lab, iou_ in enumerate(ious):
                if lab in torch.unique(label) and lab > 0:
                    prev_count = camera_iou_with_count[lab][1]
                    current_count = prev_count + 1
                    camera_iou_with_count[lab][1] = current_count
                    prev_iou = camera_iou_with_count[lab][0]
                    camera_iou_with_count[lab][0] = (prev_iou * prev_count + iou_) / current_count
                    camera_iou_without_count[lab] = (prev_iou * prev_count + iou_) / current_count

            if epoch > data['SAVE_IMG_VALID'] and epoch % data['SAVE_IMG_EVERY'] == 0:
                for j in range(batch_size):
                    file = filename[j]
                    if file in monitoring_images_val:
                        img = image[j]
                        sem_out_batch = torch.argmax(torch.softmax(sem_out[j], dim=0), dim=0)
                        cen_out_batch = obj_out[j]
                        embedding_pred = prediction[-1][j]
                        instance_pred = save_partials(img, sem_out_batch, label[j], cen_out_batch, centers[j],
                                                      embedding_pred, instance[j], epoch, i, j, out_folder, mod='val')
                        instance_predictions[j] = instance_pred
                        control_image += 1
                        computed_instances = True

            # panoptic quality
            if epoch >= PQ_wait_epochs:
                sem_out_batch = torch.argmax(torch.softmax(sem_out, dim=1), dim=1)
                cen_out_batch = obj_out.squeeze()
                embedding_pred = prediction[-1]
                if batch_size == 1:
                    cen_out_batch = cen_out_batch.unsqueeze(0)

                if not computed_instances:
                    for j in range(batch_size):
                        start = time.time()
                        
                        _, instance_pred, _ = get_instance(sem_out_batch[j],
                                                           cen_out_batch[j],
                                                           embedding_pred[j])
                        # instance_pred = post_processing(sem_out_batch[j], embedding_pred[j])
                        end = time.time()
                        times.append(end-start)
                        instance_predictions[j] = instance_pred

                panoptic_quality, panoptic_qualities = PQ.panoptic_quality_forward(sem_out_batch,
                                                                                   inst_semantic,
                                                                                   instance_predictions,
                                                                                   instance)
                pq_val.append(panoptic_quality)

            cen_loss, sem_loss, emb_loss, attraction_loss, repelling_loss, _, _ = loss_function_valid(
                prediction,
                [label],
                centers,
                instance,
                center_coordinates)
            loss_valid += cen_loss + sem_loss + emb_loss

            if debug_mode:
                # only one batch while debugging
                break

    for label in camera_iou_with_count:
        if camera_iou_with_count[label][1] > 0:
            iou_camera += camera_iou_with_count[label][0]
            valid_labels += 1
    miou = iou_camera / valid_labels

    # After all examples of camera are passed through the model,
    # we can compute miou and ious and pq
    print(f'mIoU {valid_split}: {miou}')

    if epoch > PQ_wait_epochs:
        panoptic_quality_epoch = 0.
        for entry in pq_val:
            panoptic_quality_epoch += entry
        panoptic_quality_epoch /= len(pq_val)
        print(f'PQ {valid_split}: {panoptic_quality_epoch}')
    else:
        panoptic_quality_epoch = -1

    miou_list.append(miou)
    pq_list.append(panoptic_quality_epoch)
    writer.add_scalars('Metrics/mIoU', {'MeanIntersectionOverUnion': miou_list[-1]}, epoch)
    writer.add_scalars('Metrics/PQ', {'PanopticQuality': pq_list[-1]}, epoch)

    validation_time = time.time() - validation_start_time

    logs[f'mIoU_{valid_split}'] = miou.item()
    logs[f'PQ_{valid_split}'] = panoptic_quality_epoch

    logs['time_validation'] = validation_time
    logs['time_forward'] = forward_time
    logs['time_copy_to_gpu'] = copy_to_gpu_time

    logs[f'loss_valid'] = loss_valid
    val_loss.append(loss_valid)
    writer.add_scalars('Loss/validation', {'ValidationLoss': val_loss[-1]}, epoch)

    print('Validation loss:', loss_valid.item())

    # write iou value of every class to logs
    for i in camera_iou_without_count:
        try:
            logs[f'IoU_{valid_split}_class_{i}'] = camera_iou_without_count[i].item()
        except AttributeError:
            logs[f'IoU_{valid_split}_class_{i}'] = camera_iou_without_count[i]

    if epoch >= PQ_wait_epochs:
        for entry in panoptic_qualities:
            sem_class = panoptic_qualities[entry]
            if sem_class['count'] > 0:
                pq_value = sem_class['pq']
            else:
                pq_value = -1
            logs[f'PQ_{valid_split}_class_{entry}'] = pq_value

    # log post-processing time
    postproc_time = np.mean(times)
    logs['time_post_processing'] = postproc_time

    return miou, logs


def reduce_data():
    dropping = random.random() > 0.5
    image, depth = 1, 1
    if dropping:
        drop_rgb = random.random() > 0.5
        if drop_rgb:
            image = 0
        else:
            depth = 0
    return image, depth


def compute_center_coordinates(center_tensor):
    center_coordinates = []
    batch_size = center_tensor.shape[0]

    for i in range(batch_size):
        center_batch = []
        centers = torch.where(center_tensor[i] >= 1. - 1e-3, )  # extract coordinates of points == 1
        xc, yc = centers[0], centers[1]

        assert len(xc) == len(yc)

        for j in range(len(xc)):
            # add tuple to list
            cen = (int(xc[j]), int(yc[j]))
            center_batch.append(cen)

        center_coordinates.append(center_batch)

    return center_coordinates

if __name__ == '__main__':
    train_main()
