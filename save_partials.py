import skimage.io as io
import matplotlib.pyplot as plt
from src.post_processing import *


def save_partials(input, outputs, sem_targets, cen_out, cen_target, embedding_pred, instance_mask, epoch, iteration, sample_number, out_folder, camera='camera1', mod='train'):
    image = input.cpu().detach().permute(1, 2, 0)

    if mod == 'train':
        instance_mask = instance_mask.cpu().detach()
        mask = sem_targets.cpu().detach()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(image)
        ax1.title.set_text('input')
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(mask)
        ax2.title.set_text('target')
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(outputs[0].cpu().detach())
        ax3.title.set_text('output')

        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_semantic.png'
        fig.savefig(out_folder + name)
        plt.close(fig)

        name2 = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_semantic_mask.png'
        io.imsave(out_folder + name2, outputs[0].cpu().detach())

        center = cen_target.cpu().detach()
        center_out = cen_out.cpu().detach().squeeze()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(center)
        ax1.title.set_text('target')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(center_out)
        ax2.title.set_text('output')

        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_center' + '.png'
        fig.savefig(out_folder + name)
        plt.close(fig)

        center_confidences_new, instance_pred, n_inst = get_instance(sem_targets,
                                                                     cen_out.squeeze(),
                                                                     embedding_pred)
        # instance_pred = post_processing(outputs, embedding_pred)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(instance_mask)
        ax1.title.set_text('target')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(instance_pred.cpu().detach())
        ax2.title.set_text('output')
        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_instance' + '.png'

        fig.savefig(out_folder + name)
        plt.close(fig)

        name2 = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_instance_mask.png'
        io.imsave(out_folder + name2, instance_pred.cpu().detach())

    elif mod == 'val':
        sem_out = outputs.cpu().detach()
        cen_out = cen_out.cpu().detach().squeeze()
        emb_out = embedding_pred.cpu().detach()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(sem_targets.cpu().detach())
        ax1.title.set_text('input')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(sem_out)
        ax2.title.set_text('semantic_segmentation')

        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_semantic_out_val.png'
        fig.savefig(out_folder + '/' + name)
        plt.close(fig)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(instance_mask.cpu().detach())
        ax1.title.set_text('input')
        ax2 = fig.add_subplot(1, 2, 2)
        _, instance_pred, _ = get_instance(sem_out, cen_out, emb_out)

        ax2.imshow(instance_pred.cpu().detach())
        ax2.title.set_text('instance_segmentation')

        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_instance_out.png'
        fig.savefig(out_folder + '/' + name)
        plt.close(fig)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cen_target.cpu().detach())
        ax1.title.set_text('input')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(cen_out.cpu().detach())
        ax2.title.set_text('center_pred')

        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_instance_out_val.png'
        fig.savefig(out_folder + '/' + name)
        plt.close(fig)

        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_semantic_out.png'
        io.imsave(out_folder + '/' + name, sem_out.cpu().detach().numpy().astype('uint8'))
        name = 'epoch' + str(epoch) + '_it' + str(iteration) + '_img' + str(sample_number) + '_instance_out.png'
        io.imsave(out_folder + '/' + name, instance_pred.cpu().detach().numpy().astype('uint8'))

        return instance_pred
    else:
        raise NotImplementedError('Only train and val are supported. Got '+mod+'.')

