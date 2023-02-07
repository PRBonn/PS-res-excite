"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>

Parts of this code are taken and adapted from:
https://github.com/TUI-NICR/ESANet/blob/main/src/prepare_data.py
"""

import os
import pickle

from torch.utils.data import DataLoader

from src import preprocessing
from src.datasets import SUNRGBD
from src.datasets import ScanNet
from src.datasets import HyperSim


def prepare_data(config, ckpt_dir=None, with_input_orig=False, split=None):
    paths, data, hyperparams, model_param, dataset, other = config
    train_preprocessor_kwargs = {}
    if dataset['DATASET'] == 'sunrgbd':
        Dataset = SUNRGBD
        dataset_kwargs = {}
        valid_set = 'test'
    elif dataset['DATASET'] == 'scannet':
        Dataset = ScanNet
        dataset_kwargs = {}
        valid_set = 'test'
    elif dataset['DATASET'] == 'hypersim':
        Dataset = HyperSim
        dataset_kwargs = {}
        valid_set = 'test'
    else:
        raise ValueError(f"Unknown dataset: `{dataset['DATASET']}`")

    if split in ['valid', 'test']:
        valid_set = split

    # train data
    train_data = Dataset(
        data_dir=dataset['DATASET_DIR'],
        split='train',
        with_input_orig=with_input_orig,
        debug=data['DEBUG'],
        **dataset_kwargs
    )

    train_preprocessor = preprocessing.get_preprocessor(
        height=data['HEIGHT'],
        width=data['WIDTH'],
        depth_mean=train_data.depth_mean,
        depth_std=train_data.depth_std,
        phase='train',
        **train_preprocessor_kwargs
    )
    train_data.preprocessor = train_preprocessor

    if ckpt_dir is not None:
        pickle_file_path = os.path.join(ckpt_dir, 'depth_mean_std.pickle')
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                depth_stats = pickle.load(f)
            print(f'Loaded depth mean and std from {pickle_file_path}')
            print(depth_stats)
        else:
            # dump depth stats
            depth_stats = {'mean': train_data.depth_mean,
                           'std': train_data.depth_std}
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(depth_stats, f)
    else:
        depth_stats = {'mean': train_data.depth_mean,
                       'std': train_data.depth_std}

    # valid data
    valid_preprocessor = preprocessing.get_preprocessor(
        height=data['HEIGHT'],
        width=data['WIDTH'],
        depth_mean=depth_stats['mean'],
        depth_std=depth_stats['std'],
        phase='test'
    )

    valid_data = Dataset(
        data_dir=dataset['DATASET_DIR'],
        split=valid_set,
        with_input_orig=with_input_orig,
        debug=data['DEBUG'],
        **dataset_kwargs
    )

    valid_data.preprocessor = valid_preprocessor

    if dataset['DATASET_DIR'] is None:
        # no path to the actual data was passed -> we cannot create dataloader,
        # return the valid dataset and preprocessor object for inference only
        return valid_data, valid_preprocessor

    # create the data loaders
    train_loader = DataLoader(train_data,
                              batch_size=data['BATCH_SIZE'],
                              num_workers=other['WORKERS'],
                              drop_last=True,
                              shuffle=True)

    # for validation we can use higher batch size as activations do not
    # need to be saved for the backwards pass
    batch_size_valid = data['BATCH_SIZE_VALID'] or data['BATCH_SIZE']
    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size_valid,
                              num_workers=other['WORKERS'],
                              shuffle=False)

    return train_loader, valid_loader
