import os
import random
import argparse


class ArgumentParser(argparse.ArgumentParser):
    def set_common_args(self):
        self.add_argument('--path', default='', type=str,
                          help='absolute path of folder where "scans" is')
        self.add_argument('--train_split', default=0.8, type=float,
                          help='percentage of data [0, 1] for training')


def split_dataset(data_dir, train_split):
    scan_fold = os.path.join(data_dir, 'scans/')
    scans = os.listdir(scan_fold)
    n_scans = len(scans)

    train_scans = int(train_split * n_scans)
    test_scans = n_scans - train_scans

    random.shuffle(scans)
    train = scans[:train_scans]
    test = scans[train_scans:]
    assert len(train) == train_scans
    assert len(test) == test_scans

    train = [os.path.join(scan_fold, scan) for scan in train]
    test = [os.path.join(scan_fold, scan) for scan in test]

    train_file = open(os.path.join(data_dir, 'train.txt'), 'w')
    train_file.write('/\n'.join(train))
    train_file.write('/')
    train_file.close()

    test_file = open(os.path.join(data_dir, 'test.txt'), 'w')
    test_file.write('/\n'.join(test))
    test_file.write('/')
    test_file.close()

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()

    data_dir = args.path
    train_split = args.train_split
    
    split_dataset(data_dir, train_split)
