"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>
"""

import csv


class SCANNETBase:
    SPLITS = ['train', 'test']

    labels_numbers = []
    labels_names = []
    with open('/home/matteo/Code/Panoptic_Segmentation/rgbd/src/datasets/scannet/nyu_labels.txt') as f:
        fs = csv.reader(f, delimiter=' ')
        for line in fs:
            labels_numbers.append(int(line[0]))
            labels_names.append(line[1])

    labels_mapping = {}
    with open('/home/matteo/Code/Panoptic_Segmentation/rgbd/src/datasets/scannet/scannetv2-labels.combined.tsv') as f:
        rd = csv.reader(f, delimiter='\t')
        for i, row in enumerate(rd):
            if i > 0:
                scannet_label = int(row[0])
                nyu_label = int(row[4])
                if nyu_label in labels_numbers:
                    labels_mapping[scannet_label] = nyu_label

    # number of classes
    N_CLASSES = len(labels_numbers)

    CLASS_NAMES_ENGLISH = labels_names

    CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0), (255, 120, 70), (117, 41, 121),
                    (150, 255, 0), (132, 0, 255), (24, 209, 255),
                    (191, 130, 35), (219, 200, 109)]
