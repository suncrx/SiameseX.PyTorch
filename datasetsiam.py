# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:14:15 2022

@author: Renxi Chen
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *

# Data folder structure
# VOT2018 Dataset------------------------------------
# ---root_dir
# -----|-folder1
#------|   |-img
#------|   |   |-000001.jpg
#------|   |   |-000002.jpg
#------|   |   |- ....
#------|   |-groundtruth.txt
# -----|-folder2
#------|   |-img
#------|   |   |-000001.jpg
#------|   |   |-000002.jpg
#------|   |   |- ....
#------|   |-groundtruth.txt
# Groundtruth.txt format
# bounding_box_format = 1
# x-let-top, y-left-top, width, height
# e.g.
# 556.0,203.0,222.0,209.0
# 555.0,202.0,222.0,209.0

class DatasetSiam(Dataset):
    def __init__(self, root_dir, data_type='RPN', bounding_box_format=1,
                 transform=None,  train=False,
                 batch_size=1, num_workers=4, rpnpp=False,
                 max_interval = 50):
        self.coco = 0
        self.data_type = data_type
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rpnpp = rpnpp

        # max interval between a pair of frames
        self.max_interval = max_interval
        self.bounding_box_format = bounding_box_format

        # video sequence list
        # [ [ [[imgframe1],[x1,y1,w,h]], [[imgframe2],[x1,y1,w,h]],  ... ],  first sequence
        #   []   second sequence
        #   ...
        # ]
        self.video_seqs = []

        self.search_video_seqs(root_dir)

    # search all video sequences from root_dir
    def search_video_seqs(self, root_dir):
        if not os.path.exists(root_dir):
            print('ERROR: Can not find directory: ' + root_dir)

        #print('Searching video sequences ...')
        items = os.listdir(root_dir)
        for item in items:
            path = os.path.join(root_dir, item)
            if os.path.isdir(path):
                #print('Sequence ' + item)
                # check sub-folder 'img' and 'groundtruth.txt' file.
                imgdir = os.path.join(path, 'img')
                gt_filepath = os.path.join(path, 'groundtruth.txt')
                if os.path.exists(imgdir) and os.path.isdir(imgdir) and os.path.exists(gt_filepath):
                    #print(gt_filepath)
                    # list images in 'img' folder
                    imgfiles = os.listdir(imgdir)
                    imgPaths = [os.path.join(imgdir, x) for x in imgfiles]
                    #print(pathlist)

                    # read regions from 'groundtruth.txt'
                    regions = []
                    is_valid = []
                    with open(gt_filepath,'r') as gtfile:
                        while True:
                            strs = gtfile.readline().split(',')
                            # at least 4 numbers in each line, if len(str) is
                            # less than 4, it reaches the end of the file.
                            if len(strs) < 4:
                                break

                            reg = list(map(float, strs))
                            regions.append(reg)
                            # A region is invalid if 'nan' is found in the strs
                            if 'nan' in strs:
                                is_valid.append(False)
                            else:
                                is_valid.append(True)

                    # create list to store the image and region,
                    # removing invalid regions and corresponding frames
                    seq = []
                    for i in range(min(len(imgPaths), len(regions))):
                        if is_valid[i]:
                            seq.append([imgPaths[i], regions[i]])
                    # store current sequence
                    self.video_seqs.append(seq)
                else:
                    print('WARNING: Check the files in ' + imgdir)

        #print('Total video sequences: ', len(self.video_seqs))


    def print_video_seqs(self):
        #for i in range(len(self.img_files)):
        #    print(self.img_files[i], self.lab_regions[i])
        print(self.video_seqs)

    # get frame pairs from sequence[index]
    def _get_pairs(self, index):
        assert index < len(self.video_seqs), 'index exceeds the range of sequence'
        seq = self.video_seqs[index]
        nframes = len(seq)
        idx1 = random.randint(0, nframes-1)
        idx2 = idx1 + random.randint(10, self.max_interval)
        idx2 = min(idx2, nframes-1)
        return [seq[idx1], seq[idx2]]

    def __len__(self):
        return len(self.video_seqs)


    def __getitem__(self, index):
        assert index < len(self.video_seqs), 'index range error'

        pair_infos = self._get_pairs(index)

        if self.data_type == 'NORPN':
            z, x, template, gt_box = load_data(pair_infos, self.bounding_box_format)
            if self.transform is not None:
                z = self.transform(z)
                x = self.transform(x)
            template = torch.from_numpy(template)
            gt_box = torch.from_numpy(gt_box)
            return z, x, template, gt_box

        elif self.data_type == 'RPN':
            z, x, gt_box, regression_target, label = load_data_rpn(pair_infos, self.bounding_box_format,
                                                                   rpnpp=self.rpnpp)
            if self.transform is not None:
                z = self.transform(z)
                x = self.transform(x)
            regression_target = torch.from_numpy(regression_target)
            label = torch.from_numpy(label)
            return z, x, regression_target, label


#--------------------------------------------------
if __name__ == '__main__':
    ds = DatasetSiam('D:\\GeoData\\Benchmark\\VIDEOS\\VOT2018', data_type='NORPN',
                     bounding_box_format=1)

    print('Sequence: ', len(ds))
    for iter in range(100):
        print('Iteration: ', iter)
        for i in range(len(ds)):
            d = ds[i]

    print('OK')