#!/usr/bin/python

import torch
import argparse
import sys
import os
import cv2
import numpy as np
import time
import demo_utils.vot as vot
from demo_utils.siamvggtracker import SiamVGGTracker

# =====================================================
#Try SiamFC:

#Clone this repo and run
#python demo.py --model SiamFC

#You can change --mdoel to other models like
# python demo.py --model SiamFCNext22
# =====================================================

# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************

#--------------------------------------------------------
# parameters
# data_dir =
#img_dir = './data/bag'
#gt_file = './data/bag/groundtruth.txt'

root_dir   = 'D:\\GeoData\\Videos\\Drone\\drone1'
img_folder = 'frames2'
gt_file    = 'groundtruth.txt'

root_dir   = 'D:\\GeoData\\Benchmark\\VIDEOS\\VTB\\Car24'
img_folder = 'img'
gt_file    = 'groundtruth_rect.txt'
#------------------------------------------------------

parser = argparse.ArgumentParser(description='PyTorch SiameseX demo')

parser.add_argument('--model', metavar='model', 
                    default='SiamFC', #default='SiamFCNext22', 
                    type=str,
                    help='which model to use.')

args = parser.parse_args()

handle = vot.VOT("rectangle", os.path.join(root_dir, img_folder), os.path.join(root_dir, gt_file))
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
tracker = SiamVGGTracker(args.model, imagefile, selection)

if not imagefile:
    sys.exit(0)

toc = 0

frame_count = 1
while True:
    # *****************************************
    # VOT: Call frame method to get path of the 
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************

    tic = cv2.getTickCount()

    imagefile = handle.frame()
    image = cv2.imread(imagefile)
    if not imagefile:
        break
    region, confidence = tracker.track(imagefile)
    toc += cv2.getTickCount() - tic

    region = vot.Rectangle(region.x, region.y, region.width, region.height)
    # *****************************************
    # VOT: Report the position of the object
    #      every frame using report method.
    # *****************************************
    handle.report(region, confidence)
    cv2.rectangle(image, (int(region.x), int(region.y)), (int(region.x + region.width), int(region.y + region.height)),
                  (0, 255, 0), 2)
    cv2.putText(image, 'Frame:%d' % frame_count, (20,20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, [0,0,255], thickness=1)
    cv2.imshow('SiameseX', image)
    frame_count += 1
    #cv2.waitKey(1)
    if cv2.waitKey(5) == 27:
        break

    print('Tracking Speed {:.1f}fps'.format((len(handle) - 1) / (toc / cv2.getTickFrequency())))

cv2.destroyAllWindows()
print('Tracking finished.')