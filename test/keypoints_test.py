import sys
import os
import yaml
import queue
import threading as th
import time
import multiprocessing as mp

import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sensor import RealSense
from utils.utils import get_cam_points

from mask_tracker import MaskTrackerProcess
from keypoint_tracker import KeypointTrackerProcess

def main():
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mp.set_start_method('spawn')
    rs = RealSense(config['realsense'])
    rs.start()

    mask_tracker = MaskTrackerProcess(config['mask_tracker'])
    mask_tracker.start()

    data = None
    while data is None:
        data = rs.get()
    mask_tracker.send(data['color'])
    mask = mask_tracker.get()
    
    keypoint_tracker = KeypointTrackerProcess(config['keypoint_tracker'])
    if mask_tracker.sam_done():
        keypoint_tracker.start()

    keypoint_tracker.send({
        'rgb': cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB),
        'points': get_cam_points(data['depth'], config['realsense']['instrinsics']),
        'masks': mask,
    })
    keypoint_tracker.get()

    # data = None
    # while data is None:
    #     data = rs.get()
    # mask_tracker.send(data['color'])
    # mask = mask_tracker.get()
    # points = get_cam_points(data['depth'], config['realsense']['instrinsics'])
    # keypoint_tracker.send({
    #     'rgb': cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB),
    #     'points': points,
    #     'masks': mask,
    # })
    # keypoint_tracker.get()
    # cv2.imshow("Keypoints", projected)
    # cv2.waitKey(0)

    rs.stop()
    mask_tracker.stop()
    keypoint_tracker.stop()
    # exit()

if __name__ == "__main__":
    main()