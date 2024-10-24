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

    mask_tracker = MaskTrackerProcess(config['mask_tracker'])
    mask_tracker.start()
    keypoint_tracker = KeypointTrackerProcess(config['keypoint_tracker'])
    keypoint_tracker.start()

    mask_tracker.stop()
    keypoint_tracker.stop()

