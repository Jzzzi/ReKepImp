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
from mask_tracker import MaskTrackerProcess

def main():
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    rs = RealSense(config['realsense'])
    rs.start()


    mp.set_start_method('spawn')
    tracker_process = MaskTrackerProcess(config['mask_tracker'])
    tracker_process.start()

    while True:
        data = None
        while data is None:
            data = rs.get()
        tracker_process.send(data['color'])
        mask = tracker_process.get()

        # exlude the background 0
        objects = np.unique(mask)
        bgr = data['color']

        for obj in objects:
            mask_obj = (mask == obj)
            color_mask = np.zeros_like(bgr)
            color_mask[mask_obj] = [0, 255, 255]
            bgr = cv2.addWeighted(bgr, 1, color_mask, 0.5, 0)

        cv2.imshow("Mask", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs.stop()            
    tracker_process.stop()

if __name__ == "__main__":
    main()