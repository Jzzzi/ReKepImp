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
from utils.utils import get_points
from mask_tracker import MaskTrackerProcess

def main():
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    rs = RealSense(config['realsense'])
    rs.start()
    time.sleep(3)


    mp.set_start_method('spawn')
    tracker_process = MaskTrackerProcess(config['mask_tracker'], manual=True)
    tracker_process.start()

    obj_masks = [[0,0,255],
                 [0,255,0],
                 [255,0,0],
                 [0,255,255],
                 [255,0,255],
                 [255,255,0],]

    while True:
        data = None
        while data is None:
            data = rs.get()
        bgr = data['color']
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = data['depth']
        extrinsics = data['extrinsics']
        instrinsics = rs.get_instrinsics()
        points = get_points(depth, instrinsics, extrinsics)
        send_data = {
            "rgb": rgb,
            "points": points,
        }
        tracker_process.send(send_data)
        mask = tracker_process.get()

        # exlude the background 0
        objects = np.unique(mask)
        objects = objects[objects != 0]
        bgr = data['color']

        for obj in objects:
            mask_obj = (mask == obj)
            color_mask = np.zeros_like(bgr)
            color_mask[mask_obj] = obj_masks[obj]
            bgr = cv2.addWeighted(bgr, 1, color_mask, 0.5, 0)

        cv2.imshow("Mask", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs.stop()            
    tracker_process.stop()

if __name__ == "__main__":
    main()