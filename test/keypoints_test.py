import sys
import os
import yaml
import multiprocessing as mp

import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sensor import RealSense
from utils.utils import get_points

from mask_tracker import MaskTrackerProcess
from keypoint_tracker import KeypointTrackerProcess

def main():
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mp.set_start_method('spawn')
    rs = RealSense(config['realsense'])
    rs.start()

    obj_masks = [[0,0,255],
                 [0,255,0],
                 [255,0,0],
                 [0,255,255],
                 [255,0,255],
                 [255,255,0]]

    mask_tracker = MaskTrackerProcess(config['mask_tracker'])
    mask_tracker.start()

    data = None
    while data is None:
        data = rs.get()
    rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
    depth = data['depth']
    insrinsics = rs.get_instrinsics()
    extrinsics = data['extrinsics']
    points = get_points(depth, insrinsics, extrinsics)
    mask_tracker.send({
        'rgb': rgb,
        'points': points,
    })
    mask = mask_tracker.get()
    keypoint_tracker = KeypointTrackerProcess(config['keypoint_tracker'])
    keypoint_tracker.start()

    keypoint_tracker.send({
        'rgb': rgb,
        'points': points,
        'masks': mask,
    })
    result = keypoint_tracker.get()
    projected_init = result['projected']

    while True:
        data = None
        while data is None:
            data = rs.get()
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth']
        points = get_points(depth, insrinsics, extrinsics)
        mask_tracker.send({
        'rgb': rgb,
        'points': points,
        })
        mask = mask_tracker.get()
        keypoint_tracker.send({
            'rgb': rgb,
            'points': points,
            'masks': mask,
        })
        result = keypoint_tracker.get()
        projected = result['projected']
        ids = result['obj_ids']
    
        objects = np.unique(mask)
        objects = objects[objects != 0]
        for obj in objects:
            mask_obj = (mask == obj)
            color_mask = np.zeros_like(projected)
            color_mask[mask_obj] = obj_masks[obj]
            projected = cv2.addWeighted(projected, 1, color_mask, 0.5, 0)

        show = cv2.hconcat([projected_init, projected])
        show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
        cv2.imshow('Projected Keypoints', show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs.stop()
    mask_tracker.stop()
    keypoint_tracker.stop()
    # exit()

if __name__ == "__main__":
    main()