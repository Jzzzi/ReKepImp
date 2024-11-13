import os
import sys
import yaml
import time
import cv2
import multiprocessing as mp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import get_cam_points
from utils.sensor import RealSense
from keypoint_tracker import KeypointTrackerProcess

def main():
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mp.set_start_method('spawn')
    rs = RealSense(config['realsense'])
    rs.start()
    # wait for the camera to warm up
    time.sleep(3)

    tracker = KeypointTrackerProcess(config['keypoint_tracker'])
    tracker.start()

    left, top, right, bottom = 200, 100, 400, 300

    while True:
        data = None
        while data is None:
            data = rs.get()

        rgb_image = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        points = get_cam_points(data['depth'], config['realsense']['instrinsics'])
        masks = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        masks[top:bottom, left:right] = 1
        tracker.send({
            'rgb': rgb_image,
            'points': points,
            'masks': masks
        })

        result = tracker.get()
        projected = result['projected']
        projected = cv2.cvtColor(projected, cv2.COLOR_RGB2BGR)
        projected = cv2.rectangle(projected, (left, top), (right, bottom), (0, 0, 0), 2)

        cv2.imshow('Projected Keypoints', projected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs.stop()
    tracker.stop()

if __name__ == "__main__":
    main()