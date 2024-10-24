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
from utils.sensor import RealSense, sensor_thread
from mask_tracker import MaskTrackerProcess

def main():
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    realsense_config = config["realsense"]

    sensor = RealSense(realsense_config)

    realsense_stream = queue.Queue()
    realsense_event = th.Event()
    realsense_thread = th.Thread(target=sensor_thread, args=(realsense_stream, realsense_event, sensor, 5, 10))
    realsense_thread.start()


    mp.set_start_method('spawn')
    tracker_process = MaskTrackerProcess(2)
    tracker_process.start()
    time.sleep(3)


    while True:
        data = realsense_stream.get()
        tracker_process.send(data)
        mask = tracker_process.get()

        # exlude the background 0
        objects = torch.unique(mask)[1:].tolist()
        mask = mask.cpu().numpy()
        bgr = data['color']

        for obj in objects:
            mask_obj = (mask == obj)
            color_mask = np.zeros_like(bgr)
            color_mask[mask_obj] = [0, 255, 255]
            bgr = cv2.addWeighted(bgr, 1, color_mask, 0.5, 0)

        cv2.imshow("Mask", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    realsense_event.set()
    realsense_thread.join()
    tracker_process.stop()

if __name__ == "__main__":
    main()