from threading import Event, Thread
from queue import Queue
import yaml
import sys
import os
import cv2
import cv2 as cv
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sensor import RealSense

def main():
    with open('config/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    realsense_config = config['realsense']

    rs = RealSense(realsense_config)
    rs.start()

    while True:
        data = rs.get()
        color = data['color']
        cv.imshow('color', color)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    rs.stop()
if __name__ == '__main__':
    main()