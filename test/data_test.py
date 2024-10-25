from threading import Event, Thread
from queue import Queue
import yaml
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sensor import RealSense

def main():
    with open('config/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    realsense_config = config['realsense']

    rs = RealSense(realsense_config)
    rs.start()

    while True:
        data = None
        while data is None:
            data = rs.get()
        color = data['color']
        cv2.imshow('color', color)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    rs.stop()
if __name__ == '__main__':
    main()