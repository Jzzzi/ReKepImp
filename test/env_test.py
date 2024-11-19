import sys
import os
import yaml
import multiprocessing as mp
from threading import Thread

import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from enviroment import RealEnviroment

actions = [[0, 0, 0.4, 0, 0, 0, 1, 0],
           [-0.2, 0, 0.4, 0, 0, 0, 1, 1],
           [0, -0.2, 0.4, 0, 0, 0, 1, 1],
           [-0.2, -0.2, 0.4, 0, 0, 0, 1, 1]]


def image_show_thread():
    while True:
        keypoints, projeted = rw.get_keypoints()
        cv2.imshow('keypoints', projeted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    with open('./config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    rw = RealEnviroment(config)
    rw.reset()
    rw.register_keypoints()
    step = 0
    t = Thread(target=image_show_thread)
    t.start()
    while True:
        try:
            keypoints, projeted = rw.get_keypoints()
            rw.execute_action(actions[step], wait=True)
            step = (step + 1) % len(actions)
        except KeyboardInterrupt:
            break
    rw.execute_action([0, 0, 0.4, 0, 0, 0, 1, 0], wait=True)
    t.join()
    rw.stop()