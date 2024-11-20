import sys
import os
import yaml
import multiprocessing as mp
import threading

import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from enviroment import RealEnviroment

actions = [[0, 0, 0.4, 0, 0, 0, 1, 0],
           [-0.2, 0, 0.4, 0, 0, 0, 1, 1],
           [0, -0.2, 0.4, 0, 0, 0, 1, 1],
           [-0.2, -0.2, 0.4, 0, 0, 0, 1, 1]]


def image_show_thread(stop_event):
    while not stop_event.is_set():
        observation = rw.observe()
        projeted = observation['projected']
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
    stop_event = threading.Event()
    t = threading.Thread(target=image_show_thread, args=(stop_event,))
    t.start()
    while True:
        try:
            rw.execute_action(actions[step], wait=True)
            step = (step + 1) % len(actions)
        except KeyboardInterrupt:
            break
    rw.execute_action([0, 0, 0.4, 0, 0, 0, 1, 0], wait=True)
    stop_event.set()
    t.join()
    rw.stop()