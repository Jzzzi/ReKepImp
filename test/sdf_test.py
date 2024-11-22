import sys
import os
import yaml
import multiprocessing as mp
import threading

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from enviroment import RealEnviroment

actions = [[0, 0, 0.4, 0, 0, 0, 1, 0],
           [-0.2, 0, 0.4, 0, 0, 0, 1, 1],
           [0, -0.2, 0.4, 0, 0, 0, 1, 1],
           [-0.2, -0.2, 0.4, 0, 0, 0, 1, 1]]


stop_event = threading.Event()
def control_arm_thread():
    # print('start control arm thread')
    step = 0
    while not stop_event.is_set():
        try:
            rw.execute_action(actions[step], wait=True)
            step = (step + 1) % len(actions)
        except KeyboardInterrupt:
            break
    # print('stop control arm thread')

def visualize_sdf(sdf):
    vertices, faces, _, _ = marching_cubes(sdf, level=0.05)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    with open('./config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    rw = RealEnviroment(config)
    rw.reset()
    step = 0
    rw.execute_action([0, 0, 0.4, 0, 0, 0, 1, 0], wait=False)
    import time
    time.sleep(3)
    try:
        sdf = - rw.get_sdf_voxels(resolution=0.02)
        visualize_sdf(sdf)
    except KeyboardInterrupt:
        pass
    rw.stop()