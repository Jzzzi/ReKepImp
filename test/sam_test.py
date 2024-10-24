from threading import Event, Thread
from queue import Queue
import yaml

import cv2
import torch
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

from utils.sensor import RealSense, sensor_thread
from utils.cameras import get_cam_points, get_sam_mask

def _points_vis(points, rgb):
    # filter the points with depth > 0 and < 1.5
    mask = (points[:, :, 2] > 0) & (points[:, :, 2] < 10)
    points = points[mask].reshape(-1, 3)
    rgb = rgb[mask].reshape(-1, 3).astype(np.float64) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])

def main():
    with open('config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    realsense_config = config['realsense']
    # rs = RealSense(realsense_config)
    # realsense_stream = Queue()
    # realsense_stop = Event()
    # realsense_thread = Thread(target=sensor_thread, args=(realsense_stream, realsense_stop, rs, 3, 60))
    # realsense_thread.start()

    # keypoint_proposer_config = config['keypoint_proposer']
    # keypoint_proposer = KeypointProposer(keypoint_proposer_config)
    bgr = cv2.imread('data/color_0.jpg')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    masks = get_sam_mask(rgb)
    image = rgb.copy()
    for i in range(len(masks)):
        color_mask = np.zeros_like(image)
        color_mask[masks[i]['segmentation']] = [0, 255, 0]
        tmp = cv2.addWeighted(image, 1, color_mask, 0.5, 0)
        plt.imshow(tmp)
        plt.show()

if __name__ == "__main__":
    main()