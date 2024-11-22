import sys
import os
import yaml
import multiprocessing as mp
import threading
import time

import cv2
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from enviroment import RealEnviroment
from utils.sensor import RealSense
from utils.utils import get_points, get_cam_points

actions = [[0, 0, 0.4, 0, 0, 0, 1, 0],
           [-0.2, 0, 0.4, 0, 0, 0, 1, 1],
           [0, -0.2, 0.4, 0, 0, 0, 1, 1],
           [-0.2, -0.2, 0.4, 0, 0, 0, 1, 1]]

def visualize_points(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud', width=800, height=600)
    vis.add_geometry(point_cloud)
    o3d.visualization.draw_geometries([point_cloud])

    return vis, point_cloud

if __name__ == "__main__":
    with open('./config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    rw = RealEnviroment(config)
    rw.reset()
    # rw.register_keypoints()
    step = 0
    rw.execute_action(actions[2], wait=True)

    vis = None
    point_cloud = None
    while True:
        try:
            observation = rw.observe()
            projected = observation['projected']
            mask = observation['mask']
            points = observation['points'].reshape(-1, 3)
            colors = observation['rgb'].reshape(-1, 3)
            # visualize points by Open3D
            if vis is None:
                vis, point_cloud = visualize_points(points, colors)
            else:
                point_cloud.points = o3d.utility.Vector3dVector(points)
                point_cloud.colors = o3d.utility.Vector3dVector(colors/255.0)
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()
        except KeyboardInterrupt:
            break
    rw.execute_action([0, 0, 0.4, 0, 0, 0, 1, 0], wait=True)
    rw.stop()
    # rs = RealSense(config['realsense'])
    # rs.start()
    # time.sleep(3)
    # vis = None
    # while True:
    #     try:
    #         data = None
    #         while data is None:
    #             data = rs.get()
    #         bgr = data['color']
    #         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #         depth = data['depth']
    #         extrinsics = data['extrinsics']
    #         instrinsics = data['instrinsics']
    #         points = get_points(depth, instrinsics, extrinsics).reshape(-1, 3)
    #         colors = rgb.reshape(-1, 3)
    #         if vis is None:
    #             vis, point_cloud = visualize_points(points, colors)
    #             # save point cloud
    #             o3d.io.write_point_cloud('./data/point_cloud.ply', point_cloud)
    #         else:
    #             point_cloud.points = o3d.utility.Vector3dVector(points)
    #             point_cloud.colors = o3d.utility.Vector3dVector(colors/255.0)
    #             vis.update_geometry(point_cloud)
    #             vis.poll_events()
    #             vis.update_renderer()
    #     except KeyboardInterrupt:
    #         break
    # vis = None
    # data = None
    # while data is None:
    #     data = rs.get()
    # bgr = data['color']
    # cv2.imwrite('./data/ex.png', bgr)
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # depth = data['depth']
    # extrinsics = data['extrinsics']
    # instrinsics = data['instrinsics']
    # points = get_points(depth, instrinsics, extrinsics).reshape(-1, 3)
    # # points = get_cam_points(depth, instrinsics).reshape(-1, 3)
    # colors = rgb.reshape(-1, 3)
    # if vis is None:
    #     vis, point_cloud = visualize_points(points, colors)
    #     # save points
    #     o3d.io.write_point_cloud("./data/point_cloud.ply", point_cloud)
    # rs.stop()