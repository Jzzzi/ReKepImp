import yaml
import time
import sys
import os

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sensor import RealSense

with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

rs = RealSense(config['realsense'])
rs.start()

config = config['realsense']['extrinsic_calibrator']
instrinsics = config['instrinsics']
length = config['tag_length']
instrinsics = np.array(instrinsics).reshape(3, 3)
distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
distortion = np.array(distortion)
obj_points = np.array([
            [+length / 2, -length / 2, 0],
            [+length / 2, +length / 2, 0],
            [+length / 2, -length / 2, 0],
            [-length / 2, -length / 2, 0]
        ])
if config['tag_family'] == 'DICT_APRILTAG_16H5':
    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
else:
    raise ValueError("Invalid tag family")

params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict, params)

while True:
    data = None
    while data is None:
        data = rs.get()
    color = data['color']
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img)
    corners = np.array(corners)
    ids = np.array(ids)

    if ids is not None and len(corners) > 0:
        
        # 定义 3D 世界坐标系的标记角点 (以米为单位)
        obj_points = np.array([
            [-length / 2, +length / 2, 0],
            [+length / 2, +length / 2, 0],
            [+length / 2, -length / 2, 0],
            [-length / 2, -length / 2, 0]
        ])

        corners = corners.reshape(4, 2)
        success, rvec, tvec = cv2.solvePnP(obj_points, corners, instrinsics, distortion)
        
        rmat, _ = cv2.Rodrigues(rvec)
        c2w = np.eye(4)
        c2w[:3, :3] = rmat.T  # 旋转矩阵的转置
        c2w[:3, 3] = -rmat.T @ tvec.flatten()  # 平移向量的逆变换
        # clear the screen
        print("\033[H\033[J")
        print(c2w)
        color = cv2.aruco.drawDetectedMarkers(color, corners.reshape(1,1,4,2), ids)
        color = cv2.drawFrameAxes(color, instrinsics, distortion, rvec, tvec, length)
        cv2.imshow("Pose Estimation", color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("\033[H\033[J")
        print("No markers detected.")
        cv2.imshow("Pose Estimation", color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
rs.stop()