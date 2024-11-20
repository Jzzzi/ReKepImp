import time
import random
from typing import Union
import threading
import queue

import pyrealsense2 as rs
import cv2
import numpy as np

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

class RealSense():
    '''
    Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
    Width:      	640
    Height:     	480
    PPX:        	326.660308837891
    PPY:        	247.326507568359
    Fx:         	386.740875244141
    Fy:         	386.278381347656
    Distortion: 	Inverse Brown Conrady
    Coeffs:     	-0.0556738413870335  	0.0655610412359238  	-0.000339997815899551  	0.000355891766957939  	-0.021365724503994  
    FOV (deg):  	79.2 x 63.69
    '''
    def __init__(self, config):
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        h = config['height']
        w = config['width']
        fps = config['fps']
        self._max_queue_size = config['max_queue_size']
        self._config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)        

        self._extrinsics = None
        self._tag_length = config['extrinsic_calibrator']['tag_length']
        self._instrinsics = np.asanyarray(config['instrinsics']).reshape(3, 3)
        self._distortion = np.asanyarray(config['distortions']).reshape(-1)
        
        self._queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        # start the sensor in the thread
        print(GREEN + "[RealSense]: Starting RealSense sensor..." + RESET)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        time.sleep(1) # wait for auto-exposure to stabilize
        print(GREEN + "[RealSense]: RealSense sensor started." + RESET)

    def get(self):
        '''
        return a dict
        {
            "timestamp": float,
            "color": np.ndarray, [H, W, 3], BGR
            "depth": np.ndarray, [H, W], z16
            "extrinsics": np.ndarray, [4, 4], c2w
        }
        '''
        try:
            return self._queue.get(timeout=1)
        except:
            print(YELLOW + "[RealSense]: Failed to get RealSense data within 1 second." + RESET)
            return None
    
    def stop(self):
        print(GREEN + "[RealSense]: Stopping RealSense sensor..." + RESET)
        self._stop_event.set()
        self._thread.join()
        print(GREEN + "[RealSense]: RealSense sensor stopped." + RESET)    

    def get_instrinsics(self):
        '''
        return instrinsics matrix, np.ndarray, [3, 3]
        '''
        return self._instrinsics
    
    def _run(self):
        try:
            self._pipeline.start(self._config)
        except Exception as e:
            print(RED + "[RealSense]: Failed to start the pipeline." + RESET)
            print(e)

        self._extrinsics = self._get_extrinsic()
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames()
            except Exception as e:
                print(e)
                continue
            
            timestamp = time.time()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            self._queue.put({
                "timestamp": timestamp,
                "color": color_image,
                "depth": depth_image,
                "extrinsics": self._extrinsics,
                "instrinsics": self._instrinsics,
            })

            if self._queue.qsize() > self._max_queue_size:
                self._queue.get()

        self._pipeline.stop()

    def _get_extrinsic(self):
        try:
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        except Exception as e:
            print(RED + "[RealSense]: Failed to capture frames for extrinsic calibration." + RESET)
            return None

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(tag_dict, params)

        corners, ids, _ = detector.detectMarkers(gray_image)
        if ids is None or len(corners) == 0:
            print(YELLOW + "[RealSense]: No markers detected for extrinsic calibration." + RESET)
            return None

        obj_points = np.array([
            [-self._tag_length / 2, self._tag_length / 2, 0],
            [self._tag_length / 2, self._tag_length / 2, 0],
            [self._tag_length / 2, -self._tag_length / 2, 0],
            [-self._tag_length / 2, -self._tag_length / 2, 0]
        ], dtype=np.float32)

        corners = corners[0].reshape(4, 2)
        success, rvec, tvec = cv2.solvePnP(obj_points, corners, self._instrinsics, self._distortion)

        if not success:
            print(RED + "[RealSense]: Failed to estimate pose." + RESET)
            return None

        # 将 rvec 转换为旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)

        # 计算 c2w 变换矩阵
        c2w = np.eye(4)
        c2w[:3, :3] = rmat.T  # 旋转矩阵的转置
        c2w[:3, 3] = -rmat.T @ tvec.flatten()  # 平移向量的逆变换
        print(GREEN + "[RealSense]: Successfully estimated extrinsic matrix." + RESET)
        print(GREEN + "[RealSense]: Extrinsic matrix:" + RESET)
        print(GREEN + str(c2w) + RESET)

        return c2w