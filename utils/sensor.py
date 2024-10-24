import time
import random
from typing import Union
import threading

import pyrealsense2 as rs
import numpy as np

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
        self._config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)        
        self._pipeline_started = False
        
        self._queue = threading.Queue()
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        # start the sensor in the thread
        print("Starting RealSense sensor...")
        self._thread = threading.Thread(target=self._run, args=(None, self._stop_event))
        self._thread.start()

    def get(self):
        return self._queue.get()
    
    def stop(self):
        self._stop_event.set()
        self._pipeline.stop()
        self._pipeline_started = False
        print("RealSense sensor stopped.")
    
    def _run(self, stop_event):
        try:
            self._pipeline.start(self._config)
            self._pipeline_started = True
            print("RealSense sensor started.")
        except Exception as e:
            print("Failed to start RealSense sensor.")
            print(e)

        while not stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames()
            except Exception as e:
                print("Failed to read RealSense sensor.")
                print(e)
                return
            
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
                "depth": depth_image
            })

