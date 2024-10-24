import time
import random
import pyrealsense2 as rs

import numpy as np
from typing import Union

class Sensor:
    def __init__(self):
        pass
    def start(self):
        pass
    def close(self):
        pass
    def read(self)->Union[dict, None]:
        pass

class RealSense(Sensor):
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
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        h = config['height']
        w = config['width']
        fps = config['fps']
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)        
        self.pipeline_started = False

    def start(self):
        try:
            self.pipeline.start(self.config)
            self.pipeline_started = True
            print("RealSense sensor started.")
        except Exception as e:
            print("Failed to start RealSense sensor.")
            print(e)

    def read(self):
        if self.pipeline_started:
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as e:
                print("Failed to read RealSense sensor.")
                print(e)
                return None
            
            timestamp = time.time()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return {
                "timestamp": timestamp,
                "color": color_image,
                "depth": depth_image
            }
        
        else:
            return None

    def close(self):
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False
            print("RealSense sensor closed.")

def sensor_thread(q, s, sensor: Sensor, max_size: int = 3, freq: float = 1):
    sensor.start()
    tic = time.time()
    while not s.is_set() and sensor.pipeline_started:
        if time.time() - tic < 1/freq:
            time.sleep(1/freq - (time.time() - tic))
            continue
        tic = time.time()
        q.put(sensor.read())
        if q.qsize() > max_size:
            q.get()
    sensor.close()
    return